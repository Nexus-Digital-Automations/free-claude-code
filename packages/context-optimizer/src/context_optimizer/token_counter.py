"""Owns: token estimation for message lists.

Does NOT own: the decision of whether to compact (that lives in optimizer.py).
Called by: optimizer.py.
Calls: tiktoken (bundled dep) or tokenizers (HuggingFace, optional).
"""

from __future__ import annotations

import json

import tiktoken
from loguru import logger

_ENCODER_CACHE: dict[str, object] = {}


class _HFEncoder:
    """Wraps tokenizers.Tokenizer to expose tiktoken-compatible encode()."""

    def __init__(self, tok: object) -> None:
        self._tok = tok

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids  # type: ignore[attr-defined]


def _get_encoder(tokenizer_name: str) -> object:
    """Return a cached encoder for tokenizer_name.

    Supports tiktoken encoding names (e.g. 'cl100k_base') and HuggingFace
    model IDs (e.g. 'deepseek-ai/DeepSeek-V3'). Falls back to cl100k_base
    if a HuggingFace tokenizer cannot be loaded.
    """
    if tokenizer_name in _ENCODER_CACHE:
        return _ENCODER_CACHE[tokenizer_name]

    encoder: object
    if "/" in tokenizer_name:
        try:
            from tokenizers import Tokenizer  # type: ignore[import]

            tok = Tokenizer.from_pretrained(tokenizer_name)
            encoder = _HFEncoder(tok)
            logger.info("TOKEN_COUNTER: loaded HuggingFace tokenizer {!r}", tokenizer_name)
        except Exception as exc:
            logger.warning(
                "TOKEN_COUNTER: failed to load {!r} ({}); falling back to cl100k_base",
                tokenizer_name,
                exc,
            )
            encoder = tiktoken.get_encoding("cl100k_base")
    else:
        encoder = tiktoken.get_encoding(tokenizer_name)

    _ENCODER_CACHE[tokenizer_name] = encoder
    return encoder


def count_tokens(
    messages: list[dict],
    system: str | list | None = None,
    tools: list | None = None,
    tokenizer_name: str = "cl100k_base",
) -> int:
    """Estimate total tokens for a request payload.

    Approximate — accuracy depends on tokenizer_name matching the actual
    model. For threshold decisions; callers should not treat the result as
    exact billing figures.
    """
    enc = _get_encoder(tokenizer_name)
    total = 0

    if system:
        if isinstance(system, str):
            total += len(enc.encode(system))  # type: ignore[union-attr]
        elif isinstance(system, list):
            for block in system:
                text = block.get("text", "") if isinstance(block, dict) else str(block)
                total += len(enc.encode(str(text)))  # type: ignore[union-attr]
        total += 4  # system block framing

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(enc.encode(content))  # type: ignore[union-attr]
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    total += len(enc.encode(str(block)))  # type: ignore[union-attr]
                    continue
                btype = block.get("type")
                if btype == "text":
                    total += len(enc.encode(str(block.get("text", ""))))  # type: ignore[union-attr]
                elif btype == "thinking":
                    total += len(enc.encode(str(block.get("thinking", ""))))  # type: ignore[union-attr]
                elif btype == "tool_use":
                    total += len(enc.encode(str(block.get("name", ""))))  # type: ignore[union-attr]
                    total += len(enc.encode(json.dumps(block.get("input", {}))))  # type: ignore[union-attr]
                    total += len(enc.encode(str(block.get("id", ""))))  # type: ignore[union-attr]
                    total += 15
                elif btype == "tool_result":
                    raw = block.get("content", "") or ""
                    if isinstance(raw, list):
                        raw = " ".join(
                            b.get("text", "") for b in raw if isinstance(b, dict)
                        )
                    total += len(enc.encode(str(raw)))  # type: ignore[union-attr]
                    total += 8
                else:
                    try:
                        total += len(enc.encode(json.dumps(block)))  # type: ignore[union-attr]
                    except (TypeError, ValueError):
                        total += len(enc.encode(str(block)))  # type: ignore[union-attr]
        total += 4  # per-message framing

    if tools:
        for tool in tools:
            if isinstance(tool, dict):
                tool_str = (
                    tool.get("name", "")
                    + (tool.get("description") or "")
                    + json.dumps(tool.get("input_schema", {}))
                )
            else:
                tool_str = str(tool)
            total += len(enc.encode(tool_str))  # type: ignore[union-attr]
        total += len(tools) * 5

    total += len(messages) * 4
    return total
