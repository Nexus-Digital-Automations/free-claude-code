"""Request utility functions for API route handlers.

Owns: token counting for API requests using configurable per-provider tokenizer.

Does NOT own: compaction decisions, provider selection, or request routing.
Called by: api/routes.py.
Calls: tiktoken, and optionally HuggingFace tokenizers library.
"""

import json

import tiktoken
from loguru import logger

from providers.common import get_block_attr

_ENCODER_CACHE: dict[str, object] = {}

__all__ = ["get_token_count"]


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


def get_token_count(
    messages: list,
    system: str | list | None = None,
    tools: list | None = None,
    tokenizer_name: str = "cl100k_base",
) -> int:
    """Estimate token count for a request.

    Uses tokenizer_name to encode — pass settings.context_tokenizer_model
    to match the actual provider's vocabulary. Falls back to cl100k_base if
    the named tokenizer cannot be loaded.
    """
    enc = _get_encoder(tokenizer_name)
    total_tokens = 0

    if system:
        if isinstance(system, str):
            total_tokens += len(enc.encode(system))  # type: ignore[union-attr]
        elif isinstance(system, list):
            for block in system:
                text = get_block_attr(block, "text", "")
                if text:
                    total_tokens += len(enc.encode(str(text)))  # type: ignore[union-attr]
        total_tokens += 4  # system block formatting overhead

    for msg in messages:
        if isinstance(msg.content, str):
            total_tokens += len(enc.encode(msg.content))  # type: ignore[union-attr]
        elif isinstance(msg.content, list):
            for block in msg.content:
                b_type = get_block_attr(block, "type") or None

                if b_type == "text":
                    text = get_block_attr(block, "text", "")
                    total_tokens += len(enc.encode(str(text)))  # type: ignore[union-attr]
                elif b_type == "thinking":
                    thinking = get_block_attr(block, "thinking", "")
                    total_tokens += len(enc.encode(str(thinking)))  # type: ignore[union-attr]
                elif b_type == "tool_use":
                    name = get_block_attr(block, "name", "")
                    inp = get_block_attr(block, "input", {})
                    block_id = get_block_attr(block, "id", "")
                    total_tokens += len(enc.encode(str(name)))  # type: ignore[union-attr]
                    total_tokens += len(enc.encode(json.dumps(inp)))  # type: ignore[union-attr]
                    total_tokens += len(enc.encode(str(block_id)))  # type: ignore[union-attr]
                    total_tokens += 15
                elif b_type == "image":
                    source = get_block_attr(block, "source")
                    if isinstance(source, dict):
                        data = source.get("data") or source.get("base64") or ""
                        if data:
                            total_tokens += max(85, len(data) // 3000)
                        else:
                            total_tokens += 765
                    else:
                        total_tokens += 765
                elif b_type == "tool_result":
                    content = get_block_attr(block, "content", "")
                    tool_use_id = get_block_attr(block, "tool_use_id", "")
                    if isinstance(content, str):
                        total_tokens += len(enc.encode(content))  # type: ignore[union-attr]
                    else:
                        total_tokens += len(enc.encode(json.dumps(content)))  # type: ignore[union-attr]
                    total_tokens += len(enc.encode(str(tool_use_id)))  # type: ignore[union-attr]
                    total_tokens += 8
                else:
                    logger.debug(
                        "Unexpected block type %r, falling back to json/str encoding",
                        b_type,
                    )
                    try:
                        total_tokens += len(enc.encode(json.dumps(block)))  # type: ignore[union-attr]
                    except (TypeError, ValueError):
                        total_tokens += len(enc.encode(str(block)))  # type: ignore[union-attr]

    if tools:
        for tool in tools:
            tool_str = (
                tool.name + (tool.description or "") + json.dumps(tool.input_schema)
            )
            total_tokens += len(enc.encode(tool_str))  # type: ignore[union-attr]

    total_tokens += len(messages) * 4
    if tools:
        total_tokens += len(tools) * 5

    return max(1, total_tokens)
