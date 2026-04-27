"""Owns: token estimation for message lists using tiktoken cl100k_base.

Does NOT own: the decision of whether to compact (that lives in optimizer.py).
Called by: optimizer.py.
Calls: tiktoken (bundled dep).
"""

from __future__ import annotations

import json

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(
    messages: list[dict],
    system: str | list | None = None,
    tools: list | None = None,
) -> int:
    """Estimate total tokens for a request payload.

    Approximate — uses cl100k_base regardless of model. Good enough
    for threshold decisions; callers should not treat the result as
    exact billing figures.
    """
    total = 0

    if system:
        if isinstance(system, str):
            total += len(_ENCODER.encode(system))
        elif isinstance(system, list):
            for block in system:
                text = block.get("text", "") if isinstance(block, dict) else str(block)
                total += len(_ENCODER.encode(str(text)))
        total += 4  # system block framing

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(_ENCODER.encode(content))
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    total += len(_ENCODER.encode(str(block)))
                    continue
                btype = block.get("type")
                if btype == "text":
                    total += len(_ENCODER.encode(str(block.get("text", ""))))
                elif btype == "thinking":
                    total += len(_ENCODER.encode(str(block.get("thinking", ""))))
                elif btype == "tool_use":
                    total += len(_ENCODER.encode(str(block.get("name", ""))))
                    total += len(_ENCODER.encode(json.dumps(block.get("input", {}))))
                    total += len(_ENCODER.encode(str(block.get("id", ""))))
                    total += 15
                elif btype == "tool_result":
                    raw = block.get("content", "") or ""
                    if isinstance(raw, list):
                        raw = " ".join(
                            b.get("text", "") for b in raw if isinstance(b, dict)
                        )
                    total += len(_ENCODER.encode(str(raw)))
                    total += 8
                else:
                    try:
                        total += len(_ENCODER.encode(json.dumps(block)))
                    except (TypeError, ValueError):
                        total += len(_ENCODER.encode(str(block)))
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
            total += len(_ENCODER.encode(tool_str))
        total += len(tools) * 5

    total += len(messages) * 4
    return total
