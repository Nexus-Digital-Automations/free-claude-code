"""Owns: deterministic NLP cleanup — always-on, free, no LLM calls.

Three sub-operations (in order):
  1. Strip ANSI escape sequences from text and tool-result content.
  2. Deduplicate identical tool-result content (content-hash identity).
  3. Truncate tool-result outputs that exceed tier0_max_lines.

Does NOT own: thinking-block removal (tier1.py) or LLM summarization.
Called by: optimizer.py.
"""

from __future__ import annotations

import hashlib

from .._core import strip_ansi

_DEDUPE_PLACEHOLDER = "[identical to earlier tool result -- omitted]"


def apply(
    messages: list[dict],
    max_lines: int = 200,
    head_lines: int = 50,
    tail_lines: int = 50,
) -> list[dict]:
    """Return NLP-cleaned copy of messages. Returns input unchanged if nothing changed.

    head_lines/tail_lines bound the head+tail kept when truncating tool results
    longer than max_lines. Sourced from ContextOptimizerSettings via optimizer.py.
    """
    msgs = _strip_ansi(messages)
    msgs = _dedupe_tool_results(msgs)
    msgs = _truncate_long_outputs(msgs, max_lines, head_lines, tail_lines)
    return msgs


# ---- sub-operations ----

def _strip_ansi(messages: list[dict]) -> list[dict]:
    result = []
    changed = False
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            cleaned = strip_ansi(content)
            if cleaned != content:
                msg = {**msg, "content": cleaned}
                changed = True
        elif isinstance(content, list):
            new_blocks, block_changed = _strip_ansi_blocks(content)
            if block_changed:
                msg = {**msg, "content": new_blocks}
                changed = True
        result.append(msg)
    return result if changed else messages


def _strip_ansi_blocks(blocks: list[dict]) -> tuple[list[dict], bool]:
    result = []
    changed = False
    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            cleaned = strip_ansi(block.get("text", ""))
            if cleaned != block.get("text", ""):
                block = {**block, "text": cleaned}
                changed = True
        elif btype == "tool_result" and isinstance(block.get("content"), str):
            cleaned = strip_ansi(block["content"])
            if cleaned != block["content"]:
                block = {**block, "content": cleaned}
                changed = True
        result.append(block)
    return result, changed


def _dedupe_tool_results(messages: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    changed = False
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_blocks = []
            msg_changed = False
            for block in content:
                if block.get("type") == "tool_result":
                    raw = block.get("content", "") or ""
                    text = raw if isinstance(raw, str) else str(raw)
                    h = hashlib.sha256(text.encode()).hexdigest()
                    if text.strip() and h in seen:
                        block = {**block, "content": _DEDUPE_PLACEHOLDER}
                        msg_changed = True
                    else:
                        seen.add(h)
                new_blocks.append(block)
            if msg_changed:
                msg = {**msg, "content": new_blocks}
                changed = True
        result.append(msg)
    return result if changed else messages


def _truncate_long_outputs(
    messages: list[dict], max_lines: int, head_lines: int, tail_lines: int,
) -> list[dict]:
    result = []
    changed = False
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_blocks = []
            msg_changed = False
            for block in content:
                if block.get("type") == "tool_result":
                    raw = block.get("content", "") or ""
                    if isinstance(raw, str):
                        lines = raw.split("\n")
                        if len(lines) > max_lines:
                            omitted = len(lines) - head_lines - tail_lines
                            truncated = (
                                "\n".join(lines[:head_lines])
                                + f"\n... [{omitted} lines omitted] ...\n"
                                + "\n".join(lines[-tail_lines:])
                            )
                            block = {**block, "content": truncated}
                            msg_changed = True
                new_blocks.append(block)
            if msg_changed:
                msg = {**msg, "content": new_blocks}
                changed = True
        result.append(msg)
    return result if changed else messages
