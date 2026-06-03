"""Owns: deterministic NLP cleanup — always-on, free, no LLM calls.

Four sub-operations (in order):
  1. Strip ANSI escape sequences from text and tool-result content.
  2. Deduplicate identical tool-result content (content-hash identity).
  3. Deduplicate repeated <system-reminder> blocks across messages.
  4. Truncate tool-result outputs that exceed tier0_max_lines.

Does NOT own: thinking-block removal (tier1.py) or LLM summarization.
Called by: optimizer.py.
"""

from __future__ import annotations

import hashlib
import re

from .._core import strip_ansi

_DEDUPE_PLACEHOLDER = "[identical to earlier tool result -- omitted]"
_REMINDER_DEDUPE_PLACEHOLDER = "<system-reminder>[elided — identical to earlier]</system-reminder>"
_REMINDER_RE = re.compile(r"<system-reminder>(.*?)</system-reminder>", re.DOTALL)


def apply(
    messages: list[dict],
    max_lines: int = 120,
    head_lines: int = 60,
    tail_lines: int = 60,
) -> list[dict]:
    """Return NLP-cleaned copy of messages. Returns input unchanged if nothing changed.

    head_lines/tail_lines bound the head+tail kept when truncating tool results
    longer than max_lines. Sourced from ContextOptimizerSettings via optimizer.py.
    """
    msgs = _strip_ansi(messages)
    msgs = _dedupe_tool_results(msgs)
    msgs = _dedupe_system_reminders(msgs)
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


def _dedupe_system_reminders(messages: list[dict]) -> list[dict]:
    """Replace repeat <system-reminder> blocks with a marker after the first.

    PreToolUse and UserPromptSubmit hooks re-inject the same reminder text
    on every turn (CODE STANDARDS, PROTOCOL CHECKPOINT, etc.). The first
    copy carries the signal; the rest is pure padding the model already
    saw. Lossless because the marker preserves the structural <system-reminder>
    container so any downstream parser still sees a well-formed block.
    """
    seen: set[str] = set()
    result = []
    changed = False
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            new_text, msg_changed = _dedupe_reminders_in_text(content, seen)
            if msg_changed:
                msg = {**msg, "content": new_text}
                changed = True
        elif isinstance(content, list):
            new_blocks = []
            msg_changed = False
            for block in content:
                if block.get("type") == "text":
                    new_text, block_changed = _dedupe_reminders_in_text(
                        block.get("text", ""), seen,
                    )
                    if block_changed:
                        block = {**block, "text": new_text}
                        msg_changed = True
                new_blocks.append(block)
            if msg_changed:
                msg = {**msg, "content": new_blocks}
                changed = True
        result.append(msg)
    return result if changed else messages


def _dedupe_reminders_in_text(text: str, seen: set[str]) -> tuple[str, bool]:
    """Replace already-seen reminder bodies with the marker. Mutates `seen`."""
    changed = False

    def _sub(match: re.Match) -> str:
        nonlocal changed
        body = match.group(1)
        h = hashlib.sha256(body.encode()).hexdigest()
        if h in seen:
            changed = True
            return _REMINDER_DEDUPE_PLACEHOLDER
        seen.add(h)
        return match.group(0)

    new_text = _REMINDER_RE.sub(_sub, text)
    return new_text, changed


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
