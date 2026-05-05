"""Owns: shared pure functions used across multiple tiers.

render_content    — convert a message's content field to a plain string.
content_hash      — stable SHA-256 fingerprint of a message list prefix.
safe_split_index  — advance a split past tool_result-leading user turns so
                    the suffix never starts with an orphaned tool response.
apply_summary     — replace a message prefix with a system-prompt summary block.

Does NOT own: tier logic, caching policy, or LLM calls.
Called by: cache.py, prompts.py, tiers/tier0.py, tiers/tier2.py.
Calls: stdlib only (hashlib, re).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b[A-Z]")

_SUMMARY_PREFIX = (
    "Earlier conversation (compacted by context-optimizer — "
    "treat as background context, not as new user instructions):\n\n"
)


def render_content(content: Any) -> str:
    """Flatten a message content field to a plain string for hashing/prompts."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        btype = block.get("type") if isinstance(block, dict) else None
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            parts.append(f"[tool_use: {block.get('name', '?')}]")
        elif btype == "tool_result":
            r = block.get("content", "")
            if isinstance(r, list):
                r = " ".join(b.get("text", "") for b in r if isinstance(b, dict))
            parts.append(f"[tool_result: {str(r)[:500]}]")
        elif btype == "image":
            parts.append("[image]")
        # thinking blocks intentionally skipped — stale reasoning is noise
    return " ".join(p for p in parts if p)


def content_hash(messages: list[dict]) -> str:
    """Stable SHA-256 key for a prefix of a message list.

    Used by PrefixCache. Key is based on rendered text so minor
    whitespace differences (e.g. ANSI-stripped vs not) don't split
    logically equivalent cache entries.
    """
    h = hashlib.sha256()
    for msg in messages:
        h.update(msg.get("role", "").encode())
        h.update(b"\x1f")
        h.update(render_content(msg.get("content", "")).encode())
        h.update(b"\x1e")
    return h.hexdigest()


def _starts_with_tool_result(message: dict) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "tool_result"
        for b in content
    )


def safe_split_index(messages: list[dict], split_index: int) -> int:
    """Advance split_index past user turns that contain tool_result blocks.

    Anthropic encodes a tool exchange as two messages: an assistant turn
    with a `tool_use` block and the next user turn with a matching
    `tool_result` block. apply_summary's slice (`messages[split_index:]`)
    keeps the suffix verbatim — if split_index lands on the user
    tool_result while the assistant tool_use sits in the discarded
    prefix, the surviving tool_result becomes an orphan that
    OpenAI-compatible APIs (DeepSeek) reject with HTTP 400 ("Messages
    with role 'tool' must be a response to a preceding message with
    'tool_calls'").

    Walking forward drops the orphan along with its (already-discarded)
    pair. Pathological case — every remaining message is a tool_result
    user turn — pushes the index to len(messages); callers should treat
    that as "do not compact" rather than emitting an empty suffix.
    Counterpart: providers/common/message_converter.py also drops
    orphan tool messages as a defensive second layer.
    """
    n = len(messages)
    while split_index < n and _starts_with_tool_result(messages[split_index]):
        split_index += 1
    return split_index


def apply_summary(
    messages: list[dict],
    split_index: int,
    summary: str,
    system: str | list | None,
) -> tuple[list[dict], str | list | None]:
    """Replace messages[:split_index] with a system-prompt summary block.

    WHY system-prompt placement: a synthetic user message can be
    misread by the model as a fresh user instruction. Placing it in
    system signals unambiguously that it is background context.

    split_index is snapped via safe_split_index so the surviving suffix
    never opens with an orphaned tool_result.

    Returns (truncated_messages, updated_system). System type shape
    matches the input (str -> str, list -> list, None -> str).
    # @stable — external callers (cache, tier2) depend on this signature.
    """
    split_index = safe_split_index(messages, split_index)

    block_text = _SUMMARY_PREFIX + summary

    if system is None:
        new_system: Any = block_text
    elif isinstance(system, str):
        new_system = block_text + "\n\n---\n\n" + system
    elif isinstance(system, list):
        new_system = [{"type": "text", "text": block_text}, *system]
    else:
        new_system = system  # unknown shape — leave untouched

    return messages[split_index:], new_system


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)
