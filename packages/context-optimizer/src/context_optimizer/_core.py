"""Owns: shared pure functions used across multiple tiers.

render_content — convert a message's content field to a plain string.
content_hash   — stable SHA-256 fingerprint of a message list prefix.
apply_summary  — replace a message prefix with a system-prompt summary block.

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

    Returns (truncated_messages, updated_system). System type shape
    matches the input (str -> str, list -> list, None -> str).
    """
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
