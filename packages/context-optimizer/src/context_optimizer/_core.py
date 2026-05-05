"""Owns: shared pure functions used across multiple tiers and the block tower.

render_content    — convert a message's content field to a plain string.
content_hash      — stable SHA-256 fingerprint of a message list prefix.
strip_ansi        — remove ANSI escape codes from text.

Does NOT own: tier logic, caching policy, or LLM calls.
Called by: tiers/tier0.py, block_tower/session_key.py, block_tower/selector.py.
Calls: stdlib only (hashlib, re).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b[A-Z]")


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

    Used by block_tower/session_key.py for the session-scoping key and by
    block_tower/selector.py for the inclusion-decision cache. Hashing
    rendered text (not raw dicts) means minor whitespace differences
    (e.g. ANSI-stripped vs not) don't split logically equivalent entries.
    """
    h = hashlib.sha256()
    for msg in messages:
        h.update(msg.get("role", "").encode())
        h.update(b"\x1f")
        h.update(render_content(msg.get("content", "")).encode())
        h.update(b"\x1e")
    return h.hexdigest()


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)
