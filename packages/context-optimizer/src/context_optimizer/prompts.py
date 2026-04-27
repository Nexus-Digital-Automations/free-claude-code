"""Owns: compaction prompt construction and LLM response parsing.

Does NOT own: LLM calls (tier2.py owns those) or caching (cache.py).
Called by: tiers/tier2.py.
Calls: _core.render_content.
"""

from __future__ import annotations

import re

from ._core import render_content

_SPLIT_TAG = re.compile(r"<split_index>\s*(\d+)\s*</split_index>")
_SUMMARY_TAG = re.compile(r"<summary>\s*(.*?)\s*</summary>", re.DOTALL)


def build_prompt(messages: list[dict], render_preview_chars: int = 2_000) -> str:
    lines = []
    for i, msg in enumerate(messages):
        text = render_content(msg.get("content", ""))
        if len(text) > render_preview_chars:
            text = text[:render_preview_chars] + " ... [truncated]"
        lines.append(f"[{i}] {msg.get('role', '?')}: {text}")
    n = len(messages)
    return (
        "You are a conversation compactor for a coding assistant. Compress this "
        "history to reduce token usage while preserving everything the assistant "
        "needs to keep working without losing context.\n\n"
        "STEP 1 - Choose split_index. Messages BEFORE split_index get replaced by "
        "your summary; messages FROM split_index onward are kept verbatim. Pick a "
        "natural breakpoint where messages[split_index] is a user message that "
        "starts a new sub-task or topic.\n\n"
        "STEP 2 - Write a summary of the messages before split_index. PRESERVE "
        "VERBATIM (do not paraphrase):\n"
        "  - Exact file paths, function/class names, variable names\n"
        "  - Error messages, stack traces, and command outputs the user reacted to\n"
        "  - User-stated goals, constraints, and explicit decisions\n"
        "  - Open questions, unresolved tasks, and TODOs\n"
        "  - Specific numbers, IDs, URLs, line numbers\n"
        "Compress freely: small talk, redundant tool output, intermediate reasoning "
        "the assistant has already concluded.\n\n"
        "EXAMPLE (10-message conversation about a bug fix):\n"
        "  Good: <split_index>6</split_index> - message 6 is the user saying "
        '"now let\'s add tests". Messages 0-5 (the original bugfix) get summarized; '
        "6-9 (the test work in progress) stay verbatim.\n"
        "  Bad: <split_index>9</split_index> - too late, almost nothing is kept.\n\n"
        "Output ONLY these tags, nothing else:\n"
        "<split_index>NUMBER</split_index>\n"
        "<summary>SUMMARY TEXT</summary>\n\n"
        f"Conversation has {n} messages. split_index must be between 4 and {max(4, n - 2)}.\n\n"
        "CONVERSATION:\n" + "\n\n".join(lines)
    )


def parse_response(content: str, num_messages: int) -> tuple[int, str] | None:
    """Parse LLM compaction output. Returns (split_index, summary) or None."""
    idx_match = _SPLIT_TAG.search(content)
    sum_match = _SUMMARY_TAG.search(content)
    if not idx_match or not sum_match:
        return None
    split_index = int(idx_match.group(1))
    if split_index < 4 or split_index >= num_messages - 1:
        return None
    summary = sum_match.group(1).strip()
    return (split_index, summary) if summary else None
