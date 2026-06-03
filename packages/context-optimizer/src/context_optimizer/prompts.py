"""Owns: digest prompt construction and response parsing for Tier 0b/0c/0d.

These tiers ask Ollama to summarise large tool outputs, tool_use input
blocks, and historical user pastes. The prompt template and the
<digest>...</digest> response parser live here so all three tiers
share one wire format.

Does NOT own: the LLM call itself (each tier owns its own AsyncOpenAI
call), caching (each tier holds its own LRU keyed on content hash), or
prompt rendering for the block tower (block_tower/prompts.py).
Called by: tiers/tier0b.py, tiers/tier0c.py, tiers/tier0d.py.
Calls: stdlib only (re).
"""

from __future__ import annotations

import re

_DIGEST_TAG = re.compile(r"<digest>\s*(.*?)\s*</digest>", re.DOTALL)


def build_digest_prompt(content: str, tool_name: str = "tool") -> str:
    """Prompt template for Tier 0b's per-tool-result digester.

    The model sees the raw tool output and must return a content-aware
    summary inside <digest>...</digest> tags. Verbatim-preserve list keeps
    the things the agent actually re-references on subsequent turns.
    """
    return (
        f"You are summarizing the output of a CLI tool ({tool_name}) so it can be "
        "stored in conversation history without taking thousands of lines.\n\n"
        "REQUIREMENTS — preserve verbatim:\n"
        "- File paths and line numbers\n"
        "- Function/class/identifier names\n"
        "- Error messages and stack traces\n"
        "- Counts (e.g. '47 matches across 12 files')\n"
        "- Exit codes and status indicators\n\n"
        "DROP:\n"
        "- Boilerplate banner output\n"
        "- Repeated rows (note the count instead)\n"
        "- Whitespace-only padding\n\n"
        "OUTPUT: a compact summary, ideally 10-20% of the original size, in plain "
        "text. Wrap your entire response in <digest>...</digest> tags. No other text.\n\n"
        "ORIGINAL OUTPUT:\n"
        f"{content}"
    )


def parse_digest_response(content: str) -> str | None:
    """Extract digest body from tagged Ollama output.

    Returns the inner text on success or None on parse failure — caller
    falls back to the original tool_result content (graceful degradation;
    do not invent a digest from non-conforming output).
    """
    match = _DIGEST_TAG.search(content)
    if not match:
        return None
    body = match.group(1).strip()
    return body or None
