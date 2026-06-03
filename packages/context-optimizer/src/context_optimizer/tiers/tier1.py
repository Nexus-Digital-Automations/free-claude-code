"""Owns: stripping stale thinking blocks from old assistant turns.

Free, always-on. Old reasoning has no bearing on current quality, but every
thinking block costs tokens on every subsequent request. Only the last
keep_last_n assistant turns retain their thinking blocks.

Does NOT own: NLP cleanup (tier0) or LLM summarization (tier2).
Called by: optimizer.py.
"""

from __future__ import annotations

from loguru import logger


def apply(messages: list[dict], keep_last_n: int) -> list[dict]:
    """Strip thinking blocks from all but the last keep_last_n assistant turns.

    keep_last_n=0 strips every thinking block. The strip-all path matters
    for prefix-cache stability — relative "last N" rules mutate the prefix
    every time a new turn arrives, while "strip all" is positionally stable
    across consecutive requests.
    """
    if keep_last_n < 0:
        return messages

    assistant_indices = [
        i for i, m in enumerate(messages) if m.get("role") == "assistant"
    ]
    if not assistant_indices:
        return messages
    if keep_last_n > 0 and len(assistant_indices) <= keep_last_n:
        return messages

    keep = set(assistant_indices[-keep_last_n:]) if keep_last_n > 0 else set()
    result = []
    stripped_blocks = 0

    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and i not in keep:
            content = msg.get("content")
            if isinstance(content, list):
                pruned = [b for b in content if b.get("type") != "thinking"]
                if len(pruned) < len(content):
                    stripped_blocks += len(content) - len(pruned)
                    msg = {**msg, "content": pruned or content}
        result.append(msg)

    if stripped_blocks:
        logger.info(
            "CONTEXT_OPT: tier1 stripped={} thinking_blocks old_turns={}",
            stripped_blocks,
            len(assistant_indices) - keep_last_n,
        )
    return result
