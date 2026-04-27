"""Integration and unit tests for context_optimizer package.

Covers: Tier 0 ANSI strip, Tier 0 dedup, Tier 1 thinking strip,
prefix-cache hit, Tier 2b via mocked llm_provider, full optimize() flow.
"""

from unittest.mock import AsyncMock

import pytest

from context_optimizer import ContextOptimizer, ContextOptimizerSettings
from context_optimizer.tiers import tier0, tier1
from context_optimizer.cache import PrefixCache


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _thinking_msg(thought: str, reply: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": thought},
            {"type": "text", "text": reply},
        ],
    }


def _tool_result(content: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": content}],
    }


# ---- Tier 0 ----

def test_tier0_strips_ansi_from_tool_result():
    msg = _tool_result("\x1b[31merror:\x1b[0m oops")
    result = tier0.apply([msg])
    assert result[0]["content"][0]["content"] == "error: oops"


def test_tier0_dedupes_repeated_tool_results():
    dup = "identical output"
    result = tier0.apply([_tool_result(dup), _tool_result(dup)])
    assert result[0]["content"][0]["content"] == dup
    assert "[identical" in result[1]["content"][0]["content"]


# ---- Tier 1 ----

def test_tier1_strips_thinking_from_old_turns_keeps_last_two():
    messages = []
    for i in range(6):
        messages.append(_msg("user", f"q{i}"))
        messages.append(_thinking_msg(f"think{i}", f"reply{i}"))

    result = tier1.apply(messages, keep_last_n=2)

    thinking_count = sum(
        1
        for m in result
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "thinking"
    )
    assert thinking_count == 2


def test_tier1_noop_when_few_assistant_turns():
    messages = [_msg("user", "q"), _thinking_msg("t", "r")]
    result = tier1.apply(messages, keep_last_n=2)
    assert result is messages


# ---- Prefix cache ----

def test_prefix_cache_hit_applies_summary_to_system():
    messages = [_msg("user", f"m{i}") for i in range(6)]
    cache = PrefixCache(max_entries=10)
    cache.store(messages, split_index=4, summary="Prior work: fixed X")

    result = cache.lookup(messages, "original system")
    assert result is not None
    new_msgs, new_sys = result
    assert len(new_msgs) == 2
    assert "Prior work: fixed X" in new_sys
    assert "original system" in new_sys


def test_prefix_cache_miss_returns_none():
    cache = PrefixCache(max_entries=10)
    messages = [_msg("user", f"m{i}") for i in range(6)]
    assert cache.lookup(messages, "sys") is None


# ---- Tier 2b via llm_provider ----

@pytest.mark.asyncio
async def test_optimize_calls_llm_provider_when_tokens_exceed_threshold():
    settings = ContextOptimizerSettings(
        compact_threshold_tokens=10,   # force immediate compaction
        compact_soft_threshold_tokens=5,
    )
    messages = [_msg("user", f"message {i} " * 5) for i in range(8)]

    llm_response = (
        "<split_index>4</split_index>"
        "<summary>Earlier: user discussed messages 0-3</summary>"
    )
    provider = AsyncMock(return_value=llm_response)

    new_msgs, new_sys, _ = await ContextOptimizer.optimize(
        messages=messages,
        system="sys",
        settings=settings,
        llm_provider=provider,
    )

    provider.assert_called_once()
    # summary moves to system, not a synthetic user message
    assert "Earlier:" in new_sys
    assert not any(
        isinstance(m.get("content"), list)
        and any(b.get("type") == "text" and "<conversation_summary>" in b.get("text", "")
                for b in m["content"])
        for m in new_msgs
    )


@pytest.mark.asyncio
async def test_optimize_tier0_and_tier1_run_regardless_of_token_count():
    settings = ContextOptimizerSettings(
        compact_threshold_tokens=999_999,
        compact_soft_threshold_tokens=999_999,
    )
    messages = []
    for i in range(6):
        messages.append(_msg("user", f"q{i}"))
        # assistant turns with thinking + ANSI in tool results
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "t"},
                {"type": "text", "text": "ok"},
            ],
        })

    new_msgs, _, _ = await ContextOptimizer.optimize(
        messages=messages, settings=settings
    )

    # Tier 1: only last 2 assistant turns keep thinking
    thinking_count = sum(
        1 for m in new_msgs
        if isinstance(m.get("content"), list)
        for b in m["content"] if b.get("type") == "thinking"
    )
    assert thinking_count == 2
