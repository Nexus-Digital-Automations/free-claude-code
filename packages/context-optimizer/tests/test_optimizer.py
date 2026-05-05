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


def test_tier1_strips_all_thinking_when_keep_last_n_is_zero():
    messages = []
    for i in range(4):
        messages.append(_msg("user", f"q{i}"))
        messages.append(_thinking_msg(f"think{i}", f"reply{i}"))

    result = tier1.apply(messages, keep_last_n=0)

    thinking_count = sum(
        1
        for m in result
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "thinking"
    )
    assert thinking_count == 0


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
        max_thinking_turns=2,
    )
    messages = []
    for i in range(6):
        messages.append(_msg("user", f"q{i}"))
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

    thinking_count = sum(
        1 for m in new_msgs
        if isinstance(m.get("content"), list)
        for b in m["content"] if b.get("type") == "thinking"
    )
    assert thinking_count == 2


# ---- Tier 2 keep-recent-turns floor ----

@pytest.mark.asyncio
async def test_tier2_clamps_aggressive_split_to_keep_recent_floor():
    """LLM picks split=26 (only 4 verbatim); floor=8 clamps to 22 (8 verbatim)."""
    settings = ContextOptimizerSettings(
        compact_threshold_tokens=10,
        compact_soft_threshold_tokens=5,
        tier2_keep_recent_turns=8,
    )
    messages = [_msg("user", f"message {i} " * 5) for i in range(30)]

    llm_response = (
        "<split_index>26</split_index>"
        "<summary>Compacted prefix</summary>"
    )
    provider = AsyncMock(return_value=llm_response)

    new_msgs, _, _ = await ContextOptimizer.optimize(
        messages=messages, system="sys", settings=settings, llm_provider=provider,
    )

    # split clamped to 30-8=22, so 30-22=8 verbatim messages survive
    assert len(new_msgs) == 8


@pytest.mark.asyncio
async def test_tier2_respects_llm_split_when_already_within_floor():
    """LLM picks split=20 (10 verbatim); floor=8 allows up to 22, so 20 stands."""
    settings = ContextOptimizerSettings(
        compact_threshold_tokens=10,
        compact_soft_threshold_tokens=5,
        tier2_keep_recent_turns=8,
    )
    messages = [_msg("user", f"message {i} " * 5) for i in range(30)]

    llm_response = (
        "<split_index>20</split_index>"
        "<summary>Compacted prefix</summary>"
    )
    provider = AsyncMock(return_value=llm_response)

    new_msgs, _, _ = await ContextOptimizer.optimize(
        messages=messages, system="sys", settings=settings, llm_provider=provider,
    )

    # split=20 within floor (max_allowed=22), no clamp, 30-20=10 verbatim survive
    assert len(new_msgs) == 10


# ---- Tier 0 system-reminder dedup ----

def test_tier0_dedupes_repeated_system_reminders_across_messages():
    reminder = "<system-reminder>CODE STANDARDS: keep files small</system-reminder>"
    other = "<system-reminder>different content</system-reminder>"
    messages = [
        _msg("user", f"{reminder}\nfirst question"),
        _msg("user", f"{reminder}\nsecond question"),
        _msg("user", f"{reminder}\n{other}\nthird question"),
    ]

    result = tier0.apply(messages)

    # First message keeps the reminder; later duplicates are replaced
    assert "CODE STANDARDS: keep files small" in result[0]["content"][0]["text"]
    assert "CODE STANDARDS: keep files small" not in result[1]["content"][0]["text"]
    assert "[elided" in result[1]["content"][0]["text"]
    # Distinct reminder in msg 3 survives; the duplicate one does not
    assert "different content" in result[2]["content"][0]["text"]
    assert "CODE STANDARDS" not in result[2]["content"][0]["text"]


def test_tier0_keeps_unique_system_reminders():
    messages = [
        _msg("user", "<system-reminder>A</system-reminder>"),
        _msg("user", "<system-reminder>B</system-reminder>"),
    ]
    result = tier0.apply(messages)
    assert "<system-reminder>A</system-reminder>" in result[0]["content"][0]["text"]
    assert "<system-reminder>B</system-reminder>" in result[1]["content"][0]["text"]


# ---- Tier 0b: Ollama tool-result digester ----

def _long_tool_result(content: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": content},
        ],
    }


@pytest.mark.asyncio
async def test_tier0b_returns_cached_digest_on_repeat_input(monkeypatch):
    """Identical tool_result content must always produce identical digest bytes."""
    from context_optimizer.tiers import tier0b

    tier0b.reset_for_test()

    settings = ContextOptimizerSettings(
        tier0b_digest_enabled=True,
        tier0b_digest_min_bytes=100,
        compact_threshold_tokens=999_999,
        compact_soft_threshold_tokens=999_999,
    )

    long_content = "match line " * 200  # ~2400 bytes
    msgs = [_long_tool_result(long_content)]

    call_count = {"n": 0}

    async def fake_digest_one(candidate, _settings):
        call_count["n"] += 1
        return "digest-text"

    async def fake_ensure_ready(_settings):
        return True

    from context_optimizer.ollama_supervisor import OllamaSupervisor
    monkeypatch.setattr(OllamaSupervisor, "ensure_ready", fake_ensure_ready)
    monkeypatch.setattr(tier0b, "_digest_one", fake_digest_one)

    result1 = await tier0b.apply(msgs, settings)
    result2 = await tier0b.apply(msgs, settings)

    assert call_count["n"] == 1, "second call must hit the cache, not Ollama"
    assert result1[0]["content"][0]["content"] == "digest-text"
    assert result2[0]["content"][0]["content"] == "digest-text"


@pytest.mark.asyncio
async def test_tier0b_passes_through_when_ollama_unavailable(monkeypatch):
    from context_optimizer.tiers import tier0b
    from context_optimizer.ollama_supervisor import OllamaSupervisor

    tier0b.reset_for_test()

    settings = ContextOptimizerSettings(
        tier0b_digest_enabled=True,
        tier0b_digest_min_bytes=100,
    )

    long_content = "x" * 500
    msgs = [_long_tool_result(long_content)]

    async def fake_ensure_ready(_settings):
        return False

    monkeypatch.setattr(OllamaSupervisor, "ensure_ready", fake_ensure_ready)

    result = await tier0b.apply(msgs, settings)
    assert result[0]["content"][0]["content"] == long_content


@pytest.mark.asyncio
async def test_tier0b_skips_short_tool_results():
    from context_optimizer.tiers import tier0b

    tier0b.reset_for_test()

    settings = ContextOptimizerSettings(
        tier0b_digest_enabled=True,
        tier0b_digest_min_bytes=8000,
    )

    short = "small output"
    msgs = [_long_tool_result(short)]

    result = await tier0b.apply(msgs, settings)
    assert result is msgs or result[0]["content"][0]["content"] == short


# ---- Tier 0c: tool_use input compaction ----

def _tool_use(call_id: str, name: str, input_dict: dict) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": call_id, "name": name, "input": input_dict}],
    }


@pytest.mark.asyncio
async def test_tier0c_keeps_recent_tool_use_calls_verbatim(monkeypatch):
    """Last `keep_recent_calls` tool_use blocks must not be digested."""
    from context_optimizer.tiers import tier0c
    from context_optimizer.ollama_supervisor import OllamaSupervisor

    tier0c.reset_for_test()

    settings = ContextOptimizerSettings(
        tier0c_digest_enabled=True,
        tier0c_digest_min_bytes=100,
        tier0c_keep_recent_calls=2,
    )

    big_input = {"new_string": "x" * 500, "file_path": "/foo.py"}
    msgs = [
        _tool_use("a", "Edit", big_input),
        _tool_use("b", "Edit", big_input),
        _tool_use("c", "Edit", big_input),
        _tool_use("d", "Edit", big_input),
    ]

    async def fake_ensure_ready(_settings):
        return True

    async def fake_digest_one(candidate, _settings):
        return "edit-digest"

    monkeypatch.setattr(OllamaSupervisor, "ensure_ready", fake_ensure_ready)
    monkeypatch.setattr(tier0c, "_digest_one", fake_digest_one)

    result = await tier0c.apply(msgs, settings)

    # Last 2 must be untouched
    assert result[2]["content"][0]["input"] == big_input
    assert result[3]["content"][0]["input"] == big_input
    # First 2 should have been digested
    assert "_compacted_summary" in result[0]["content"][0]["input"]
    assert "_compacted_summary" in result[1]["content"][0]["input"]


# ---- Tier 0d: long historical user-paste digester ----

@pytest.mark.asyncio
async def test_tier0d_skips_active_last_user_message(monkeypatch):
    """The last user message (the active request) must never be digested."""
    from context_optimizer.tiers import tier0d
    from context_optimizer.ollama_supervisor import OllamaSupervisor

    tier0d.reset_for_test()

    settings = ContextOptimizerSettings(
        tier0d_digest_enabled=True,
        tier0d_digest_min_bytes=100,
    )

    big_text = "log line\n" * 200
    msgs = [
        _msg("user", big_text),
        _msg("assistant", "ack"),
        _msg("user", big_text),  # last user — must not be digested
    ]

    async def fake_ensure_ready(_settings):
        return True

    async def fake_digest_one(candidate, _settings):
        return "user-paste-digest"

    monkeypatch.setattr(OllamaSupervisor, "ensure_ready", fake_ensure_ready)
    monkeypatch.setattr(tier0d, "_digest_one", fake_digest_one)

    result = await tier0d.apply(msgs, settings)

    # The historical user message gets digested
    assert result[0]["content"][0]["text"] == "user-paste-digest"
    # The active (last) user message stays verbatim
    assert result[2]["content"][0]["text"] == big_text


@pytest.mark.asyncio
async def test_tier0d_skips_short_user_text():
    from context_optimizer.tiers import tier0d

    tier0d.reset_for_test()

    settings = ContextOptimizerSettings(
        tier0d_digest_enabled=True,
        tier0d_digest_min_bytes=16_000,
    )

    msgs = [_msg("user", "what time is it"), _msg("user", "now")]
    result = await tier0d.apply(msgs, settings)
    assert result is msgs
