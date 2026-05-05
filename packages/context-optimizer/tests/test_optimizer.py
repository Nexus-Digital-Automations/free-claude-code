"""Integration and unit tests for context_optimizer package.

Covers: Tier 0 cleanup, Tier 1 thinking strip, Tier 0b/0c/0d Ollama digests,
and the block tower's emergency-seal placeholder fallback.
"""

import pytest

from context_optimizer import ContextOptimizer, ContextOptimizerSettings
from context_optimizer.tiers import tier0, tier1


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


# ---- Full optimize() with block tower disabled ----

@pytest.mark.asyncio
async def test_optimize_runs_tier0_and_tier1_when_layer0_disabled():
    """With block_selection_mode='off' the tower never loads, so tier0/1 are the only state mutators."""
    settings = ContextOptimizerSettings(
        compact_threshold_tokens=999_999,
        max_thinking_turns=2,
        block_selection_mode="off",
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

    assert "CODE STANDARDS: keep files small" in result[0]["content"][0]["text"]
    assert "CODE STANDARDS: keep files small" not in result[1]["content"][0]["text"]
    assert "[elided" in result[1]["content"][0]["text"]
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
        block_selection_mode="off",
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

    assert result[2]["content"][0]["input"] == big_input
    assert result[3]["content"][0]["input"] == big_input
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
        _msg("user", big_text),
    ]

    async def fake_ensure_ready(_settings):
        return True

    async def fake_digest_one(candidate, _settings):
        return "user-paste-digest"

    monkeypatch.setattr(OllamaSupervisor, "ensure_ready", fake_ensure_ready)
    monkeypatch.setattr(tier0d, "_digest_one", fake_digest_one)

    result = await tier0d.apply(msgs, settings)

    assert result[0]["content"][0]["text"] == "user-paste-digest"
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


# ---- Block tower seal_sync emergency placeholder fallback ----

@pytest.mark.asyncio
async def test_seal_sync_writes_placeholder_when_ollama_unreachable(tmp_path, monkeypatch):
    """When Ollama is unreachable, seal_sync writes a deterministic placeholder block.

    The placeholder must satisfy the same immutability invariant as a real
    block (range_start = previous range_end, body bytes deterministic). Two
    seals with the same tail produce byte-identical placeholders so prefix
    caches stay stable.
    """
    from context_optimizer.block_tower import seal_sync
    from context_optimizer.block_tower.store import BlockStore
    from context_optimizer.ollama_supervisor import OllamaSupervisor

    BlockStore.reset_for_test()

    async def fake_ensure_ready(_settings):
        return False

    monkeypatch.setattr(OllamaSupervisor, "ensure_ready", fake_ensure_ready)

    settings = ContextOptimizerSettings(block_storage_dir=str(tmp_path))
    messages = [_msg("user", "hello world") for _ in range(3)]
    store = BlockStore.get_or_build("session_aaa", tmp_path)

    sealed_real = await seal_sync(store, messages, settings)

    assert sealed_real is False, "ollama-down path returns False"
    assert len(store.blocks) == 1
    assert store.blocks[0].range_start == 0
    assert store.blocks[0].range_end == 3
    assert "truncation" in store.blocks[0].header.lower()
    body = store.read_body(store.blocks[0])
    assert "3 messages omitted" in body


@pytest.mark.asyncio
async def test_seal_sync_returns_false_for_empty_session_key(tmp_path, monkeypatch):
    """seal_sync is a no-op for the placeholder 'empty' session key."""
    from context_optimizer.block_tower import seal_sync
    from context_optimizer.block_tower.store import BlockStore

    BlockStore.reset_for_test()

    settings = ContextOptimizerSettings(block_storage_dir=str(tmp_path))
    store = BlockStore.get_or_build("empty", tmp_path)

    sealed = await seal_sync(store, [], settings)

    assert sealed is False
    assert store.blocks == []
