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


# ---- Block tower acceptance criteria (block-tower-compaction.md AC1-AC10) ----
#
# WHY here and not in a separate file: the existing seal_sync tests already
# wire `BlockStore.reset_for_test`, the `_msg` helper, and the conftest
# `ContextOptimizer._reset_for_test` autouse fixture. Co-locating keeps the
# test surface uniform without a new conftest layer.

import hashlib  # noqa: E402 — placed near the block-tower tests that use it
import json  # noqa: E402
from pathlib import Path  # noqa: E402


def _seal_real_block(store, range_start, range_end, body, header):
    """Helper: write a real block synchronously without the Ollama round trip.

    Used by AC2/AC3/AC4 tests that need ≥1 sealed block as a precondition.
    """
    return store.seal(range_start, range_end, body, header)


@pytest.mark.asyncio
async def test_ac1_block_files_appear_after_emergency_seal(tmp_path, monkeypatch):
    """AC1: low-threshold + multi-turn convo causes block-0001.txt + meta.json with content > 0."""
    from context_optimizer.block_tower import seal_sync
    from context_optimizer.block_tower.store import BlockStore
    from context_optimizer.ollama_supervisor import OllamaSupervisor

    BlockStore.reset_for_test()

    async def fake_ensure_ready(_settings):
        return False  # forces the deterministic placeholder path — still a real file write

    monkeypatch.setattr(OllamaSupervisor, "ensure_ready", fake_ensure_ready)

    settings = ContextOptimizerSettings(
        block_storage_dir=str(tmp_path),
        block_seal_min_tail_tokens=500,
    )
    messages = [_msg("user", f"turn {i}") for i in range(6)]
    store = BlockStore.get_or_build("session_ac1", tmp_path)

    await seal_sync(store, messages, settings)

    body_path = tmp_path / "session_ac1" / "block-0001.txt"
    meta_path = tmp_path / "session_ac1" / "block-0001.meta.json"
    assert body_path.is_file(), "block-0001.txt missing"
    assert meta_path.is_file(), "block-0001.meta.json missing"
    assert body_path.stat().st_size > 0, "block body empty"
    assert meta_path.stat().st_size > 0, "block meta empty"


@pytest.mark.asyncio
async def test_ac2_block_one_sha256_unchanged_after_block_two_sealed(tmp_path):
    """AC2: sealing block 2 must not modify block 1's bytes (immutability)."""
    from context_optimizer.block_tower.store import BlockStore

    BlockStore.reset_for_test()
    store = BlockStore.get_or_build("session_ac2", tmp_path)

    block1 = _seal_real_block(store, 0, 5, "first body bytes", "header one")
    sha_before = hashlib.sha256(block1.body_path.read_bytes()).hexdigest()

    _seal_real_block(store, 5, 10, "second body bytes", "header two")
    sha_after = hashlib.sha256(block1.body_path.read_bytes()).hexdigest()

    assert sha_before == sha_after, "block-0001.txt mutated after block 2 sealed"


@pytest.mark.asyncio
async def test_ac3_block_two_range_start_equals_block_one_range_end(tmp_path):
    """AC3: block 2's range.start == block 1's range.end (no gap, no overlap)."""
    from context_optimizer.block_tower.store import BlockStore

    BlockStore.reset_for_test()
    store = BlockStore.get_or_build("session_ac3", tmp_path)

    _seal_real_block(store, 0, 7, "body 1", "h1")
    _seal_real_block(store, 7, 12, "body 2", "h2")

    meta1 = json.loads((tmp_path / "session_ac3" / "block-0001.meta.json").read_text())
    meta2 = json.loads((tmp_path / "session_ac3" / "block-0002.meta.json").read_text())

    assert meta2["range_start"] == meta1["range_end"] == 7


@pytest.mark.asyncio
async def test_ac4_selector_skips_irrelevant_block(tmp_path, monkeypatch):
    """AC4: a query unrelated to a block's content causes the selector to skip it."""
    from context_optimizer.block_tower import selector
    from context_optimizer.block_tower.store import BlockStore
    from context_optimizer.ollama_supervisor import OllamaSupervisor

    BlockStore.reset_for_test()
    selector.reset_for_test()
    store = BlockStore.get_or_build("session_ac4", tmp_path)
    block1 = _seal_real_block(store, 0, 3, "kafka producer config", "kafka producer setup")
    block2 = _seal_real_block(store, 3, 6, "react state hook bug", "react hook bug")

    async def fake_ensure_ready(_s):
        return True

    async def fake_ask_ollama(blocks, current_user_text, _settings):
        return [block2.block_index]  # only react block included

    monkeypatch.setattr(OllamaSupervisor, "ensure_ready", fake_ensure_ready)
    monkeypatch.setattr(selector, "_ask_ollama", fake_ask_ollama)

    settings = ContextOptimizerSettings(block_storage_dir=str(tmp_path))
    selected = await selector.select_blocks(
        [block1, block2], "fix my react useEffect", "session_ac4", settings,
    )

    assert [b.block_index for b in selected] == [block2.block_index]


@pytest.mark.asyncio
async def test_ac5_identical_queries_hit_selection_cache(tmp_path, monkeypatch):
    """AC5: identical inputs ⇒ cached selection ⇒ Ollama not called twice."""
    from context_optimizer.block_tower import selector
    from context_optimizer.block_tower.store import BlockStore

    BlockStore.reset_for_test()
    selector.reset_for_test()
    store = BlockStore.get_or_build("session_ac5", tmp_path)
    block1 = _seal_real_block(store, 0, 3, "body 1", "h1")
    block2 = _seal_real_block(store, 3, 6, "body 2", "h2")

    call_count = {"n": 0}

    async def fake_ask_ollama(blocks, current_user_text, _settings):
        call_count["n"] += 1
        return [block1.block_index, block2.block_index]

    monkeypatch.setattr(selector, "_ask_ollama", fake_ask_ollama)

    settings = ContextOptimizerSettings(block_storage_dir=str(tmp_path))
    await selector.select_blocks([block1, block2], "same query", "session_ac5", settings)
    await selector.select_blocks([block1, block2], "same query", "session_ac5", settings)

    assert call_count["n"] == 1, "second identical call must hit the LRU cache"


def test_ac6_no_seal_when_request_count_below_threshold(tmp_path):
    """AC6: should_seal() returns False when requests_since_last_seal < min_requests."""
    from context_optimizer.block_tower.sealer import should_seal

    settings = ContextOptimizerSettings(
        block_storage_dir=str(tmp_path),
        block_seal_min_requests=10,
        block_seal_min_tail_tokens=10,  # so tail size never blocks the check
    )
    # 5 requests, lots of tail content — still below the request floor.
    tail = [_msg("user", "x" * 500) for _ in range(20)]

    assert should_seal(tail, requests_since_last_seal=5, settings=settings) is False


@pytest.mark.asyncio
async def test_ac7_selector_falls_back_to_all_blocks_when_ollama_unavailable(tmp_path, monkeypatch):
    """AC7: stopping ollama ⇒ selector returns every block (request still has full context)."""
    from context_optimizer.block_tower import selector
    from context_optimizer.block_tower.store import BlockStore
    from context_optimizer.ollama_supervisor import OllamaSupervisor

    BlockStore.reset_for_test()
    selector.reset_for_test()
    store = BlockStore.get_or_build("session_ac7", tmp_path)
    block1 = _seal_real_block(store, 0, 3, "body 1", "h1")
    block2 = _seal_real_block(store, 3, 6, "body 2", "h2")

    async def fake_ensure_ready(_s):
        return False  # ollama down

    monkeypatch.setattr(OllamaSupervisor, "ensure_ready", fake_ensure_ready)

    settings = ContextOptimizerSettings(block_storage_dir=str(tmp_path))
    selected = await selector.select_blocks(
        [block1, block2], "any query", "session_ac7", settings,
    )

    assert [b.block_index for b in selected] == [block1.block_index, block2.block_index]


# AC8 obsolete after Phase 2 — Tier 2 was deleted (see spec progress 2026-05-05),
# so its premise ("block_tower_enabled=False leaves Tier 2 unchanged") no longer
# applies. AC9 covers the remaining gating behaviour: mode="off" skips Layer 0.


@pytest.mark.asyncio
async def test_ac9_selection_mode_off_skips_layer_zero(tmp_path):
    """AC9: block_selection_mode='off' returns no blocks regardless of tower contents."""
    from context_optimizer.block_tower import selector
    from context_optimizer.block_tower.store import BlockStore

    BlockStore.reset_for_test()
    selector.reset_for_test()
    store = BlockStore.get_or_build("session_ac9", tmp_path)
    block1 = _seal_real_block(store, 0, 3, "body 1", "h1")
    block2 = _seal_real_block(store, 3, 6, "body 2", "h2")

    settings = ContextOptimizerSettings(
        block_storage_dir=str(tmp_path),
        block_selection_mode="off",
    )
    selected = await selector.select_blocks(
        [block1, block2], "any query", "session_ac9", settings,
    )

    assert selected == []


def test_ac10_ruff_clean_on_block_tower_module():
    """AC10: ruff check exits 0 on the block_tower module."""
    import subprocess

    pkg_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["ruff", "check", "src/context_optimizer/block_tower/"],
        cwd=pkg_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ruff failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
