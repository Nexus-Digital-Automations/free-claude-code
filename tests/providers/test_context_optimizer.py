"""Proxy-level tests for context_optimizer tier logic (via package imports).

All tier logic now lives in the context_optimizer package. These tests import
from the package directly. The proxy adapter itself is a thin delegation layer
with no internal state.

For the authoritative package-level test suite, see:
packages/context-optimizer/tests/test_optimizer.py
"""

import pytest
from context_optimizer import ContextOptimizer as _PkgOptimizer
from context_optimizer.cache import PrefixCache
from context_optimizer.prompts import parse_response
from context_optimizer.tiers import tier0, tier1


@pytest.fixture(autouse=True)
def clear_optimizer_state():
    _PkgOptimizer._reset_for_test()
    yield
    _PkgOptimizer._reset_for_test()


def _user_dict(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _thinking_dict(thought: str, reply: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": thought},
            {"type": "text", "text": reply},
        ],
    }


def _tool_result_dict(content: str, tool_use_id: str = "t1") -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
    }


def test_strip_old_thinking_keeps_last_two_turns():
    messages = []
    for i in range(6):
        messages.append(_user_dict(f"q{i}"))
        messages.append(_thinking_dict(f"think{i}", f"reply{i}"))

    result = tier1.apply(messages, keep_last_n=2)

    thinking_count = sum(
        1
        for m in result
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "thinking"
    )
    assert thinking_count == 2


def test_strip_old_thinking_noop_when_few_turns():
    messages = [
        _user_dict("q0"),
        _thinking_dict("t0", "r0"),
        _user_dict("q1"),
        _thinking_dict("t1", "r1"),
    ]
    result = tier1.apply(messages, keep_last_n=2)
    assert result is messages


def test_apply_prefix_cache_hits_and_augments_system():
    messages = [_user_dict(f"m{i}") for i in range(6)]
    summary = "Earlier work: investigated path X"
    cache = PrefixCache(max_entries=10)
    cache.store(messages, split_index=4, summary=summary)

    result = cache.lookup(messages, "user-supplied system")

    assert result is not None
    out_messages, out_system = result
    assert len(out_messages) == 2
    assert summary in out_system
    assert "user-supplied system" in out_system


def test_apply_prefix_cache_with_list_system_prepends_block():
    messages = [_user_dict(f"m{i}") for i in range(6)]
    summary = "Prior context"
    cache = PrefixCache(max_entries=10)
    cache.store(messages, split_index=4, summary=summary)

    original_system = [{"type": "text", "text": "original instructions"}]
    result = cache.lookup(messages, original_system)

    assert result is not None
    _, out_system = result
    assert isinstance(out_system, list)
    assert summary in out_system[0]["text"]
    assert out_system[1]["text"] == "original instructions"


def test_apply_prefix_cache_misses_returns_inputs_unchanged():
    messages = [_user_dict(f"m{i}") for i in range(6)]
    cache = PrefixCache(max_entries=10)
    assert cache.lookup(messages, "sys") is None


def test_parse_response_returns_none_for_malformed_output():
    assert parse_response("just prose, no tags", num_messages=10) is None
    assert parse_response("<split_index>5</split_index>", num_messages=10) is None
    assert parse_response(
        "<split_index>2</split_index><summary>x</summary>", num_messages=10
    ) is None
    assert parse_response(
        "<split_index>10</split_index><summary>x</summary>", num_messages=10
    ) is None


def test_parse_response_accepts_well_formed_tags():
    parsed = parse_response(
        "garbage <split_index>5</split_index> more garbage <summary>ok</summary> end",
        num_messages=10,
    )
    assert parsed == (5, "ok")


def test_apply_tier0_dedupes_repeated_tool_results():
    duplicated = "lots of identical output here"
    msg1 = _tool_result_dict(duplicated, "t1")
    msg2 = _tool_result_dict(duplicated, "t2")

    result = tier0.apply([msg1, msg2])

    assert result[0]["content"][0]["content"] == duplicated
    assert "[identical" in result[1]["content"][0]["content"]


def test_apply_tier0_strips_ansi_from_tool_results():
    ansi_text = "\x1b[31merror:\x1b[0m something failed"

    result = tier0.apply([_tool_result_dict(ansi_text)])

    assert result[0]["content"][0]["content"] == "error: something failed"
