"""Unit tests for providers/common/context_optimizer.py.

Owns: behaviour of each tier's pure function in isolation.
Does NOT own: end-to-end optimize() integration with real providers
              (covered by tests/api/test_routes_optimizations.py).

Each test is a tier-level specification — names read as the contract under test,
not the function being called.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from api.models.anthropic import (
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolResult,
    Message,
    SystemContent,
)
from providers.common.context_optimizer import ContextOptimizer


@pytest.fixture(autouse=True)
def clear_optimizer_state():
    """Reset class-level mutable state so tests don't leak into each other."""
    ContextOptimizer._summary_cache.clear()
    ContextOptimizer._inflight.clear()
    yield
    ContextOptimizer._summary_cache.clear()
    ContextOptimizer._inflight.clear()


def _user_text(text: str) -> Message:
    return Message(role="user", content=[ContentBlockText(type="text", text=text)])


def _assistant_with_thinking(thought: str, reply: str) -> Message:
    return Message(
        role="assistant",
        content=[
            ContentBlockThinking(type="thinking", thinking=thought),
            ContentBlockText(type="text", text=reply),
        ],
    )


def test_strip_old_thinking_keeps_last_two_turns():
    # Six assistant turns each with one thinking block → keep_last_n=2 means
    # only the last two retain their thinking; four older turns lose it.
    messages = []
    for i in range(6):
        messages.append(_user_text(f"q{i}"))
        messages.append(_assistant_with_thinking(f"think{i}", f"reply{i}"))

    result = ContextOptimizer._strip_old_thinking(messages, keep_last_n=2)

    thinking_blocks = sum(
        1
        for m in result
        if isinstance(m.content, list)
        for b in m.content
        if isinstance(b, ContentBlockThinking)
    )
    assert thinking_blocks == 2


def test_strip_old_thinking_noop_when_few_turns():
    messages = [
        _user_text("q0"),
        _assistant_with_thinking("t0", "r0"),
        _user_text("q1"),
        _assistant_with_thinking("t1", "r1"),
    ]
    result = ContextOptimizer._strip_old_thinking(messages, keep_last_n=2)
    assert result is messages  # unchanged identity when nothing to strip


def test_apply_prefix_cache_hits_and_augments_system():
    # Build 6 messages and pre-populate the cache for messages[:4].
    # _apply_prefix_cache picks k from {n-2, n/2, n/3, 4} clamped to [4, n-2];
    # for n=6 that yields {4, 3, 2, 4} → {4} (only 4 is in [4, n-2]=[4, 4]).
    messages = [_user_text(f"m{i}") for i in range(6)]
    summary = "Earlier work: investigated path X"
    key = ContextOptimizer._cache_key(messages[:4])
    ContextOptimizer._summary_cache[key] = (4, summary)

    out_messages, out_system = ContextOptimizer._apply_prefix_cache(
        messages, "user-supplied system"
    )

    # The 4-message prefix is replaced (gone), the last 2 stay verbatim.
    assert len(out_messages) == 2
    # Summary text now lives in the system prompt, not in messages.
    assert summary in out_system
    assert "user-supplied system" in out_system
    # No synthetic user message with old <conversation_summary> wrapper.
    assert not any(
        isinstance(b, ContentBlockText) and "<conversation_summary>" in b.text
        for m in out_messages
        if isinstance(m.content, list)
        for b in m.content
    )


def test_apply_prefix_cache_with_list_system_prepends_block():
    messages = [_user_text(f"m{i}") for i in range(6)]
    summary = "Prior context"
    key = ContextOptimizer._cache_key(messages[:4])
    ContextOptimizer._summary_cache[key] = (4, summary)

    original_system = [SystemContent(type="text", text="original instructions")]
    _, out_system = ContextOptimizer._apply_prefix_cache(messages, original_system)

    assert isinstance(out_system, list)
    assert summary in out_system[0].text
    assert out_system[1].text == "original instructions"


def test_apply_prefix_cache_misses_returns_inputs_unchanged():
    messages = [_user_text(f"m{i}") for i in range(6)]
    out_messages, out_system = ContextOptimizer._apply_prefix_cache(messages, "sys")

    assert out_messages is messages
    assert out_system == "sys"


def test_parse_response_returns_none_for_malformed_output():
    # Missing <split_index> tag entirely.
    assert ContextOptimizer._parse_response("just prose, no tags", num_messages=10) is None
    # Has tag but no summary.
    assert ContextOptimizer._parse_response(
        "<split_index>5</split_index>", num_messages=10
    ) is None
    # split_index out of allowed range [4, n-1).
    assert ContextOptimizer._parse_response(
        "<split_index>2</split_index><summary>x</summary>", num_messages=10
    ) is None
    assert ContextOptimizer._parse_response(
        "<split_index>10</split_index><summary>x</summary>", num_messages=10
    ) is None


def test_parse_response_accepts_well_formed_tags():
    parsed = ContextOptimizer._parse_response(
        "garbage <split_index>5</split_index> more garbage <summary>ok</summary> end",
        num_messages=10,
    )
    assert parsed == (5, "ok")


@pytest.mark.asyncio
async def test_compact_via_ollama_returns_false_on_network_error(monkeypatch):
    """Network failure must not raise, must not cache anything, must return False."""
    settings = MagicMock()
    settings.ollama_base_url = "http://localhost:11434/v1"
    settings.ollama_model = "qwen2.5:7b"

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("network down")
    )
    fake_client.close = AsyncMock()
    monkeypatch.setattr(
        "providers.common.context_optimizer.AsyncOpenAI", lambda **_: fake_client
    )

    messages = [_user_text(f"m{i}") for i in range(8)]
    result = await ContextOptimizer._compact_via_ollama(messages, settings)

    assert result is False
    assert len(ContextOptimizer._summary_cache) == 0


def test_apply_tier0_dedupes_repeated_tool_results():
    # Two messages each containing a tool result with identical content;
    # the second one must be replaced with the dedup placeholder.
    duplicated = "lots of identical output here"
    msg1 = Message(
        role="user",
        content=[ContentBlockToolResult(
            type="tool_result", tool_use_id="t1", content=duplicated
        )],
    )
    msg2 = Message(
        role="user",
        content=[ContentBlockToolResult(
            type="tool_result", tool_use_id="t2", content=duplicated
        )],
    )

    result = ContextOptimizer._apply_tier0([msg1, msg2])

    assert result[0].content[0].content == duplicated
    assert "[identical to earlier tool result" in result[1].content[0].content


def test_apply_tier0_strips_ansi_from_tool_results():
    ansi_text = "\x1b[31merror:\x1b[0m something failed"
    msg = Message(
        role="user",
        content=[ContentBlockToolResult(
            type="tool_result", tool_use_id="t1", content=ansi_text
        )],
    )

    result = ContextOptimizer._apply_tier0([msg])

    assert result[0].content[0].content == "error: something failed"
