"""Owns: tier0f span-level Rabin-Karp dedup behavioural tests.

Counterpart: src/context_optimizer/tiers/tier0f.py.

Verifies first-occurrence preservation, greedy maximal extension, threshold
gating, system/last-user definer protection, idempotence, determinism, and
structural-field integrity. Pure-Python; no Ollama/network dependencies.
"""

from __future__ import annotations

import tiktoken

from context_optimizer.settings import ContextOptimizerSettings
from context_optimizer.tiers import tier0f

_ENC = tiktoken.get_encoding("cl100k_base")


def _text_of_n_tokens(n: int, seed: str = "alpha") -> str:
    """Return text whose cl100k_base tokenization equals exactly n tokens."""
    words = " ".join(f"{seed}-{i}" for i in range(max(n, 1) * 3))
    return _ENC.decode(_ENC.encode(words)[:n])


def _user_text(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _user_tool_result(call_id: str, content: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": call_id, "content": content},
        ],
    }


def _settings(**overrides) -> ContextOptimizerSettings:
    base: dict = {"tier0f_enabled": True, "tier0f_min_tokens": 70}
    base.update(overrides)
    return ContextOptimizerSettings(**base)


# ---- gating ----

def test_returns_input_identity_when_disabled():
    msgs = [_user_text("hello")]
    settings = _settings(tier0f_enabled=False)
    assert tier0f.apply(msgs, settings) is msgs


def test_returns_input_identity_when_no_duplicates():
    msgs = [
        _user_text("unique short prompt"),
        _user_text("different unique prompt"),
    ]
    settings = _settings()
    assert tier0f.apply(msgs, settings) is msgs


def test_returns_input_identity_when_min_tokens_below_two():
    msgs = [_user_text("hello")]
    settings = _settings(tier0f_min_tokens=1)
    assert tier0f.apply(msgs, settings) is msgs


# ---- core dedup ----

def test_drops_second_occurrence_of_repeated_span_above_threshold():
    repeat = _text_of_n_tokens(80)
    msgs = [
        _user_tool_result("a", f"prefix\n{repeat}\nsuffix"),
        _user_tool_result("b", f"different prefix\n{repeat}\ndifferent suffix"),
        _user_text("active question"),
    ]
    result = tier0f.apply(msgs, _settings())

    assert repeat in result[0]["content"][0]["content"]
    assert repeat not in result[1]["content"][0]["content"]
    assert "different prefix" in result[1]["content"][0]["content"]
    assert "different suffix" in result[1]["content"][0]["content"]


def test_greedy_extension_collapses_full_repeated_run():
    repeat = _text_of_n_tokens(200)
    msgs = [
        _user_tool_result("a", f"x\n{repeat}\ny"),
        _user_tool_result("b", f"p\n{repeat}\nq"),
        _user_text("active"),
    ]
    result = tier0f.apply(msgs, _settings())

    first_tokens = len(_ENC.encode(result[0]["content"][0]["content"]))
    second_tokens = len(_ENC.encode(result[1]["content"][0]["content"]))
    assert first_tokens - second_tokens >= 180


def test_skips_repeats_below_threshold():
    repeat = _text_of_n_tokens(40)
    msgs = [
        _user_tool_result("a", f"alpha {repeat} beta"),
        _user_tool_result("b", f"gamma {repeat} delta"),
        _user_text("active"),
    ]
    assert tier0f.apply(msgs, _settings()) is msgs


# ---- definer protection ----

def test_system_prompt_acts_as_definer_messages_get_dedup():
    repeat = _text_of_n_tokens(90)
    system = f"--- system instructions ---\n{repeat}\n--- end ---"
    msgs = [
        _user_tool_result("a", f"earlier output:\n{repeat}\nmore output"),
        _user_text("active"),
    ]
    result = tier0f.apply(msgs, _settings(), system=system)

    assert repeat not in result[0]["content"][0]["content"]
    assert repeat in system


def test_last_user_message_protected_from_deletion():
    repeat = _text_of_n_tokens(90)
    msgs = [
        _user_tool_result("a", f"history: {repeat}"),
        _user_text(f"please review: {repeat}"),
    ]
    result = tier0f.apply(msgs, _settings())

    assert repeat in result[-1]["content"][0]["text"]


# ---- determinism / idempotence ----

def test_idempotent_across_repeated_application():
    repeat = _text_of_n_tokens(100)
    msgs = [
        _user_tool_result("a", f"x\n{repeat}\ny"),
        _user_tool_result("b", f"p\n{repeat}\nq"),
        _user_text("active"),
    ]
    settings = _settings()
    once = tier0f.apply(msgs, settings)
    twice = tier0f.apply(once, settings)
    assert once == twice


def test_deterministic_byte_for_byte_across_runs():
    repeat = _text_of_n_tokens(100)
    msgs = [
        _user_tool_result("a", f"x\n{repeat}\ny"),
        _user_tool_result("b", f"p\n{repeat}\nq"),
        _user_text("active"),
    ]
    settings = _settings()
    assert tier0f.apply(msgs, settings) == tier0f.apply(msgs, settings)


# ---- structural integrity ----

def test_tool_use_input_id_name_untouched():
    repeat = _text_of_n_tokens(90)
    msgs = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "Bash",
                    "input": {"command": repeat},
                },
            ],
        },
        _user_tool_result("call-1", f"result containing {repeat} again"),
        _user_text("active"),
    ]
    result = tier0f.apply(msgs, _settings())
    tu = result[0]["content"][0]
    assert tu["id"] == "call-1"
    assert tu["name"] == "Bash"
    assert tu["input"] == {"command": repeat}


def test_text_blocks_in_assistant_messages_get_dedup():
    repeat = _text_of_n_tokens(90)
    msgs = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"first: {repeat}"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"second: {repeat}"}],
        },
        _user_text("active"),
    ]
    result = tier0f.apply(msgs, _settings())
    assert repeat in result[0]["content"][0]["text"]
    assert repeat not in result[1]["content"][0]["text"]


def test_tool_result_with_list_content_is_left_alone():
    """List-shaped tool_result content is out of scope (handled upstream)."""
    repeat = _text_of_n_tokens(90)
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "a",
                    "content": [{"type": "text", "text": repeat}],
                },
            ],
        },
        _user_tool_result("b", f"plain string version: {repeat}"),
        _user_text("active"),
    ]
    result = tier0f.apply(msgs, _settings())
    assert result[0]["content"][0]["content"] == [{"type": "text", "text": repeat}]
