"""Owns: tier0e error-aware tool-call filter behavioural tests.

Counterpart: src/context_optimizer/tiers/tier0e.py.

Verifies the three-signal error classifier and the stub-input transform.
Pure-Python; no Ollama/network dependencies.
"""

from __future__ import annotations

from context_optimizer.settings import ContextOptimizerSettings
from context_optimizer.tiers import tier0e


def _tool_use(call_id: str, name: str, input_dict: dict) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": call_id, "name": name, "input": input_dict}
        ],
    }


def _tool_result(
    call_id: str,
    content: object,
    *,
    is_error: bool | None = None,
) -> dict:
    block: dict = {"type": "tool_result", "tool_use_id": call_id, "content": content}
    if is_error is not None:
        block["is_error"] = is_error
    return {"role": "user", "content": [block]}


def test_returns_messages_unchanged_when_tier_disabled():
    settings = ContextOptimizerSettings(tier0e_enabled=False)
    msgs = [
        _tool_use("a", "Bash", {"command": "ls /foo"}),
        _tool_result("a", "file1\nfile2\n"),
    ]
    result = tier0e.apply(msgs, settings)
    assert result is msgs


def test_stubs_input_when_successful_bash_call_paired_with_clean_output():
    settings = ContextOptimizerSettings(tier0e_enabled=True)
    msgs = [
        _tool_use("a", "Bash", {"command": "ls /foo", "description": "list dir"}),
        _tool_result("a", "file1\nfile2\nfile3\n"),
    ]

    result = tier0e.apply(msgs, settings)

    assert result[0]["content"][0]["input"] == {}
    assert result[0]["content"][0]["name"] == "Bash"
    assert result[0]["content"][0]["id"] == "a"
    assert result[1]["content"][0]["content"] == "file1\nfile2\nfile3\n"


def test_keeps_tool_use_input_when_is_error_flag_is_true():
    settings = ContextOptimizerSettings(tier0e_enabled=True)
    original_input = {"command": "rm /nope"}
    msgs = [
        _tool_use("a", "Bash", original_input),
        _tool_result("a", "rm: /nope: No such file", is_error=True),
    ]

    result = tier0e.apply(msgs, settings)

    assert result[0]["content"][0]["input"] == original_input


def test_keeps_tool_use_input_when_bash_output_carries_exit_code_marker():
    settings = ContextOptimizerSettings(tier0e_enabled=True)
    original_input = {"command": "false"}
    msgs = [
        _tool_use("a", "Bash", original_input),
        _tool_result("a", "command failed\n<bash-stderr>error</bash-stderr>"),
    ]

    result = tier0e.apply(msgs, settings)

    assert result[0]["content"][0]["input"] == original_input


def test_does_not_treat_read_result_with_error_word_as_failure():
    """Source-file content can legitimately contain 'Error:' or 'Traceback' — must not trigger keyword scan."""
    settings = ContextOptimizerSettings(tier0e_enabled=True)
    original_input = {"file_path": "/src/foo.py"}
    msgs = [
        _tool_use("a", "Read", original_input),
        _tool_result("a", "def handle_error():\n    raise Error('boom')\n"),
    ]

    result = tier0e.apply(msgs, settings)

    assert result[0]["content"][0]["input"] == {}


def test_leaves_tool_use_alone_when_no_matching_tool_result_exists():
    """Orphan tool_use (e.g. last message of an in-flight turn) must not be touched."""
    settings = ContextOptimizerSettings(tier0e_enabled=True)
    original_input = {"command": "ls"}
    msgs = [_tool_use("orphan", "Bash", original_input)]

    result = tier0e.apply(msgs, settings)

    assert result[0]["content"][0]["input"] == {}


def test_keeps_webfetch_result_when_keyword_scan_finds_error_substring():
    settings = ContextOptimizerSettings(tier0e_enabled=True)
    original_input = {"url": "https://example.com"}
    msgs = [
        _tool_use("a", "WebFetch", original_input),
        _tool_result("a", "HTTP 500\nError: upstream timed out"),
    ]

    result = tier0e.apply(msgs, settings)

    assert result[0]["content"][0]["input"] == original_input


def test_handles_structured_text_block_content_in_tool_result():
    """tool_result.content may be a list of {type:'text', text:...} blocks, not just a string."""
    settings = ContextOptimizerSettings(tier0e_enabled=True)
    msgs = [
        _tool_use("a", "Bash", {"command": "echo hi"}),
        _tool_result("a", [{"type": "text", "text": "hi\n"}]),
    ]

    result = tier0e.apply(msgs, settings)

    assert result[0]["content"][0]["input"] == {}


def test_only_stubs_successful_pair_when_mixed_with_errored_pair():
    settings = ContextOptimizerSettings(tier0e_enabled=True)
    success_input = {"command": "ls"}
    error_input = {"command": "rm /nope"}
    msgs = [
        _tool_use("ok", "Bash", success_input),
        _tool_result("ok", "file1\n"),
        _tool_use("bad", "Bash", error_input),
        _tool_result("bad", "rm: cannot remove", is_error=True),
    ]

    result = tier0e.apply(msgs, settings)

    assert result[0]["content"][0]["input"] == {}
    assert result[2]["content"][0]["input"] == error_input
