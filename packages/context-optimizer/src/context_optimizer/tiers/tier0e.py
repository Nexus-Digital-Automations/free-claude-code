"""Owns: tier0e — error-aware tool-call filter.

Replaces the `input` payload of every assistant tool_use whose paired
tool_result is NOT classified as an error with `{}`. The tool_result body
itself is preserved untouched — the codebase signal the model needs to
reason about survives; only the command/args wrapper is dropped.

Errored pairs pass through verbatim so the model can debug from them.

Does NOT own:
  - orphan-tool-message cleanup (the OpenAI converter at
    providers/common/message_converter.py handles that)
  - any byte-size truncation (tier0/0b/0c/0d already do)
  - removal of the tool_use block itself (would orphan the tool_result
    at conversion time and silently drop the codebase signal — see plan
    note: stub-input is intentionally preferred over full-drop)

Called by: optimizer.optimize() between tier0d and tier1.
Calls: nothing — pure data transformation, no I/O, no Ollama.
"""

from __future__ import annotations

import re

from ..settings import ContextOptimizerSettings

# Tools whose error-shaped output reliably indicates failure. For
# Read/Edit/Write/Grep/Glob/etc., the same strings ("Error:", "Traceback")
# routinely appear inside legitimate file contents — applying the keyword
# scan to those tools would produce false-positive "errored" classifications
# and defeat the filter on every codebase exploration turn.
_KEYWORD_SCAN_TOOLS: frozenset[str] = frozenset({
    "Bash",
    "WebFetch",
    "WebSearch",
})

# Lowercase substrings; matched case-insensitively against tool_result text.
_ERROR_KEYWORDS: tuple[str, ...] = (
    "traceback",
    "error:",
    "enoent",
    "permission denied",
    "failed:",
    "fatal:",
    "command not found",
)

# Bash-specific failure patterns. The harness usually sets is_error on
# Bash failures, but stdout/stderr formats vary across tool emitters and
# is_error is occasionally absent — these patterns catch the misses.
_BASH_FAILURE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<bash-stderr>", re.IGNORECASE),
    re.compile(r"\bexit code:\s*([1-9]\d*)\b", re.IGNORECASE),
    re.compile(r"\bexit status\s*([1-9]\d*)\b", re.IGNORECASE),
)


def apply(
    messages: list[dict], settings: ContextOptimizerSettings,
) -> list[dict]:
    """Stub successful tool_use inputs; pass errored pairs through.

    No-op when settings.tier0e_enabled is False. Returns the input list
    object unchanged (same identity) when no message qualifies, so callers
    can detect a no-op via `result is messages` if needed.

    Never raises; pure data transformation over plain dicts/lists.
    # @stable
    """
    if not settings.tier0e_enabled:
        return messages

    errored_ids = _collect_errored_tool_use_ids(messages)
    return _stub_successful_tool_uses(messages, errored_ids)


def _collect_errored_tool_use_ids(messages: list[dict]) -> set[str]:
    """Return the set of tool_use_ids whose result was classified as failed.

    Walks user messages (where tool_result blocks live) and applies the
    three-signal classifier from `_is_error`. Builds a tool_use_id ->
    tool_name map up front so the keyword-scan denylist can be applied
    correctly even when the tool_use lives several messages earlier.
    """
    name_by_id = _index_tool_use_names(messages)
    errored: set[str] = set()
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str):
                continue
            if _is_error(block, name_by_id.get(tool_use_id, "")):
                errored.add(tool_use_id)
    return errored


def _index_tool_use_names(messages: list[dict]) -> dict[str, str]:
    """Map every assistant tool_use id to its tool name."""
    out: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            block_id = block.get("id")
            if isinstance(block_id, str):
                out[block_id] = str(block.get("name", ""))
    return out


def _is_error(tool_result_block: dict, tool_name: str) -> bool:
    """Three-signal classifier; any positive signal means errored.

    Signal 1 — `is_error: true` on the tool_result. Anthropic's canonical
    flag, set by the harness when the underlying tool returns nonzero or
    raises. Cheapest check; runs first.

    Signal 2 — Bash exit-code pattern in result text. Catches failures
    where signal 1 is missing because the tool emitter (e.g. third-party
    Bash wrapper) didn't propagate is_error. Only applied when tool_name
    == "Bash" so file contents that legitimately contain shell-shaped
    text never trigger.

    Signal 3 — keyword scan, scoped to `_KEYWORD_SCAN_TOOLS`. Read/Edit/
    Write/Grep/Glob are excluded by design: source files routinely contain
    "Error:" and "Traceback" inside legitimate code.
    """
    if tool_result_block.get("is_error") is True:
        return True

    text = _content_to_text(tool_result_block.get("content"))
    if not text:
        return False

    if tool_name == "Bash" and _looks_like_bash_failure(text):
        return True

    if tool_name in _KEYWORD_SCAN_TOOLS:
        lowered = text.lower()
        if any(kw in lowered for kw in _ERROR_KEYWORDS):
            return True

    return False


def _content_to_text(content: object) -> str:
    """Normalise tool_result.content to plain text.

    Anthropic accepts both `content: "string"` and the structured form
    `content: [{"type": "text", "text": "..."}, ...]`. We must handle both;
    other block types (image, document) carry no error-signal text and
    are ignored.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _looks_like_bash_failure(text: str) -> bool:
    """True when text carries any non-zero-exit shell signal."""
    return any(p.search(text) for p in _BASH_FAILURE_PATTERNS)


def _stub_successful_tool_uses(
    messages: list[dict], errored_ids: set[str],
) -> list[dict]:
    """Return a copy of messages with successful tool_use inputs replaced by {}.

    Preserves tool_use `id` and `name` so the OpenAI converter's
    orphan-pair filter still matches the corresponding role:"tool" message
    and the tool_result content survives downstream.
    Counterpart: providers/common/message_converter.py orphan filter.
    """
    changed = False
    out: list[dict] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            out.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue

        new_content: list = []
        msg_changed = False
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                new_content.append(block)
                continue
            block_id = block.get("id")
            if not isinstance(block_id, str) or block_id in errored_ids:
                new_content.append(block)
                continue
            new_content.append({**block, "input": {}})
            msg_changed = True

        if msg_changed:
            out.append({**msg, "content": new_content})
            changed = True
        else:
            out.append(msg)

    return out if changed else messages
