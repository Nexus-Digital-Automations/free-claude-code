"""Request builder for DeepSeek provider.

Owns: Anthropic→OpenAI request shape for DeepSeek, including the two
parallel-tool-call encouragement levers (system-prompt nudge and trailing
reminder) and the `_count_tool_results` helper used by both the request
builder and the parallel-miss observability hook in DeepSeekProvider.

Does NOT own: response/streaming logic (that lives in the shared
OpenAICompatibleProvider). Counterpart: providers/deepseek/client.py.
"""

from typing import Any

from loguru import logger

from providers.common.message_converter import build_base_request_body, get_block_type

# Models with built-in chain-of-thought that must NOT receive the thinking param.
# EXTENSION POINT: add future deepseek-vN-reasoner variants here.
_BUILTIN_REASONER_MODELS: frozenset[str] = frozenset({"deepseek-reasoner"})

# Appended to the system prompt to encourage the model to batch independent
# tool calls. parallel_tool_calls=True *permits* multi-tool emission; this
# nudge *encourages* it. Cost framing + 2 mini-examples chosen because models
# imitate concrete examples better than abstract instructions (empirically:
# ~38% multi-tool baseline pre-strengthening on DeepSeek).
# Counterpart: settings.deepseek_parallel_tool_call_nudge (kill-switch).
_PARALLEL_TOOL_CALL_NUDGE = (
    "Tool-call parallelism: when multiple tool calls in a turn are independent "
    "(no data dependency between them), emit them all in a single response so "
    "they can run in parallel. Each separate turn re-ships the full chat "
    "history at meaningful token cost — batching is a free speed and cost win.\n"
    "Examples of correct batching:\n"
    "  - Reading three unrelated files: emit Read, Read, Read in one response.\n"
    "  - Searching for two unrelated patterns: emit Grep, Grep in one response.\n"
    "Sequential calls are required only when one tool's output is the input "
    "to the next."
)

# Trailing user-message reminder injected only on turns that carry ≥2
# tool_results — i.e. the model is about to respond to multiple parallel
# results and the next turn is the highest-yield opportunity to repeat the
# pattern. Trailing position is post-cache-prefix, so this does not
# invalidate the system-prompt cache. ~25 tokens.
# Counterpart: settings.deepseek_parallel_tool_call_nudge gates both nudge
# and reminder so the kill-switch covers A+B together.
_PARALLEL_TOOL_CALL_REMINDER = (
    "Reminder: if your next tool calls don't depend on each other's output, "
    "emit them in one response so they run in parallel."
)


def _count_tool_results(request_data: Any) -> int:
    """Count tool_result blocks in the latest user turn (Anthropic format).

    Pre-conversion mirror of how `_convert_user_message` later splits this
    turn into one role=tool message per tool_result. Used by:
      - build_request_body (decides whether to inject the trailing reminder)
      - DeepSeekProvider._on_stream_finish (decides parallel_miss truthiness)

    Returns 0 (never None) when there is no last user message or the
    content is a plain string. Empty-collection-as-absence per CLAUDE.md.
    """
    messages = getattr(request_data, "messages", None) or []
    if not messages:
        return 0
    last = messages[-1]
    if getattr(last, "role", None) != "user":
        return 0
    content = getattr(last, "content", None)
    if not isinstance(content, list):
        return 0
    return sum(1 for block in content if get_block_type(block) == "tool_result")


def build_request_body(
    request_data: Any,
    *,
    thinking_enabled: bool,
    parallel_tool_call_nudge: bool = True,
) -> dict:
    """Build OpenAI-format request body from Anthropic request for DeepSeek."""
    logger.debug(
        "DEEPSEEK_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    body = build_base_request_body(
        request_data,
        include_reasoning_content=True,
    )

    # parallel_tool_calls only meaningful when tools are present (OpenAI spec).
    # Set explicitly so we don't depend on undocumented upstream defaults; logs
    # on 2026-05-04..05 already showed up to 9 parallel tool_calls per response,
    # so this preserves observed behavior under a stable contract.
    if body.get("tools"):
        body["parallel_tool_calls"] = True
        # Both nudge (A) and trailing reminder (B) fire under the same gate so
        # the kill-switch covers them together. System-prompt nudge is skipped
        # when no system message exists at index 0 — never invent one.
        if parallel_tool_call_nudge:
            messages = body.get("messages") or []
            if messages and messages[0].get("role") == "system":
                existing = messages[0].get("content") or ""
                messages[0] = {
                    **messages[0],
                    "content": f"{existing}\n\n{_PARALLEL_TOOL_CALL_NUDGE}",
                }
            # Trailing reminder fires only when this turn carries ≥2 tool_results.
            # Rationale: the system-prompt nudge gets diluted across long
            # contexts; the highest-yield moment to repeat the rule is right
            # before the model responds to multiple parallel results — it's
            # already demonstrating it can produce parallel calls.
            # Cache-safe: trailing position is post-cache-prefix.
            if _count_tool_results(request_data) >= 2:
                body.setdefault("messages", []).append(
                    {"role": "user", "content": _PARALLEL_TOOL_CALL_REMINDER}
                )

    extra_body: dict[str, Any] = {}
    request_extra = getattr(request_data, "extra_body", None)
    if request_extra:
        extra_body.update(request_extra)

    if thinking_enabled and body.get("model") not in _BUILTIN_REASONER_MODELS:
        extra_body.setdefault("thinking", {"type": "enabled"})

    if extra_body:
        body["extra_body"] = extra_body

    logger.debug(
        "DEEPSEEK_REQUEST: conversion done model={} msgs={} tools={} parallel_tool_calls={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
        body.get("parallel_tool_calls"),
    )
    return body
