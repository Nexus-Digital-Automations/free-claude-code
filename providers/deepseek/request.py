"""Request builder for DeepSeek provider."""

from typing import Any

from loguru import logger

from providers.common.message_converter import build_base_request_body

# Models with built-in chain-of-thought that must NOT receive the thinking param.
# EXTENSION POINT: add future deepseek-vN-reasoner variants here.
_BUILTIN_REASONER_MODELS: frozenset[str] = frozenset({"deepseek-reasoner"})

# Short factual instruction appended to the system prompt to encourage the
# model to batch independent tool calls. parallel_tool_calls=True permits
# multi-tool emission; this nudge encourages it. Counterpart:
# settings.deepseek_parallel_tool_call_nudge.
_PARALLEL_TOOL_CALL_NUDGE = (
    "Tool-call parallelism: when multiple tool calls in a turn are independent "
    "(no data dependency between them), emit them all in a single response so "
    "they can run in parallel. Examples of independent calls: reading several "
    "unrelated files, running multiple greps with different patterns, looking "
    "up several pieces of context. Sequential calls are only required when one "
    "tool's output is needed as input to the next."
)


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
        # Nudge fires on the same gate so the system-prompt bytes don't shift
        # between tool-bearing and tool-less requests for the same session.
        # Skip when no system message exists at index 0 — never invent one.
        if parallel_tool_call_nudge:
            messages = body.get("messages") or []
            if messages and messages[0].get("role") == "system":
                existing = messages[0].get("content") or ""
                messages[0] = {
                    **messages[0],
                    "content": f"{existing}\n\n{_PARALLEL_TOOL_CALL_NUDGE}",
                }

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
