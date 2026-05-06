"""DeepSeek provider implementation."""

from typing import Any

from loguru import logger

from providers.base import ProviderConfig
from providers.common.sse_builder import SSEBuilder
from providers.openai_compat import OpenAICompatibleProvider

from .request import _count_tool_results, build_request_body

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider using OpenAI-compatible chat completions."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        parallel_tool_call_nudge: bool = True,
    ):
        super().__init__(
            config,
            provider_name="DEEPSEEK",
            base_url=config.base_url or DEEPSEEK_BASE_URL,
            api_key=config.api_key,
        )
        # Counterpart: settings.deepseek_parallel_tool_call_nudge, threaded
        # through api/dependencies.py at provider construction time so the
        # value is fixed for the provider's lifetime.
        self._parallel_tool_call_nudge = parallel_tool_call_nudge

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request),
            parallel_tool_call_nudge=self._parallel_tool_call_nudge,
        )

    def _on_stream_finish(
        self,
        request: Any,
        sse: SSEBuilder,
        request_id: str | None,
        error_occurred: bool,
    ) -> None:
        """Emit parallel-miss observability for tool-bearing turns.

        Logs `parallel_miss=true` when the model responded to ≥2 tool_results
        with a single tool_call — the signature of a parallel-call opportunity
        the model declined to take. Errored streams are skipped so transient
        upstream failures don't pollute the rate. Counterpart:
        providers/deepseek/request.py:_count_tool_results.

        Independent of the parallel_tool_call_nudge flag — observability must
        keep emitting when the nudge is in kill-switch state so we can see
        whether disabling it changed the rate.
        """
        if error_occurred:
            return
        tool_calls_emitted = len(sse.blocks.tool_states)
        tool_results_in = _count_tool_results(request)
        parallel_miss = tool_calls_emitted == 1 and tool_results_in >= 2
        logger.info(
            "PROVIDER: parallel_miss provider=DEEPSEEK request_id={} "
            "tool_calls_emitted={} tool_results_in={} parallel_miss={}",
            request_id,
            tool_calls_emitted,
            tool_results_in,
            parallel_miss,
        )

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Retry without thinking when compacted history drops reasoning_content.

        DeepSeek requires every assistant turn's reasoning_content to be echoed
        back in subsequent requests. Claude Code's auto-compaction strips thinking
        blocks from history, breaking the round-trip. On that specific 400, retry
        with thinking disabled so the request succeeds without the stale history.
        """
        if "reasoning_content" not in str(error):
            return None
        retry = {**body}
        # Remove thinking param from extra_body; keep any other extra fields.
        extra = {k: v for k, v in (body.get("extra_body") or {}).items() if k != "thinking"}
        if extra:
            retry["extra_body"] = extra
        else:
            retry.pop("extra_body", None)
        # Strip reasoning_content from all history messages so they're clean.
        retry["messages"] = [
            {k: v for k, v in msg.items() if k != "reasoning_content"}
            for msg in body.get("messages", [])
        ]
        return retry
