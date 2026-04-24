"""DeepSeek provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider using OpenAI-compatible chat completions."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="DEEPSEEK",
            base_url=config.base_url or DEEPSEEK_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Build request body, disabling thinking when history lacks reasoning_content.

        DeepSeek requires that every prior assistant turn's reasoning_content is
        echoed back when thinking mode is on. Claude Code's auto-compaction strips
        thinking blocks, so we detect the gap pre-flight and fall back to no-thinking
        rather than letting DeepSeek return a 400 mid-stream (where our retry can't
        intercept it because the OpenAI SDK connects lazily).
        """
        thinking_enabled = self._is_thinking_enabled(request)
        body = build_request_body(request, thinking_enabled=thinking_enabled)

        if thinking_enabled:
            assistant_msgs = [m for m in body.get("messages", []) if m.get("role") == "assistant"]
            # Prior assistant turns exist but none carry reasoning_content → compaction
            # stripped the thinking blocks. Disable thinking so DeepSeek accepts the request.
            if assistant_msgs and not any(m.get("reasoning_content") for m in assistant_msgs):
                body = build_request_body(request, thinking_enabled=False)

        return body

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
