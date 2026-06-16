"""Tier 0b alternative: delegate tool-output compression to a Headroom sidecar.

Owns: POSTing the message list to a running Headroom proxy's ``/v1/compress``
endpoint and returning the compressed messages. Selected in place of the Ollama
Tier 0b digester when ``settings.headroom_enabled`` is True — the two are
mutually exclusive (see ``optimizer.optimize``), so only one tool-output
compressor ever runs per request.

Headroom compresses tool_result / assistant tool-output content with its
deterministic structural compressors (SmartCrusher / log / code) and leaves
user and system messages untouched (its ``compress_user_messages`` /
``compress_system_messages`` default to False), matching this pipeline's
protect-the-active-turn philosophy.

Never raises. On any sidecar failure — timeout, connection refused, non-200,
or a malformed body — the original messages pass through unchanged, so a
sidecar outage degrades to "no tool-output compression this request" rather
than a broken request. Counterpart upstream: packages/headroom ``/v1/compress``.
"""

from __future__ import annotations

import httpx
from loguru import logger

from ..settings import ContextOptimizerSettings

# The /v1/compress body requires a model field; it only steers token-counting
# inside Headroom and never reaches a provider, so a stable Anthropic id is fine.
_COMPRESS_MODEL = "claude-3-5-sonnet-20241022"


async def apply(
    messages: list[dict],
    settings: ContextOptimizerSettings,
) -> list[dict]:
    """Return messages with tool outputs compressed by the Headroom sidecar.

    Falls back to the unmodified ``messages`` on any failure. Never raises.
    """
    if not settings.headroom_enabled:
        return messages

    url = f"{settings.headroom_url.rstrip('/')}/v1/compress"
    payload = {"model": _COMPRESS_MODEL, "messages": messages}
    try:
        async with httpx.AsyncClient(
            timeout=settings.headroom_timeout_seconds
        ) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            logger.warning(
                "HEADROOM: sidecar status={} — passing messages through uncompressed",
                resp.status_code,
            )
            return messages
        body = resp.json()
    except Exception as exc:
        logger.warning(
            "HEADROOM: sidecar call failed ({}: {}) — passing messages "
            "through uncompressed",
            type(exc).__name__,
            exc,
        )
        return messages

    out = body.get("messages")
    if not isinstance(out, list) or not out:
        logger.warning(
            "HEADROOM: sidecar returned no messages — passing through uncompressed"
        )
        return messages

    # Log a real, tokenizer-independent savings figure: the sidecar's own
    # tokens_before/after summary is unreliable, so measure the content delta.
    before = sum(len(str(m.get("content", ""))) for m in messages)
    after = sum(len(str(m.get("content", ""))) for m in out)
    if after < before:
        logger.info(
            "HEADROOM: chars_before={} chars_after={} saved={} transforms={}",
            before,
            after,
            before - after,
            body.get("transforms_applied"),
        )
    return out
