"""Owns: per-request prompt-cache hit/miss accounting.

Does NOT own: API calls, routing, or logging configuration.
Called by: optimizer.py (after each LLM response completes) or provider-level callers.
Calls: loguru.

WHY here and not in openai_compat.py: openai_compat.py already logs raw cached_tokens at
the provider level. This tracker accumulates across streaming chunks and provides a
repo-index-scoped summary that correlates cache performance with prefix size.
"""

from __future__ import annotations

from loguru import logger

from ._types import RequestCacheStats


class CacheStatsTracker:
    """Accumulates prompt-cache token counts from API usage objects across streaming chunks.

    States: accumulating ──log_summary()──> (still accumulating, summary already logged)
    Not thread-safe — one instance per request (asyncio is single-threaded per process).
    """

    def __init__(self, request_id: str, prefix_bytes: int = 0) -> None:
        self._stats = RequestCacheStats(
            request_id=request_id,
            prefix_bytes=prefix_bytes,
        )

    def record_api_usage(self, usage: object, *, provider: str = "unknown") -> None:
        """Extract cache token counts from a usage object and accumulate.

        Handles two shapes:
          DeepSeek/OpenAI: usage.prompt_tokens_details.cached_tokens
          Anthropic:       usage.cache_read_input_tokens (hit)
                           usage.cache_creation_input_tokens (miss/write)
        Silently ignores usage objects that carry neither shape.
        """
        if usage is None:
            return

        hit = _extract_attr(usage, "cache_read_input_tokens")
        miss = _extract_attr(usage, "cache_creation_input_tokens")

        if hit is None and miss is None:
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                hit = _extract_attr(details, "cached_tokens")

        if hit is not None:
            self._stats.prompt_cache_hit_tokens += hit
        if miss is not None:
            self._stats.prompt_cache_miss_tokens += miss

    def log_summary(self) -> None:
        s = self._stats
        total = s.prompt_cache_hit_tokens + s.prompt_cache_miss_tokens
        hit_pct = f"{s.prompt_cache_hit_tokens / total * 100:.1f}" if total else "n/a"
        logger.info(
            "REPO_INDEX: cache_stats request_id={} hit={} miss={} hit_pct={}% prefix_bytes={}",
            s.request_id,
            s.prompt_cache_hit_tokens,
            s.prompt_cache_miss_tokens,
            hit_pct,
            s.prefix_bytes,
        )

    @property
    def stats(self) -> RequestCacheStats:
        return self._stats


def _extract_attr(obj: object, name: str) -> int | None:
    val = getattr(obj, name, None)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None
