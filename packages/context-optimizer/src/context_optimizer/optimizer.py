"""Owns: ContextOptimizer — orchestrates all four tiers for a single request.

Tier pipeline per request:
  raw -> prefix cache check (cheapest first) ->
    on hit:  return cached truncation immediately, skipping tier0/tier1
    on miss: tier0 (NLP) -> tier1 (thinking strip) ->
      tokens >= hard_threshold:  await tier2.compact_sync (blocking)
      tokens >= soft_threshold:  tier2.schedule_background (fire-and-forget)
      else:                      return as-is

WHY cache-first: tier0 hashes every tool result for dedup; on warm
conversations that work is wasted when a cached prefix would have
short-circuited the request. Hashing the raw messages costs ~µs.

Does NOT own: individual tier logic, cache storage, Ollama management,
token counting, or prompt construction — all delegated to submodules.
Called by: any application code that imports context_optimizer.
Calls: tiers/tier0, tiers/tier1, tiers/tier2, cache.PrefixCache,
       token_counter.count_tokens.

# @stable — external callers depend on ContextOptimizer.optimize() signature.
# EXTENSION POINT: add new tiers inside optimize() between tier1 and the
#   tier2 thresholds.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import ClassVar

from loguru import logger

from .cache import PrefixCache
from .settings import ContextOptimizerSettings
from .tiers import tier0, tier1, tier2
from .token_counter import count_tokens
from ._core import content_hash

LLMProvider = Callable[[str], Awaitable[str]]

_DEFAULT_SETTINGS = ContextOptimizerSettings()


class ContextOptimizer:
    """Stateless entry point — all mutable state lives in the module-level
    cache and tier2 inflight/semaphore sets.

    Multiple simultaneous callers share the same cache (class-level),
    which is safe because asyncio is single-threaded per process.
    """

    _cache: ClassVar[PrefixCache | None] = None

    @classmethod
    def _get_cache(cls, settings: ContextOptimizerSettings) -> PrefixCache:
        if cls._cache is None or cls._cache._max_entries != settings.prefix_cache_max_entries:
            cls._cache = PrefixCache(settings.prefix_cache_max_entries)
        return cls._cache

    @classmethod
    async def optimize(
        cls,
        messages: list[dict],
        system: str | list | None = None,
        settings: ContextOptimizerSettings | None = None,
        llm_provider: LLMProvider | None = None,
        tools: list | None = None,
    ) -> tuple[list[dict], str | list | None, int]:
        """Apply all optimization tiers. Returns (messages, system, token_count).

        Never raises — failures degrade gracefully to the previous tier's result.
        The returned token_count reflects the state after all optimizations.

        llm_provider: async callable (prompt: str) -> str, used for Tier 2b
        and as a fallback when Ollama is near-capacity. Pass None to disable
        Tier 2b (Tier 0/1 and background Ollama still run).
        # @stable
        """
        if settings is None:
            settings = _DEFAULT_SETTINGS

        cache = cls._get_cache(settings)

        # --- Prefix cache (cheapest path first) ---
        # Hashing raw messages avoids paying tier0+tier1 cost on warm
        # conversations whose prefix is already cached.
        cache_result = cache.lookup(messages, system)
        if cache_result is not None:
            msgs, sys = cache_result
            tokens = count_tokens(msgs, sys, tools)
            return msgs, sys, tokens

        # --- Tier 0: free NLP cleanup ---
        before_bytes = sum(
            len(str(m.get("content", ""))) for m in messages
        )
        msgs = tier0.apply(
            messages,
            settings.tier0_max_lines,
            settings.tier0_head_lines,
            settings.tier0_tail_lines,
        )
        after_bytes = sum(len(str(m.get("content", ""))) for m in msgs)
        if before_bytes != after_bytes:
            logger.info(
                "CONTEXT_OPT: tier0 bytes_before={} bytes_after={} saved={}",
                before_bytes, after_bytes, before_bytes - after_bytes,
            )

        # --- Tier 1: thinking-block strip ---
        msgs = tier1.apply(msgs, settings.max_thinking_turns)
        sys = system

        tokens = count_tokens(msgs, sys, tools)

        # --- Tier 2: LLM compaction ---
        if tokens >= settings.compact_threshold_tokens and llm_provider is not None:
            logger.info(
                "CONTEXT_OPT: triggering sync compaction tokens={} threshold={} msgs={}",
                tokens, settings.compact_threshold_tokens, len(msgs),
            )
            result = await tier2.compact_sync(msgs, sys, settings, llm_provider, cache)
            if result is not None:
                new_msgs, new_sys = result
                new_tokens = count_tokens(new_msgs, new_sys, tools)
                logger.info(
                    "CONTEXT_OPT: compacted {} -> {} messages, tokens {} -> {}",
                    len(msgs), len(new_msgs), tokens, new_tokens,
                )
                msgs, sys, tokens = new_msgs, new_sys, new_tokens

        elif tokens >= settings.compact_soft_threshold_tokens:
            near_hard = tokens >= settings.compact_deepseek_fallback_threshold_tokens
            inflight_key = content_hash(msgs)
            tier2.schedule_background(
                msgs, settings, llm_provider, cache, inflight_key,
                use_provider_fallback=near_hard,
            )

        return msgs, sys, tokens

    @classmethod
    def _reset_for_test(cls) -> None:
        # @internal — tests/conftest uses this to isolate cache state
        cls._cache = None
        tier2.reset_for_test()
