"""Owns: ContextOptimizer — orchestrates all optimization layers for a single request.

Pipeline per request (Layer -1 first, then message tiers):
  Layer -1 (repo index, optional):
    git HEAD SHA → load/build stable prefix → cosine search → prepend to system
  Message tiers:
    tier0 (NLP truncation) → tier0b/0c/0d (Ollama digests) → tier1 (thinking strip)
    → prefix cache lookup → [tier2 compaction if needed]

WHY Layer -1 runs first: the system prompt prefix must be established before any
message-level token counting so tier2 thresholds account for its size.

WHY cache-first: tier0 hashes every tool result for dedup; on warm
conversations that work is wasted when a cached prefix would have
short-circuited the request. Hashing the raw messages costs ~µs.

Does NOT own: individual tier logic, cache storage, Ollama management,
token counting, repo indexing, or prompt construction — all delegated.
Called by: any application code that imports context_optimizer.
Calls: tiers/tier0, tiers/tier1, tiers/tier2, cache.PrefixCache,
       token_counter.count_tokens, repo_index.RepoIndex (when enabled).

# @stable — external callers depend on ContextOptimizer.optimize() signature.
# EXTENSION POINT: add new tiers inside optimize() between tier1 and the
#   tier2 thresholds.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Awaitable, Callable
from typing import ClassVar

from loguru import logger

from .cache import PrefixCache
from .settings import ContextOptimizerSettings
from .tiers import tier0, tier0b, tier0c, tier0d, tier1, tier2
from .token_counter import count_tokens
from ._core import content_hash

LLMProvider = Callable[[str], Awaitable[str]]

_DEFAULT_SETTINGS = ContextOptimizerSettings()

# ── Cache-path resolution ──────────────────────────────────────────────
# The persist-path is scoped to the working directory so two projects never
# share cache entries.  Resolution order:
#   1. Explicit settings.context_cache_dir (if non-None and non-empty).
#   2. Git root of cwd → <git_root>/.claude/data/context-cache.json
#   3. Bare cwd (not a git repo) → <cwd>/.claude/data/context-cache.json
#   4. context_cache_dir="" disables persistence explicitly.
# Counterpart: providers/common/context_optimizer.py maps from proxy settings.


def _resolve_repo_root(settings: ContextOptimizerSettings) -> str:
    if settings.repo_index_root:
        return settings.repo_index_root
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return os.getcwd()


def _extract_last_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
    return ""


def _prepend_repo_context(prefix_text: str, suffix: str, system: str | list | None) -> str | list:
    block = prefix_text + ("\n\n---\n\n" + suffix if suffix else "") + "\n\n---\n\n"
    if system is None:
        return block
    if isinstance(system, str):
        return block + system
    if isinstance(system, list):
        return [{"type": "text", "text": block}, *system]
    return system


def _resolve_cache_path(settings: ContextOptimizerSettings) -> str | None:
    """Return the absolute cache file path, or None to disable persistence."""
    if settings.context_cache_dir is not None:
        # Explicitly set — even "" means "disable", anything else is a dir
        if settings.context_cache_dir == "":
            return None
        return os.path.join(settings.context_cache_dir, "context-cache.json")
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        root = r.stdout.strip() if r.returncode == 0 else os.getcwd()
    except Exception:
        root = os.getcwd()
    cache_dir = os.path.join(root, ".claude", "data")
    return os.path.join(cache_dir, "context-cache.json")


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
            persist_path = _resolve_cache_path(settings)
            cls._cache = PrefixCache(
                settings.prefix_cache_max_entries, persist_path=persist_path,
            )
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

        # --- Layer -1: Repo index stable prefix + dynamic suffix ---
        # Lazy import so torch/networkx are never loaded when the feature is off.
        # run_in_executor wraps the synchronous build() to avoid blocking the loop.
        # Counterpart: repo_index/index.py RepoIndex.get_or_build stores to disk.
        if settings.repo_index_enabled:
            try:
                from .repo_index import RepoIndex  # noqa: PLC0415
                repo_root = _resolve_repo_root(settings)
                loop = asyncio.get_running_loop()
                loaded = await loop.run_in_executor(
                    None, RepoIndex.get_or_build, repo_root, settings
                )
                if loaded is not None:
                    last_user_text = _extract_last_user_text(messages)
                    results = loaded.query(last_user_text, top_k=settings.repo_index_query_top_k)
                    system = _prepend_repo_context(
                        loaded.prefix_text, loaded.format_suffix(results), system
                    )
                    logger.info(
                        "REPO_INDEX: prefix_applied prefix_bytes={} suffix_chunks={} commit={}",
                        len(loaded.prefix_text), len(results), loaded.commit_hash[:7],
                    )
            except Exception as exc:
                logger.warning(
                    "REPO_INDEX: layer-1 failed, skipping — {}: {}", type(exc).__name__, exc
                )

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

        # --- Tier 0b: Ollama tool-result digest ---
        # Runs after Tier 0's mechanical truncation. Ollama produces a
        # content-aware summary for tool_results above the byte threshold;
        # the digest is content-hashed so identical inputs always yield the
        # same bytes, keeping DeepSeek's prefix cache hot.
        msgs = await tier0b.apply(msgs, settings)

        # --- Tier 0c: Ollama tool_use input compaction ---
        # Old assistant tool_use blocks with large input dicts (Edit, Write,
        # MultiEdit) get their input replaced by a digest. Recent calls are
        # preserved verbatim so the model can still reference its latest args.
        msgs = await tier0c.apply(msgs, settings)

        # --- Tier 0d: Ollama long-user-paste digest ---
        # Historical user-message text blocks above the high byte threshold
        # get digested. The active (last) user message is always skipped.
        msgs = await tier0d.apply(msgs, settings)

        # --- Tier 1: thinking-block strip ---
        msgs = tier1.apply(msgs, settings.max_thinking_turns)
        sys = system

        # --- Prefix cache lookup (post-cleanup) ---
        # Tier 2a stores summaries computed *after* the cleanup tiers ran,
        # so the cache key is post-cleanup. Looking up here (after tier0/0b/0c/0d/1
        # have run) is what makes Tier 2a's background work actually findable.
        # Counterpart: tier2._do_ollama_call / compact_sync stores via cache.store.
        cache_result = cache.lookup(msgs, sys)
        if cache_result is not None:
            msgs, sys = cache_result
            tokens = count_tokens(msgs, sys, tools, tokenizer_name=settings.tokenizer_name)
            return msgs, sys, tokens

        tokens = count_tokens(msgs, sys, tools, tokenizer_name=settings.tokenizer_name)

        # --- Tier 2: LLM compaction ---
        if tokens >= settings.compact_threshold_tokens and llm_provider is not None:
            logger.info(
                "CONTEXT_OPT: triggering sync compaction tokens={} threshold={} msgs={}",
                tokens, settings.compact_threshold_tokens, len(msgs),
            )
            result = await tier2.compact_sync(msgs, sys, settings, llm_provider, cache)
            if result is not None:
                new_msgs, new_sys = result
                new_tokens = count_tokens(new_msgs, new_sys, tools, tokenizer_name=settings.tokenizer_name)
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
        # @internal — tests/conftest uses this to isolate cache state.
        # Clear the on-disk cache file so the next PrefixCache creation
        # starts with an empty store instead of picking up stale entries
        # from a previous test run.
        if cls._cache is not None:
            cls._cache.clear()
        cls._cache = None
        tier2.reset_for_test()
        tier0b.reset_for_test()
        tier0c.reset_for_test()
        tier0d.reset_for_test()
