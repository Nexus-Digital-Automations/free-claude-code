"""Owns: LLM-based compaction — both sync (Tier 2b) and background (Tier 2a).

Tier 2b: called inline when tokens >= hard threshold. Calls llm_provider
         (caller-supplied async callable) and applies the summary.
Tier 2a: scheduled as a background asyncio task at soft threshold.
         Calls local Ollama (via openai SDK) and stores in the prefix cache.

Does NOT own: Ollama daemon management (ollama_supervisor.py), caching
logic (cache.py), or the compaction prompt (prompts.py).
Called by: optimizer.py.
Calls: prompts.build_prompt/parse_response, cache.PrefixCache.store,
       ollama_supervisor.OllamaSupervisor.ensure_ready.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger
from openai import AsyncOpenAI

from .._core import apply_summary
from ..prompts import build_prompt, parse_response

if TYPE_CHECKING:
    from ..cache import PrefixCache
    from ..settings import ContextOptimizerSettings


LLMProvider = Callable[[str], Awaitable[str]]

# Serialises all background Ollama calls so multiple concurrent callers
# don't pile requests onto a single Ollama process.
_ollama_semaphore: asyncio.Semaphore | None = None
_inflight: set[str] = set()
_background_tasks: set[asyncio.Task] = set()


def _get_semaphore() -> asyncio.Semaphore:
    global _ollama_semaphore
    if _ollama_semaphore is None:
        _ollama_semaphore = asyncio.Semaphore(1)
    return _ollama_semaphore


def _clamp_split_index(split_index: int, n: int, keep_recent: int) -> int:
    """Clamp the LLM-chosen split so the last `keep_recent` messages survive.

    The LLM is instructed to pick split_index in [4, n-2] but has no view of
    the keep_recent floor. Without this clamp an aggressive summary could
    collapse the most recent context the next turn depends on.

    Floor of 4 preserved because parse_response rejects anything lower —
    keeping fewer than n-4 verbatim is preferable to skipping compaction
    when n is small relative to keep_recent.
    """
    return max(4, min(split_index, n - keep_recent))


def _classify_exception(exc: BaseException) -> str:
    """Single-token reason label for COMPACTION: logs.

    Heuristic on exception class names so we don't pull in the openai/httpx
    exception types as direct dependencies. Order matters — Timeout subclasses
    of ConnectionError must hit the timeout branch first.
    """
    name = type(exc).__name__
    msg = str(exc)
    if "Timeout" in name or "timed out" in msg.lower():
        return "timeout"
    if "NotFound" in name or "404" in msg:
        return "model_missing"
    if "RateLimit" in name or "429" in msg:
        return "busy"
    if "Connection" in name or "Network" in name:
        return "network"
    return "error"


# ---- Tier 2b: sync compaction via llm_provider ----

async def compact_sync(
    messages: list[dict],
    system: Any,
    settings: "ContextOptimizerSettings",
    llm_provider: LLMProvider,
    cache: "PrefixCache",
) -> tuple[list[dict], Any] | None:
    """Compact via caller-supplied LLM. Returns (messages, system) or None on failure.

    Always stores in cache on success for subsequent requests.
    Never raises — returns None on any failure.
    """
    prompt = build_prompt(messages, settings.render_preview_chars)
    try:
        content = await llm_provider(prompt)
    except Exception as exc:
        logger.warning(
            "CONTEXT_OPT: tier=2b outcome=fallback reason={} {}: {}",
            _classify_exception(exc), type(exc).__name__, exc,
        )
        return None

    parsed = parse_response(content, len(messages))
    if parsed is None:
        logger.warning(
            "CONTEXT_OPT: tier=2b outcome=fallback reason=parse_error first_200={!r}",
            content[:200],
        )
        return None

    raw_split, summary = parsed
    split_index = _clamp_split_index(raw_split, len(messages), settings.tier2_keep_recent_turns)
    if split_index != raw_split:
        logger.info(
            "CONTEXT_OPT: clamped split_index llm={} clamped={} keep_recent={}",
            raw_split, split_index, settings.tier2_keep_recent_turns,
        )
    cache.store(messages, split_index, summary)
    logger.info(
        "CONTEXT_OPT: provider compacted split_index={} msgs_before={} summary_chars={}",
        split_index, len(messages), len(summary),
    )
    return apply_summary(messages, split_index, summary, system)


# ---- Tier 2a: background Ollama ----

def schedule_background(
    messages: list[dict],
    settings: "ContextOptimizerSettings",
    llm_provider: LLMProvider | None,
    cache: "PrefixCache",
    cache_key: str,
    *,
    use_provider_fallback: bool,
) -> None:
    """Fire-and-forget background compaction. Noop if same prefix is already in-flight."""
    if cache_key in _inflight:
        logger.debug("CONTEXT_OPT: background compaction already in-flight, skipping")
        return
    _inflight.add(cache_key)
    logger.info(
        "CONTEXT_OPT: scheduling background compaction msgs={} provider_fallback={}",
        len(messages), use_provider_fallback,
    )
    task = asyncio.create_task(
        _run_background(messages, settings, llm_provider, cache, cache_key, use_provider_fallback)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _run_background(
    messages: list[dict],
    settings: "ContextOptimizerSettings",
    llm_provider: LLMProvider | None,
    cache: "PrefixCache",
    cache_key: str,
    use_provider_fallback: bool,
) -> None:
    try:
        if use_provider_fallback and llm_provider is not None:
            sem = _get_semaphore()
            if not sem.locked():
                async with sem:
                    succeeded = await _do_ollama_call(messages, settings, cache)
                if not succeeded:
                    logger.info("CONTEXT_OPT: Ollama failed near hard limit, using provider fallback")
                    succeeded = await _compact_for_cache(messages, settings, llm_provider, cache)
            else:
                logger.info("CONTEXT_OPT: Ollama busy near hard limit, using provider fallback")
                succeeded = await _compact_for_cache(messages, settings, llm_provider, cache)
        else:
            async with _get_semaphore():
                succeeded = await _do_ollama_call(messages, settings, cache)

        if succeeded:
            logger.info(
                "CONTEXT_OPT: background compaction cached summary for {} message prefix",
                len(messages),
            )
    except Exception as exc:
        logger.warning(
            "CONTEXT_OPT: tier=2a outcome=crash reason={} {}: {}",
            _classify_exception(exc), type(exc).__name__, exc,
        )
    finally:
        _inflight.discard(cache_key)


async def _do_ollama_call(
    messages: list[dict],
    settings: "ContextOptimizerSettings",
    cache: "PrefixCache",
) -> bool:
    """Raw Ollama call; caller must hold the semaphore. Returns True if cached."""
    from ..ollama_supervisor import OllamaSupervisor

    if not await OllamaSupervisor.ensure_ready(settings):
        return False

    prompt = build_prompt(messages, settings.render_preview_chars)
    # async with ensures close() runs even when chat.completions.create raises
    # mid-call. The previous form leaked the underlying httpx connection on
    # exception because client.close() ran only on the success path.
    try:
        async with AsyncOpenAI(
            api_key="ollama",  # pragma: allowlist secret — placeholder for local Ollama
            base_url=settings.ollama_base_url,
        ) as client:
            resp = await client.chat.completions.create(
                model=settings.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.compaction_max_tokens,
                temperature=settings.compaction_temperature,
                stream=False,
            )
        content = resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning(
            "CONTEXT_OPT: tier=2a outcome=fallback reason={} {}: {}",
            _classify_exception(exc), type(exc).__name__, exc,
        )
        return False

    parsed = parse_response(content, len(messages))
    if parsed is None:
        logger.warning(
            "CONTEXT_OPT: tier=2a outcome=fallback reason=parse_error first_200={!r}",
            content[:200],
        )
        return False

    raw_split, summary = parsed
    split_index = _clamp_split_index(raw_split, len(messages), settings.tier2_keep_recent_turns)
    cache.store(messages, split_index, summary)
    logger.info(
        "CONTEXT_OPT: ollama compacted split_index={} msgs_before={} summary_chars={}",
        split_index, len(messages), len(summary),
    )
    return True


async def _compact_for_cache(
    messages: list[dict],
    settings: "ContextOptimizerSettings",
    llm_provider: LLMProvider,
    cache: "PrefixCache",
) -> bool:
    """Background-mode provider call: just cache, don't apply."""
    prompt = build_prompt(messages, settings.render_preview_chars)
    try:
        content = await llm_provider(prompt)
    except Exception as exc:
        logger.warning(
            "CONTEXT_OPT: tier=2a-fallback outcome=fallback reason={} {}: {}",
            _classify_exception(exc), type(exc).__name__, exc,
        )
        return False
    parsed = parse_response(content, len(messages))
    if parsed is None:
        logger.warning(
            "CONTEXT_OPT: tier=2a-fallback outcome=fallback reason=parse_error first_200={!r}",
            content[:200],
        )
        return False
    raw_split, summary = parsed
    split_index = _clamp_split_index(raw_split, len(messages), settings.tier2_keep_recent_turns)
    cache.store(messages, split_index, summary)
    return True


def reset_for_test() -> None:
    # @internal — test isolation only
    global _ollama_semaphore
    _ollama_semaphore = None
    _inflight.clear()
    _background_tasks.clear()
