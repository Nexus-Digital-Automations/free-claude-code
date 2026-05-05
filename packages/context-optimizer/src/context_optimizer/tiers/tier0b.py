"""Owns: per-tool-result Ollama digestion. Replaces long tool_result content
with a content-aware summary so subsequent requests carry a smaller history.

The cache is the load-bearing piece. Identical bytes-in MUST produce
identical bytes-out across calls, otherwise DeepSeek's prefix cache
invalidates on every turn and we lose more than we save.

State diagram (per process, single asyncio loop):

    miss ──Ollama success──> cached  (LRU, bounded)
         ──Ollama failure──> passthrough  (no cache write — try again next time)
    cached ──evicted on overflow──> miss

Does NOT own: Ollama daemon management (ollama_supervisor.py), the digest
prompt (prompts.py), or whole-conversation summarization (tier2.py).
Called by: optimizer.py, between tier0 and tier1.
Calls: prompts.build_digest_prompt/parse_digest_response,
       ollama_supervisor.OllamaSupervisor.ensure_ready, AsyncOpenAI.

# @stable — optimizer.py depends on apply()'s signature.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from loguru import logger
from openai import AsyncOpenAI

from ..prompts import build_digest_prompt, parse_digest_response

if TYPE_CHECKING:
    from ..settings import ContextOptimizerSettings


_digest_cache: OrderedDict[str, str] = OrderedDict()


def _cache_get(key: str, max_entries: int) -> str | None:
    if key not in _digest_cache:
        return None
    _digest_cache.move_to_end(key)
    return _digest_cache[key]


def _cache_put(key: str, value: str, max_entries: int) -> None:
    _digest_cache[key] = value
    _digest_cache.move_to_end(key)
    while len(_digest_cache) > max_entries:
        _digest_cache.popitem(last=False)


def reset_for_test() -> None:
    """# @internal — clears the digest cache for test isolation."""
    _digest_cache.clear()


async def apply(
    messages: list[dict], settings: ContextOptimizerSettings,
) -> list[dict]:
    """Replace eligible tool_results with cached or freshly-digested summaries.

    Eligibility: block.type == 'tool_result' and content is a string longer
    than settings.tier0b_digest_min_bytes. Non-string tool_result content
    (lists of blocks, image references) is left alone — too varied to digest
    safely without per-shape handling.

    Never raises. On Ollama unreachable, parse failure, or timeout the
    affected tool_result passes through unchanged.
    """
    if not settings.tier0b_digest_enabled:
        return messages

    candidates = _find_candidates(messages, settings.tier0b_digest_min_bytes)
    if not candidates:
        return messages

    cached_results, miss_indices = _split_cached(
        candidates, settings.tier0b_digest_cache_max_entries,
    )

    fresh_results: dict[int, str] = {}
    if miss_indices:
        fresh_results = await _digest_misses(
            candidates, miss_indices, settings,
        )

    digests = {**cached_results, **fresh_results}
    if not digests:
        return messages

    return _apply_digests(messages, candidates, digests)


def _find_candidates(
    messages: list[dict], min_bytes: int,
) -> list[tuple[int, int, str, str]]:
    """Return [(msg_idx, block_idx, tool_name, content), ...] for eligible blocks."""
    out: list[tuple[int, int, str, str]] = []
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            raw = block.get("content")
            if not isinstance(raw, str):
                continue
            if len(raw.encode("utf-8", errors="ignore")) < min_bytes:
                continue
            tool_name = str(block.get("tool_use_id", "tool"))
            out.append((mi, bi, tool_name, raw))
    return out


def _split_cached(
    candidates: list[tuple[int, int, str, str]], max_entries: int,
) -> tuple[dict[int, str], list[int]]:
    """Return ({cand_idx: cached_digest}, [cand_idx_to_fetch, ...])."""
    cached: dict[int, str] = {}
    misses: list[int] = []
    for ci, (_mi, _bi, _tool, raw) in enumerate(candidates):
        key = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()
        hit = _cache_get(key, max_entries)
        if hit is not None:
            cached[ci] = hit
            logger.debug("CONTEXT_OPT: tier0b cache_hit hash={}", key[:12])
        else:
            misses.append(ci)
    return cached, misses


async def _digest_misses(
    candidates: list[tuple[int, int, str, str]],
    miss_indices: list[int],
    settings: ContextOptimizerSettings,
) -> dict[int, str]:
    """Fire all Ollama digest calls concurrently, bounded by timeout."""
    from ..ollama_supervisor import OllamaSupervisor

    if not await OllamaSupervisor.ensure_ready(settings):
        logger.warning(
            "CONTEXT_OPT: tier0b ollama_unavailable misses={} — passthrough",
            len(miss_indices),
        )
        return {}

    started = time.monotonic()
    coros = [_digest_one(candidates[ci], settings) for ci in miss_indices]
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=False),
            timeout=settings.tier0b_digest_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "CONTEXT_OPT: tier0b batch_timeout misses={} timeout_s={}",
            len(miss_indices), settings.tier0b_digest_timeout_seconds,
        )
        return {}

    out: dict[int, str] = {}
    bytes_before = 0
    bytes_after = 0
    for ci, digest in zip(miss_indices, results, strict=True):
        if digest is None:
            continue
        out[ci] = digest
        raw = candidates[ci][3]
        key = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()
        _cache_put(key, digest, settings.tier0b_digest_cache_max_entries)
        bytes_before += len(raw)
        bytes_after += len(digest)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    if out:
        logger.info(
            "CONTEXT_OPT: tier0b digesting bytes_before={} bytes_after={} "
            "count={} latency_ms={}",
            bytes_before, bytes_after, len(out), elapsed_ms,
        )
    return out


async def _digest_one(
    candidate: tuple[int, int, str, str],
    settings: ContextOptimizerSettings,
) -> str | None:
    """Single Ollama digest call. Returns the digest text or None on failure."""
    _mi, _bi, tool_name, raw = candidate
    prompt = build_digest_prompt(raw, tool_name)
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
            "CONTEXT_OPT: tier0b digest_failed {}: {}",
            type(exc).__name__, exc,
        )
        return None

    digest = parse_digest_response(content)
    if digest is None:
        logger.warning(
            "CONTEXT_OPT: tier0b parse_error first_200={!r}", content[:200],
        )
        return None
    return digest


def _apply_digests(
    messages: list[dict],
    candidates: list[tuple[int, int, str, str]],
    digests: dict[int, str],
) -> list[dict]:
    """Materialize the digest replacements into a new message list."""
    by_msg: dict[int, dict[int, str]] = {}
    for ci, digest in digests.items():
        mi, bi, _tool, _raw = candidates[ci]
        by_msg.setdefault(mi, {})[bi] = digest

    if not by_msg:
        return messages

    result = []
    for mi, msg in enumerate(messages):
        if mi not in by_msg:
            result.append(msg)
            continue
        new_blocks = []
        replacements = by_msg[mi]
        for bi, block in enumerate(msg.get("content", [])):
            if bi in replacements:
                new_blocks.append({**block, "content": replacements[bi]})
            else:
                new_blocks.append(block)
        result.append({**msg, "content": new_blocks})
    return result
