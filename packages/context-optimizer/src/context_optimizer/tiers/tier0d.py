"""Owns: per-historical-user-text Ollama compaction. For text blocks inside
old user messages whose byte size exceeds a high threshold, replaces the
text with an Ollama-generated digest. The LAST user message — which carries
the active request — is always skipped to avoid mutating what the model is
about to respond to.

State diagram (per process, single asyncio loop):

    miss ──Ollama success──> cached  (LRU, bounded)
         ──Ollama failure──> passthrough  (no cache write)
    cached ──evicted on overflow──> miss

The high default threshold (16 KB) is intentional. Typical conversational
prompts must never trip the digester; only genuine pastes (large logs,
file dumps, error transcripts) are eligible.

Does NOT own: tool_result digestion (tier0b.py) or tool_use compaction
(tier0c.py).
Called by: optimizer.py, between tier0c and tier1.
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


def _cache_get(key: str) -> str | None:
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
    messages: list[dict],
    settings: ContextOptimizerSettings,
) -> list[dict]:
    """Replace eligible historical user-text blocks with digests.

    Eligibility: user message whose index is NOT the last user message in
    the conversation, containing a text block of size >= settings.tier0d_digest_min_bytes.

    Never raises. On Ollama unavailability or parse failure the affected
    text passes through unchanged.
    """
    if not settings.tier0d_digest_enabled:
        return messages

    candidates = _find_candidates(messages, settings.tier0d_digest_min_bytes)
    if not candidates:
        return messages

    cached_results, miss_indices = _split_cached(candidates)

    fresh_results: dict[int, str] = {}
    if miss_indices:
        fresh_results = await _digest_misses(candidates, miss_indices, settings)

    digests = {**cached_results, **fresh_results}
    if not digests:
        return messages

    return _apply_digests(messages, candidates, digests)


def _find_candidates(
    messages: list[dict],
    min_bytes: int,
) -> list[tuple[int, int, str]]:
    """Return [(msg_idx, block_idx, text), ...] for eligible blocks.

    Locates the last user-message index up front so we can exclude it: the
    final user message is the active request the model is about to answer
    and must never be digested.
    """
    last_user_idx = -1
    for mi, msg in enumerate(messages):
        if msg.get("role") == "user":
            last_user_idx = mi
    if last_user_idx < 0:
        return []

    out: list[tuple[int, int, str]] = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "user" or mi == last_user_idx:
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text", "")
            if not isinstance(text, str):
                continue
            if len(text.encode("utf-8", errors="ignore")) < min_bytes:
                continue
            out.append((mi, bi, text))
    return out


def _split_cached(
    candidates: list[tuple[int, int, str]],
) -> tuple[dict[int, str], list[int]]:
    cached: dict[int, str] = {}
    misses: list[int] = []
    for ci, (_mi, _bi, text) in enumerate(candidates):
        key = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        hit = _cache_get(key)
        if hit is not None:
            cached[ci] = hit
            logger.debug("CONTEXT_OPT: tier0d cache_hit hash={}", key[:12])
        else:
            misses.append(ci)
    return cached, misses


async def _digest_misses(
    candidates: list[tuple[int, int, str]],
    miss_indices: list[int],
    settings: ContextOptimizerSettings,
) -> dict[int, str]:
    from ..ollama_supervisor import OllamaSupervisor

    if not await OllamaSupervisor.ensure_ready(settings):
        logger.warning(
            "CONTEXT_OPT: tier0d ollama_unavailable misses={} — passthrough",
            len(miss_indices),
        )
        return {}

    started = time.monotonic()
    coros = [_digest_one(candidates[ci], settings) for ci in miss_indices]
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=False),
            timeout=settings.tier0d_digest_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "CONTEXT_OPT: tier0d batch_timeout misses={} timeout_s={}",
            len(miss_indices),
            settings.tier0d_digest_timeout_seconds,
        )
        return {}

    out: dict[int, str] = {}
    bytes_before = 0
    bytes_after = 0
    for ci, digest in zip(miss_indices, results, strict=True):
        if digest is None:
            continue
        out[ci] = digest
        text = candidates[ci][2]
        key = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        _cache_put(key, digest, settings.tier0d_digest_cache_max_entries)
        bytes_before += len(text)
        bytes_after += len(digest)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    if out:
        logger.info(
            "CONTEXT_OPT: tier0d digesting bytes_before={} bytes_after={} "
            "count={} latency_ms={}",
            bytes_before,
            bytes_after,
            len(out),
            elapsed_ms,
        )
    return out


async def _digest_one(
    candidate: tuple[int, int, str],
    settings: ContextOptimizerSettings,
) -> str | None:
    _mi, _bi, text = candidate
    prompt = build_digest_prompt(text, "user-paste")
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
            "CONTEXT_OPT: tier0d digest_failed {}: {}",
            type(exc).__name__,
            exc,
        )
        return None

    digest = parse_digest_response(content)
    if digest is None:
        logger.warning(
            "CONTEXT_OPT: tier0d parse_error first_200={!r}",
            content[:200],
        )
        return None
    return digest


def _apply_digests(
    messages: list[dict],
    candidates: list[tuple[int, int, str]],
    digests: dict[int, str],
) -> list[dict]:
    by_msg: dict[int, dict[int, str]] = {}
    for ci, digest in digests.items():
        mi, bi, _text = candidates[ci]
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
                new_blocks.append({**block, "text": replacements[bi]})
            else:
                new_blocks.append(block)
        result.append({**msg, "content": new_blocks})
    return result
