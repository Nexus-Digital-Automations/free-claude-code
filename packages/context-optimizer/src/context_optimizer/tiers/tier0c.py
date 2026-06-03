"""Owns: per-tool_use Ollama input compaction. For old assistant tool_use
blocks whose serialized input exceeds a byte threshold, replaces the input
dict with an Ollama-generated single-string summary. The most recent N
tool_use blocks pass through verbatim so the model can still reference its
own latest call args.

State diagram (per process, single asyncio loop):

    miss ──Ollama success──> cached  (LRU, bounded)
         ──Ollama failure──> passthrough  (no cache write)
    cached ──evicted on overflow──> miss

Cache stability is load-bearing: identical input dicts (sorted-keys JSON)
must always yield identical digest bytes, otherwise DeepSeek's prefix cache
invalidates on every turn.

Does NOT own: Ollama daemon management (ollama_supervisor.py) or tool_result
content (tier0b.py).
Called by: optimizer.py, between tier0b and tier1.
Calls: prompts.build_digest_prompt/parse_digest_response,
       ollama_supervisor.OllamaSupervisor.ensure_ready, AsyncOpenAI.

# @stable — optimizer.py depends on apply()'s signature.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

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
    """Replace eligible tool_use inputs with Ollama-generated summaries.

    Eligibility: assistant block with type=tool_use whose serialised input
    exceeds settings.tier0c_digest_min_bytes AND is older than the last
    settings.tier0c_keep_recent_calls tool_use blocks in the conversation.

    Never raises. On Ollama unavailability or parse failure the affected
    tool_use passes through unchanged.
    """
    if not settings.tier0c_digest_enabled:
        return messages

    candidates = _find_candidates(
        messages,
        settings.tier0c_digest_min_bytes,
        settings.tier0c_keep_recent_calls,
    )
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
    keep_recent: int,
) -> list[tuple[int, int, str, str]]:
    """Return [(msg_idx, block_idx, tool_name, serialized_input), ...].

    Walks the message list left-to-right collecting eligible tool_use blocks,
    then drops the last `keep_recent` to preserve recency.
    """
    found: list[tuple[int, int, str, str]] = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            inp = block.get("input")
            if not isinstance(inp, dict):
                continue
            serialized = json.dumps(inp, sort_keys=True)
            if len(serialized.encode("utf-8", errors="ignore")) < min_bytes:
                continue
            tool_name = str(block.get("name", "tool"))
            found.append((mi, bi, tool_name, serialized))

    if keep_recent > 0 and len(found) > keep_recent:
        return found[:-keep_recent]
    if keep_recent > 0 and len(found) <= keep_recent:
        return []
    return found


def _split_cached(
    candidates: list[tuple[int, int, str, str]],
) -> tuple[dict[int, str], list[int]]:
    cached: dict[int, str] = {}
    misses: list[int] = []
    for ci, (_mi, _bi, _tool, serialized) in enumerate(candidates):
        key = hashlib.sha256(serialized.encode("utf-8", errors="ignore")).hexdigest()
        hit = _cache_get(key)
        if hit is not None:
            cached[ci] = hit
            logger.debug("CONTEXT_OPT: tier0c cache_hit hash={}", key[:12])
        else:
            misses.append(ci)
    return cached, misses


async def _digest_misses(
    candidates: list[tuple[int, int, str, str]],
    miss_indices: list[int],
    settings: ContextOptimizerSettings,
) -> dict[int, str]:
    from ..ollama_supervisor import OllamaSupervisor

    if not await OllamaSupervisor.ensure_ready(settings):
        logger.warning(
            "CONTEXT_OPT: tier0c ollama_unavailable misses={} — passthrough",
            len(miss_indices),
        )
        return {}

    started = time.monotonic()
    coros = [_digest_one(candidates[ci], settings) for ci in miss_indices]
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=False),
            timeout=settings.tier0c_digest_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "CONTEXT_OPT: tier0c batch_timeout misses={} timeout_s={}",
            len(miss_indices),
            settings.tier0c_digest_timeout_seconds,
        )
        return {}

    out: dict[int, str] = {}
    bytes_before = 0
    bytes_after = 0
    for ci, digest in zip(miss_indices, results, strict=True):
        if digest is None:
            continue
        out[ci] = digest
        serialized = candidates[ci][3]
        key = hashlib.sha256(serialized.encode("utf-8", errors="ignore")).hexdigest()
        _cache_put(key, digest, settings.tier0c_digest_cache_max_entries)
        bytes_before += len(serialized)
        bytes_after += len(digest)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    if out:
        logger.info(
            "CONTEXT_OPT: tier0c digesting bytes_before={} bytes_after={} "
            "count={} latency_ms={}",
            bytes_before,
            bytes_after,
            len(out),
            elapsed_ms,
        )
    return out


async def _digest_one(
    candidate: tuple[int, int, str, str],
    settings: ContextOptimizerSettings,
) -> str | None:
    _mi, _bi, tool_name, serialized = candidate
    prompt = build_digest_prompt(serialized, f"{tool_name}-call-args")
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
            "CONTEXT_OPT: tier0c digest_failed {}: {}",
            type(exc).__name__,
            exc,
        )
        return None

    digest = parse_digest_response(content)
    if digest is None:
        logger.warning(
            "CONTEXT_OPT: tier0c parse_error first_200={!r}",
            content[:200],
        )
        return None
    return digest


def _apply_digests(
    messages: list[dict],
    candidates: list[tuple[int, int, str, str]],
    digests: dict[int, str],
) -> list[dict]:
    """Replace each candidate's input dict with {"_compacted_summary": digest}.

    Preserves block.type/name/id so the model still recognises the tool_use
    structure; the model just sees a digest in place of the verbose input.
    """
    by_msg: dict[int, dict[int, str]] = {}
    for ci, digest in digests.items():
        mi, bi, _tool, _ser = candidates[ci]
        by_msg.setdefault(mi, {})[bi] = digest

    if not by_msg:
        return messages

    result = []
    for mi, msg in enumerate(messages):
        if mi not in by_msg:
            result.append(msg)
            continue
        new_blocks: list[Any] = []
        replacements = by_msg[mi]
        for bi, block in enumerate(msg.get("content", [])):
            if bi in replacements:
                new_blocks.append(
                    {**block, "input": {"_compacted_summary": replacements[bi]}}
                )
            else:
                new_blocks.append(block)
        result.append({**msg, "content": new_blocks})
    return result
