"""Owns: hash-keyed LRU cache + single-shot Ollama digest call.

Generic over content shape and prompt template — callers supply both.
tier0b uses this for tool_result digestion; test-digester (free-claude-code
package) digests test-failure bodies; memory-recall digests plan/spec
markdown for a recall index.

Does NOT own: candidate selection, batch orchestration, OllamaSupervisor
readiness, or splicing digests back into the caller's data structure.
Each caller handles its own shape and concurrency policy.

State diagram (per process, single asyncio loop):

    miss --Ollama success--> cached  (LRU, bounded by max_entries)
         --Ollama failure--> passthrough  (no cache write — try again)
    cached --evicted on overflow--> miss

# @stable — tier0b, test-digester, memory-recall depend on digest()'s signature.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Callable, Protocol

from loguru import logger
from openai import AsyncOpenAI


class DigestConfig(Protocol):
    """Minimal config shape digest() needs.

    ContextOptimizerSettings satisfies this; so do small per-tool dataclasses
    in test-digester and memory-recall — no shared dependency on the
    optimizer's full settings object.
    """

    ollama_base_url: str
    ollama_model: str
    compaction_max_tokens: int
    compaction_temperature: float


_digest_cache: OrderedDict[str, str] = OrderedDict()


def reset_for_test() -> None:
    """# @internal — clears the digest cache for test isolation."""
    _digest_cache.clear()


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


async def digest(
    content: str,
    build_prompt: Callable[[str], str],
    parse_response: Callable[[str], str | None],
    config: DigestConfig,
    *,
    cache_max_entries: int,
) -> str | None:
    """Return a cached or freshly-digested summary of `content`, or None on failure.

    Cache key is sha256(content) so identical bytes-in produce identical
    bytes-out across calls — load-bearing for DeepSeek prefix-cache hits.

    Caller is responsible for: ensuring Ollama is reachable (call
    OllamaSupervisor.ensure_ready first), wrapping in asyncio.wait_for for
    timeouts, and any logging beyond the debug-level cache trace here.

    Never raises — all failures (Ollama unreachable, parse error, network
    timeout) return None and the caller falls back to original content.
    """
    key = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()

    hit = _cache_get(key)
    if hit is not None:
        logger.debug("DIGEST_CORE: cache_hit hash={}", key[:12])
        return hit

    body = await _call_ollama(content, build_prompt, config)
    if body is None:
        return None

    parsed = parse_response(body)
    if parsed is None:
        logger.warning("DIGEST_CORE: parse_error first_200={!r}", body[:200])
        return None

    _cache_put(key, parsed, cache_max_entries)
    return parsed


async def _call_ollama(
    content: str,
    build_prompt: Callable[[str], str],
    config: DigestConfig,
) -> str | None:
    """One Ollama chat call. Returns the raw response body or None on failure.

    Logged failures: connection refused, timeout, non-200, model-not-found.
    Never raises; caller treats None as "passthrough this content unchanged".
    """
    prompt = build_prompt(content)
    try:
        async with AsyncOpenAI(
            api_key="ollama",  # pragma: allowlist secret — local Ollama placeholder
            base_url=config.ollama_base_url,
        ) as client:
            resp = await client.chat.completions.create(
                model=config.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.compaction_max_tokens,
                temperature=config.compaction_temperature,
                stream=False,
            )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning(
            "DIGEST_CORE: ollama_call_failed type={} msg={}",
            type(exc).__name__,
            str(exc)[:200],
        )
        return None
