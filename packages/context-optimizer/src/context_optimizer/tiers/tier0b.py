"""Owns: per-tool-result Ollama digestion. Replaces long tool_result content
with a content-aware summary so subsequent requests carry a smaller history.

Delegates the cache + Ollama call mechanics to digest_core; this module owns
candidate selection (which tool_results are eligible) and message splicing
(how digests replace original content in the message list).

Does NOT own: the digest cache (digest_core.py), Ollama daemon management
(ollama_supervisor.py), the digest prompt (prompts.py), or whole-conversation
summarization (tier2.py).
Called by: optimizer.py, between tier0 and tier1.
Calls: digest_core.digest, OllamaSupervisor.ensure_ready,
       prompts.build_digest_prompt/parse_digest_response.

# @stable — optimizer.py depends on apply()'s signature.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from loguru import logger

from .. import digest_core
from ..prompts import build_digest_prompt, parse_digest_response

if TYPE_CHECKING:
    from ..settings import ContextOptimizerSettings


def reset_for_test() -> None:
    """# @internal — clears the digest cache for test isolation.

    Counterpart: digest_core.reset_for_test (the cache lives there now).
    Kept here for backward-compatible imports from tier0b's existing tests.
    """
    digest_core.reset_for_test()


async def apply(
    messages: list[dict],
    settings: ContextOptimizerSettings,
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

    digests = await _digest_all(candidates, settings)
    if not digests:
        return messages

    return _apply_digests(messages, candidates, digests)


def _find_candidates(
    messages: list[dict],
    min_bytes: int,
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


async def _digest_all(
    candidates: list[tuple[int, int, str, str]],
    settings: ContextOptimizerSettings,
) -> dict[int, str]:
    """Concurrently digest every candidate, bounded by a single batch timeout.

    Returns {candidate_index: digest_text}. Cache hits are returned for free
    by digest_core; only cache misses incur Ollama calls. Aggregate
    bytes_before / bytes_after / latency are logged for observability.
    """
    from ..ollama_supervisor import OllamaSupervisor

    if not await OllamaSupervisor.ensure_ready(settings):
        logger.warning(
            "CONTEXT_OPT: tier0b ollama_unavailable count={} — passthrough",
            len(candidates),
        )
        return {}

    started = time.monotonic()
    coros = [
        digest_core.digest(
            content=raw,
            build_prompt=lambda c, tn=tool_name: build_digest_prompt(c, tn),
            parse_response=parse_digest_response,
            config=settings,
            cache_max_entries=settings.tier0b_digest_cache_max_entries,
        )
        for (_mi, _bi, tool_name, raw) in candidates
    ]
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=False),
            timeout=settings.tier0b_digest_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "CONTEXT_OPT: tier0b batch_timeout count={} timeout_s={}",
            len(candidates),
            settings.tier0b_digest_timeout_seconds,
        )
        return {}

    out: dict[int, str] = {}
    bytes_before = 0
    bytes_after = 0
    for ci, result in enumerate(results):
        if result is None:
            continue
        out[ci] = result
        raw = candidates[ci][3]
        bytes_before += len(raw)
        bytes_after += len(result)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    if out:
        logger.info(
            "CONTEXT_OPT: tier0b digesting bytes_before={} bytes_after={} "
            "count={} latency_ms={}",
            bytes_before,
            bytes_after,
            len(out),
            elapsed_ms,
        )
    return out


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
