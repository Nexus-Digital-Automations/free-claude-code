"""Owns: per-request block-inclusion decision.

Layer 0 calls `select_blocks` once per request with the loaded tower
and the current user message. The selector returns the ordered list of
BlockHandles that should be prepended to the system prompt.

Selection mode (from settings.block_selection_mode):
  "off"       — caller should not even invoke the selector. Defensive
                guard returns [] if invoked anyway.
  "all"       — return every block, in chronological order, no Ollama call.
  "selective" — call Ollama once with all block headers + the current
                message; return the indices Ollama marks INCLUDE.
                Falls back to "all" on any Ollama failure (preserves
                request reliability — token-cost regression is preferable
                to a blocked or truncated request).

WHY in-process selection cache: identical (session, block_count,
current_message_hash) inputs produce identical inclusion patterns,
which keep prefix-cache bytes stable for the upstream LLM. A small
LRU is enough; selection is idempotent so persistence isn't worth
the complexity.

Does NOT own: BlockStore (store.py), Ollama daemon supervision
(ollama_supervisor.py), prompt rendering (prompts.py).
Called by: optimizer.py Layer 0.
Calls: prompts.build_select_prompt / parse_select_response,
       OllamaSupervisor.ensure_ready, AsyncOpenAI.

# @stable — optimizer.py depends on select_blocks.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from loguru import logger
from openai import AsyncOpenAI

from .._core import content_hash
from .prompts import build_select_prompt, parse_select_response
from .store import BlockHandle

if TYPE_CHECKING:
    from ..settings import ContextOptimizerSettings


_SELECTION_CACHE_MAX = 50

# session_key -> (cache_key -> selected_indices)
# Per-session OrderedDict so an LRU eviction in one session can't
# starve another session's hot pattern.
_selection_cache: dict[str, "OrderedDict[str, list[int]]"] = {}


async def select_blocks(
    blocks: list[BlockHandle],
    current_user_text: str,
    session_key: str,
    settings: "ContextOptimizerSettings",
) -> list[BlockHandle]:
    """Return the BlockHandles to include in this request's prefix.

    Never raises — all failures degrade to "include every block" so
    the request always has the most context available.
    """
    if not blocks:
        return []
    if settings.block_selection_mode == "off":
        return []
    if settings.block_selection_mode == "all" or len(blocks) == 1:
        return list(blocks)

    cache_key = _selection_cache_key(blocks, current_user_text)
    cached = _cache_get(session_key, cache_key)
    if cached is not None:
        return [b for b in blocks if b.block_index in cached]

    selected_indices = await _ask_ollama(blocks, current_user_text, settings)
    if selected_indices is None:
        # Ollama failed — fall back to all so the request still has context.
        return list(blocks)

    _cache_put(session_key, cache_key, selected_indices)
    skipped = [b.block_index for b in blocks if b.block_index not in selected_indices]
    logger.info(
        "BLOCK_TOWER: selected session={} include={} skip={}",
        session_key[:7], selected_indices, skipped,
    )
    return [b for b in blocks if b.block_index in selected_indices]


async def _ask_ollama(
    blocks: list[BlockHandle],
    current_user_text: str,
    settings: "ContextOptimizerSettings",
) -> list[int] | None:
    from ..ollama_supervisor import OllamaSupervisor  # noqa: PLC0415 — lazy import keeps cold path light

    if not await OllamaSupervisor.ensure_ready(settings):
        logger.warning("BLOCK_TOWER: ollama not ready, falling back to include-all")
        return None

    prompt = build_select_prompt(
        [(b.block_index, b.header) for b in blocks],
        current_user_text,
    )
    try:
        async with AsyncOpenAI(
            api_key="ollama",  # pragma: allowlist secret — placeholder for local Ollama
            base_url=settings.ollama_base_url,
        ) as client:
            resp = await client.chat.completions.create(
                model=settings.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,  # deterministic — same input must give same selection for cache stability
                stream=False,
            )
        content = resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning(
            "BLOCK_TOWER: select call failed reason={} {}: {}",
            type(exc).__name__, type(exc).__name__, exc,
        )
        return None

    max_index = max(b.block_index for b in blocks)
    parsed = parse_select_response(content, max_index)
    if parsed is None:
        logger.warning(
            "BLOCK_TOWER: select parse failed first_200={!r}", content[:200],
        )
        return None
    return parsed


def _selection_cache_key(blocks: list[BlockHandle], current_user_text: str) -> str:
    block_count = len(blocks)
    message_hash = content_hash([{"role": "user", "content": current_user_text}])[:16]
    return f"{block_count}:{message_hash}"


def _cache_get(session_key: str, cache_key: str) -> list[int] | None:
    bucket = _selection_cache.get(session_key)
    if bucket is None:
        return None
    if cache_key not in bucket:
        return None
    bucket.move_to_end(cache_key)
    return bucket[cache_key]


def _cache_put(session_key: str, cache_key: str, indices: list[int]) -> None:
    bucket = _selection_cache.setdefault(session_key, OrderedDict())
    bucket[cache_key] = indices
    bucket.move_to_end(cache_key)
    while len(bucket) > _SELECTION_CACHE_MAX:
        bucket.popitem(last=False)


def reset_for_test() -> None:
    # @internal — test isolation only.
    _selection_cache.clear()
