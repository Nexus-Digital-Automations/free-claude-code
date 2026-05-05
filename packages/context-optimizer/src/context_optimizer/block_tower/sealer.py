"""Owns: deciding when to seal + the actual sealing pipeline.

Two entry points exist:

  schedule_seal_if_due  — fire-and-forget background seal at the math
                          threshold. The triggering request never pays
                          the latency; the new block is picked up on the
                          next request.
  seal_sync             — bounded synchronous seal for emergencies (cold
                          start with a tail already over the hard token
                          budget). Blocks the live request; on Ollama
                          timeout/failure writes a placeholder truncation
                          block so the boundary moves forward and the
                          request can complete safely.

WHY math-based trigger for the background path: a fixed token threshold
burns budget on short sessions whose tail will never be referenced enough
to amortise the seal cost. The trigger here requires both a sizeable tail
AND enough prior requests to suggest the session is long-lived.

WHY a placeholder fallback in seal_sync: when the tower has zero blocks
yet and tokens already exceed the hard limit, we can't proceed without
moving the boundary. A real summary is preferred but if Ollama is down
or slow we still need to ship the request rather than block forever.
The placeholder is deterministic (same range → same bytes), so the
prefix-cache hit story remains intact.

Does NOT own: BlockStore (store.py), Ollama daemon supervision
(ollama_supervisor.py), or selection (selector.py).
Called by: optimizer.py Layer 0.
Calls: BlockStore.seal, OllamaSupervisor.ensure_ready, AsyncOpenAI.

# @stable — optimizer.py depends on schedule_seal_if_due / seal_sync /
# should_seal.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger
from openai import AsyncOpenAI

from ..token_counter import count_tokens
from .prompts import build_seal_prompt, parse_seal_response
from .store import BlockStore

if TYPE_CHECKING:
    from ..settings import ContextOptimizerSettings


# Hard ceiling on seal_sync's Ollama call. Picked at the upper end of typical
# qwen2.5:7b summarisation time on warm hardware; long enough that healthy
# Ollama almost always finishes in time, short enough that a stuck daemon
# cannot wedge an entire user request.
_SYNC_SEAL_TIMEOUT_SECONDS = 12.0

_inflight_sessions: set[str] = set()
_background_tasks: set[asyncio.Task] = set()


def should_seal(
    tail: list[dict],
    requests_since_last_seal: int,
    settings: "ContextOptimizerSettings",
) -> bool:
    """Mathematical seal-eligibility check.

    Returns True iff sealing is expected to be net-token-positive:
    tail is large enough that summarising saves significant tokens AND
    the session has run enough turns to amortise the one-time write.

    Tail token count is measured with the same tiktoken encoding used
    elsewhere; floor of zero on count failure (tokenizer not loadable)
    is intentional — we'd rather skip a seal than seal blindly.
    """
    if requests_since_last_seal < settings.block_seal_min_requests:
        return False
    try:
        tail_tokens = count_tokens(tail, None, None, tokenizer_name=settings.tokenizer_name)
    except Exception as exc:
        logger.warning(
            "BLOCK_TOWER: tail token count failed, skipping seal reason={} {}",
            type(exc).__name__, exc,
        )
        return False
    return tail_tokens > settings.block_seal_min_tail_tokens


def schedule_seal_if_due(
    store: BlockStore,
    messages: list[dict],
    settings: "ContextOptimizerSettings",
) -> None:
    """Fire-and-forget background seal of the current uncompacted tail.

    Noop if a seal for this session is already in flight, the session is
    the empty key (no first user message yet), or the math check fails.
    """
    if store.session_key == "empty":
        return
    if store.session_key in _inflight_sessions:
        return
    tail_start = store.last_end()
    tail = messages[tail_start:]
    if not tail:
        return
    if not should_seal(tail, store.requests_since_last_seal, settings):
        return

    _inflight_sessions.add(store.session_key)
    logger.info(
        "BLOCK_TOWER: scheduling seal session={} tail_msgs={} prior_blocks={}",
        store.session_key[:7], len(tail), len(store.blocks),
    )
    task = asyncio.create_task(_run_seal(store, tail_start, tail, settings))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _run_seal(
    store: BlockStore,
    tail_start: int,
    tail: list[dict],
    settings: "ContextOptimizerSettings",
) -> None:
    try:
        result = await _compact_tail(tail, settings)
        if result is None:
            return
        header, body = result
        try:
            store.seal(tail_start, tail_start + len(tail), body, header)
        except (ValueError, OSError) as exc:
            logger.warning(
                "BLOCK_TOWER: seal apply failed session={} reason={} {}",
                store.session_key[:7], type(exc).__name__, exc,
            )
    finally:
        _inflight_sessions.discard(store.session_key)


async def seal_sync(
    store: BlockStore,
    messages: list[dict],
    settings: "ContextOptimizerSettings",
) -> bool:
    """Block until the uncompacted tail is sealed. Used as the cold-start
    emergency safety net when token count exceeds the hard threshold.

    Returns True iff a real Ollama-summarised block was written; False iff
    the call timed out / failed and a deterministic placeholder block was
    written instead. Either outcome moves the immutability boundary
    forward and lets the live request continue with a smaller tail.

    Never raises — even a write failure is logged and yields False, so
    callers can still serve the request (uncompacted) instead of erroring.

    Concurrency: ignores the background-seal in-flight guard. The async
    `schedule_seal_if_due` path explicitly avoids overlapping with seal_sync
    by skipping when blocks already exist; seal_sync only fires when
    `len(store.blocks) == 0`, which is mutually exclusive.
    Counterpart: optimizer.py Layer 0b emergency check.
    """
    if store.session_key == "empty":
        return False
    tail_start = store.last_end()
    tail = messages[tail_start:]
    if not tail:
        return False

    logger.info(
        "BLOCK_TOWER: seal_sync session={} tail_msgs={} timeout={}s",
        store.session_key[:7], len(tail), _SYNC_SEAL_TIMEOUT_SECONDS,
    )
    try:
        result = await asyncio.wait_for(
            _compact_tail(tail, settings),
            timeout=_SYNC_SEAL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "BLOCK_TOWER: seal_sync timeout session={} timeout={}s — writing placeholder",
            store.session_key[:7], _SYNC_SEAL_TIMEOUT_SECONDS,
        )
        result = None

    if result is None:
        return _write_placeholder_block(store, tail_start, len(tail))

    header, body = result
    try:
        store.seal(tail_start, tail_start + len(tail), body, header)
        return True
    except (ValueError, OSError) as exc:
        logger.warning(
            "BLOCK_TOWER: seal_sync apply failed session={} reason={} {} — writing placeholder",
            store.session_key[:7], type(exc).__name__, exc,
        )
        return _write_placeholder_block(store, tail_start, len(tail))


def _write_placeholder_block(store: BlockStore, tail_start: int, tail_len: int) -> bool:
    """Seal a deterministic truncation marker. Returns False to signal "not real summary".

    The body bytes are a function of the message-count only, so two requests
    that hit the same emergency boundary produce byte-identical placeholders
    and prefix-cache hits remain stable.
    """
    body = (
        f"Earlier conversation truncated due to emergency context-window pressure; "
        f"{tail_len} messages omitted. The original content was not summarised "
        f"because the local summariser did not respond in time."
    )
    header = f"Truncation placeholder ({tail_len} messages)"
    try:
        store.seal(tail_start, tail_start + tail_len, body, header)
    except (ValueError, OSError) as exc:
        logger.error(
            "BLOCK_TOWER: placeholder seal failed session={} reason={} {}",
            store.session_key[:7], type(exc).__name__, exc,
        )
    return False


async def _compact_tail(
    tail: list[dict],
    settings: "ContextOptimizerSettings",
) -> tuple[str, str] | None:
    from ..ollama_supervisor import OllamaSupervisor  # noqa: PLC0415 — lazy import keeps cold path light

    if not await OllamaSupervisor.ensure_ready(settings):
        logger.warning("BLOCK_TOWER: ollama not ready, skipping seal")
        return None

    prompt = build_seal_prompt(
        tail,
        target_tokens=settings.block_target_summary_tokens,
        preview_chars=settings.render_preview_chars,
    )
    try:
        async with AsyncOpenAI(
            api_key="ollama",  # pragma: allowlist secret — placeholder for local Ollama
            base_url=settings.ollama_base_url,
        ) as client:
            resp = await client.chat.completions.create(
                model=settings.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                # 1.5× the body target gives Ollama room for the header section
                # plus delimiter overhead; tighter caps cut bodies mid-sentence.
                max_tokens=int(settings.block_target_summary_tokens * 1.5) + 200,
                temperature=settings.compaction_temperature,
                stream=False,
            )
        content = resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning(
            "BLOCK_TOWER: ollama seal call failed reason={} {}: {}",
            type(exc).__name__, type(exc).__name__, exc,
        )
        return None

    parsed = parse_seal_response(content)
    if parsed is None:
        logger.warning(
            "BLOCK_TOWER: seal parse failed first_200={!r}", content[:200],
        )
        return None
    return parsed


def reset_for_test() -> None:
    # @internal — test isolation only.
    _inflight_sessions.clear()
    _background_tasks.clear()
