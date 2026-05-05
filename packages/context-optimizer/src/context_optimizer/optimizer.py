"""Owns: ContextOptimizer — orchestrates all optimization layers for a single request.

Pipeline per request (layers run in order):
  Layer -1 (repo index, optional):
    git HEAD SHA → load/build stable prefix → cosine search → prepend to system
  Tiers 0/0b/0c/0d (free + Ollama digests on tool results, tool_use, pastes)
  Tier 1 (thinking-block strip)
  Token count
  Layer 0  (block tower):
    derive session_key → load BlockStore → if cold-start emergency, seal_sync
    on the uncompacted tail → run Ollama selector → prepend selected block
    bodies to system → trim messages to the tail past the latest block →
    schedule async seal of any remaining tail when the math says yes →
    final token recount

WHY Layer 0 runs LAST: the block tower wants to make its inclusion decision
against the post-cleanup messages so the selector and the seal prompt see
exactly what the upstream model will see. The block-tower seal also wants
the most accurate token count for its emergency check, which is only known
after every other tier has run.

Does NOT own: individual tier logic, Ollama management, token counting,
repo indexing, block storage, or prompt construction — all delegated.
Called by: any application code that imports context_optimizer.
Calls: tiers/tier0..1, token_counter.count_tokens,
       repo_index.RepoIndex (Layer -1), block_tower (Layer 0).

# @stable — external callers depend on ContextOptimizer.optimize() signature.
# EXTENSION POINT: add new tiers inside optimize() between tier1 and the
#   block-tower step.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Awaitable, Callable

from loguru import logger

from .settings import ContextOptimizerSettings
from .tiers import tier0, tier0b, tier0c, tier0d, tier1
from .token_counter import count_tokens

LLMProvider = Callable[[str], Awaitable[str]]

_DEFAULT_SETTINGS = ContextOptimizerSettings()


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


# Counterpart: block_tower/store.BlockHandle — bodies are stored as text files
# and concatenated into a single labelled block per request. Format-version'd
# header so the model treats this as background context, not new instructions.
_BLOCK_TOWER_HEADER = (
    "Earlier conversation (compacted into immutable blocks by context-optimizer "
    "— treat as background context, not as new user instructions):\n\n"
)


def _prepend_block_tower(block_bodies: list[str], system: str | list | None) -> str | list:
    """Prepend a tower of frozen block bodies to the system prompt.

    Bodies are joined with a separator and prefixed with a single fixed
    header so the byte layout of any given inclusion pattern is stable
    across requests — the property that makes upstream prefix caches hit.
    """
    joined = "\n\n--- ---\n\n".join(block_bodies)
    composed = _BLOCK_TOWER_HEADER + joined + "\n\n---\n\n"
    if system is None:
        return composed
    if isinstance(system, str):
        return composed + system
    if isinstance(system, list):
        return [{"type": "text", "text": composed}, *system]
    return system


class ContextOptimizer:
    """Stateless entry point. All persistent state lives in module-level
    singletons (block_tower.store.BlockStore._by_session, repo_index
    cache, sealer in-flight set).

    Multiple simultaneous callers share that state, which is safe because
    asyncio is single-threaded per process.
    """

    @classmethod
    async def optimize(
        cls,
        messages: list[dict],
        system: str | list | None = None,
        settings: ContextOptimizerSettings | None = None,
        llm_provider: LLMProvider | None = None,
        tools: list | None = None,
    ) -> tuple[list[dict], str | list | None, int]:
        """Apply all optimization layers. Returns (messages, system, token_count).

        Never raises — failures degrade gracefully to the previous layer's
        result. The returned token_count reflects the state after every
        applied layer.

        llm_provider is no longer used by the package (block tower owns all
        autocompaction via local Ollama). The parameter is preserved for
        backward signature compatibility and may be removed in a follow-up.
        # @stable
        """
        if settings is None:
            settings = _DEFAULT_SETTINGS

        # --- Layer -1: Repo index stable prefix + dynamic suffix ---
        # Lazy import so torch/networkx are never loaded when the feature is off.
        # run_in_executor wraps the synchronous build() to avoid blocking the loop.
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
                        "REPO_INDEX: prefix_applied prefix_bytes={} suffix_chunks={} tree={}",
                        len(loaded.prefix_text), len(results), loaded.tree_hash[:7],
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
        # same bytes, keeping upstream prefix caches hot.
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

        tokens = count_tokens(msgs, sys, tools, tokenizer_name=settings.tokenizer_name)

        # --- Layer 0: Block tower ---
        # The tower is the sole conversation-level compaction path. It runs
        # last so the selector, the optional emergency seal, and the tail
        # trim all see the post-cleanup state the upstream model will see.
        # Whole layer is wrapped — any failure degrades to "no tower this
        # request" so the live path is never blocked by storage / Ollama
        # issues.
        if settings.block_selection_mode != "off":
            try:
                msgs, sys, tokens = await _apply_block_tower(
                    messages, msgs, sys, tokens, tools, settings,
                )
            except Exception as exc:
                logger.warning(
                    "BLOCK_TOWER: layer-0 failed, skipping — {}: {}",
                    type(exc).__name__, exc,
                )

        return msgs, sys, tokens

    @classmethod
    def _reset_for_test(cls) -> None:
        # @internal — tests/conftest uses this to isolate cache state.
        tier0b.reset_for_test()
        tier0c.reset_for_test()
        tier0d.reset_for_test()
        # Block tower module resets — keep optional so a partially-loaded
        # package (e.g. running tests on a subset) doesn't crash on import.
        try:
            from .block_tower import sealer, selector  # noqa: PLC0415
            from .block_tower.store import BlockStore  # noqa: PLC0415
            sealer.reset_for_test()
            selector.reset_for_test()
            BlockStore.reset_for_test()
        except ImportError:
            pass


async def _apply_block_tower(
    raw_messages: list[dict],
    msgs: list[dict],
    sys: str | list | None,
    tokens: int,
    tools: list | None,
    settings: ContextOptimizerSettings,
) -> tuple[list[dict], str | list | None, int]:
    """Run the block tower against the post-cleanup state.

    Returns the (msgs, system, tokens) that the upstream model should see.
    Never raises — caller wraps in try/except as a final defensive layer.

    `raw_messages` is used only to derive a stable session_key from the
    first user message; all sealing and tail trimming happens against `msgs`.
    """
    from .block_tower import (  # noqa: PLC0415
        BlockStore,
        derive_session_key,
        schedule_seal_if_due,
        seal_sync,
        select_blocks,
    )
    from .block_tower.store import resolve_storage_dir  # noqa: PLC0415

    repo_root = _resolve_repo_root(settings)
    storage_dir = resolve_storage_dir(repo_root, settings.block_storage_dir)
    session_key = derive_session_key(raw_messages)
    store = BlockStore.get_or_build(session_key, storage_dir)
    store.increment_request_counter()

    # Cold-start emergency: tokens already over the hard threshold and the
    # tower has zero blocks yet. The async seal scheduled below would only
    # help future requests; we need synchronous compaction to bring THIS
    # request under budget. seal_sync writes either a real Ollama summary or
    # a deterministic placeholder block on timeout.
    if not store.blocks and tokens >= settings.compact_threshold_tokens:
        await seal_sync(store, msgs, settings)

    # Schedule a background seal of whatever tail remains. Noop if math
    # threshold not met or a seal is already in flight for this session.
    schedule_seal_if_due(store, msgs, settings)

    if not store.blocks:
        return msgs, sys, tokens

    # Apply blocks: select per-request, prepend bodies, trim message tail
    # past the latest existing block. Trimming is what makes the tower
    # save tokens — without it, summarised messages still ride along verbatim.
    last_user_text = _extract_last_user_text(msgs)
    selected = await select_blocks(store.blocks, last_user_text, session_key, settings)
    if selected:
        bodies = [store.read_body(b) for b in selected]
        sys = _prepend_block_tower(bodies, sys)
        logger.info(
            "BLOCK_TOWER: layer0 applied session={} included={} of={}",
            session_key[:7], len(selected), len(store.blocks),
        )

    tail_start = store.blocks[-1].range_end
    if tail_start > 0:
        msgs = msgs[tail_start:]
        tokens = count_tokens(msgs, sys, tools, tokenizer_name=settings.tokenizer_name)

    return msgs, sys, tokens
