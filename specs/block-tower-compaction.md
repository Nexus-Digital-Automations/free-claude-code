---
title: Block Tower — Immutable Compaction with Selective Inclusion
status: completed
created: 2026-05-05
---

## Vision

Replace Tier 2's rolling-summary autocompaction (which moves the
system-prompt boundary on every compaction and forces the LLM to
re-summarise already-summarised content) with an immutable "block tower":

- Compaction blocks are sealed once, then never re-touched.
- Each compaction processes only the **uncompacted tail** since the last
  block was sealed.
- An Ollama relevance selector decides per-request which blocks to include,
  so blocks irrelevant to the current query don't burn input tokens.
- A new block is sealed only when the token economics are net-positive
  (`tail_tokens > 2 × C` AND `requests_since_last_seal ≥ 4`).

The result: byte-identical block bytes (cache-friendly within a given
inclusion pattern), no redundant LLM compaction work, and per-request
token savings on irrelevant historical context.

## Requirements

**R1 — Layer placement.** A new "Layer 0" runs in
`ContextOptimizer.optimize()` after Layer -1 (repo index) and before
Tier 0. It is gated on `settings.block_tower_enabled` (default `False`,
backward-compatible).

**R2 — Block immutability.** Once written, a block file is never
modified. Subsequent compactions only read `messages[last_block.end:]`.

**R3 — Mathematical seal trigger.** A new block is sealed only when
`tail_tokens > settings.block_seal_min_tail_tokens` (default 3000) AND
`requests_since_last_seal ≥ settings.block_seal_min_requests` (default 4).

**R4 — Ollama relevance selector.** Per request, the selector receives
each block's one-line header plus the current user message and returns a
subset of block indices to include. Selected blocks are concatenated in
chronological order. Failure of Ollama or selector ⇒ include all blocks
(safe fallback).

**R5 — Selection mode setting.** `settings.block_selection_mode` ∈
`{"all", "selective", "off"}`. `"off"` skips Layer 0 entirely; `"all"`
loads the tower but skips the selector; `"selective"` is the default.

**R6 — Tier 2 bypass.** When `block_tower_enabled=True`, Tier 2 is
bypassed. When `False`, Tier 2 runs unchanged (regression safety).

**R7 — Storage layout.** Blocks are written to
`<repo_root>/.context/blocks/<session_key>/block-{N:04d}.txt` and
`block-{N:04d}.meta.json`. Atomic write via temp-file + rename.

**R8 — Session key.** Derived from `content_hash(messages[0])` so
distinct conversations get distinct towers and resumed conversations
rejoin theirs.

**R9 — Reuse Ollama supervisor.** Selector and sealer call
`OllamaSupervisor.ensure_ready(settings)` before each Ollama request;
no new daemon-management code.

**R10 — Reuse single Ollama model.** Use `settings.ollama_model` for
both block sealing and selection. Three separate model settings would
be unrequested configurability — add only when measurement shows a
small selection model is worth the configuration surface.

## Acceptance Criteria

- [x] **AC1**: `block_tower_enabled=True` plus `block_seal_min_tail_tokens=500`
      causes a real `block-0001.txt` and `block-0001.meta.json` to appear
      under `.context/blocks/<session>/` after a multi-turn conversation
      (verified by file existence + content > 0 bytes).
      Test: `test_ac1_block_files_appear_after_emergency_seal`.
- [x] **AC2**: After `block-0002.txt` is sealed, `sha256(block-0001.txt)`
      is identical to its value when first written. (Immutability.)
      Test: `test_ac2_block_one_sha256_unchanged_after_block_two_sealed`.
- [x] **AC3**: `block-0002.meta.json.range.start` equals
      `block-0001.meta.json.range.end`. (No overlap, no gap.)
      Test: `test_ac3_block_two_range_start_equals_block_one_range_end`.
- [x] **AC4**: A query whose intent is unrelated to block 1's content
      causes the selector to return a `skip` list containing block 1
      (logged at info level for verification).
      Test: `test_ac4_selector_skips_irrelevant_block`.
- [x] **AC5**: Two identical queries produce identical selection
      decisions (deterministic given same inputs ⇒ cache hit on second
      request, measurable via `prompt_cache_hit_tokens` rising).
      Test: `test_ac5_identical_queries_hit_selection_cache`.
- [x] **AC6**: With `block_seal_min_requests=10` and only 5 requests
      issued, no block is sealed even if `tail_tokens > min_tail_tokens`.
      Test: `test_ac6_no_seal_when_request_count_below_threshold`.
- [x] **AC7**: Stopping `ollama` does not break requests — selector
      falls back to "include all blocks" and the request succeeds.
      Test: `test_ac7_selector_falls_back_to_all_blocks_when_ollama_unavailable`.
- [~] **AC8**: ~~`block_tower_enabled=False` (default) leaves Tier 2's
      rolling summary path completely unchanged~~ — Obsolete after
      Phase 2 deleted Tier 2 entirely. The remaining gating behaviour
      is covered by AC9 (`block_selection_mode="off"`).
- [x] **AC9**: `block_selection_mode="off"` skips Layer 0 entirely (no
      block files read or written).
      Test: `test_ac9_selection_mode_off_skips_layer_zero`.
- [x] **AC10**: `ruff check` passes on
      `packages/context-optimizer/src/context_optimizer/block_tower/`.
      Test: `test_ac10_ruff_clean_on_block_tower_module`. (mypy was not
      configured at the package level; ruff clean is the enforced bar.)

## Technical Decisions

**TD1 — Free functions over classes.** `BlockStore` warrants a class
(it owns mutable singleton state per session_key). `BlockSealer` and
`BlockSelector` are pure data-transformation pipelines and ship as
free functions in `sealer.py` / `selector.py`. Class wrappers were
considered and rejected as unrequested abstraction.

**TD2 — One Ollama model.** Reuse `settings.ollama_model` for both
sealing and selection. Differentiated models add three settings
fields and a model-routing decision with no measured benefit yet.
If selection latency turns out to dominate, add
`ollama_block_select_model` later as a single targeted addition.

**TD3 — Async sealing.** `seal()` is scheduled via the same pattern
Tier 2 uses for background compaction (`tier2.schedule_background`):
fire-and-forget asyncio task, the sealed block appears on the *next*
request. Sealing never blocks the live request path.

**TD4 — Selection cache key.** The selector's input is
`(session_key, block_count, hash(current_user_message))`. Identical
inputs ⇒ identical inclusion ⇒ stable system-prompt bytes ⇒
prefix-cache hit. A small in-memory LRU (~50 entries) is enough; no
disk persistence needed for selection decisions.

**TD5 — Block header source.** The seal step asks Ollama for both the
block body (~500 tokens) and a one-line header (≤120 chars) in a single
call, parsed from delimited sections. Two separate Ollama calls per
seal would double latency for marginal quality gain.

## Phase 2 — Tier 2 Removal + Sync Emergency Seal

After Phase 1 shipped (`bc1a805`), the only remaining unique value of Tier 2
was synchronous emergency compaction at the hard token threshold. Phase 2
replaces it with a bounded `seal_sync` in the block tower and rips Tier 2
out entirely.

### Phase 2 Requirements

**R11 — Sync emergency seal.** `block_tower/sealer.py` exports
`seal_sync(store, messages, settings)` that calls Ollama under
`asyncio.wait_for(timeout=12s)` and writes a real summary block on
success or a deterministic placeholder block on timeout/failure.

**R12 — Cold-start trigger.** `optimizer.py` Layer 0 calls `seal_sync`
when `not store.blocks and tokens >= settings.compact_threshold_tokens`.
The `compact_threshold_tokens` setting is repurposed (no longer Tier 2
sync threshold).

**R13 — Tier 2 deletion.** `tiers/tier2.py`, `cache.py` (PrefixCache),
and the Tier 2-only halves of `prompts.py` and `_core.py`
(`build_prompt`, `parse_response`, `apply_summary`, `safe_split_index`,
`_SUMMARY_PREFIX`) are removed. Settings deleted at both the package
and proxy levels: `compact_soft_threshold_tokens`,
`compact_deepseek_fallback_threshold_tokens`, `tier2_keep_recent_turns`,
`prefix_cache_max_entries`, `context_cache_dir`, `block_tower_enabled`.

**R14 — Tail trim bug fix.** Layer 0 must trim `messages` to the tail
past the latest existing block's `range_end` before forwarding. The
Phase 1 code prepended block bodies to `system` but failed to trim the
already-summarised messages from `messages`, double-paying tokens.

### Phase 2 Acceptance Criteria

- [x] **AC11**: `seal_sync` writes a real block when Ollama is reachable
      and a placeholder block (containing "truncation" + N omitted)
      when Ollama is unreachable. (Verified by
      `test_seal_sync_writes_placeholder_when_ollama_unreachable`.)
- [x] **AC12**: After deletion, `tier2.py`, `cache.py`,
      `compact_soft_threshold_tokens`, `compact_deepseek_fallback_threshold_tokens`,
      `tier2_keep_recent_turns`, `prefix_cache_max_entries`,
      `context_cache_dir`, `block_tower_enabled` are absent from the
      codebase (`grep -r` returns nothing).
- [x] **AC13**: `ruff check` clean, `mypy` reports no new errors,
      `pytest` 16/16 pass.
- [x] **AC14**: Layer 0 trims `messages` to `messages[last_block.range_end:]`
      after applying selected block bodies.

## Progress

- 2026-05-05: Phase 1 plan approved via ExitPlanMode. Spec drafted from plan.
- 2026-05-05: Phase 1 shipped in `bc1a805` (added Layer 0, kept Tier 2 as fallback).
- 2026-05-05: Phase 2 plan approved (replace Tier 2 with sync emergency seal).
- 2026-05-05: Phase 2 implementation complete; all checks green.
- 2026-05-05: AC1-AC10 acceptance tests added to `tests/test_optimizer.py`
  (9 new tests, AC8 obsoleted post Phase 2). Suite: 25/25 passing, ruff clean.
