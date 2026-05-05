---
title: Block Tower — Immutable Compaction with Selective Inclusion
status: planning
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

- [ ] **AC1**: `block_tower_enabled=True` plus `block_seal_min_tail_tokens=500`
      causes a real `block-0001.txt` and `block-0001.meta.json` to appear
      under `.context/blocks/<session>/` after a multi-turn conversation
      (verified by file existence + content > 0 bytes).
- [ ] **AC2**: After `block-0002.txt` is sealed, `sha256(block-0001.txt)`
      is identical to its value when first written. (Immutability.)
- [ ] **AC3**: `block-0002.meta.json.range.start` equals
      `block-0001.meta.json.range.end`. (No overlap, no gap.)
- [ ] **AC4**: A query whose intent is unrelated to block 1's content
      causes the selector to return a `skip` list containing block 1
      (logged at info level for verification).
- [ ] **AC5**: Two identical queries produce identical selection
      decisions (deterministic given same inputs ⇒ cache hit on second
      request, measurable via `prompt_cache_hit_tokens` rising).
- [ ] **AC6**: With `block_seal_min_requests=10` and only 5 requests
      issued, no block is sealed even if `tail_tokens > min_tail_tokens`.
- [ ] **AC7**: Stopping `ollama` does not break requests — selector
      falls back to "include all blocks" and the request succeeds.
- [ ] **AC8**: `block_tower_enabled=False` (default) leaves Tier 2's
      rolling summary path completely unchanged (regression check via
      existing Tier 2 tests).
- [ ] **AC9**: `block_selection_mode="off"` skips Layer 0 entirely (no
      block files read or written).
- [ ] **AC10**: `ruff check` and `mypy` pass on
      `packages/context-optimizer/`.

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

## Progress

- 2026-05-05: Plan approved via ExitPlanMode. Spec drafted from plan.
