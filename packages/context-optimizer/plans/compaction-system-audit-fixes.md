---
title: Compaction system audit — bug fixes
status: planning
created: 2026-05-05
---

## Context

User asked for a thorough audit of the context-optimizer package and fix anything broken. Three parallel Explore agents audited the per-message tiers + orchestrator, the block tower, and the repo-index subsystem — total ~4200 lines across 28 modules.

After triaging the agent findings against the actual code (some were false positives — e.g. the placeholder body IS deterministic; `BlockStore.seal()` has no `await` so the singleton race the agent flagged can't happen), four real bugs and three small hardening items are worth fixing.

Out of scope after triage:
- Cross-platform `math.sqrt` determinism (1-token diff, no incident).
- Comma-in-filename Repomix `--include` injection (path-validation rabbit hole).
- Tier 0c/0d sharing tier0b's timeout/cache settings (intentional design — adding three new fields is unrequested configurability per the standards' "no flexibility not requested" rule; will leave a clarifying comment instead).

## Bugs

### B1 — `repo_index/index.py:309-312` — token-cap loop has a 5-file floor (HIGH)

```python
while count(prefix_text) > token_ceiling and len(top_files) > 5:
    top_files = ranker.get_top_n_files(ranked, len(top_files) - 5)
    prefix_text = _render(repo_root, top_files, settings)
```

If 5 files alone exceed the ceiling (large monorepo files), the loop exits with the prefix still over budget.

Fix: after the coarse loop exits, do a single-file shrink down to floor 1 if the prefix is still over the ceiling; if even one file exceeds it, log a warning and accept (a single hub file is what the user asked for).

### B2 — `repo_index/index.py:95-114` — concurrent `get_or_build` race (HIGH)

`RepoIndex.get_or_build()` is dispatched from `optimizer.py:159` via `loop.run_in_executor(None, RepoIndex.get_or_build, ...)`. Two concurrent requests → two threads → both can see `_loaded_tree_hash != sha`, both call `cls.build()`, both write `repo-<sha>.{txt,npy,json}`. 20-60s of duplicated work plus an intermediate non-atomic state visible to the disk-cache reader.

Fix: a module-level `threading.Lock` guarding the cache check and assignment. Pattern matches `OllamaSupervisor`'s existing `_lock`.

### B3 — `repo_index/embedder.py:244-253` — `load_index` doesn't verify `vectors.shape[0] == len(chunks)` (HIGH)

If a partial write or external edit leaves `.npy` and `.json` out of sync, `load_index()` returns a `LoadedIndex` whose `query()` produces silently wrong results — cosine search picks index `i` from `vectors`, returns `chunks[i]`, but the two arrays disagree on what `i` means.

Fix: assert `vectors.shape[0] == len(chunks)` before returning; on mismatch, log and return `None` so the caller rebuilds.

### B4 — `block_tower/sealer.py:117-119` — background seal task crashes are silently swallowed (MED-HIGH)

```python
task = asyncio.create_task(_run_seal(...))
_background_tasks.add(task)
task.add_done_callback(_background_tasks.discard)
```

`_run_seal`'s `try/finally` guarantees `_inflight_sessions.discard()` (so no leak — agent's HIGH leak claim was a false positive). But asyncio's default done-callback path swallows uncaught exceptions; production seals could be silently failing.

Fix: a second done-callback that logs `task.exception()` when set.

## Hardening (small, piggy-backed)

### H1 — `block_tower/store.py:104-106` — `_load_from_disk` race on `is_dir`/`listdir`

If `session_dir` is deleted between the `is_dir()` check and the `os.listdir()` call, raises `FileNotFoundError`. Wrap `listdir` in `try/except FileNotFoundError: return`.

### H2 — `repo_index/embedder.py:260-264` — `prune_old_indexes` undefined order on identical mtime

Add a secondary sort key by name: `key=lambda p: (p.stat().st_mtime, p.name)`.

### H3 — `repo_index/cache_stats.py:75-83` — `_extract_attr` silently drops malformed values

Add `logger.debug(...)` so cache-stats incompleteness is visible during debugging.

## Files modified

1. `packages/context-optimizer/src/context_optimizer/repo_index/index.py` — B1, B2
2. `packages/context-optimizer/src/context_optimizer/repo_index/embedder.py` — B3, H2
3. `packages/context-optimizer/src/context_optimizer/block_tower/sealer.py` — B4
4. `packages/context-optimizer/src/context_optimizer/block_tower/store.py` — H1
5. `packages/context-optimizer/src/context_optimizer/repo_index/cache_stats.py` — H3
6. `packages/context-optimizer/src/context_optimizer/tiers/tier0c.py` — clarifying comment that `tier0b_*` settings are intentionally shared
7. `packages/context-optimizer/src/context_optimizer/tiers/tier0d.py` — same comment
8. `packages/context-optimizer/tests/test_optimizer.py` — two regression tests (see acceptance criteria)

## Acceptance Criteria

- [ ] **AC1**: `repo_index/index.py:_enforce_token_cap` shrinks below 5 files when needed; new test `test_token_cap_shrinks_below_five_files_when_one_exceeds_budget` (gated on `pytest.importorskip` for ML deps) passes both before-fix-fails / after-fix-passes.
- [ ] **AC2**: `RepoIndex.get_or_build` is guarded by a module-level `threading.Lock`; concurrent calls from two threads see at most one `cls.build()` invocation. Verified by reading the lock's import + usage; no test (concurrency tests for executor pools are flaky).
- [ ] **AC3**: `embedder.load_index` returns `None` (and logs) when on-disk `vectors.shape[0] != len(chunks)`; new test `test_load_index_returns_none_on_chunk_vector_shape_mismatch` (gated on `pytest.importorskip`) passes.
- [ ] **AC4**: `sealer._run_seal` task failures are logged via a `task.exception()` done-callback.
- [ ] **AC5**: `BlockStore._load_from_disk` no longer crashes on `FileNotFoundError` race.
- [ ] **AC6**: `prune_old_indexes` sort key is deterministic on identical mtimes (secondary sort by name).
- [ ] **AC7**: `_extract_attr` in `cache_stats.py` emits a debug log on conversion failure.
- [ ] **AC8**: `ruff check src/ tests/` exits 0 and `pytest tests/` is 35/35 (33 existing + 2 new) with the `[repo-index]` extras installed (29/33 pass + 4 skip on a base install).

## Verification

1. `cd packages/context-optimizer && ruff check src/ tests/` — exits 0
2. `cd packages/context-optimizer && TOKENIZERS_PARALLELISM=false python -m pytest tests/ -q` — all pass
3. The two new regression tests must fail when their respective fix is reverted (sanity).

## What this plan is NOT

- Not a settings-API refactor for tier0c/0d (would be unrequested API churn).
- Not a path-traversal hardening on Repomix `--include`.
- Not a cross-platform `math.sqrt` determinism fix.
- Not splitting `tests/test_optimizer.py` (already justified via the file-size registry).
