---
title: Out-of-scope fixes — sqrt determinism, repomix include hardening, tier0c/0d settings
status: active
created: 2026-05-05
---

## Vision

Close out three items previously deferred from the compaction-system audit:

1. Cross-platform `math.sqrt` determinism in repo-index.
2. Robust handling of unusual filenames (commas, glob meta-chars) in the repomix invocation.
3. Independent timeout / cache-size settings for `tier0c` and `tier0d` instead of the shared `tier0b_*` knobs.

## Requirements

### 1. math.sqrt → deterministic alternative
- All non-test `math.sqrt` calls in `packages/context-optimizer/src/context_optimizer/repo_index/` are replaced with `math.isqrt` (which returns an exact `int` on integer inputs and is bit-identical across platforms).
- `_compute_token_ceiling` keeps its sub-linear scaling shape; minor magnitude changes are acceptable since the function is heuristic. The 56_000 cap and 8_000 floor are preserved.
- `rank_files` continues to weight edges by a sub-linear function of `num_refs`. `isqrt(0)` is impossible here (`num_refs >= 1` from `Counter`).
- `aider/` is vendored upstream code and is NOT modified.

### 2. repomix --include hardening
- `render_with_repomix` no longer relies on comma-joining file paths into a single `--include` argument.
- Switched to a temp `repomix.config.json` written under a temp dir, with `include` as a JSON array of patterns — eliminates comma-split mangling for any filename.
- Each filename is glob-escaped (escapes `*?[]{}!()+@\`) so meta-chars in filenames don't accidentally widen or narrow the match.
- Temp config file is always cleaned up (success, repomix non-zero exit, timeout, or any exception).

### 3. tier0c / tier0d independent settings
- Add `tier0c_digest_timeout_seconds`, `tier0c_digest_cache_max_entries`, `tier0d_digest_timeout_seconds`, `tier0d_digest_cache_max_entries` to `ContextOptimizerSettings`.
- Defaults match the current shared values (5.0s / 500 entries) so behaviour is unchanged out of the box.
- `tier0c.py` reads its own knobs; `tier0d.py` reads its own knobs; the "intentionally share" comment blocks are removed.
- `tier0b.py` is untouched.

## Acceptance Criteria

- [x] `grep -rn "math\.sqrt" packages/context-optimizer/src` returns 0 hits.
- [x] `grep -n "math\.isqrt" packages/context-optimizer/src/context_optimizer/repo_index/index.py packages/context-optimizer/src/context_optimizer/repo_index/ranker.py` shows both call sites converted.
- [x] `render_with_repomix` writes a `repomix.config.json`, passes `--config <path>`, and never joins filenames with commas — verified by reading the file.
- [x] Temp config dir is removed even on `RuntimeError` or `subprocess.TimeoutExpired` — verified by inspection (try/finally with `shutil.rmtree(..., ignore_errors=True)`).
- [x] `ContextOptimizerSettings` exposes 4 new fields: `tier0c_digest_timeout_seconds`, `tier0c_digest_cache_max_entries`, `tier0d_digest_timeout_seconds`, `tier0d_digest_cache_max_entries`.
- [x] `tier0c.py` references zero `tier0b_digest_timeout_seconds` / `tier0b_digest_cache_max_entries`; same for `tier0d.py`.
- [x] `ruff check packages/context-optimizer/src` passes (zero new findings vs. baseline).
- [x] Existing `pytest packages/context-optimizer/tests` suite passes (no behavioural regression).

## Technical Decisions

- **`isqrt` vs Decimal vs rounding:** `math.isqrt` is the smallest, fastest deterministic option for integer-input call sites. Decimal sqrt would be overkill for a heuristic budget formula and a graph edge weight.
- **Config file vs `--include-from`:** repomix doesn't ship a stable `--include-from` flag. Its supported config schema (`{"include": [...]}` array) is the documented robust mechanism and side-steps the comma issue entirely.
- **Glob escaping policy:** include entries are treated as literal file paths (the ranker emits real `git ls-files` paths, never globs), so escaping meta-chars is correct — we don't need glob expansion.
- **Settings defaults match old shared values:** keeps existing deployments byte-identical until they choose to tune per-tier.

## Progress

- 2026-05-05 — spec drafted from user request.
- 2026-05-05 — implemented all three fixes.
  - `math.sqrt` → `math.isqrt` in `repo_index/index.py` (`_compute_token_ceiling`) and `repo_index/ranker.py` (`rank_files` edge weight).
  - `render_with_repomix` switched to a temp `repomix.config.json` with a JSON `include` array; `_escape_glob` neutralises glob meta-chars; temp dir always removed via `try/finally`.
  - Added `tier0c_digest_timeout_seconds`, `tier0c_digest_cache_max_entries`, `tier0d_digest_timeout_seconds`, `tier0d_digest_cache_max_entries` (all defaulting to 5.0 / 500); rewrote tier0c/0d to use them; deleted "intentionally share" comments.
  - Verified: `grep math\.sqrt` 0 actual calls, tier0c/0d reference no `tier0b_*` knobs, `ruff check` clean, `pytest` 36 passed.
