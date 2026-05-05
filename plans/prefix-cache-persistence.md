---
title: Prefix-Cache Persistence — Implementation Plan
status: archived
created: 2026-05-05
---

## Context
Pre-existing plan note from prior compaction work. Frontmatter added retroactively so the plans-directory spec contract is satisfied; original content preserved unchanged below.

## Acceptance Criteria
- [x] Pre-existing scope captured in body.

# Prefix-Cache Persistence — Implementation Plan

## Problem

`PrefixCache` lives as a class-level `OrderedDict` on `ContextOptimizer._cache`. It is entirely in-memory. Every time the process restarts (new Claude Code session, proxy restart), the cache starts cold. Session restarts are the single largest source of the ~57% cache-miss rate.

## Key Constraints

1. **Filepath-scoped, not global.** Two different projects at different working directories must NEVER share cache entries. The cache file is tied to the git root (or working directory if not a git repo).
2. **No hook changes.** Hooks run as subprocesses via `uv run --script`. They cannot access the optimizer's in-memory state. Persistence must be self-contained within the optimizer package.
3. **Backward compatible.** `persist_path = None` (default) → zero file I/O, same behavior as today.

## Design: Self-Saving PrefixCache

The `PrefixCache` class handles its own persistence via safe file writes. No hook modifications needed.

### Storage Path Resolution

```
context_cache_dir provided?  →  {dir}/context-cache.json
context_cache_dir is None    →  resolve from cwd:
                                 git root found  →  {git_root}/.claude/data/context-cache.json
                                 no git root     →  {cwd}/.claude/data/context-cache.json
```

The resolution happens once, lazily on the first `optimize()` call (not at import time, since cwd can change). The optimizer computes the path and passes it to `PrefixCache.__init__()`.

### File Format

Single JSON file in the project's `.claude/data/` directory (already gitignored):

```json
{
  "version": 2,
  "created_at": "2025-05-14T12:00:00",
  "entries": [
    {
      "key": "sha256hex...",
      "split_index": 42,
      "summary": "...",
      "created_at": "2025-05-14T12:00:00",
      "hit_count": 3
    }
  ]
}
```

### Write Strategy

- **Atomic:** Write to `<path>.tmp`, then `os.rename()` to `<path>`. Atomic on POSIX. Never corrupts the live file.
- **On store():** After inserting a new entry into the LRU, call `_save()` if the dirty flag is set. `store()` is called 0–2 times per prompt (background Tier 2a, sync Tier 2b).
- **Dirty flag:** Set on every `store()`. Cleared after successful `_save()`. Avoids rewriting the same file on no-op lookups.
- **mkdir ensured:** `os.makedirs(Path(path).parent, exist_ok=True)` in both `_save()` and `_load()`.

### Load Strategy

On `PrefixCache.__init__()`:
1. If `persist_path` is None → no-op (backward compatible)
2. Try to open and read the file
3. Validate `version >= 2`
4. Load entries into `self._store` (insertion order = created_at sequence)
5. Log count: `"CONTEXT_OPT: loaded N cached entries from {path}"`
6. File missing or corrupt → log debug, start empty, no crash

### File Size Bounds

- Max 100 entries (existing config)
- Per entry: ~256B key + ~2000B summary + ~60B metadata ≈ ~2.3 KB
- Max file: ~230 KB + JSON overhead → well under 1 MB
- Writes complete in <5ms

### What Changes (4 files)

| File | Change |
|------|--------|
| `packages/context-optimizer/src/context_optimizer/settings.py` | Add `context_cache_dir: str \| None = None` |
| `packages/context-optimizer/src/context_optimizer/cache.py` | Add `persist_path`, `_save()`, `_load()`, dirty flag, `clear()` |
| `packages/context-optimizer/src/context_optimizer/optimizer.py` | Compute cache path from cwd/git root → pass to `PrefixCache` |
| `providers/common/context_optimizer.py` | Map through from proxy settings |

### Details Per File

**`settings.py`** — one new field:
```python
context_cache_dir: str | None = None
"""Directory for the persisted prefix-cache file.
If None, the optimizer auto-resolves from the current working directory
(git root → .claude/data/, or cwd → .claude/data/ if not a git repo).
Set to an explicit path to override, or to an empty sentinel to disable.
The cache file itself is always named 'context-cache.json' inside this dir."""
```

**`cache.py`** — the core change:
- `__init__(max_entries=100, persist_path=None)`: accept optional path. If set, call `_load()`.
- `_load()`: read JSON, validate, populate `_store`.
- `_save()`: serialize `_store` + metadata to JSON, temp+rename write. Set dirty flag.
- `store()`: call `_save()` after inserting (if persist_path set).
- `clear()`: reset dirty flag, delete file if persist_path set.
- Module or class-level `_dirty: bool` flag.

**`optimizer.py`** — resolve + wire:
```python
def _resolve_cache_dir(settings: ContextOptimizerSettings) -> str | None:
    if settings.context_cache_dir is not None:
        return settings.context_cache_dir
    # Auto-resolve from cwd
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, timeout=5)
        root = r.stdout.strip() if r.returncode == 0 else os.getcwd()
    except Exception:
        root = os.getcwd()
    cache_dir = os.path.join(root, ".claude", "data")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir
```

Then `_get_cache()` builds the full path:
```python
cache_dir = _resolve_cache_dir(settings)
persist_path = os.path.join(cache_dir, "context-cache.json") if cache_dir else None
cls._cache = PrefixCache(settings.prefix_cache_max_entries, persist_path=persist_path)
```

**`providers/common/context_optimizer.py`** — add one mapping line:
```python
context_cache_dir=getattr(settings, "context_cache_dir", None),
```

### Acceptance Criteria

1. **Project isolation:** Two sessions in different working directories produce separate cache files with no shared state
2. **Restart recovery:** Kill + restart process → next `optimize()` loads persisted entries → cache hit works
3. **Safe corruption:** Delete or corrupt the cache file → log warning, empty cache, no crash
4. **Backward compatible:** `context_cache_dir = None` (default) → no file I/O, existing callers unchanged
5. **Atomic writes:** Kill during `_save()` → at most lose the current write, never corrupt prior data
6. **No-ops don't thrash:** Lookups without stores don't touch disk
7. **LRU persists:** Evicted entries are removed from the file on next store

### Verification

1. `ruff check packages/context-optimizer/` — no new lint errors
2. Run existing tests: `cd packages/context-optimizer && python -m pytest tests/ -x -q`
3. Manual: start session → trigger background compaction → kill process → restart → verify `"CONTEXT_OPT: loaded N cached entries"` in logs → verify `context-cache.json` exists in `.claude/data/`
