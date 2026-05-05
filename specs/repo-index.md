---
title: repo-index — Stable Prefix Layer for context-optimizer
status: active
created: 2026-05-05
---

## Vision

Add Layer -1 to context-optimizer: a commit-hash-keyed stable system-prompt prefix that makes the LLM's prefix cache hit rate independent of session boundaries. Same commit → byte-identical prefix → cache hit. Dynamic embedding retrieval appends relevant file chunks as a suffix after the stable block.

## Requirements

- tree-sitter + PageRank ranks files by architectural centrality (port of aider's algorithm)
- Repomix renders the top-N ranked files as full content; output is deterministic (sorted include list)
- Result saved to `.context/repo-<hash>.txt` per repo; only regenerated on new commit
- Embedding index (numpy flat) built once per commit; cosine search retrieves relevant chunks at query time
- `CacheStatsTracker` extracts `prompt_cache_hit_tokens` per API response and logs them
- All new settings in `ContextOptimizerSettings`; disabled by default (`repo_index_enabled=False`)
- No breaking changes to existing `ContextOptimizer.optimize()` callers

## Acceptance Criteria

- [ ] `from context_optimizer.repo_index import RepoIndex, LoadedIndex, CacheStatsTracker` works
- [ ] `RepoIndex.build(repo_root, settings, force=True)` returns a `LoadedIndex` with non-empty `prefix_text`
- [ ] `RepoIndex.build()` on same commit (no force) returns in < 200ms (disk cache hit)
- [ ] `LoadedIndex.query("how does the cache work?")` returns a non-empty results list
- [ ] `.context/repo-<hash>.txt`, `.context/repo-<hash>.npy`, `.context/repo-<hash>.json` written atomically
- [ ] `ContextOptimizer.optimize()` with `repo_index_enabled=True` prepends prefix to system prompt
- [ ] `ruff check packages/context-optimizer/src/` exits 0
- [ ] `CacheStatsTracker.record_api_usage()` handles both DeepSeek and Anthropic usage shapes

## Technical Decisions

- **PageRank with no personalization**: stable prefix has no session-specific context to personalize toward
- **Alphabetical sort of top-N files before Repomix**: ensures byte-identical output for same file set
- **sentence-transformers/all-MiniLM-L6-v2**: 384-dim, MIT license, ~80MB, good quality/speed tradeoff
- **Flat numpy cosine search**: adequate for < 10K chunks (~5ms on CPU); avoids FAISS C++ dependency
- **Repomix markdown format**: `## File: <path>` headers are easy to parse for chunk source attribution
- **render_fallback()**: pure Python renderer preserves stability guarantee if Node.js unavailable

## Progress

Layer -1 extends `optimizer.py::optimize()` — inserts before tier0, modifying only `system` parameter.
