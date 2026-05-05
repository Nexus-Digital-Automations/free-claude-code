# Repo Index — Stable Prefix Layer

Layer -1 of the context-optimizer pipeline. Builds a commit-hash-keyed system-prompt
prefix from the top-N architecturally central files in a git repo, so the prefix bytes
are byte-identical across every session on the same commit — collapsing cold-start
prefix-cache misses.

---

## Why This Exists

DeepSeek and Anthropic's prefix caching works by hashing the leading bytes of the
system prompt. If those bytes change between sessions, the cache misses and the provider
reprocesses the full prompt. Aider-style dynamic repo-maps change every session (files
are ranked by recency, the map shifts). This system instead keys the prefix off
`git rev-parse HEAD`: same commit SHA → same bytes → cache hit, always.

The prefix is a full-content Repomix render of the top-N architecturally central files.
At query time, an embedding cosine search appends a short dynamic suffix of the most
relevant chunks for the current user message. Only the suffix changes per-request; the
prefix is frozen until the next commit.

---

## Enabling It

**Install the extra dependencies:**

```bash
pip install "context-optimizer[repo-index]"
```

**Enable in settings:**

```python
from context_optimizer import ContextOptimizer, ContextOptimizerSettings

settings = ContextOptimizerSettings(
    repo_index_enabled=True,        # off by default
    repo_index_top_n=20,            # files in the stable prefix
    repo_index_max_prefix_tokens=8_000,  # soft token cap
)

msgs, sys, tokens = await ContextOptimizer.optimize(
    messages, system=system, settings=settings, llm_provider=provider
)
```

The first call on a new commit triggers a full build (10–60 s depending on repo size).
Subsequent calls on the same commit return in microseconds from the in-memory singleton.

**Verifying the prefix loaded:**

Look for this log line:

```
REPO_INDEX: prefix_applied prefix_bytes=<N> suffix_chunks=<K> commit=<sha7>
```

If Layer -1 fails for any reason (Repomix not installed, git not found, etc.), it logs
a warning and falls through — the rest of the optimizer pipeline runs unchanged.

---

## Build Pipeline

```
git rev-parse HEAD  →  SHA
   │
   ├─ .context/repo-<SHA>.txt exists?  ──YES──▶  load from disk  ──▶  embed → singleton
   │
   NO
   │
   ▼
git ls-files --cached              all tracked files
   │
   ▼
tagger.get_tags_for_repo()         tree-sitter def/ref extraction (ThreadPoolExecutor)
   │
   ▼
ranker.rank_files()                PageRank on MultiDiGraph of symbol references
   │
   ▼
ranker.get_top_n_files()           top-N files, alphabetically sorted
   │
   ▼
renderer.render_with_repomix()     npx repomix → full file content with ## File: headers
   │  (fallback: renderer.render_fallback() if Node.js not available)
   │
   ▼
_enforce_token_cap()               reduce N by 5 until prefix ≤ max_prefix_tokens
   │
   ▼
embedder.chunk_text()              overlapping 200-token chunks per ## File: section
   │
   ▼
embedder.embed_chunks()            sentence-transformers all-MiniLM-L6-v2 (384-dim)
   │
   ▼
embedder.save_index()              atomic writes: .txt, .npy, .json to .context/
   │
   ▼
LoadedIndex singleton              cached in-process on _loaded_commit_hash
```

---

## Query-Time Path (per request)

```
optimize() called
   │
   ▼
settings.repo_index_enabled?  ──NO──▶  skip (no overhead)
   │
   YES
   │
   ▼
_resolve_repo_root()           settings.repo_index_root or git rev-parse --show-toplevel
   │
   ▼
loop.run_in_executor(          wraps synchronous build() to avoid blocking the event loop
  RepoIndex.get_or_build
)
   │
   ├─ _loaded_commit_hash == current SHA?  ──YES──▶  return singleton (µs)
   │
   ├─ .context/repo-<SHA>.{txt,npy,json} exist?  ──YES──▶  load from disk (~100 ms)
   │
   └─ NO  ──▶  full build (10–60 s, runs once per commit)
   │
   ▼
loaded.query(last_user_text, top_k=K)
   │  embed query → dot product against index_vectors → top-K chunks
   │
   ▼
loaded.format_suffix(results)  →  ## Relevant context section
   │
   ▼
_prepend_repo_context(prefix_text, suffix, system)
   │  prepends block to system (handles str | list | None)
   │
   ▼
system is now:  [stable prefix] [---] [dynamic suffix] [---] [original system]
```

The modified `system` flows into all downstream tiers (Tier 0–2) and token counting,
so the prefix tokens are correctly accounted for in compaction thresholds.

---

## Module Breakdown

### `repo_index/_types.py`

Pure dataclasses, no external deps.

| Type | Purpose |
|------|---------|
| `Tag` | namedtuple: `(rel_path, name, kind, line)` — a tree-sitter def or ref |
| `FileRank` | `(file_path, score)` — output of PageRank |
| `Chunk` | `(source_file, chunk_index, text, token_count)` — embedding unit |
| `IndexManifest` | metadata persisted to `.json`: commit_hash, model, files, timestamp |
| `RequestCacheStats` | per-request prefix cache accounting (for `CacheStatsTracker`) |

---

### `repo_index/git_watcher.py`

**`get_head_sha(repo_root) → str | None`**

Tries GitPython first (if installed), falls back to `git rev-parse HEAD` subprocess.
Returns `None` if `repo_root` is not a git repo or has no commits.

**`GitWatcher`** (optional — not used in the main hot path)

Asyncio poller that calls `on_commit_change(new_sha)` when HEAD changes.
States: `stopped → running → stopped`. Start with `.start()`, stop with `.stop()`.
Useful for background index rebuilds when `repo_index_watch_enabled=True`.

---

### `repo_index/tagger.py`

Extracts `Tag` records from source files using tree-sitter and vendored `.scm` query files.

**Supported languages:** Python, JavaScript, TypeScript, Go, Rust.

**`get_tags_for_file(abs_path, rel_path) → list[Tag]`**

- Detects language from file extension (`_EXT_TO_LANG` map)
- Loads the matching `.scm` query from `repo_index/queries/`
- Runs `_run_captures()` — compat shim for tree-sitter 0.23 (`query.captures()`) vs
  0.24+ (`QueryCursor.captures()`)
- Filters capture names: `name.definition.*` → `"def"`, `name.reference.*` → `"ref"`
- Returns `[]` on any error — never raises

**`get_tags_for_repo(repo_root, file_paths, *, max_workers=4) → dict[str, list[Tag]]`**

Runs `get_tags_for_file` in a `ThreadPoolExecutor`. Returns `{rel_path: [Tag, ...]}`.

---

### `repo_index/ranker.py`

PageRank on a symbol import graph. Port of Aider's algorithm from `aider/repomap.py`.

**Graph construction:**

- Node = file path
- Edge = `(referencer → definer)` for each `(ref_tag, def_tag)` pair where `name` matches
- Edge weight = `_edge_multiplier(ident, definer_count) * sqrt(ref_count)`

**`_edge_multiplier` rules:**

| Condition | Multiplier | Rationale |
|-----------|------------|-----------|
| Ident ≥ 8 chars, snake/camel/kebab | 10× | Long names are domain-specific |
| Ident starts with `_` | 0.1× | Private symbols rarely indicate architecture |
| More than 5 definers for same name | 0.1× | Very common builtins (e.g. `__init__`) |
| Otherwise | 1× | |

Self-edges are added for files that only define symbols (no inbound refs), keeping them
in the graph so they receive a baseline PageRank score.

**`rank_files(tags_by_file) → list[FileRank]`**

Runs `nx.pagerank(G, weight="weight")`. Falls back to uniform ranking on
`ZeroDivisionError` (disconnected graph with no edges).

**`get_top_n_files(ranked, n) → list[str]`**

Returns the top-N file paths sorted **alphabetically**. Alphabetical sort is critical:
Repomix must receive the same file list in the same order every time to produce
byte-identical output — which is what keeps the prefix cache hot.

---

### `repo_index/renderer.py`

**`render_with_repomix(repo_root, include_files, *, timeout_seconds, extra_args)`**

Runs:
```
npx repomix \
  --include <comma-sep-absolute-paths> \
  --output-format markdown \
  --output /dev/stdout
```

Output format: `## File: <relative-path>` headers followed by file content.
Raises `FileNotFoundError` if `npx` is not on PATH, `RuntimeError` on non-zero exit,
`subprocess.TimeoutExpired` on timeout.

**`render_fallback(repo_root, include_files)`**

Pure Python fallback — reads files directly and emits the same `## File:` header format.
Used when Node.js/npx is not available. Output is always deterministic.

---

### `repo_index/embedder.py`

**Chunking**

`chunk_text(text, *, chunk_size_tokens=200, overlap_tokens=20) → list[Chunk]`

- Splits on `## File: <path>` headers to assign `source_file` attribution per chunk
- Respects token budget line-by-line (never splits mid-line)
- Trailing chunks overlap by `overlap_tokens` from the previous chunk

Token counting uses tiktoken `cl100k_base` (same as the rest of the package).
Falls back to `len(s) // 4` if tiktoken is unavailable.

**Embedding**

`embed_chunks(chunks, *, model_name="sentence-transformers/all-MiniLM-L6-v2") → np.ndarray`

- Model: 384-dim, MIT license, ~80 MB download on first use
- Batch size 64, `normalize_embeddings=True` → unit-normalised float32
- Module-level singleton avoids re-loading the model across requests

**Search**

`cosine_search(query_vector, index_vectors, *, top_k) → list[tuple[int, float]]`

Flat dot product (= cosine similarity on unit vectors). `O(N × dim)`.
Adequate for <10K chunks (~20 MB index, <10 ms on CPU). No FAISS/ANN needed at this scale.

**Persistence** (atomic writes via `.tmp` + `os.rename`)

| File | Content |
|------|---------|
| `.context/repo-<sha>.txt` | Frozen Repomix render (the stable prefix text) |
| `.context/repo-<sha>.npy` | float32 embedding matrix, shape `(N_chunks, 384)` |
| `.context/repo-<sha>.json` | `IndexManifest` + full chunk texts for reload |

`prune_old_indexes(output_dir, keep=3)` deletes old triplets by `.json` mtime, keeping
only the 3 most recent commits. Prevents unbounded disk growth on active repos.

`.context/` is always gitignored (a `.gitignore` containing `*` is written into it on
first save, and the root `.gitignore` also excludes `.context/`).

---

### `repo_index/index.py`

**`LoadedIndex`** — immutable dataclass, in-memory representation of one built index.

```
commit_hash: str
prefix_text: str        # frozen Repomix render
chunks: list[Chunk]
vectors: np.ndarray     # (N, 384), unit-normalised float32
manifest: IndexManifest

query(query_text, *, top_k) → list[tuple[Chunk, float]]
format_suffix(results)  → str   # ## Relevant context markdown block
```

**`RepoIndex`** — stateless class, all mutable state in module-level singletons.

```python
_loaded_index: LoadedIndex | None       # hot-path cache
_loaded_commit_hash: str | None
```

`get_or_build(repo_root, settings) → LoadedIndex | None`

Performance tiers:
1. In-memory hit (`_loaded_commit_hash == sha`): **µs**
2. Disk hit (`.context/repo-<sha>.*` all present): **~100 ms**
3. Full build: **10–60 s** (runs once per commit, then cached)

Never raises — all failures are logged as warnings and return `None`, allowing the
optimizer to degrade gracefully.

---

### `repo_index/cache_stats.py`

**`CacheStatsTracker`**

Records prefix-cache accounting from provider API responses. Call after each API call:

```python
tracker = CacheStatsTracker(request_id="...", prefix_bytes=len(loaded.prefix_text))
tracker.record_api_usage(response.usage, provider="deepseek")
tracker.log_summary()
```

Handles two usage shapes:
- **DeepSeek/OpenAI**: `usage.prompt_tokens_details.cached_tokens`
- **Anthropic**: `usage.cache_read_input_tokens` (hit), `usage.cache_creation_input_tokens` (write)

---

## Configuration Reference

All fields on `ContextOptimizerSettings` (defaults shown):

```python
# Feature flag
repo_index_enabled: bool = False

# Repo root — None = auto-detect from cwd via git rev-parse --show-toplevel
repo_index_root: str | None = None

# Where to write .context/ files — None = <git_root>/.context/
repo_index_context_dir: str | None = None

# How many files to include in the stable prefix
repo_index_top_n: int = 20

# Embedding chunk parameters
repo_index_chunk_size_tokens: int = 200
repo_index_chunk_overlap_tokens: int = 20

# sentence-transformers model — must be HuggingFace compatible
repo_index_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

# How many embedding-search chunks to append as dynamic suffix
repo_index_query_top_k: int = 10

# Repomix subprocess timeout
repo_index_repomix_timeout: float = 120.0

# GitWatcher poll interval (only used if repo_index_watch_enabled=True)
repo_index_poll_interval_seconds: float = 30.0
repo_index_watch_enabled: bool = False

# Soft token cap on the stable prefix; build() drops files until it fits
# 0 = no cap
repo_index_max_prefix_tokens: int = 8_000

# Extra CLI args passed verbatim to repomix (e.g. ["--ignore", "*.lock"])
repo_index_repomix_extra_args: list[str] = field(default_factory=list)
```

---

## Dependency Installation

The repo-index deps are an optional extra to keep the base package lightweight:

```bash
pip install "context-optimizer[repo-index]"
```

This installs: `networkx`, `numpy`, `sentence-transformers` (pulls torch), `gitpython`,
`tree-sitter`, `tree-sitter-language-pack`.

The `sentence-transformers` model (~80 MB) is downloaded on first use. Subsequent calls
use the module-level singleton — no re-download.

Repomix requires Node.js:

```bash
npm install -g repomix
# or use npx (no global install needed)
```

If `npx`/`repomix` is not available, `render_fallback()` is used automatically — pure
Python, no Node.js required. Output is slightly less structured but uses the same
`## File:` header format, so chunking and embedding work identically.

---

## Disk Layout

```
<git_root>/
  .context/                    ← gitignored
    .gitignore                 ← contains "*", written on first save
    repo-<sha7digits>.txt      ← stable prefix text (Repomix render)
    repo-<sha7digits>.npy      ← embedding matrix (float32, shape N×384)
    repo-<sha7digits>.json     ← IndexManifest + chunk texts
    repo-<old_sha>.txt         ← pruned after 3 commits accumulate
    repo-<old_sha>.npy
    repo-<old_sha>.json
```

All three files for a commit are written atomically (`.tmp` + `os.rename`). A partial
write (e.g. process killed mid-build) leaves a `.tmp` file and the old index intact —
`load_index()` checks all three extensions exist before returning, so a partial write
is never served.

---

## Tradeoffs and Rejected Alternatives

**Why Repomix instead of reading files directly?**
Repomix normalises line endings, strips binary files, and produces a consistent header
format. The fallback renderer does the same in pure Python. The key invariant is that
the output format is deterministic for a fixed file list — both renderers guarantee this.

**Why flat numpy search instead of FAISS?**
At <10K chunks the flat dot product costs <10 ms on CPU and needs zero C++ build
infrastructure. FAISS adds ANN approximation error and a binary dependency. Re-evaluate
at >50K chunks.

**Why `all-MiniLM-L6-v2` instead of a larger model?**
384-dim, MIT license, ~80 MB, fast inference. Retrieval quality is adequate for
"find files relevant to this user message". A 768-dim model doubles index size and
inference time for marginal quality gain at this task.

**Why alphabetical sort before Repomix?**
Repomix's output order follows its `--include` argument order. Alphabetical sort
ensures the same file list always produces the same byte sequence — critical for the
prefix cache hit. Any sort that is a pure function of file paths (not mtime, git log,
etc.) would work; alphabetical is simplest.

**Why run_in_executor instead of making build() async?**
`build()` calls tree-sitter (C extension), numpy, and sentence-transformers — all
synchronous CPU work that cannot yield to the event loop. Wrapping in `run_in_executor`
keeps the asyncio loop unblocked while the build runs in a thread pool thread.
