"""Owns: file-level dependency graph for a repo — outgoing imports + reverse importers.

Does NOT own: per-language extraction (import_extractors/* owns that),
tree-sitter parsing (tagger.parse_repo), git introspection (git_watcher),
or the MCP wire-shape (~/.claude/mcp-repo-tools/src/repo_tools/dep_graph.py).
Called by: the MCP wrapper at repo_tools.dep_graph (lazy-imported there) and
optionally by index.py if/when transparent injection ships.
Calls: tagger.parse_repo (parses every tracked file once), import_extractors
(per-language extraction), git_watcher.get_head_tree_sha (cache invalidation).

State diagram:
                    +----------+   build_for_repo()    +-----------+
    (no entry)  ──▶ |  miss    | ───────────────────▶  |   built   |
                    +----------+                       +-----------+
                                                            │
                          HEAD changes ─ stale ─◀───────────┘
                          ↓
                    rebuild on next call

Cache lives in module-level dict keyed by repo_root absolute path. Each
entry is keyed by tree SHA so post-rebase reads still hit when no source
files changed. Build is serialised per repo via per-repo lock so two
concurrent calls can't both rebuild the same graph.

Public API (the MCP wrapper depends on these names verbatim):
    imports_of(repo_root, rel_path)      -> list[Import]
    imported_by(repo_root, rel_path)     -> set[str]
    unresolved_count(repo_root, rel_path) -> int
    language_of(repo_root, rel_path)     -> str
    build_for_repo(repo_root)            -> DepGraph
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass, field

from loguru import logger

from . import tagger
from ._types import ParsedFile
from .git_watcher import get_head_tree_sha
from .import_extractors import EXTRACTORS, RawImport


# # @stable — repo_tools.dep_graph wire-shape depends on these field names
@dataclass(frozen=True)
class Import:
    """A resolved import edge as exposed to MCP clients.

    `resolved` is None when the importer's language is in the raw-only tier
    OR when resolution couldn't find a matching in-repo file (e.g. a Python
    import of `requests`). Clients use the presence/absence to decide
    whether to follow the edge.
    """

    raw: str
    line: int
    resolved: str | None


@dataclass
class DepGraph:
    """Built graph for a single tree SHA. Immutable after build_for_repo returns.

    `by_file` maps each importing file to its outgoing edges.
    `importers` maps each *resolved* path to the set of files importing it
    (the inverse of by_file restricted to edges with resolved targets).
    `unresolved_by_file` counts unresolved edges per importer — the MCP
    wrapper surfaces this so the agent knows how complete the answer is.
    `language_by_file` lets the MCP wrapper echo the per-language resolution
    tier without re-detecting from extension.
    """

    tree_sha: str
    by_file: dict[str, list[Import]]
    importers: dict[str, set[str]]
    unresolved_by_file: dict[str, int]
    language_by_file: dict[str, str]
    resolution_by_language: dict[str, str] = field(default_factory=dict)


# Module-level cache — one DepGraph per repo, keyed by absolute repo_root.
# Per-repo locks serialise concurrent build_for_repo calls without
# globally serialising different repos.
_cache: dict[str, DepGraph] = {}
_cache_lock = threading.Lock()
_repo_locks: dict[str, threading.Lock] = {}


def _lock_for(repo_root: str) -> threading.Lock:
    with _cache_lock:
        lock = _repo_locks.get(repo_root)
        if lock is None:
            lock = threading.Lock()
            _repo_locks[repo_root] = lock
        return lock


# ── public API ─────────────────────────────────────────────────────────────


# # @stable — repo_tools.dep_graph reads `edge.raw`, `edge.resolved`, `edge.line`
def imports_of(repo_root: str, rel_path: str) -> list[Import]:
    """Return outgoing import edges from `rel_path`, building if necessary.

    Returns [] if the file is not tracked or has no detectable imports.
    Raises RuntimeError on git failure (not-a-repo, no commits) — callers
    in the MCP wrapper catch and translate to error JSON.
    """
    graph = build_for_repo(repo_root)
    return list(graph.by_file.get(rel_path, []))


# # @stable
def imported_by(repo_root: str, rel_path: str) -> set[str]:
    """Return the set of files that import `rel_path` (depth 1, resolved only)."""
    graph = build_for_repo(repo_root)
    return set(graph.importers.get(rel_path, set()))


# # @stable
def unresolved_count(repo_root: str, rel_path: str) -> int:
    """Return how many of `rel_path`'s outgoing edges did not resolve to in-repo paths."""
    graph = build_for_repo(repo_root)
    return graph.unresolved_by_file.get(rel_path, 0)


# # @stable
def language_of(repo_root: str, rel_path: str) -> str:
    """Return the detected language for `rel_path`, or "" if unknown."""
    graph = build_for_repo(repo_root)
    return graph.language_by_file.get(rel_path, "")


# # @stable
def build_for_repo(repo_root: str) -> DepGraph:
    """Return the cached DepGraph for `repo_root`'s current HEAD, building on miss.

    Cache hit (same tree SHA) returns in microseconds. Build is serialised
    per repo so two concurrent callers don't both rebuild.
    Raises RuntimeError if `repo_root` is not a git repository or has no commits.
    """
    sha = get_head_tree_sha(repo_root)
    if sha is None:
        raise RuntimeError(f"not a git repo or no commits: {repo_root}")
    cached = _cache.get(repo_root)
    if cached is not None and cached.tree_sha == sha:
        return cached
    with _lock_for(repo_root):
        cached = _cache.get(repo_root)
        if cached is not None and cached.tree_sha == sha:
            return cached
        graph = _build(repo_root, sha)
        _cache[repo_root] = graph
        return graph


# ── build pipeline ─────────────────────────────────────────────────────────


def _build(repo_root: str, tree_sha: str) -> DepGraph:
    """Parse every tracked file, extract imports, resolve, build inverse map."""
    logger.info(
        "REPO_INDEX: dep_graph build_start repo={} sha={}", repo_root, tree_sha[:8]
    )
    file_paths = _list_tracked_files(repo_root)
    parsed = tagger.parse_repo(repo_root, file_paths)

    by_file: dict[str, list[Import]] = {}
    unresolved_by_file: dict[str, int] = {}
    language_by_file: dict[str, str] = {p.rel_path: p.language for p in parsed.values()}
    resolution_by_language: dict[str, str] = {}

    for rel_path, parsed_file in parsed.items():
        extractor = EXTRACTORS.get(parsed_file.language)
        if extractor is None:
            resolution_by_language.setdefault(parsed_file.language, "raw_only")
            continue
        resolution_by_language.setdefault(parsed_file.language, "resolved")
        edges = _resolve_imports(parsed_file, extractor(parsed_file), parsed)
        by_file[rel_path] = edges
        unresolved_by_file[rel_path] = sum(1 for e in edges if e.resolved is None)

    importers: dict[str, set[str]] = {}
    for src, edges in by_file.items():
        for edge in edges:
            if edge.resolved is not None:
                importers.setdefault(edge.resolved, set()).add(src)

    logger.info(
        "REPO_INDEX: dep_graph build_done repo={} files_with_edges={} importers={}",
        repo_root,
        len(by_file),
        len(importers),
    )
    return DepGraph(
        tree_sha=tree_sha,
        by_file=by_file,
        importers=importers,
        unresolved_by_file=unresolved_by_file,
        language_by_file=language_by_file,
        resolution_by_language=resolution_by_language,
    )


def _list_tracked_files(repo_root: str) -> list[str]:
    """git ls-files for the repo. Mirrors index._list_tracked_files but local
    to this module so dep_graph can be used standalone without importing the
    optimizer-facing build pipeline.
    """
    result = subprocess.run(
        ["git", "-C", repo_root, "ls-files", "--cached", "--full-name"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {result.stderr[:200]}")
    rel = [p for p in result.stdout.splitlines() if p]
    return [os.path.join(repo_root, p) for p in rel]


# ── resolution ─────────────────────────────────────────────────────────────


def _resolve_imports(
    parsed: ParsedFile,
    raw_imports: list[RawImport],
    all_parsed: dict[str, ParsedFile],
) -> list[Import]:
    """Map RawImport entries to in-repo paths where possible.

    Python and JS/TS get full resolution; other languages stay raw-only —
    cross-module resolution there requires build-system knowledge
    (go.mod, Cargo.toml, tsconfig path aliases) that's deliberately out
    of scope for this tool.
    """
    if parsed.language == "python":
        return [_resolve_python(parsed, ri, all_parsed) for ri in raw_imports]
    if parsed.language in ("javascript", "typescript"):
        return [_resolve_js(parsed, ri, all_parsed) for ri in raw_imports]
    return [Import(raw=ri.raw, line=ri.line, resolved=None) for ri in raw_imports]


def _resolve_python(
    importer: ParsedFile,
    raw: RawImport,
    all_parsed: dict[str, ParsedFile],
) -> Import:
    """Resolve a Python import to a repo-relative .py path or __init__.py.

    Algorithm:
      - Relative imports (`raw.is_relative`): start from importer's package
        directory, walk up one level per leading dot, then descend the
        remaining segments. `from . import x` resolves to `<dir>/x.py`.
      - Absolute imports (`foo.bar`): try `foo/bar.py` and `foo/bar/__init__.py`
        from the repo root.
    Returns Import.resolved=None when no in-repo file matches — typical for
    third-party packages (`requests`, `numpy`).
    """
    candidates = _python_candidates(importer.rel_path, raw)
    parsed_paths = {rel: True for rel in all_parsed}
    for candidate in candidates:
        if candidate in parsed_paths:
            return Import(raw=raw.raw, line=raw.line, resolved=candidate)
    return Import(raw=raw.raw, line=raw.line, resolved=None)


def _resolve_js(
    importer: ParsedFile,
    raw: RawImport,
    all_parsed: dict[str, ParsedFile],
) -> Import:
    """Resolve a JS/TS import to an in-repo file when it's relative.

    Bare specifiers (`react`, `@scope/foo`) stay unresolved — they refer to
    npm packages that aren't in the repo. Path-alias resolution (e.g. tsconfig
    `paths`) is out of scope; the agent gets the raw string and can fall back
    to grep-equivalent if it really needs cross-alias resolution.
    """
    if not raw.is_relative:
        return Import(raw=raw.raw, line=raw.line, resolved=None)
    candidates = _js_candidates(importer.rel_path, raw.raw)
    parsed_paths = {rel: True for rel in all_parsed}
    for candidate in candidates:
        if candidate in parsed_paths:
            return Import(raw=raw.raw, line=raw.line, resolved=candidate)
    return Import(raw=raw.raw, line=raw.line, resolved=None)


_JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")


def _js_candidates(importer_rel: str, raw: str) -> list[str]:
    """Return the rel-paths a relative JS/TS import COULD resolve to.

    Mirrors Node/TS resolution at the file-system level only:
      ./x      → ./x.ts, ./x.tsx, ./x.js, ./x.jsx, ./x.mjs, ./x.cjs,
                 ./x/index.ts, ./x/index.tsx, ./x/index.js, ./x/index.jsx
      ./x.ts   → ./x.ts (already has extension — try as-is first)
    """
    importer_dir = os.path.dirname(importer_rel)
    base = os.path.normpath(os.path.join(importer_dir, raw))
    out: list[str] = []
    if any(base.endswith(ext) for ext in _JS_EXTENSIONS):
        out.append(base)
    for ext in _JS_EXTENSIONS:
        out.append(base + ext)
    for ext in _JS_EXTENSIONS:
        out.append(os.path.join(base, "index" + ext))
    return out


def _python_candidates(importer_rel: str, raw: RawImport) -> list[str]:
    """Return the rel-paths a Python import string COULD resolve to, in priority order.

    Callers take the first that exists in the parsed set. Strategy:
      1. For each segment-prefix length from longest to shortest, try the
         module form (foo/bar.py) before the package form (foo/bar/__init__.py).
      2. Truncating gives correct fallback for `from pkg import name` where
         `name` is a symbol in pkg/__init__.py rather than a submodule —
         pkg/name.py misses, then pkg/__init__.py hits.

    Module form is preferred over package form within a length: matches
    Python's own resolution (a sibling foo.py shadows foo/__init__.py).
    """
    text = raw.raw
    if raw.is_relative:
        leading_dots = len(text) - len(text.lstrip("."))
        remainder = text[leading_dots:]
        importer_dir = os.path.dirname(importer_rel)
        for _ in range(leading_dots - 1):
            importer_dir = os.path.dirname(importer_dir)
        base_segments = importer_dir.split(os.sep) if importer_dir else []
        rest_segments = remainder.split(".") if remainder else []
        segments = [s for s in base_segments + rest_segments if s]
    else:
        segments = [s for s in text.split(".") if s]
    candidates: list[str] = []
    while segments:
        joined = "/".join(segments)
        candidates.append(joined + ".py")
        candidates.append(joined + "/__init__.py")
        segments = segments[:-1]
    return candidates


# ── test hook ──────────────────────────────────────────────────────────────


def _reset_cache_for_tests() -> None:
    """Clear the module-level cache. INTERNAL — tests use this to start fresh
    when simulating HEAD changes within a single process.
    """
    with _cache_lock:
        _cache.clear()
        _repo_locks.clear()
