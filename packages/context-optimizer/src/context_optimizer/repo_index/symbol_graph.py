"""Owns: name-keyed index of symbol definitions and references across a repo.

Does NOT own: tree-sitter parsing (tagger.parse_repo), tag extraction
(tagger.extract_tags_from_parsed), git introspection (git_watcher), or the
MCP wire-shape (~/.claude/mcp-repo-tools/src/repo_tools/symbol_graph.py).
Called by: the MCP wrapper at repo_tools.symbol_graph (lazy-imported there).
Calls: tagger.parse_repo + tagger.extract_tags_from_parsed (single parse
pass shared with dep_graph), git_watcher.get_head_tree_sha (cache key).

State diagram — mirrors dep_graph because the cache contract is identical:
                +----------+   build_for_repo()    +-----------+
    (no entry)──▶|  miss    |──────────────────────▶|   built   |
                +----------+                        +-----------+
                                                         │
                      HEAD changes ─ stale ─◀────────────┘
                      ↓
                rebuild on next call

Honest limitations (documented in README too):
  - **Name-keyed only.** `process` defined in two files matches both. Use
    the returned (file, line) pairs to disambiguate by hand. No type
    inference, no class-method scoping.
  - **`kind` fidelity tracks tagger's tags queries.** Languages whose
    tree-sitter tags query distinguishes def from ref get accurate kinds;
    languages whose query lacks the distinction return everything as ref
    (defs_by_name will be empty for those languages). The list of
    def-distinguishing languages comes from queries/<lang>-tags.scm.
  - **No transitive call graph.** "Callers of callers" requires per-
    language enclosing-definition resolution; deferred to v2.

Public API (the MCP wrapper depends on these names verbatim):
    definition_of(repo_root, name)  -> list[SymbolDef]
    references_to(repo_root, name)  -> list[SymbolRef]
    build_for_repo(repo_root)       -> SymbolGraph
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass

from loguru import logger

from . import tagger
from .git_watcher import get_head_tree_sha


# # @stable — repo_tools.symbol_graph wire-shape depends on these field names
@dataclass(frozen=True)
class SymbolDef:
    """One definition site for a symbol. `kind` is always "def" today; kept as
    a field so the wire-shape matches SymbolRef and future per-language
    sub-kinds (e.g. "def.class" vs "def.function") can land without a schema
    change.
    """
    file: str
    line: int
    kind: str = "def"


# # @stable
@dataclass(frozen=True)
class SymbolRef:
    """One reference site for a symbol. Reference here means tagger flagged the
    capture as `name.reference.*` — call sites, attribute reads, type
    annotations all qualify depending on the language's tags.scm.
    """
    file: str
    line: int
    kind: str = "ref"


@dataclass
class SymbolGraph:
    """Built index for a single tree SHA. Immutable after build_for_repo returns.

    Two reverse-indexes from name → list-of-sites. Both are populated from
    the same Tag stream tagger emits, so they're internally consistent: a
    name appears in defs_by_name iff at least one tag with kind="def" named
    it; same for refs_by_name.
    """
    tree_sha: str
    defs_by_name: dict[str, list[SymbolDef]]
    refs_by_name: dict[str, list[SymbolRef]]


# Module-level cache — one SymbolGraph per repo, keyed by absolute repo_root.
# Per-repo locks serialise concurrent build_for_repo calls without globally
# serialising different repos. Counterpart pattern: dep_graph._cache.
_cache: dict[str, SymbolGraph] = {}
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


# # @stable — repo_tools.symbol_graph reads `.file`, `.line`, `.kind`
def definition_of(repo_root: str, name: str) -> list[SymbolDef]:
    """Return every definition site for `name` (across all files in the repo).

    Returns [] when no definition is tagged. Multiple matches are returned
    when the same name is defined in different files / scopes — callers
    disambiguate by `file` and `line`.
    Raises RuntimeError on git failure (not-a-repo, no commits) — callers
    in the MCP wrapper catch and translate to error JSON.
    """
    graph = build_for_repo(repo_root)
    return list(graph.defs_by_name.get(name, []))


# # @stable
def references_to(repo_root: str, name: str) -> list[SymbolRef]:
    """Return every reference site for `name`.

    Reference semantics are language-dependent: in Python `process` will
    match call sites and attribute reads alike. The list is unfiltered;
    the agent disambiguates by inspecting (file, line) snippets.
    """
    graph = build_for_repo(repo_root)
    return list(graph.refs_by_name.get(name, []))


# # @stable
def build_for_repo(repo_root: str) -> SymbolGraph:
    """Return the cached SymbolGraph for `repo_root`'s current HEAD.

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


def _build(repo_root: str, tree_sha: str) -> SymbolGraph:
    """Parse every tracked file, extract tags, fold into name → sites maps.

    Single parse pass shared with dep_graph (when both are called) — tagger
    handles its own internal caching. Tag stream comes from the existing
    `name.definition.*` / `name.reference.*` capture distinction in the
    per-language tags.scm files.
    """
    logger.info("REPO_INDEX: symbol_graph build_start repo={} sha={}", repo_root, tree_sha[:8])
    file_paths = _list_tracked_files(repo_root)
    parsed = tagger.parse_repo(repo_root, file_paths)
    tags_by_file = tagger.extract_tags_from_parsed(parsed)

    defs_by_name: dict[str, list[SymbolDef]] = {}
    refs_by_name: dict[str, list[SymbolRef]] = {}

    for rel_path, tags in tags_by_file.items():
        for tag in tags:
            # Tag.line is 0-based from tree-sitter; expose as 1-based to match
            # editor / git_blame conventions throughout repo-tools.
            line_1based = tag.line + 1
            if tag.kind == "def":
                defs_by_name.setdefault(tag.name, []).append(
                    SymbolDef(file=rel_path, line=line_1based)
                )
            elif tag.kind == "ref":
                refs_by_name.setdefault(tag.name, []).append(
                    SymbolRef(file=rel_path, line=line_1based)
                )

    logger.info(
        "REPO_INDEX: symbol_graph build_done repo={} defs={} refs={}",
        repo_root, len(defs_by_name), len(refs_by_name),
    )
    return SymbolGraph(
        tree_sha=tree_sha,
        defs_by_name=defs_by_name,
        refs_by_name=refs_by_name,
    )


def _list_tracked_files(repo_root: str) -> list[str]:
    """git ls-files for the repo. Mirrors dep_graph._list_tracked_files
    deliberately so symbol_graph can be used standalone without coupling
    to the optimizer-facing build pipeline.
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


# ── test hook ──────────────────────────────────────────────────────────────


def _reset_cache_for_tests() -> None:
    """Clear the module-level cache. INTERNAL — tests use this to start fresh
    when simulating HEAD changes within a single process.
    """
    with _cache_lock:
        _cache.clear()
        _repo_locks.clear()
