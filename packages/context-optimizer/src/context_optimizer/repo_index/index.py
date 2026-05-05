"""Owns: top-level RepoIndex coordinator — wires the full build pipeline and in-memory cache.

Does NOT own: git introspection (git_watcher), tag extraction (tagger), ranking (ranker),
rendering (renderer), or embedding (embedder) — delegates all of those.
Called by: optimizer.py (get_or_build via run_in_executor).
Calls: git_watcher, tagger, ranker, renderer, embedder, subprocess (git ls-files).

Module-level singleton (_loaded_index / _loaded_commit_hash) caches the active LoadedIndex
in memory so get_or_build() returns in microseconds on the hot path (same commit, already
loaded). build() is synchronous so callers can wrap it in run_in_executor without an event
loop inside the thread.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from loguru import logger

from ..settings import ContextOptimizerSettings
from . import embedder, ranker, renderer, tagger
from ._types import Chunk, IndexManifest
from .git_watcher import get_head_sha

# ── Module-level hot-path cache ────────────────────────────────────────────
# Avoids disk I/O on every request when the commit hasn't changed.
_loaded_index: LoadedIndex | None = None
_loaded_commit_hash: str | None = None


@dataclass
class LoadedIndex:
    """In-memory representation of a fully built index for one commit. Immutable after creation.

    # @stable — optimizer.py depends on query() and format_suffix() signatures.
    """

    commit_hash: str
    prefix_text: str        # frozen Repomix render — the stable system-prompt prefix
    chunks: list[Chunk]
    vectors: np.ndarray     # shape (N, embedding_dim), unit-normalised float32
    manifest: IndexManifest

    def query(self, query_text: str, *, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Embed query_text and return top-k matching chunks sorted by cosine similarity.

        Uses the same model that was used during indexing (stored in manifest).
        Returns [] if the index has no chunks.
        """
        if not self.chunks:
            return []
        model = embedder._load_model(self.manifest.embedding_model)
        vec = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
        query_vec = vec[0].astype(np.float32)
        hits = embedder.cosine_search(query_vec, self.vectors, top_k=top_k)
        return [(self.chunks[i], score) for i, score in hits]

    def format_suffix(self, results: list[tuple[Chunk, float]]) -> str:
        """Render query results as a markdown suffix block for the system prompt."""
        if not results:
            return ""
        lines = ["## Relevant context (embedding search)\n"]
        seen: set[str] = set()
        for chunk, _score in results:
            key = f"{chunk.source_file}:{chunk.chunk_index}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"\n### File: {chunk.source_file}\n\n{chunk.text}\n")
        return "\n".join(lines)


class RepoIndex:
    """Stateless class — all mutable state lives in module-level singletons and files on disk.

    # @stable — optimizer.py depends on get_or_build() signature.
    """

    @classmethod
    def get_or_build(
        cls,
        repo_root: str,
        settings: ContextOptimizerSettings,
    ) -> LoadedIndex | None:
        """Return the current LoadedIndex, building if necessary.

        Hot path (in-memory hit): µs. Disk hit: ~100ms. Full build: 10-60s.
        Returns None only if repo_root is not a git repo.
        Never raises — failures are logged and return None so callers degrade gracefully.
        """
        global _loaded_index, _loaded_commit_hash
        try:
            sha = get_head_sha(repo_root)
            if sha is None:
                logger.debug("REPO_INDEX: get_or_build no git repo at root={}", repo_root)
                return None

            if _loaded_commit_hash == sha and _loaded_index is not None:
                return _loaded_index

            loaded = cls.load(repo_root, settings)
            if loaded is not None:
                _loaded_index = loaded
                _loaded_commit_hash = sha
                return loaded

            built = cls.build(repo_root, settings)
            _loaded_index = built
            _loaded_commit_hash = sha
            return built

        except Exception as exc:
            logger.warning("REPO_INDEX: get_or_build failed root={} reason={}: {}", repo_root, type(exc).__name__, exc)
            return None

    @classmethod
    def load(
        cls,
        repo_root: str,
        settings: ContextOptimizerSettings,
    ) -> LoadedIndex | None:
        """Load an existing index for the current HEAD SHA from disk. Does NOT build.

        Returns None if no index exists for the current commit (normal on first run).
        """
        sha = get_head_sha(repo_root)
        if sha is None:
            return None
        context_dir = _resolve_context_dir(repo_root, settings)
        result = embedder.load_index(context_dir, sha)
        if result is None:
            return None
        prefix_text, chunks, vectors, manifest = result
        logger.info("REPO_INDEX: loaded from disk hash={} chunks={}", sha[:7], len(chunks))
        return LoadedIndex(commit_hash=sha, prefix_text=prefix_text, chunks=chunks, vectors=vectors, manifest=manifest)

    @classmethod
    def build(
        cls,
        repo_root: str,
        settings: ContextOptimizerSettings,
        *,
        force: bool = False,
    ) -> LoadedIndex:
        """Run the full build pipeline for the current HEAD SHA.

        Skips if .context/repo-<sha>.txt already exists and force=False.
        Pipeline: git ls-files → tagger → ranker → repomix → embedder → save.

        Raises RuntimeError if the repo has no tracked files or the embed step fails.
        Never returns None — raises on unrecoverable failure.
        # @stable
        """
        sha = get_head_sha(repo_root)
        if sha is None:
            raise RuntimeError(f"Not a git repo or no commits: {repo_root}")

        context_dir = _resolve_context_dir(repo_root, settings)

        if not force:
            existing = embedder.load_index(context_dir, sha)
            if existing is not None:
                prefix_text, chunks, vectors, manifest = existing
                logger.info("REPO_INDEX: build cache_hit hash={}", sha[:7])
                return LoadedIndex(commit_hash=sha, prefix_text=prefix_text, chunks=chunks, vectors=vectors, manifest=manifest)

        file_paths = _list_tracked_files(repo_root)
        if not file_paths:
            raise RuntimeError(f"No tracked files found in {repo_root}")

        n_tracked = len(file_paths)
        token_ceiling = _compute_token_ceiling(settings, n_tracked)
        effective_top_n = _compute_effective_top_n(settings, n_tracked)
        logger.info(
            "REPO_INDEX: build start hash={} tracked={} mass_target={} top_n={} token_ceiling={}",
            sha[:7], n_tracked, settings.repo_index_pagerank_mass_target,
            effective_top_n, token_ceiling,
        )

        tags_by_file = tagger.get_tags_for_repo(repo_root, file_paths)
        ranked = ranker.rank_files(tags_by_file)
        top_files = ranker.select_by_mass(
            ranked,
            settings.repo_index_pagerank_mass_target,
            max_files=effective_top_n,
        )

        prefix_text = _render(repo_root, top_files, settings)
        prefix_text, top_files = _enforce_token_cap(
            repo_root, top_files, ranked, prefix_text, settings, token_ceiling
        )

        chunks = embedder.chunk_text(
            prefix_text,
            chunk_size_tokens=settings.repo_index_chunk_size_tokens,
            overlap_tokens=settings.repo_index_chunk_overlap_tokens,
        )
        vectors = embedder.embed_chunks(chunks, model_name=settings.repo_index_embedding_model)

        manifest = IndexManifest(
            commit_hash=sha,
            repo_root=repo_root,
            top_n=len(top_files),
            ranked_files=top_files,
            chunks=[],  # populated in save_index via chunks list
            build_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            embedding_model=settings.repo_index_embedding_model,
        )
        embedder.save_index(prefix_text, chunks, vectors, manifest, context_dir, sha)
        embedder.prune_old_indexes(context_dir, keep=3)

        logger.info("REPO_INDEX: build done hash={} files={} chunks={}", sha[:7], len(top_files), len(chunks))
        return LoadedIndex(commit_hash=sha, prefix_text=prefix_text, chunks=chunks, vectors=vectors, manifest=manifest)


# ── Helpers ────────────────────────────────────────────────────────────────


def _resolve_context_dir(repo_root: str, settings: ContextOptimizerSettings) -> str:
    if settings.repo_index_context_dir:
        return settings.repo_index_context_dir
    return os.path.join(repo_root, ".context")


def _list_tracked_files(repo_root: str) -> list[str]:
    """Return absolute paths of all git-tracked files."""
    result = subprocess.run(
        ["git", "-C", repo_root, "ls-files", "--cached", "--full-name"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {result.stderr[:200]}")
    rel_paths = [p for p in result.stdout.splitlines() if p]
    return [os.path.join(repo_root, p) for p in rel_paths]


def _render(repo_root: str, top_files: list[str], settings: ContextOptimizerSettings) -> str:
    """Render top_files via Repomix, falling back to pure-Python renderer on failure."""
    try:
        return renderer.render_with_repomix(
            repo_root,
            top_files,
            timeout_seconds=settings.repo_index_repomix_timeout,
            extra_args=settings.repo_index_repomix_extra_args or None,
        )
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as exc:
        logger.warning("REPO_INDEX: repomix unavailable, using fallback reason={}", exc)
        return renderer.render_fallback(repo_root, top_files)


def _compute_effective_top_n(settings: ContextOptimizerSettings, n_tracked_files: int) -> int:
    """Return the file-count ceiling for the mass selector.

    When repo_index_top_n == 0 (auto), scales linearly with repo size:
      effective = clamp(20 + n_tracked_files // 10, max=100)
    Examples: 100 files → 30, 300 files → 50, 800 files → 100.
    Linear scaling (vs sqrt for token ceiling) is intentional — file count is already
    a rough proxy for breadth; diminishing returns on token quality is handled separately.

    When repo_index_top_n > 0, uses that value verbatim as the explicit override.
    Counterpart: settings.repo_index_top_n docstring.
    """
    if settings.repo_index_top_n > 0:
        return settings.repo_index_top_n
    return min(20 + n_tracked_files // 10, 100)


def _compute_token_ceiling(settings: ContextOptimizerSettings, n_tracked_files: int) -> int:
    """Return the maximum prefix token budget.

    When max_prefix_tokens == 0 (auto), scales with sqrt(n_tracked_files):
      ceiling = clamp(8_000 + sqrt(n) * 1_200, max=56_000)
    Sqrt scaling matches the diminishing-returns curve of PageRank: each additional
    file beyond the top-20 adds less architectural signal, so the budget grows
    sub-linearly with repo size.

    When max_prefix_tokens > 0, uses that value verbatim as the explicit override.
    """
    if settings.repo_index_max_prefix_tokens > 0:
        return settings.repo_index_max_prefix_tokens
    import math
    return min(8_000 + int(math.sqrt(n_tracked_files) * 1_200), 56_000)


def _enforce_token_cap(
    repo_root: str,
    top_files: list[str],
    ranked: list,
    prefix_text: str,
    settings: ContextOptimizerSettings,
    token_ceiling: int,
) -> tuple[str, list[str]]:
    """Reduce file count by 5 at a time until the rendered prefix fits within token_ceiling."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def count(s: str) -> int:
            return len(enc.encode(s))
    except Exception:
        def count(s: str) -> int:  # type: ignore[misc]
            return len(s) // 4

    while count(prefix_text) > token_ceiling and len(top_files) > 5:
        top_files = ranker.get_top_n_files(ranked, len(top_files) - 5)
        prefix_text = _render(repo_root, top_files, settings)
        logger.info("REPO_INDEX: prefix_cap_reduce top_n={} tokens~={}", len(top_files), count(prefix_text))

    return prefix_text, top_files
