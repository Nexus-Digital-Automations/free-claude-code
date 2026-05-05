"""Owns: text chunking, embedding, cosine search, and atomic index persistence.

Does NOT own: file selection, rendering, or git operations.
Called by: repo_index/index.py.
Calls: sentence-transformers, numpy, tiktoken (already a package dependency), stdlib.

Index files per commit (all written atomically via tmp+rename):
  .context/repo-<hash>.txt   — frozen Repomix render (the stable prefix)
  .context/repo-<hash>.npy   — float32 embedding matrix, unit-normalised, shape (N, dim)
  .context/repo-<hash>.json  — IndexManifest including full chunk text for reload

WHY flat numpy search: 5-10K chunks × 384-dim × float32 = ~20MB, cosine search < 10ms on CPU.
FAISS adds a C++ build step and approximate-NN tradeoffs we don't need at this scale.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

from ._types import Chunk, IndexManifest

_FILE_HEADER_RE = re.compile(r"^##\s+File:\s+(.+)$", re.MULTILINE)

# Module-level singleton to avoid reloading the model for every build() call.
_model = None
_model_name: str | None = None


def chunk_text(
    text: str,
    *,
    chunk_size_tokens: int = 200,
    overlap_tokens: int = 20,
    tokenizer_name: str = "cl100k_base",
) -> list[Chunk]:
    """Split Repomix-rendered text into overlapping chunks with source file attribution.

    Parses '## File: <path>' headers (produced by both Repomix and render_fallback)
    to assign source_file. Never splits mid-line. Uses tiktoken for token counts so
    chunk sizing is consistent with the rest of the package.

    Returns [] for empty text.
    """
    if not text:
        return []

    counter = _make_token_counter(tokenizer_name)
    sections = _split_sections(text)

    chunks: list[Chunk] = []
    for source_file, content in sections:
        chunks.extend(_chunk_section(source_file, content, chunk_size_tokens, overlap_tokens, counter))
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split on '## File: <path>' headers, return [(source_file, content)] pairs.

    Uses m.start() for section ends so content between headers is sliced correctly.
    """
    matches = list(_FILE_HEADER_RE.finditer(text))
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        source_file = m.group(1).strip()
        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((source_file, text[content_start:content_end]))
    return sections


def _make_token_counter(tokenizer_name: str) -> Callable[[str], int]:
    try:
        import tiktoken
        enc = tiktoken.get_encoding(tokenizer_name)
        return lambda s: len(enc.encode(s))
    except Exception:
        return lambda s: max(1, len(s) // 4)


def _chunk_section(
    source_file: str,
    content: str,
    chunk_size: int,
    overlap: int,
    counter: Callable[[str], int],
) -> list[Chunk]:
    lines = content.splitlines(keepends=True)
    if not lines:
        return []

    chunks: list[Chunk] = []
    chunk_lines: list[str] = []
    chunk_tokens = 0
    chunk_index = 0
    overlap_lines: list[str] = []
    overlap_tokens = 0

    def flush() -> None:
        nonlocal chunk_lines, chunk_tokens, chunk_index, overlap_lines, overlap_tokens
        if not chunk_lines:
            return
        text = "".join(chunk_lines).strip()
        if text:
            chunks.append(Chunk(source_file=source_file, chunk_index=chunk_index, text=text, token_count=chunk_tokens))
        # Build overlap buffer from tail of flushed chunk
        overlap_lines = []
        overlap_tokens = 0
        for line in reversed(chunk_lines):
            t = counter(line)
            if overlap_tokens + t > overlap:
                break
            overlap_lines.insert(0, line)
            overlap_tokens += t
        chunk_lines = list(overlap_lines)
        chunk_tokens = overlap_tokens
        chunk_index += 1

    for line in lines:
        t = counter(line)
        if chunk_tokens + t > chunk_size and chunk_lines:
            flush()
        chunk_lines.append(line)
        chunk_tokens += t

    flush()
    return chunks


def embed_chunks(
    chunks: list[Chunk],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> np.ndarray:
    """Encode chunks using sentence-transformers; return unit-normalised float32 matrix.

    Shape: (len(chunks), embedding_dim). Unit-normalised so cosine similarity = dot product.
    Lazy-loads the model on first call (~80MB download for default model).

    Raises ImportError if sentence-transformers is not installed.
    """
    if not chunks:
        return np.zeros((0, 384), dtype=np.float32)

    model = _load_model(model_name)
    logger.info("REPO_INDEX: embedder encoding chunks={} model={}", len(chunks), model_name)
    vectors = model.encode(
        [c.text for c in chunks],
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def _load_model(model_name: str):
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model
    logger.info("REPO_INDEX: embedder loading model={} (first use may download ~80MB)", model_name)
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(model_name)
    _model_name = model_name
    return _model


def cosine_search(
    query_vector: np.ndarray,
    index_vectors: np.ndarray,
    *,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Flat dot-product search over unit-normalised vectors.

    query_vector: shape (dim,), must be unit-normalised.
    index_vectors: shape (n_chunks, dim), must be unit-normalised.
    Returns [(chunk_index, score), ...] sorted descending by score.
    Returns [] if index_vectors is empty.
    """
    if index_vectors.shape[0] == 0:
        return []
    scores: np.ndarray = index_vectors @ query_vector
    k = min(top_k, len(scores))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [(int(i), float(scores[i])) for i in top_idx]


def save_index(
    prefix_text: str,
    chunks: list[Chunk],
    vectors: np.ndarray,
    manifest: IndexManifest,
    output_dir: str,
    commit_hash: str,
) -> None:
    """Atomically write all three index files for commit_hash.

    Writes to .tmp siblings then renames — POSIX-safe, never corrupts live files.
    Also creates a .gitignore inside output_dir so generated files are excluded even if
    the repo's root .gitignore doesn't cover .context/.
    """
    os.makedirs(output_dir, exist_ok=True)
    _ensure_gitignore(output_dir)

    base = os.path.join(output_dir, f"repo-{commit_hash}")
    _atomic_write_text(f"{base}.txt", prefix_text)
    _atomic_write_npy(f"{base}.npy", vectors)
    _atomic_write_json(f"{base}.json", _manifest_to_dict(manifest, chunks))
    logger.info("REPO_INDEX: embedder saved hash={} chunks={}", commit_hash[:7], len(chunks))


def load_index(output_dir: str, commit_hash: str) -> tuple[str, list[Chunk], np.ndarray, IndexManifest] | None:
    """Load all three index files for commit_hash.

    Returns (prefix_text, chunks, vectors, manifest) or None if any file is missing,
    corrupt, or the manifest's commit_hash doesn't match the argument.
    """
    base = os.path.join(output_dir, f"repo-{commit_hash}")
    if not all(os.path.exists(f"{base}{ext}") for ext in (".txt", ".npy", ".json")):
        return None

    try:
        prefix_text = Path(f"{base}.txt").read_text(encoding="utf-8")
        vectors = np.load(f"{base}.npy").astype(np.float32)
        raw = json.loads(Path(f"{base}.json").read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("REPO_INDEX: embedder load_error hash={} reason={}", commit_hash[:7], exc)
        return None

    if raw.get("commit_hash") != commit_hash:
        logger.warning("REPO_INDEX: embedder manifest_hash_mismatch expected={} got={}",
                       commit_hash[:7], str(raw.get("commit_hash", ""))[:7])
        return None

    chunks = [Chunk(**c) for c in raw["chunks"]]
    manifest = IndexManifest(
        commit_hash=raw["commit_hash"],
        repo_root=raw["repo_root"],
        top_n=raw["top_n"],
        ranked_files=raw["ranked_files"],
        chunks=raw["chunks"],
        build_timestamp_utc=raw["build_timestamp_utc"],
        embedding_model=raw["embedding_model"],
    )
    return prefix_text, chunks, vectors, manifest


def prune_old_indexes(output_dir: str, keep: int = 3) -> None:
    """Delete all but the `keep` most-recent index triplets (by .json mtime)."""
    if not os.path.isdir(output_dir):
        return
    json_files = sorted(
        Path(output_dir).glob("repo-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old_json in json_files[keep:]:
        stem = old_json.stem  # e.g. repo-abc123
        for ext in (".txt", ".npy", ".json"):
            target = old_json.parent / f"{stem}{ext}"
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("REPO_INDEX: embedder prune_error file={} reason={}", target, exc)


# ── helpers ────────────────────────────────────────────────────────────────


def _ensure_gitignore(output_dir: str) -> None:
    p = os.path.join(output_dir, ".gitignore")
    if not os.path.exists(p):
        Path(p).write_text("*\n", encoding="utf-8")


def _atomic_write_text(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    Path(tmp).write_text(text, encoding="utf-8")
    os.rename(tmp, path)


def _atomic_write_npy(path: str, array: np.ndarray) -> None:
    tmp = f"{path}.tmp"
    np.save(tmp, array)
    os.rename(tmp, path)


def _atomic_write_json(path: str, obj: dict) -> None:
    tmp = f"{path}.tmp"
    Path(tmp).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.rename(tmp, path)


def _manifest_to_dict(manifest: IndexManifest, chunks: list[Chunk]) -> dict:
    return {
        "commit_hash": manifest.commit_hash,
        "repo_root": manifest.repo_root,
        "top_n": manifest.top_n,
        "ranked_files": manifest.ranked_files,
        "chunks": [{"source_file": c.source_file, "chunk_index": c.chunk_index,
                    "text": c.text, "token_count": c.token_count} for c in chunks],
        "build_timestamp_utc": manifest.build_timestamp_utc,
        "embedding_model": manifest.embedding_model,
    }
