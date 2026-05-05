"""Owns: shared dataclasses for the repo-index subpackage.

Does NOT own: any logic, I/O, or external dependencies.
Called by: all repo_index modules (tagger, ranker, embedder, index, cache_stats).
Calls: stdlib only (dataclasses, collections).
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

# A symbol extracted from a source file by tree-sitter.
# kind: "def" (definition) or "ref" (reference)
Tag = namedtuple("Tag", ["rel_path", "name", "kind", "line"])


@dataclass
class FileRank:
    rel_path: str
    pagerank_score: float
    tags_count: int


@dataclass
class Chunk:
    """One chunk of text from the Repomix render, used for embedding-based retrieval."""

    source_file: str   # repo-relative path of the originating file section
    chunk_index: int   # 0-based index within source_file's section
    text: str          # raw text of this chunk (stripped)
    token_count: int


@dataclass
class IndexManifest:
    """Metadata saved alongside .npy and .txt for a single commit's index.

    chunks stores full Chunk content so load_index can reconstruct without re-parsing.
    """

    commit_hash: str
    repo_root: str
    top_n: int
    ranked_files: list[str]       # rel_paths in alphabetical order (deterministic)
    chunks: list[dict]             # serialisable: {source_file, chunk_index, text, token_count}
    build_timestamp_utc: str       # ISO-8601
    embedding_model: str


@dataclass
class RequestCacheStats:
    request_id: str
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    prefix_bytes: int = 0
