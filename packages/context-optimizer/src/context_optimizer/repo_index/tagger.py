"""Owns: tree-sitter tag extraction — parse source files into def/ref Tag namedtuples.

Does NOT own: file ranking, rendering, embedding, or caching.
Called by: repo_index/index.py (via get_tags_for_repo).
Calls: tree_sitter, tree_sitter_language_pack, concurrent.futures, stdlib.

WHY vendored .scm queries: aider maintains these per-language patterns and we vendor
them so our package has no runtime dependency on aider. Update them by copying from
aider/aider/queries/tree-sitter-language-pack/ on aider version bumps.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from ._types import Tag

_QUERIES_DIR = Path(__file__).parent / "queries"

# Maps file extension → tree_sitter_language_pack language name.
# Keys are lowercase; caller must normalise before lookup.
# EXTENSION POINT: add languages here when corresponding <lang>-tags.scm exists in
# queries/ AND tree_sitter_language_pack ships a parser for it. Files in unmapped
# languages silently get zero tags and lose architectural signal in PageRank, so
# every language used by the indexed repo should be covered.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
}


def _filename_to_lang(abs_path: str) -> str | None:
    return _EXT_TO_LANG.get(os.path.splitext(abs_path)[1].lower())


def _scm_path(lang: str) -> Path | None:
    p = _QUERIES_DIR / f"{lang}-tags.scm"
    return p if p.exists() else None


def _run_captures(query, node) -> dict:
    """Compatible with tree-sitter < 0.24 (captures on Query) and >= 0.24 (QueryCursor).

    tree-sitter 0.24.0 moved captures from Query to a separate QueryCursor class.
    We probe for the old API first so the same code runs on both versions.
    Counterpart: same shim in aider/aider/repomap.py:_run_captures().
    """
    if hasattr(query, "captures"):
        return query.captures(node)
    from tree_sitter import QueryCursor

    return QueryCursor(query).captures(node)


def get_tags_for_file(abs_path: str, rel_path: str) -> list[Tag]:
    """Parse one file and return its def/ref Tags.

    Returns [] for unsupported languages, unreadable files, or parse errors.
    Never raises — callers treat [] as "no tags available".
    """
    lang = _filename_to_lang(abs_path)
    if not lang:
        return []

    scm = _scm_path(lang)
    if scm is None:
        return []

    try:
        from tree_sitter import Query
        from tree_sitter_language_pack import get_language, get_parser

        language = get_language(lang)
        parser = get_parser(lang)
    except Exception as exc:
        logger.debug("REPO_INDEX: tagger lang_load_error file={} lang={} reason={}", rel_path, lang, exc)
        return []

    try:
        code = Path(abs_path).read_bytes()
    except OSError as exc:
        logger.debug("REPO_INDEX: tagger read_error file={} reason={}", rel_path, exc)
        return []

    try:
        tree = parser.parse(code)
        query = Query(language, scm.read_text(encoding="utf-8"))
        captures = _run_captures(query, tree.root_node)
    except Exception as exc:
        logger.debug("REPO_INDEX: tagger query_error file={} reason={}", rel_path, exc)
        return []

    return list(_extract_tags(captures, rel_path))


def _extract_tags(captures: dict, rel_path: str) -> Iterator[Tag]:
    """Yield Tags from a captures dict, filtering to def/ref patterns."""
    for capture_name, nodes in captures.items():
        if capture_name.startswith("name.definition."):
            kind = "def"
        elif capture_name.startswith("name.reference."):
            kind = "ref"
        else:
            continue
        for node in nodes:
            try:
                name = node.text.decode("utf-8")
            except Exception:
                continue
            yield Tag(rel_path=rel_path, name=name, kind=kind, line=node.start_point[0])


def get_tags_for_repo(
    repo_root: str,
    file_paths: list[str],
    *,
    max_workers: int = 4,
) -> dict[str, list[Tag]]:
    """Parse all files in parallel, returning {rel_path: [Tag, ...]} for every file.

    Files that fail to parse are logged and included as empty lists rather than
    omitted — this lets ranker.py still include them as untagged nodes.
    tree-sitter releases the GIL during parse, so ThreadPoolExecutor is effective.
    """
    tags_by_file: dict[str, list[Tag]] = {}

    def _parse(abs_path: str) -> tuple[str, list[Tag]]:
        try:
            rel = os.path.relpath(abs_path, repo_root)
        except ValueError:
            rel = abs_path
        return rel, get_tags_for_file(abs_path, rel)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_parse, p): p for p in file_paths}
        for future in as_completed(futures):
            try:
                rel_path, tags = future.result()
                tags_by_file[rel_path] = tags
            except Exception as exc:
                logger.warning("REPO_INDEX: tagger file_error path={} reason={}", futures[future], exc)

    return tags_by_file
