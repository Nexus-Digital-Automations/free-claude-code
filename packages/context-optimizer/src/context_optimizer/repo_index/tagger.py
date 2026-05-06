"""Owns: tree-sitter parsing and tag (def/ref) extraction.

Does NOT own: file ranking, rendering, embedding, caching, or import-edge
extraction (dep_graph.py owns that).
Called by: repo_index/index.py (build pipeline) and repo_index/dep_graph.py
(reuses the ParsedFile output to avoid a second parse for non-Python files).
Calls: tree_sitter, tree_sitter_language_pack, concurrent.futures, stdlib.

Public surface (in dependency order — top builds the input the rest consume):
    parse_repo(repo_root, file_paths) -> dict[rel_path, ParsedFile]
    extract_tags_from_parsed(parsed: dict[rel_path, ParsedFile]) -> dict[rel_path, list[Tag]]
    get_tags_for_repo(repo_root, file_paths) -> dict[rel_path, list[Tag]]
        # thin compose of parse_repo + extract_tags_from_parsed; kept for
        # back-compat with index.py callers that don't need the trees.

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

from ._types import ParsedFile, Tag

_QUERIES_DIR = Path(__file__).parent / "queries"

# Maps file extension → tree_sitter_language_pack language name.
# Keys are lowercase; caller must normalise before lookup.
# EXTENSION POINT: add languages here when corresponding <lang>-tags.scm exists in
# queries/ AND tree_sitter_language_pack ships a parser for it. Files in unmapped
# languages silently get zero tags and lose architectural signal in PageRank, so
# every language used by the indexed repo should be covered.
# Counterpart: dep_graph.import_extractors uses the same mapping to decide which
# extractor to dispatch to per file.
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


# ── parse_repo ─────────────────────────────────────────────────────────────


# # @stable — dep_graph.py and tests depend on this signature
def parse_repo(
    repo_root: str,
    file_paths: list[str],
    *,
    max_workers: int = 4,
) -> dict[str, ParsedFile]:
    """Parse every file in `file_paths` in parallel; return successful parses.

    Files in unsupported languages, unreadable files, or parse-time errors
    are silently omitted — the dict only contains rel_paths that produced a
    valid Tree. This is intentional: callers (tagger / dep_graph) treat the
    absence of a key as "skip this file" rather than failing the whole pipeline.

    tree-sitter releases the GIL during parse, so ThreadPoolExecutor is effective.
    Memory: roughly len(file_paths) * (avg_source_size + tree_overhead). For a
    500-file Python repo this is ~50MB peak — well within budget for an
    indexing process. If that becomes a constraint, this becomes a streaming
    iterator and downstream consumers fold-as-they-go.
    """
    parsed: dict[str, ParsedFile] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_parse_one, repo_root, p): p for p in file_paths}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                logger.warning("REPO_INDEX: tagger parse_error path={} reason={}", futures[future], exc)
                continue
            if result is not None:
                parsed[result.rel_path] = result
    return parsed


def _parse_one(repo_root: str, abs_path: str) -> ParsedFile | None:
    try:
        rel = os.path.relpath(abs_path, repo_root)
    except ValueError:
        rel = abs_path
    lang = _filename_to_lang(abs_path)
    if lang is None or _scm_path(lang) is None:
        return None
    try:
        from tree_sitter_language_pack import get_parser
        parser = get_parser(lang)
    except Exception as exc:
        logger.debug("REPO_INDEX: tagger lang_load_error file={} lang={} reason={}", rel, lang, exc)
        return None
    try:
        source = Path(abs_path).read_bytes()
    except OSError as exc:
        logger.debug("REPO_INDEX: tagger read_error file={} reason={}", rel, exc)
        return None
    try:
        tree = parser.parse(source)
    except Exception as exc:
        logger.debug("REPO_INDEX: tagger parse_error file={} reason={}", rel, exc)
        return None
    return ParsedFile(rel_path=rel, abs_path=abs_path, language=lang, source=source, tree=tree)


# ── tag extraction ─────────────────────────────────────────────────────────


# # @stable — index.py / dep_graph.py both call this
def extract_tags_from_parsed(parsed: dict[str, ParsedFile]) -> dict[str, list[Tag]]:
    """Run the per-language tags.scm query against each ParsedFile.

    Returns one entry per input rel_path; on query failure the value is an
    empty list (NOT a missing key) so ranker.py still includes the file as
    an untagged node in the graph.
    """
    from tree_sitter import Query
    from tree_sitter_language_pack import get_language

    tags_by_file: dict[str, list[Tag]] = {}
    for rel_path, parsed_file in parsed.items():
        scm = _scm_path(parsed_file.language)
        if scm is None:
            tags_by_file[rel_path] = []
            continue
        try:
            language = get_language(parsed_file.language)
            query = Query(language, scm.read_text(encoding="utf-8"))
            captures = _run_captures(query, parsed_file.tree.root_node)
        except Exception as exc:
            logger.debug("REPO_INDEX: tagger query_error file={} reason={}", rel_path, exc)
            tags_by_file[rel_path] = []
            continue
        tags_by_file[rel_path] = list(_extract_tags(captures, rel_path))
    return tags_by_file


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


# ── back-compat: get_tags_for_repo ─────────────────────────────────────────


# # @stable — index.py and external callers depend on this signature
def get_tags_for_repo(
    repo_root: str,
    file_paths: list[str],
    *,
    max_workers: int = 4,
) -> dict[str, list[Tag]]:
    """Parse and extract tags for every file. Returns one entry per input path.

    Files that fail to parse are included as empty lists rather than omitted —
    this lets ranker.py still include them as untagged nodes.
    Internally a thin compose: parse_repo → extract_tags_from_parsed, with
    parse_repo's omitted files reinstated as empty-list entries here so the
    output schema matches the legacy contract.
    """
    parsed = parse_repo(repo_root, file_paths, max_workers=max_workers)
    tags_by_file = extract_tags_from_parsed(parsed)
    # Legacy contract: every input path appears in the output, even if parse
    # failed. Fill the gaps with empty lists.
    for abs_path in file_paths:
        try:
            rel = os.path.relpath(abs_path, repo_root)
        except ValueError:
            rel = abs_path
        tags_by_file.setdefault(rel, [])
    return tags_by_file


# ── back-compat: get_tags_for_file ─────────────────────────────────────────


# # @stable — kept for any external callers; reimplemented in terms of parse_repo
def get_tags_for_file(abs_path: str, rel_path: str) -> list[Tag]:
    """Parse one file and return its def/ref Tags.

    Returns [] for unsupported languages, unreadable files, or parse errors.
    Never raises — callers treat [] as "no tags available".
    """
    parsed_file = _parse_one(os.path.dirname(abs_path), abs_path)
    if parsed_file is None:
        return []
    # _parse_one derives rel_path from `repo_root`; here the caller asks for
    # a specific rel_path, so we substitute it before extraction.
    parsed_file_at_rel = ParsedFile(
        rel_path=rel_path,
        abs_path=parsed_file.abs_path,
        language=parsed_file.language,
        source=parsed_file.source,
        tree=parsed_file.tree,
    )
    tags = extract_tags_from_parsed({rel_path: parsed_file_at_rel})
    return tags.get(rel_path, [])
