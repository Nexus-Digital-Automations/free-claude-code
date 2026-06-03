"""Owns: per-language tree-sitter node-type registry for import extraction.

Does NOT own: the walking algorithm (treesitter_imports.py owns that), or
resolution of the extracted strings (dep_graph.py owns that).
Called by: import_extractors.treesitter_imports.make_extractor — looks up
ImportNodeShape by language id.
Calls: nothing (pure data table + small extractor functions).

WHY a small data table rather than a big if/elif:
  Adding a new language is one entry. The walker stays untouched. The
  extractor functions are small and per-language because each language's
  AST shape for "module string" is slightly different.
# EXTENSION POINT: add languages here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .raw_import import RawImport

# Node and source bytes — typed as object/bytes so this module doesn't pull
# in tree_sitter at import time. The walker passes whatever Node it has.
_ExtractorFn = Callable[[object, bytes], "RawImport | None"]


@dataclass(frozen=True)
class ImportNodeShape:
    """Describes how to find and parse import statements for one language.

    `statement_types` is the set of tree-sitter node `type` strings that
    enclose an import (e.g. `{"import_statement"}` for JS).
    `extract_module` runs on each matched node and returns one RawImport
    or None when the node lacks a string child (parse-recovery cases).
    """
    statement_types: frozenset[str]
    extract_module: _ExtractorFn


# ── JS / TS ────────────────────────────────────────────────────────────────


def _extract_js_import(node: object, source: bytes) -> RawImport | None:
    """ES module imports: `import x from "./y"` and `import "./side-effect"`.

    The module string lives in a `string` child of the `import_statement`
    node. Strip the outer quotes from the captured text.
    """
    string_node = _first_child_of_type(node, {"string"})
    if string_node is None:
        return None
    raw = _string_text(string_node, source)
    if raw is None:
        return None
    return RawImport(raw=raw, line=node.start_point[0] + 1, is_relative=raw.startswith("."))


# ── Go ─────────────────────────────────────────────────────────────────────


def _extract_go_import(node: object, source: bytes) -> RawImport | None:
    """Go imports: `import "fmt"` or `import alias "path"` inside `import_spec`.

    Tree-sitter Go wraps each individual import in `import_spec`; the path
    string is `interpreted_string_literal` (or `raw_string_literal` for
    backtick-quoted, rare for imports).
    """
    string_node = _first_child_of_type(
        node, {"interpreted_string_literal", "raw_string_literal"}
    )
    if string_node is None:
        return None
    raw = _string_text(string_node, source)
    if raw is None:
        return None
    return RawImport(raw=raw, line=node.start_point[0] + 1, is_relative=False)


# ── helpers ────────────────────────────────────────────────────────────────


def _first_child_of_type(node: object, types: frozenset[str] | set[str]) -> object | None:
    """Return the first direct child whose `type` is in `types`, or None."""
    for child in getattr(node, "children", []):
        if child.type in types:
            return child
    return None


def _string_text(string_node: object, source: bytes) -> str | None:
    """Decode a tree-sitter string node's text and strip outer quotes.

    Handles single, double, and backtick quotes — covers JS/TS/Go cases.
    Returns None on decode failure (binary content shouldn't happen in
    a string literal, but we guard rather than raise).
    """
    try:
        text = source[string_node.start_byte:string_node.end_byte].decode("utf-8")
    except UnicodeDecodeError:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
        return text[1:-1]
    return text


# # @stable — treesitter_imports.make_extractor reads this map by language
# Languages NOT listed here go through the raw-only path automatically (they
# get skipped in import_extractors.EXTRACTORS too unless registered there).
IMPORT_NODE_SHAPES: dict[str, ImportNodeShape] = {
    "javascript": ImportNodeShape(
        statement_types=frozenset({"import_statement"}),
        extract_module=_extract_js_import,
    ),
    "typescript": ImportNodeShape(
        statement_types=frozenset({"import_statement"}),
        extract_module=_extract_js_import,
    ),
    "go": ImportNodeShape(
        statement_types=frozenset({"import_spec"}),
        extract_module=_extract_go_import,
    ),
}
