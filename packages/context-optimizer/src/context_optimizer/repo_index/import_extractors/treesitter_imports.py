"""Owns: generic tree-sitter import extraction — walks the parsed Tree
looking for language-specific import nodes and pulls out the module string.

Does NOT own: per-language resolution to in-repo paths (dep_graph.py owns
that), or detection of which node types matter (node_types.py owns that
mapping).
Called by: import_extractors.__init__.EXTRACTORS dispatch for every
language in node_types.IMPORT_NODE_SHAPES.
Calls: tree_sitter.Tree.walk (cursor traversal), loguru.

WHY a single walker rather than per-language modules:
  - The shape "find these node types, pull out a string" is identical for
    every language's import syntax. Splitting into one module per language
    would be ~20 lines of dispatch each, mostly identical.
  - When an extractor needs language-specific logic (e.g. Python's relative
    imports), it gets its own module (python_ast.py). Until that's needed,
    the table-driven walker covers JS/TS/Go/Rust/Ruby/Java/C#/C/C++ in one place.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from .._types import ParsedFile
from .node_types import IMPORT_NODE_SHAPES, ImportNodeShape
from .raw_import import RawImport


def make_extractor(language: str) -> Callable[[ParsedFile], list[RawImport]]:
    """Return a closure that extracts imports for `language`.

    Caller is responsible for ensuring `language` has an entry in
    IMPORT_NODE_SHAPES — KeyError is raised here intentionally so a missing
    registration fails loudly at module import rather than silently producing
    empty edges at query time.
    """
    shape = IMPORT_NODE_SHAPES[language]
    return partial(_extract_with_shape, shape=shape)


def _extract_with_shape(parsed: ParsedFile, *, shape: ImportNodeShape) -> list[RawImport]:
    """Walk parsed.tree, yielding one RawImport per matched node.

    Uses tree-sitter's cursor traversal (iterative, avoids deep recursion on
    large files) to find every node whose `type` is in shape.statement_types,
    then asks shape.extract_module to pull the import string out of it.
    """
    if parsed.tree is None:
        return []
    out: list[RawImport] = []
    cursor = parsed.tree.walk()
    visited_children = False
    while True:
        node = cursor.node
        if not visited_children:
            if node.type in shape.statement_types:
                edge = shape.extract_module(node, parsed.source)
                if edge is not None:
                    out.append(edge)
            if cursor.goto_first_child():
                visited_children = False
                continue
        if cursor.goto_next_sibling():
            visited_children = False
            continue
        if not cursor.goto_parent():
            return out
        visited_children = True
