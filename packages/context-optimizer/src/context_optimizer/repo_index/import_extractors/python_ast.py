"""Owns: Python import extraction via stdlib `ast`.

Does NOT own: resolution to in-repo paths (dep_graph.py owns that — this
file only emits RawImport DTOs).
Called by: import_extractors.__init__.EXTRACTORS dispatch.
Calls: stdlib ast, loguru.

WHY stdlib ast over tree-sitter for Python:
  - Python's relative-import semantics (`from . import x`, `from ..pkg import y`)
    need explicit dot-counting; tree-sitter captures don't distinguish dots.
  - Stdlib ast is in every Python install — no version drift across the
    tree-sitter-language-pack release cadence.
  - Faster: skips parser load, parses ~200% as fast for small files.
The cost is one extra parse for Python files (tagger has already parsed them
with tree-sitter). For typical repos that's ~50ms total — well below the
budget for the full indexing pass.

WHY one RawImport per imported NAME (not per statement):
  `from . import sibling` is a file-level edge to pkg/sibling.py — losing the
  imported name would make it look like an edge to the parent package. Same
  for `from pkg.sub import helper` → must be able to resolve to pkg/sub/helper.py.
  The resolver in dep_graph.py truncates segments back-to-front when a
  more-specific candidate doesn't exist, so this also yields correct
  resolution when the imported name is a symbol in __init__.py.
"""

from __future__ import annotations

import ast

from loguru import logger

from .._types import ParsedFile
from .raw_import import RawImport


def extract_python_imports(parsed: ParsedFile) -> list[RawImport]:
    """Walk the AST of a Python file, yielding one RawImport per imported NAME.

    Returns [] on syntax error rather than raising — the file is still useful
    to ranker as an untagged-but-tracked node, and the agent gets a partial
    graph rather than a hard failure on one syntactically-broken file.
    """
    try:
        tree = ast.parse(parsed.source, filename=parsed.rel_path)
    except SyntaxError as exc:
        logger.debug("REPO_INDEX: dep_graph python_parse_error file={} reason={}", parsed.rel_path, exc)
        return []
    imports: list[RawImport] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(_imports_from_import(node))
        elif isinstance(node, ast.ImportFrom):
            imports.extend(_imports_from_import_from(node))
    return imports


def _imports_from_import(node: ast.Import) -> list[RawImport]:
    """`import a, b.c as d` → one RawImport per dotted name (a, b.c)."""
    return [
        RawImport(raw=alias.name, line=node.lineno, is_relative=False)
        for alias in node.names
    ]


def _imports_from_import_from(node: ast.ImportFrom) -> list[RawImport]:
    """`from .pkg.sub import a, b as c` → one RawImport per name.

    Each emitted raw is `<dots><module>.<name>` (or `<dots><name>` when
    module is None, i.e. `from . import x`). The resolver in dep_graph
    walks the segments and falls back to truncated forms if the most-
    specific candidate doesn't match a real file.
    """
    dots = "." * node.level
    module = node.module or ""
    prefix = dots + module
    is_relative = node.level > 0
    out: list[RawImport] = []
    for alias in node.names:
        raw = f"{prefix}.{alias.name}" if module else f"{prefix}{alias.name}"
        out.append(RawImport(raw=raw, line=node.lineno, is_relative=is_relative))
    return out
