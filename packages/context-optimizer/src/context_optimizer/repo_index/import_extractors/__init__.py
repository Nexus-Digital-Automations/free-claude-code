"""Owns: language → import-extractor dispatch for the dep_graph build.

Does NOT own: extraction logic itself (each language has its own module),
import resolution to in-repo paths (dep_graph.py owns that), or caching
(dep_graph.py).
Called by: repo_index/dep_graph.py during build_for_repo.
Calls: per-language extractor modules in this package.

WHY a registry rather than if/elif chains in dep_graph.py: each language's
extractor has subtly different behaviour (Python uses stdlib ast for
relative-import correctness; tree-sitter languages use captured node types).
Routing through a dict keeps dep_graph.py free of language detail and makes
it obvious where to add a language: a new module + one entry here.
"""

from __future__ import annotations

from collections.abc import Callable

from .._types import ParsedFile
from .python_ast import extract_python_imports
from .raw_import import RawImport
from .treesitter_imports import make_extractor

# # @stable — dep_graph.py iterates this map
# Signature: (parsed: ParsedFile) -> list[RawImport]
# Languages absent from this map produce no imports — neither resolved nor
# raw-only. Add an entry here when you ship a new extractor.
# EXTENSION POINT: add languages here as their extractors land.
EXTRACTORS: dict[str, Callable[[ParsedFile], list[RawImport]]] = {
    "python": extract_python_imports,
    "javascript": make_extractor("javascript"),
    "typescript": make_extractor("typescript"),
    "go": make_extractor("go"),
}


__all__ = ["EXTRACTORS", "RawImport"]
