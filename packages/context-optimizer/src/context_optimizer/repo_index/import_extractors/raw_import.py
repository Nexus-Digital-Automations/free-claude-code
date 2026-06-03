"""Owns: RawImport — the dataclass an extractor yields per import statement.

Does NOT own: resolution to in-repo paths (dep_graph.py does that with the
file's directory + the project root). RawImport is the unresolved DTO that
crosses the extractor → resolver boundary.
Called by: import_extractors/* (produce); dep_graph.py (consumes).

Three fields by design — `is_relative` tells the resolver how to interpret
`raw`. Without it, the resolver would have to re-parse Python relative-dot
syntax it already discarded.
"""

from __future__ import annotations

from dataclasses import dataclass


# # @stable — dep_graph.Import is built from this; renaming a field breaks the wire shape
@dataclass(frozen=True)
class RawImport:
    """An unresolved import as it appears in source.

    `raw` is the textual module path as written (e.g. "foo.bar", "./utils").
    `line` is 1-based for editor compatibility.
    `is_relative` is True for Python `from .x import y` / JS `./x` style;
    the resolver uses it to decide whether to start from the importing
    file's directory or the repo root.
    """
    raw: str
    line: int
    is_relative: bool
