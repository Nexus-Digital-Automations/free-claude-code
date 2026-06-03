"""Owns: Jedi-backed implementations of hover and definition queries.

Why Jedi (not pyright via LSP): Jedi is pure Python, has no subprocess to
manage, and ships infer/get_signatures/help/search in one library. The
LSP-via-pyright path would buy us slightly better type inference at the
cost of process lifecycle, stdio framing, and rootUri management — not a
trade we want for v1.

Does NOT own: the MCP surface (server.py) or any cross-language dispatch.
Called by: server.handle_hover, server.handle_definition.
Calls: jedi.Script, jedi.Project.

# EXTENSION POINT — to add TypeScript/Go/Rust support, introduce a
# `Backend` protocol with hover/definition methods and dispatch on file
# extension or an explicit `language` argument from the MCP caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jedi
from loguru import logger


@dataclass
class HoverResult:
    """Type/signature/doc snapshot at one source location.

    All fields may be None when Jedi cannot resolve the cursor (bad
    coordinates, unresolved import, syntax error in the file).
    """

    symbol: str | None
    type_name: str | None
    signature: str | None
    docstring: str | None


@dataclass
class DefinitionLocation:
    """One file:line pointing at a symbol definition. Always concrete —
    callers can rely on both fields being non-None."""

    file: str
    line: int


def hover_at(file: str, line: int, column: int) -> HoverResult:
    """Return the type, signature, and docstring at (line, column).

    Coordinates are 1-indexed line, 0-indexed column — matching Jedi's API
    and the LSP convention. Callers passing 0-indexed lines will silently
    point at the wrong row, so this is documented loudly here.

    Never raises — Jedi internally swallows AST/IO errors and returns empty
    lists; we surface that as an all-None HoverResult.
    """
    try:
        script = jedi.Script(path=file)
    except Exception as exc:
        logger.warning(
            "lsp_bridge hover script_init_failed file={} type={} msg={}",
            file,
            type(exc).__name__,
            str(exc)[:120],
        )
        return HoverResult(None, None, None, None)

    inferred = _safe_call(script.infer, line, column, label="infer")
    signatures = _safe_call(script.get_signatures, line, column, label="get_signatures")

    primary = inferred[0] if inferred else None
    return HoverResult(
        symbol=primary.name if primary else None,
        type_name=_describe_type(primary) if primary else None,
        signature=signatures[0].to_string() if signatures else None,
        docstring=primary.docstring(raw=True) if primary else None,
    )


def find_definitions(symbol: str, project_root: str) -> list[DefinitionLocation]:
    """Project-wide search for symbol definitions.

    Returns concrete file:line records. Names without a known module path
    (built-ins, unresolved) are dropped — callers get only navigable hits.

    Never raises — Jedi's search swallows project-walk errors internally;
    a corrupt project returns an empty list.
    """
    try:
        project = jedi.Project(project_root)
    except Exception as exc:
        logger.warning(
            "lsp_bridge definition project_init_failed root={} type={} msg={}",
            project_root,
            type(exc).__name__,
            str(exc)[:120],
        )
        return []

    names = _safe_call(project.search, symbol, label="search")
    out: list[DefinitionLocation] = []
    for name in names:
        path = getattr(name, "module_path", None)
        line_no = getattr(name, "line", None)
        if path is None or line_no is None:
            continue
        out.append(DefinitionLocation(file=str(path), line=int(line_no)))
    return out


def _safe_call(method: Any, *args: Any, label: str) -> list[Any]:
    """Wrap a Jedi call so transient analysis failures degrade to []."""
    try:
        result = method(*args)
    except Exception as exc:
        logger.warning(
            "lsp_bridge jedi_call_failed op={} type={} msg={}",
            label,
            type(exc).__name__,
            str(exc)[:120],
        )
        return []
    return list(result) if result else []


def _describe_type(name: Any) -> str:
    """Best-effort type description from a Jedi Name.

    full_name is qualified ('mymod.MyClass'); description is the inferred
    runtime form ('class MyClass', 'instance of int'). Prefer full_name
    when available; fall back to description; final fallback is the bare
    name so callers always get something printable.
    """
    full = getattr(name, "full_name", None)
    if full:
        return str(full)
    description = getattr(name, "description", None)
    if description:
        return str(description)
    return str(name.name)
