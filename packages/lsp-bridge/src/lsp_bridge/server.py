"""Owns: FastMCP stdio server exposing `hover` and `definition` tools.

Does NOT own: any Jedi-specific logic (jedi_backend.py).
Called by: MCP client (Claude Code harness) over stdio.
Calls: jedi_backend.hover_at, jedi_backend.find_definitions.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.utilities.logging import get_logger
from pydantic import Field

from .jedi_backend import find_definitions, hover_at

_VERSION = "0.1.0"

logger = get_logger(__name__)

mcp = FastMCP(
    name="lsp-bridge",
    instructions=(
        "Python type/hover and project-wide definition lookup via Jedi. "
        "Coordinates are 1-indexed line, 0-indexed column (LSP convention)."
    ),
    version=_VERSION,
)


@mcp.tool
async def hover(
    file: Annotated[str, Field(description="Absolute path to a .py file.")],
    line: Annotated[int, Field(description="1-indexed line number.", ge=1)],
    column: Annotated[int, Field(description="0-indexed column number.", ge=0)],
) -> dict[str, Any]:
    """Resolve type, signature, and docstring at the given location.

    Returns all-null fields when the cursor is on whitespace, an unresolved
    import, or in a file with syntax errors.
    """
    started = asyncio.get_event_loop().time()
    result = await asyncio.to_thread(hover_at, file, line, column)
    duration_ms = int((asyncio.get_event_loop().time() - started) * 1000)
    status = "ok" if result.symbol else "empty"
    logger.info(
        "lsp_bridge method=hover file=%s line=%d status=%s duration_ms=%d",
        os.path.basename(file), line, status, duration_ms,
    )
    return asdict(result)


@mcp.tool
async def definition(
    symbol: Annotated[str, Field(description="Name to search for.")],
    project_root: Annotated[
        str | None,
        Field(description="Absolute path to project root. Defaults to cwd."),
    ] = None,
) -> dict[str, Any]:
    """Project-wide search for Python symbol definitions by name.

    Returns a list of {file, line} for every navigable definition; empty
    list when nothing matches or the project root is unreadable.
    """
    started = asyncio.get_event_loop().time()
    root = project_root or os.getcwd()
    locations = await asyncio.to_thread(find_definitions, symbol, root)
    duration_ms = int((asyncio.get_event_loop().time() - started) * 1000)
    logger.info(
        "lsp_bridge method=definition symbol=%s hits=%d duration_ms=%d",
        symbol, len(locations), duration_ms,
    )
    return {
        "symbol": symbol,
        "locations": [asdict(loc) for loc in locations],
    }


def cli() -> None:
    """Console entry point — `lsp-bridge` script and `python -m lsp_bridge`."""
    mcp.run()
