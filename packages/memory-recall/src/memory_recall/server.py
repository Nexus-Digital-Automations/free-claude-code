"""Owns: FastMCP stdio server exposing `find` and `refresh` tools.

`find(problem)` runs an FTS5 MATCH against the index and returns top-k
hits with snippet, status, and source_path. `refresh(roots)` walks the
given directories and re-indexes every plan/spec markdown — typically
triggered by ~/.claude/hooks/session_end.py after a plan transitions to
status: completed.

Does NOT own: SQLite/FTS5 schema or markdown parsing (index.py).
Called by: MCP client (Claude Code harness) over stdio; refresh is also
called from the session-end hook.
Calls: index.open_index, index.refresh_directory, index.search.

Recall result staleness is the caller's problem: the agent prompt should
verify each hit is still applicable, especially for plans dated months
ago. We surface `created_at` and `status` in every hit to make that easy.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.utilities.logging import get_logger
from pydantic import Field

from .index import open_index, refresh_directory, search

_VERSION = "0.1.0"
_DEFAULT_LIMIT = 5
_DEFAULT_DB_PATH = Path.home() / ".claude" / "memory-index" / "recall.db"

logger = get_logger(__name__)

mcp = FastMCP(
    name="memory-recall",
    instructions=(
        "Cross-session search over plan/spec markdown files. Use `find` to "
        "look up prior solutions before re-deriving them. Hits include "
        "status (planning|active|completed|archived) and created date — "
        "verify before applying, especially for older plans."
    ),
    version=_VERSION,
)


@mcp.tool
async def find(
    problem: Annotated[
        str,
        Field(description="Natural-language description of the problem to look up."),
    ],
    limit: Annotated[
        int,
        Field(description="Max hits to return.", ge=1, le=50),
    ] = _DEFAULT_LIMIT,
    db_path: Annotated[
        str | None,
        Field(
            description="Override index DB path. Defaults to ~/.claude/memory-index/recall.db."
        ),
    ] = None,
) -> dict[str, Any]:
    """Return ranked plan/spec hits matching `problem`.

    Empty list when nothing matches or the index is empty. Score is FTS5
    bm25 — lower is better; comparable across calls but not across DBs.
    """
    started = asyncio.get_event_loop().time()
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    hits = await asyncio.to_thread(_run_search, path, problem, limit)
    duration_ms = int((asyncio.get_event_loop().time() - started) * 1000)
    logger.info(
        "memory_recall method=find hits=%d duration_ms=%d",
        len(hits),
        duration_ms,
    )
    return {"problem": problem, "hits": [asdict(hit) for hit in hits]}


@mcp.tool
async def refresh(
    roots: Annotated[
        list[str],
        Field(description="Absolute paths to directories containing .md plans/specs."),
    ],
    db_path: Annotated[
        str | None,
        Field(
            description="Override index DB path. Defaults to ~/.claude/memory-index/recall.db."
        ),
    ] = None,
) -> dict[str, Any]:
    """Walk every root and re-index its .md files.

    Idempotent — re-indexing a file replaces its row keyed on
    source_path. Returns per-root row counts so callers can see which
    directories contributed.
    """
    started = asyncio.get_event_loop().time()
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    counts = await asyncio.to_thread(_run_refresh, path, roots)
    duration_ms = int((asyncio.get_event_loop().time() - started) * 1000)
    logger.info(
        "memory_recall method=refresh roots=%d total_rows=%d duration_ms=%d",
        len(roots),
        sum(counts.values()),
        duration_ms,
    )
    return {"counts": counts, "db_path": str(path)}


def _run_search(db_path: Path, problem: str, limit: int) -> list:
    """Synchronous body of `find` so the async tool can to_thread it."""
    conn = open_index(db_path)
    try:
        return search(conn, problem, limit)
    finally:
        conn.close()


def _run_refresh(db_path: Path, roots: list[str]) -> dict[str, int]:
    """Synchronous body of `refresh` so the async tool can to_thread it."""
    conn = open_index(db_path)
    try:
        return {root: refresh_directory(conn, Path(root)) for root in roots}
    finally:
        conn.close()


def cli() -> None:
    """Console entry point — `memory-recall` script and `python -m memory_recall`."""
    # Ensure parent of default DB exists; openers handle the file itself.
    _DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    mcp.run()
