"""Owns: SQLite FTS5 index over plan/spec markdown files.

Storage layout: one virtual table `plans` with FTS5 columns (title, status,
content) and unindexed metadata columns (source_path, created_at, indexed_at).
Idempotent — re-indexing the same file replaces its row, so the indexer can
run from `session_end.py` after every plan completion without bloating.

Does NOT own: the MCP surface (server.py) or any frontmatter parsing —
the parser is in this file because nothing else needs it.
Called by: server.find_plans, refresh_one (called from ~/.claude/hooks/session_end.py).
Calls: stdlib sqlite3, pathlib.

# @stable — server.py and session_end.py depend on the schema and the
# `refresh_one` / `search` signatures. Schema changes need a migration step.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

_FRONTMATTER = re.compile(
    r"\A---\s*\n(?P<body>.*?)\n---\s*\n",
    re.DOTALL,
)
_FRONTMATTER_FIELD = re.compile(r"^(?P<key>[\w-]+):\s*(?P<value>.*?)\s*$", re.MULTILINE)


@dataclass
class PlanRecord:
    """One plan/spec file as stored in the index."""

    source_path: str
    title: str
    status: str
    content: str
    created_at: str  # ISO date or empty
    indexed_at: float  # unix seconds


@dataclass
class SearchHit:
    """One FTS match. `score` is FTS5's bm25 (lower = better)."""

    source_path: str
    title: str
    status: str
    snippet: str
    score: float


def open_index(db_path: Path) -> sqlite3.Connection:
    """Open or create the FTS5 index DB. Caller is responsible for closing.

    The schema lives here (not in a migrations dir) because it is one
    table, never evolves at runtime, and reading it inline keeps the
    file self-contained.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS plans USING fts5("
        "  source_path UNINDEXED, "
        "  title, "
        "  status UNINDEXED, "
        "  content, "
        "  created_at UNINDEXED, "
        "  indexed_at UNINDEXED, "
        "  tokenize='porter unicode61'"
        ")",
    )
    conn.commit()
    return conn


def refresh_one(conn: sqlite3.Connection, file_path: Path) -> PlanRecord | None:
    """Index or re-index a single plan/spec markdown file.

    Returns the record on success, None if the file doesn't exist or is
    empty. Idempotent: re-indexing replaces the previous row keyed on
    source_path.
    """
    if not file_path.is_file():
        logger.warning("memory_recall index missing_file path={}", str(file_path))
        return None

    raw = file_path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return None

    record = _build_record(file_path, raw)
    _upsert(conn, record)
    return record


def refresh_directory(conn: sqlite3.Connection, root: Path) -> int:
    """Index every .md file under `root`. Returns the number of rows written.

    Caller chooses the directory: typically `~/.claude/plans/` plus any
    `<project>/specs/` and `<project>/plans/` discovered up the cwd tree.
    """
    if not root.is_dir():
        logger.info("memory_recall index skip_missing_dir path={}", str(root))
        return 0

    written = 0
    for path in root.rglob("*.md"):
        record = refresh_one(conn, path)
        if record is not None:
            written += 1
    logger.info("memory_recall index refresh root={} rows={}", str(root), written)
    return written


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
) -> list[SearchHit]:
    """Run an FTS5 MATCH query, return ranked hits.

    Empty / whitespace queries return []. We tokenize on whitespace and
    quote each token individually — wrapping the whole query in one set
    of quotes would force an exact-phrase match, missing documents where
    the words appear in any order. Punctuation inside tokens is stripped
    by the porter/unicode61 tokenizer, so we only need to neutralise FTS5
    operator characters per-token.
    """
    if not query.strip():
        return []
    safe_query = _build_match_expression(query)
    cur = conn.execute(
        "SELECT source_path, title, status, "
        "       snippet(plans, 3, '<<', '>>', '…', 16) AS snippet, "
        "       bm25(plans) AS score "
        "FROM plans "
        "WHERE plans MATCH ? "
        "ORDER BY score "
        "LIMIT ?",
        (safe_query, limit),
    )
    return [
        SearchHit(
            source_path=row[0],
            title=row[1] or "",
            status=row[2] or "",
            snippet=row[3] or "",
            score=float(row[4]),
        )
        for row in cur.fetchall()
    ]


_FTS_NEUTRALIZE = re.compile(r'["()*+\-:^]')


def _build_match_expression(query: str) -> str:
    """Turn a natural-language query into a tokens-AND FTS5 expression.

    Each whitespace-separated token gets wrapped in double quotes after
    stripping FTS5 operator characters. Empty tokens (created when the
    user types only punctuation) are dropped. Callers that pass a plain
    word like `oauth` get back `"oauth"`; multi-word queries like
    `oauth compliance` become `"oauth" "compliance"` — implicit AND.
    """
    tokens: list[str] = []
    for raw in query.split():
        stripped = _FTS_NEUTRALIZE.sub("", raw).strip()
        if stripped:
            tokens.append(f'"{stripped}"')
    return " ".join(tokens) if tokens else '""'


def _build_record(file_path: Path, raw: str) -> PlanRecord:
    frontmatter = _parse_frontmatter(raw)
    body = _strip_frontmatter(raw)
    return PlanRecord(
        source_path=str(file_path.resolve()),
        title=frontmatter.get("title", file_path.stem),
        status=frontmatter.get("status", ""),
        content=body,
        created_at=frontmatter.get("created", ""),
        indexed_at=time.time(),
    )


def _parse_frontmatter(raw: str) -> dict[str, str]:
    """Extract YAML-ish key/value pairs from a leading `---` block.

    We do not pull a YAML library for this — the plans use a flat
    key:value shape (title, status, created), nothing nested. A regex
    over those is enough; anything fancier should switch to PyYAML.
    """
    match = _FRONTMATTER.match(raw)
    if not match:
        return {}
    out: dict[str, str] = {}
    for field in _FRONTMATTER_FIELD.finditer(match.group("body")):
        out[field.group("key")] = field.group("value").strip("'\"")
    return out


def _strip_frontmatter(raw: str) -> str:
    return _FRONTMATTER.sub("", raw, count=1).lstrip()


def _upsert(conn: sqlite3.Connection, record: PlanRecord) -> None:
    """FTS5 has no UPSERT; emulate via DELETE-then-INSERT in one txn."""
    with conn:
        conn.execute(
            "DELETE FROM plans WHERE source_path = ?",
            (record.source_path,),
        )
        conn.execute(
            "INSERT INTO plans (source_path, title, status, content, "
            "                   created_at, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.source_path,
                record.title,
                record.status,
                record.content,
                record.created_at,
                record.indexed_at,
            ),
        )
