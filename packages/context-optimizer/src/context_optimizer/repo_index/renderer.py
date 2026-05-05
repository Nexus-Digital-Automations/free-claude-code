"""Owns: Repomix subprocess invocation and pure-Python fallback rendering.

Does NOT own: file selection (ranking), embedding, or caching.
Called by: repo_index/index.py.
Calls: subprocess (npx repomix), stdlib only for fallback.

WHY Repomix: deterministic full-file content with consistent `## File: <path>` headers that
embedder.py can parse. The fallback produces the same header format so embedder.py works
identically regardless of which renderer was used.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from loguru import logger


def render_with_repomix(
    repo_root: str,
    include_files: list[str],
    *,
    timeout_seconds: float = 120.0,
    extra_args: list[str] | None = None,
) -> str:
    """Invoke repomix via npx to render selected files as a markdown text block.

    include_files must already be sorted (ranker.get_top_n_files guarantees this).
    Alphabetical sort of include_files → deterministic Repomix output for the same file set.

    Raises FileNotFoundError if npx is not on PATH.
    Raises RuntimeError if repomix exits non-zero.
    Raises subprocess.TimeoutExpired if render exceeds timeout_seconds.
    """
    if not include_files:
        return ""

    cmd = [
        "npx", "--yes", "repomix",
        "--include", ",".join(include_files),
        "--output-format", "markdown",
        "--output", "/dev/stdout",
        *(extra_args or []),
    ]

    logger.info("REPO_INDEX: renderer repomix_start files={} root={}", len(include_files), repo_root)
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "npx not found — install Node.js or set repo_index_repomix_timeout to use render_fallback()"
        )

    if result.returncode != 0:
        raise RuntimeError(f"repomix exited {result.returncode}: {result.stderr[:500]}")

    logger.info("REPO_INDEX: renderer repomix_done bytes={}", len(result.stdout))
    return result.stdout


def render_fallback(repo_root: str, include_files: list[str]) -> str:
    """Pure-Python renderer — reads files and concatenates with `## File: <path>` headers.

    Produces deterministic output identical in structure to Repomix markdown format so
    embedder.chunk_text() can parse both renderers without branching.
    Used when Node.js / npx is unavailable.
    """
    if not include_files:
        return ""

    parts: list[str] = ["# Repository Context\n\n"]
    for rel_path in include_files:
        abs_path = os.path.join(repo_root, rel_path)
        try:
            content = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.debug("REPO_INDEX: renderer fallback_read_error file={} reason={}", rel_path, exc)
            content = f"<unreadable: {exc}>\n"
        parts.append(f"## File: {rel_path}\n\n```\n{content}\n```\n\n")

    return "".join(parts)
