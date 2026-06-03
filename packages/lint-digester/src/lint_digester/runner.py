"""Owns: linter detection + safe subprocess spawn for ruff/eslint.

Uses asyncio.create_subprocess_exec (argv form, no shell — execFile
equivalent). Argv list comes from the static _LINTERS table; user input
never reaches spawn args.

Each linter has an entry describing detection markers, the argv to
invoke (always with JSON output), and a parser turning that JSON into
Finding rows that server.py groups by rule_id.

Does NOT own: digest call (digest.py) or MCP surface (server.py).

# EXTENSION POINT — add a linter by appending to _LINTERS and writing its
# `parse` function below.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Finding:
    """One lint violation, parsed from JSON output."""

    linter: str
    rule_id: str
    file: str
    line: int | None
    column: int | None
    message: str


@dataclass
class RunResult:
    """Aggregated linter run outcome. None counts mean the linter crashed
    before producing parsable JSON; callers surface raw_tail instead."""

    linter: str | None
    findings: list[Finding]
    duration_s: float | None
    exit_code: int | None
    raw_tail: str


def detect_linter(cwd: Path) -> str | None:
    """Return the first linter whose marker file exists in `cwd`.

    Order: ruff before eslint only because Python is the proxy's primary
    language; for repos with both, the caller can override with `linter=`.
    """
    for name, spec in _LINTERS.items():
        if any((cwd / marker).exists() for marker in spec["markers"]):
            return name
    return None


async def run_linter(
    cwd: Path,
    linter: str,
    paths: list[str] | None,
    timeout_s: float,
) -> RunResult:
    """Spawn the linter in cwd via execFile-equivalent.

    Never raises: timeout, missing binary, JSON-parse failure all map to
    a RunResult with None counts and raw_tail explaining what happened.
    """
    if linter not in _LINTERS:
        return RunResult(linter, [], None, None, f"unsupported_linter: {linter}")
    spec = _LINTERS[linter]
    argv = list(spec["argv"]) + (paths or spec["default_paths"])

    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_s,
        )
        exit_code = proc.returncode
    except asyncio.TimeoutError:
        return RunResult(linter, [], None, None, f"{linter}_timeout after {timeout_s}s")
    except FileNotFoundError:
        return RunResult(
            linter,
            [],
            None,
            None,
            f"{linter} binary not found on PATH (needed: {argv[0]})",
        )

    elapsed = time.monotonic() - started
    text = stdout.decode("utf-8", errors="replace")
    err_tail = stderr.decode("utf-8", errors="replace").splitlines()[-10:]
    findings = spec["parse"](text)
    return RunResult(
        linter=linter,
        findings=findings,
        duration_s=elapsed,
        exit_code=exit_code,
        raw_tail="\n".join(text.splitlines()[-15:] + err_tail),
    )


def _parse_ruff(text: str) -> list[Finding]:
    """ruff --output-format json emits a top-level JSON array."""
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    out: list[Finding] = []
    for item in payload:
        location = item.get("location") or {}
        out.append(
            Finding(
                linter="ruff",
                rule_id=str(item.get("code") or ""),
                file=str(item.get("filename", "")),
                line=int(location.get("row"))
                if location.get("row") is not None
                else None,
                column=int(location.get("column"))
                if location.get("column") is not None
                else None,
                message=str(item.get("message", "")),
            )
        )
    return out


def _parse_eslint(text: str) -> list[Finding]:
    """eslint --format json emits an array of file results, each with messages."""
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    out: list[Finding] = []
    for file_result in payload:
        path = str(file_result.get("filePath", ""))
        for msg in file_result.get("messages", []):
            out.append(
                Finding(
                    linter="eslint",
                    rule_id=str(msg.get("ruleId") or "unknown"),
                    file=path,
                    line=int(msg["line"]) if msg.get("line") is not None else None,
                    column=int(msg["column"])
                    if msg.get("column") is not None
                    else None,
                    message=str(msg.get("message", "")),
                )
            )
    return out


_LINTERS = {
    "ruff": {
        "markers": ["pyproject.toml", "ruff.toml", ".ruff.toml"],
        "argv": ["ruff", "check", "--output-format", "json"],
        "default_paths": ["."],
        "parse": _parse_ruff,
    },
    "eslint": {
        "markers": [
            ".eslintrc.json",
            ".eslintrc.cjs",
            ".eslintrc.js",
            "eslint.config.js",
            "eslint.config.mjs",
        ],
        "argv": ["npx", "--no-install", "eslint", "--format", "json"],
        "default_paths": ["."],
        "parse": _parse_eslint,
    },
}


SUPPORTED_LINTERS: tuple[str, ...] = tuple(_LINTERS.keys())
"""# @stable — server's `linter=` parameter mirrors this list."""
