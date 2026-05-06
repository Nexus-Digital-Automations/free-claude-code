"""Owns: framework detection + subprocess spawn for build/typecheck.

Uses `asyncio.create_subprocess_exec` (argv form, no shell) — the safe
equivalent of execFile. The argv list comes from the static _FRAMEWORKS
table; user input never reaches the spawn args.

Each supported framework has a small entry in _FRAMEWORKS describing the
detection marker, the argv to invoke, and the parser that turns its
stdout into ErrorRecord rows.

Does NOT own: the digest call (digest.py) or the MCP surface (server.py).
Called by: server.handle_run.

# EXTENSION POINT — add a framework by appending to _FRAMEWORKS and
# defining its `parse` function below.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ErrorRecord:
    """One compiler/type-check error parsed from tool output.

    `body` is the verbatim block (including any context lines or hint
    arrows the tool emits). The digest layer collapses bodies above a
    threshold; this layer never throws information away.
    """

    framework: str
    file: str | None
    line: int | None
    code: str | None
    summary: str
    body: str


@dataclass
class RunResult:
    """Aggregated run outcome. None counts mean the tool crashed before
    reporting anything; callers surface raw_tail to the user instead."""

    framework: str | None
    errors: list[ErrorRecord]
    warnings: int | None
    duration_s: float | None
    exit_code: int | None
    raw_tail: str


def detect_framework(cwd: Path) -> str | None:
    """Return the first framework whose marker file exists in `cwd`.

    Order matters: tsc beats npm-build because a TS project's package.json
    typically has both, and tsc gives structured per-error output where
    npm-build is whatever the user wrote in their script.
    """
    for name, spec in _FRAMEWORKS.items():
        if any((cwd / marker).exists() for marker in spec["markers"]):
            return name
    return None


async def run_build(
    cwd: Path, framework: str, timeout_s: float,
) -> RunResult:
    """Spawn the framework's command in cwd via execFile-equivalent.

    Never raises: timeout, missing binary, crash before output — all map
    to a RunResult with None counts and raw_tail explaining what happened.
    """
    if framework not in _FRAMEWORKS:
        return RunResult(framework, [], None, None, None,
                         f"unsupported_framework: {framework}")
    spec = _FRAMEWORKS[framework]
    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *spec["argv"],
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        exit_code = proc.returncode
    except asyncio.TimeoutError:
        return RunResult(framework, [], None, None, None,
                         f"{framework}_timeout after {timeout_s}s")
    except FileNotFoundError:
        return RunResult(framework, [], None, None, None,
                         f"{framework} binary not found on PATH "
                         f"(needed: {spec['argv'][0]})")

    elapsed = time.monotonic() - started
    text = stdout.decode("utf-8", errors="replace")
    errors = spec["parse"](text)
    return RunResult(
        framework=framework,
        errors=errors,
        warnings=None,
        duration_s=elapsed,
        exit_code=exit_code,
        raw_tail="\n".join(text.splitlines()[-30:]),
    )


_TSC_LINE = re.compile(
    r"^(?P<file>[^\s(].*?)\((?P<line>\d+),(?P<col>\d+)\):\s+"
    r"error\s+(?P<code>TS\d+):\s+(?P<msg>.*)$",
    re.MULTILINE,
)


def _parse_tsc(text: str) -> list[ErrorRecord]:
    """tsc default text output: `file(line,col): error TS2345: message`.

    We keep one record per matched line and put the matched substring in
    body so the digest layer has context to work with.
    """
    out: list[ErrorRecord] = []
    for match in _TSC_LINE.finditer(text):
        out.append(ErrorRecord(
            framework="tsc",
            file=match.group("file"),
            line=int(match.group("line")),
            code=match.group("code"),
            summary=match.group("msg").strip(),
            body=match.group(0),
        ))
    return out


_MYPY_LINE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):(?:(?P<col>\d+):)?\s+"
    r"(?P<level>error|note):\s+(?P<msg>.*?)\s*(?:\[(?P<code>[\w-]+)\])?$",
    re.MULTILINE,
)


def _parse_mypy(text: str) -> list[ErrorRecord]:
    """mypy text output, optionally with `[error-code]` suffix when
    --show-error-codes is on. Treat each `error:` line as one record;
    `note:` lines are dropped (they're context for the preceding error
    and reading them via raw_tail when needed is cheaper than carrying
    them through the digest pipeline).
    """
    out: list[ErrorRecord] = []
    for match in _MYPY_LINE.finditer(text):
        if match.group("level") != "error":
            continue
        out.append(ErrorRecord(
            framework="mypy",
            file=match.group("file"),
            line=int(match.group("line")),
            code=match.group("code"),
            summary=match.group("msg").strip(),
            body=match.group(0),
        ))
    return out


def _parse_cargo(text: str) -> list[ErrorRecord]:
    """cargo --message-format=json emits one JSON object per line.

    Keep `compiler-message` records whose `level` is `error`. The final
    `build-finished` record carries no error context and is dropped.
    """
    out: list[ErrorRecord] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("reason") != "compiler-message":
            continue
        msg = payload.get("message") or {}
        if msg.get("level") != "error":
            continue
        spans = msg.get("spans") or []
        primary = next((s for s in spans if s.get("is_primary")), None)
        out.append(ErrorRecord(
            framework="cargo",
            file=primary.get("file_name") if primary else None,
            line=int(primary["line_start"]) if primary else None,
            code=(msg.get("code") or {}).get("code"),
            summary=str(msg.get("message", "")).strip(),
            body=msg.get("rendered") or json.dumps(msg, indent=2),
        ))
    return out


_FRAMEWORKS = {
    "tsc": {
        "markers": ["tsconfig.json"],
        "argv": ["tsc", "--noEmit", "--pretty", "false"],
        "parse": _parse_tsc,
    },
    "mypy": {
        "markers": ["mypy.ini", "pyproject.toml"],
        "argv": ["mypy", "--show-error-codes", "--no-color-output", "."],
        "parse": _parse_mypy,
    },
    "cargo": {
        "markers": ["Cargo.toml"],
        "argv": ["cargo", "check", "--message-format=json"],
        "parse": _parse_cargo,
    },
}


SUPPORTED_FRAMEWORKS: tuple[str, ...] = tuple(_FRAMEWORKS.keys())
"""# @stable — server's `framework=` parameter mirrors this list."""
