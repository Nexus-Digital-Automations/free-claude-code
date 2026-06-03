"""Owns: subprocess spawn of pytest and parsing of its short-traceback
output into structured failure records.

Does NOT own: the digest call (digest.py) or the MCP server surface (server.py).
Called by: server.handle_run.
Calls: asyncio.create_subprocess_exec (argv form, no shell — safe by design).

# @internal — server.py is the only caller; safe to refactor parser internals.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass


@dataclass
class FailureRecord:
    """One failed test as parsed from pytest output.

    `body` is the verbatim traceback block from the FAILURES section. The
    digest layer will collapse it; this layer never throws information away.
    """

    test_id: str
    file: str | None
    line: int | None
    summary: str
    body: str


@dataclass
class RunResult:
    """Aggregated pytest run outcome. None counts mean the line was never
    emitted (e.g. pytest crashed before reporting); callers should treat
    that as test-runner failure rather than zero failures."""

    passed: int | None
    failed: int | None
    skipped: int | None
    duration_s: float | None
    failures: list[FailureRecord]
    raw_tail: str


_FAILED_LINE = re.compile(
    r"^FAILED (?P<test>[^\s]+)(?:\s+-\s+(?P<summary>.*))?$",
    re.MULTILINE,
)
_FAILURE_HEADER = re.compile(r"^_+ (?P<name>[\w\.\-:\[\]]+) _+$")
_LOCATION = re.compile(r"^(?P<file>[^\s:]+):(?P<line>\d+):", re.MULTILINE)
_SUMMARY_LINE = re.compile(
    r"=+\s*(?:(?P<failed>\d+)\s+failed)?,?\s*(?:(?P<passed>\d+)\s+passed)?,?"
    r"\s*(?:(?P<skipped>\d+)\s+skipped)?.*?in\s+(?P<duration>[\d.]+)s",
)


async def run_pytest(
    cwd: str,
    filter_expr: str | None,
    paths: list[str] | None,
    timeout_s: float,
) -> RunResult:
    """Spawn pytest in `cwd`, return a structured result.

    Never raises: a non-zero exit, parser misses, or a hard timeout all return
    a RunResult with None counts and whatever failures could be parsed. The
    caller decides whether to surface the raw_tail as an error to the user.
    """
    args = ["pytest", "--tb=short", "-q", "--no-header", "--color=no"]
    if filter_expr:
        args.extend(["-k", filter_expr])
    if paths:
        args.extend(paths)

    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return RunResult(
            None, None, None, None, [], f"pytest_timeout after {timeout_s}s"
        )
    except FileNotFoundError:
        return RunResult(None, None, None, None, [], "pytest binary not found on PATH")

    elapsed = time.monotonic() - started
    text = stdout.decode("utf-8", errors="replace")
    return _parse_pytest_output(text, fallback_duration=elapsed)


def _parse_pytest_output(text: str, fallback_duration: float) -> RunResult:
    """Extract failure blocks and the summary line from pytest's stdout."""
    failures = _parse_failure_blocks(text)
    summary = _parse_summary(text)
    raw_tail = "\n".join(text.splitlines()[-30:])

    return RunResult(
        passed=summary.get("passed"),
        failed=summary.get("failed"),
        skipped=summary.get("skipped"),
        duration_s=summary.get("duration", fallback_duration),
        failures=failures,
        raw_tail=raw_tail,
    )


def _parse_failure_blocks(text: str) -> list[FailureRecord]:
    """Walk the FAILURES section, collecting one block per failed test.

    Pytest --tb=short emits:
        ============================ FAILURES =============================
        ____________________ test_pkg.test_mod.test_name ____________________
        path/to/test.py:42: in test_name
            assert x == 1
        AssertionError
        ____________________ test_pkg.test_mod.test_other ___________________
        ...
        ============== short test summary info ============================

    We slice on the section markers and chunk on the underline headers.
    """
    fail_start = text.find("= FAILURES =")
    fail_end = text.find("= short test summary info =")
    if fail_start < 0 or fail_end < 0:
        return []

    section = text[fail_start:fail_end]
    summary_lookup = _index_summaries_by_test(text)
    return _chunk_failures(section, summary_lookup)


def _chunk_failures(
    section: str,
    summary_lookup: dict[str, str],
) -> list[FailureRecord]:
    """# @internal — split the FAILURES section into per-test records."""
    lines = section.splitlines()
    blocks: list[tuple[str, list[str]]] = []
    current: list[str] = []
    current_name: str | None = None

    for line in lines:
        header_match = _FAILURE_HEADER.match(line)
        if header_match:
            if current_name is not None:
                blocks.append((current_name, current))
            current_name = header_match.group("name")
            current = []
            continue
        if current_name is not None:
            current.append(line)
    if current_name is not None:
        blocks.append((current_name, current))

    records: list[FailureRecord] = []
    for name, body_lines in blocks:
        body = "\n".join(body_lines).strip()
        loc = _LOCATION.search(body)
        records.append(
            FailureRecord(
                test_id=name,
                file=loc.group("file") if loc else None,
                line=int(loc.group("line")) if loc else None,
                summary=summary_lookup.get(name, body.splitlines()[-1] if body else ""),
                body=body,
            )
        )
    return records


def _index_summaries_by_test(text: str) -> dict[str, str]:
    """# @internal — map test_id -> one-line summary from `FAILED ... - ...` lines."""
    out: dict[str, str] = {}
    for match in _FAILED_LINE.finditer(text):
        out[match.group("test")] = (match.group("summary") or "").strip()
    return out


def _parse_summary(text: str) -> dict[str, float | int]:
    """Pull pass/fail/skip counts and duration from pytest's last line."""
    out: dict[str, float | int] = {}
    for match in _SUMMARY_LINE.finditer(text):
        if match.group("failed"):
            out["failed"] = int(match.group("failed"))
        if match.group("passed"):
            out["passed"] = int(match.group("passed"))
        if match.group("skipped"):
            out["skipped"] = int(match.group("skipped"))
        if match.group("duration"):
            out["duration"] = float(match.group("duration"))
    if "failed" not in out and "passed" in out:
        out["failed"] = 0
    return out
