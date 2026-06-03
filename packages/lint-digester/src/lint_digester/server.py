"""Owns: FastMCP stdio server exposing the `run` tool.

Runs ruff / eslint in JSON mode, groups findings by rule_id, picks one
representative finding per rule, digests its message via digest_core,
and returns `{by_rule: [{rule_id, count, files, example_digest}], total}`.

Does NOT own: linter parsing (runner.py), digest prompting (digest.py),
Ollama lifecycle (context_optimizer.ollama_supervisor).
Called by: MCP client (Claude Code harness) over stdio.
Calls: runner.detect_linter, runner.run_linter, digest.digest_rule.
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.utilities.logging import get_logger
from pydantic import Field

from .digest import LintDigestConfig, digest_rule
from .runner import (
    SUPPORTED_LINTERS,
    Finding,
    RunResult,
    detect_linter,
    run_linter,
)

_DEFAULT_TIMEOUT_S = 120.0
_VERSION = "0.1.0"

logger = get_logger(__name__)

mcp = FastMCP(
    name="lint-digester",
    instructions=(
        "Run ruff or eslint and return findings grouped by rule, with one "
        "Ollama-digested explanation per rule. Avoids the 'one line per "
        "violation × N violations' dump pattern. Auto-detects linter from "
        "cwd; pass `linter=` to override. Pass `paths=` to scope the run."
    ),
    version=_VERSION,
)


@mcp.tool
async def run(
    cwd: Annotated[
        str | None,
        Field(description="Project root to lint. Defaults to cwd."),
    ] = None,
    linter: Annotated[
        str | None,
        Field(description=f"Override auto-detection. One of: {SUPPORTED_LINTERS}."),
    ] = None,
    paths: Annotated[
        list[str] | None,
        Field(description="Specific paths to lint (relative to cwd). Defaults per-linter."),
    ] = None,
    timeout_s: Annotated[
        float,
        Field(description="Hard timeout for the linter run.", gt=0),
    ] = _DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run the linter, group + digest findings, return a structured response.

    Logged: method=run, linter, status, duration_ms, total findings, distinct rules.
    Never raises — surfaces tool errors via the response payload.
    """
    started = asyncio.get_event_loop().time()
    target = Path(cwd or os.getcwd())
    chosen = linter or detect_linter(target)

    if chosen is None:
        duration_ms = int((asyncio.get_event_loop().time() - started) * 1000)
        logger.info(
            "lint_digester method=run linter=none status=undetected duration_ms=%d",
            duration_ms,
        )
        return {
            "status": "linter_undetected",
            "supported": list(SUPPORTED_LINTERS),
            "hint": "no pyproject.toml, ruff.toml, or .eslintrc* in cwd",
        }

    result = await run_linter(target, chosen, paths, timeout_s)
    response = await _shape_response(result)

    duration_ms = int((asyncio.get_event_loop().time() - started) * 1000)
    logger.info(
        "lint_digester method=run linter=%s status=%s duration_ms=%d "
        "total=%d rules=%d",
        chosen, response["status"], duration_ms,
        response.get("total", 0), len(response.get("by_rule", [])),
    )
    return response


async def _shape_response(result: RunResult) -> dict[str, Any]:
    """Group findings by rule_id, digest one example per rule."""
    if result.exit_code is None and not result.findings:
        return {
            "status": "tool_error",
            "linter": result.linter,
            "error": result.raw_tail,
            "by_rule": [],
            "total": 0,
        }

    grouped = _group_by_rule(result.findings)
    config = LintDigestConfig()
    by_rule = await asyncio.gather(*(
        _digest_rule_entry(rule_id, items, config)
        for rule_id, items in grouped.items()
    ))
    by_rule_sorted = sorted(by_rule, key=lambda entry: -entry["count"])

    return {
        "status": "ok" if not result.findings else "findings",
        "linter": result.linter,
        "exit_code": result.exit_code,
        "duration_s": result.duration_s,
        "total": len(result.findings),
        "by_rule": by_rule_sorted,
    }


def _group_by_rule(findings: list[Finding]) -> dict[str, list[Finding]]:
    """Bucket findings by rule_id. Order within each bucket is input order."""
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        grouped[finding.rule_id].append(finding)
    return grouped


async def _digest_rule_entry(
    rule_id: str, items: list[Finding], config: LintDigestConfig,
) -> dict[str, Any]:
    """Build one rule entry: count, distinct files, an example, its digest."""
    example = items[0]
    distinct_files = sorted({finding.file for finding in items})
    payload = (
        f"rule={rule_id}\n"
        f"message={example.message}\n"
        f"file={example.file}:{example.line}"
    )
    digest_text = await digest_rule(payload, config)
    return {
        "rule_id": rule_id,
        "count": len(items),
        "files": distinct_files,
        "example": {
            "file": example.file,
            "line": example.line,
            "message": example.message,
        },
        "rule_digest": digest_text or example.message,
        "digest_status": "ok" if digest_text else "passthrough",
    }


def cli() -> None:
    """Console entry point — `lint-digester` script and `python -m lint_digester`."""
    mcp.run()
