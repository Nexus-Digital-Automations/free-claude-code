"""Owns: FastMCP stdio server exposing the `run` tool.

`run` spawns pytest in a target directory, parses the failure section,
digests each failure body via context-optimizer's digest_core (through
digest.py), and returns failures-only by default.

Does NOT own: pytest output parsing (runner.py), digest prompting
(digest.py), Ollama lifecycle (context_optimizer.ollama_supervisor).
Called by: MCP client (Claude Code harness) over stdio.
Calls: runner.run_pytest, digest.digest_failure_body.
"""

from __future__ import annotations

import asyncio
import os
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.utilities.logging import get_logger
from pydantic import Field

from .digest import FailureDigestConfig, digest_failure_body
from .runner import FailureRecord, RunResult, run_pytest

_DEFAULT_TIMEOUT_S = 300.0
_BODY_DIGEST_THRESHOLD_BYTES = 800
_VERSION = "0.1.0"

logger = get_logger(__name__)

mcp = FastMCP(
    name="test-digester",
    instructions=(
        "Run pytest and return failures-only with each error body digested to "
        "~500 tokens via local Ollama. Use `verbose=true` to include the raw "
        "tail of pytest output for debugging."
    ),
    version=_VERSION,
)


@mcp.tool
async def run(
    cwd: Annotated[
        str | None,
        Field(description="Project root to run pytest in. Defaults to cwd."),
    ] = None,
    filter: Annotated[
        str | None,
        Field(description="Pytest -k filter expression."),
    ] = None,
    paths: Annotated[
        list[str] | None,
        Field(description="Specific test paths to run (relative to cwd)."),
    ] = None,
    verbose: Annotated[
        bool,
        Field(description="Include the raw tail of pytest output."),
    ] = False,
    timeout_s: Annotated[
        float,
        Field(description="Hard timeout for the whole pytest run.", gt=0),
    ] = _DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run pytest, digest failures, return a structured response.

    Logged: method=run, cwd basename, status, duration_ms, failure count.
    Never raises — surfaces tool errors via the response payload.
    """
    started = asyncio.get_event_loop().time()
    target = cwd or os.getcwd()

    result = await run_pytest(
        cwd=target, filter_expr=filter, paths=paths, timeout_s=timeout_s,
    )
    response = await _shape_response(result, verbose=verbose)

    duration_ms = int((asyncio.get_event_loop().time() - started) * 1000)
    logger.info(
        "test_digester method=run cwd=%s status=%s duration_ms=%d "
        "failures=%d passed=%s",
        os.path.basename(target.rstrip("/")), response["status"], duration_ms,
        len(response.get("failures", [])), response.get("passed"),
    )
    return response


async def _shape_response(
    result: RunResult, verbose: bool,
) -> dict[str, Any]:
    """Map RunResult -> response, digesting each failure body in parallel."""
    if result.failed is None and result.passed is None:
        return {
            "status": "tool_error",
            "error": result.raw_tail,
            "passed": None,
            "failed": None,
            "skipped": None,
            "duration_s": result.duration_s,
            "failures": [],
        }

    digested = await _digest_each_failure(result.failures)
    response: dict[str, Any] = {
        "status": "ok" if not result.failures else "failures",
        "passed": result.passed,
        "failed": result.failed,
        "skipped": result.skipped,
        "duration_s": result.duration_s,
        "failures": digested,
    }
    if verbose:
        response["raw_tail"] = result.raw_tail
    return response


async def _digest_each_failure(
    failures: list[FailureRecord],
) -> list[dict[str, Any]]:
    """Digest every failure body concurrently; bodies under threshold pass through."""
    config = FailureDigestConfig()
    coros = [_digest_one_failure(failure, config) for failure in failures]
    return list(await asyncio.gather(*coros))


async def _digest_one_failure(
    failure: FailureRecord, config: FailureDigestConfig,
) -> dict[str, Any]:
    """Digest body when above threshold; otherwise keep verbatim."""
    if len(failure.body.encode("utf-8", errors="ignore")) < _BODY_DIGEST_THRESHOLD_BYTES:
        return {
            "test_id": failure.test_id,
            "file": failure.file,
            "line": failure.line,
            "summary": failure.summary,
            "error_digest": failure.body,
        }
    digest_text = await digest_failure_body(failure.body, config)
    return {
        "test_id": failure.test_id,
        "file": failure.file,
        "line": failure.line,
        "summary": failure.summary,
        "error_digest": digest_text or failure.body,
        "digest_status": "ok" if digest_text else "passthrough",
    }


def cli() -> None:
    """Console entry point — `test-digester` script and `python -m test_digester`."""
    mcp.run()
