"""Owns: FastMCP stdio server exposing the `run` tool.

Detects framework from cwd markers (tsconfig.json, Cargo.toml,
pyproject.toml/mypy.ini), runs the corresponding command, parses errors,
digests bodies above a byte threshold via digest_core, and returns
errors-only by default.

Does NOT own: framework detection / parsing (runner.py), digest prompting
(digest.py), Ollama lifecycle (context_optimizer.ollama_supervisor).
Called by: MCP client (Claude Code harness) over stdio.
Calls: runner.detect_framework, runner.run_build, digest.digest_error_body.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.utilities.logging import get_logger
from pydantic import Field

from .digest import BuildDigestConfig, digest_error_body
from .runner import (
    SUPPORTED_FRAMEWORKS,
    ErrorRecord,
    RunResult,
    detect_framework,
    run_build,
)

_DEFAULT_TIMEOUT_S = 600.0
_BODY_DIGEST_THRESHOLD_BYTES = 600
_VERSION = "0.1.0"

logger = get_logger(__name__)

mcp = FastMCP(
    name="build-digester",
    instructions=(
        "Run a build / typecheck (tsc, mypy, cargo check) and return "
        "errors-only with each error body digested via local Ollama. "
        "Auto-detects framework from cwd; pass `framework=` to override. "
        "Use `verbose=true` to include the raw tail of tool output."
    ),
    version=_VERSION,
)


@mcp.tool
async def run(
    cwd: Annotated[
        str | None,
        Field(description="Project root to run the build in. Defaults to cwd."),
    ] = None,
    framework: Annotated[
        str | None,
        Field(
            description=f"Override framework auto-detection. One of: {SUPPORTED_FRAMEWORKS}."
        ),
    ] = None,
    verbose: Annotated[
        bool,
        Field(description="Include the raw tail of tool output."),
    ] = False,
    timeout_s: Annotated[
        float,
        Field(description="Hard timeout for the whole build run.", gt=0),
    ] = _DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run the build/typecheck, digest errors, return a structured response.

    Logged: method=run, framework, status, duration_ms, error count.
    Never raises — surfaces tool errors via the response payload's
    `status` and `error` fields.
    """
    started = asyncio.get_event_loop().time()
    target = Path(cwd or os.getcwd())
    chosen = framework or detect_framework(target)

    if chosen is None:
        duration_ms = int((asyncio.get_event_loop().time() - started) * 1000)
        logger.info(
            "build_digester method=run framework=none status=undetected duration_ms=%d",
            duration_ms,
        )
        return {
            "status": "framework_undetected",
            "supported": list(SUPPORTED_FRAMEWORKS),
            "hint": "no tsconfig.json, Cargo.toml, mypy.ini, or pyproject.toml in cwd",
        }

    result = await run_build(target, chosen, timeout_s)
    response = await _shape_response(result, verbose=verbose)

    duration_ms = int((asyncio.get_event_loop().time() - started) * 1000)
    logger.info(
        "build_digester method=run framework=%s status=%s duration_ms=%d errors=%d",
        chosen,
        response["status"],
        duration_ms,
        len(response.get("errors", [])),
    )
    return response


async def _shape_response(
    result: RunResult,
    verbose: bool,
) -> dict[str, Any]:
    """Map RunResult -> response, digesting error bodies in parallel."""
    if result.exit_code is None:
        return {
            "status": "tool_error",
            "framework": result.framework,
            "error": result.raw_tail,
            "errors": [],
            "duration_s": result.duration_s,
        }
    digested = await _digest_each_error(result.errors)
    response: dict[str, Any] = {
        "status": "ok" if not result.errors else "errors",
        "framework": result.framework,
        "exit_code": result.exit_code,
        "duration_s": result.duration_s,
        "errors": digested,
    }
    if verbose:
        response["raw_tail"] = result.raw_tail
    return response


async def _digest_each_error(
    errors: list[ErrorRecord],
) -> list[dict[str, Any]]:
    """Digest each error body concurrently; bodies under threshold pass through."""
    config = BuildDigestConfig()
    coros = [_digest_one_error(error, config) for error in errors]
    return list(await asyncio.gather(*coros))


async def _digest_one_error(
    error: ErrorRecord,
    config: BuildDigestConfig,
) -> dict[str, Any]:
    """Digest body when above threshold; otherwise keep verbatim."""
    if len(error.body.encode("utf-8", errors="ignore")) < _BODY_DIGEST_THRESHOLD_BYTES:
        return {
            "framework": error.framework,
            "file": error.file,
            "line": error.line,
            "code": error.code,
            "summary": error.summary,
            "error_digest": error.body,
        }
    digest_text = await digest_error_body(error.body, config)
    return {
        "framework": error.framework,
        "file": error.file,
        "line": error.line,
        "code": error.code,
        "summary": error.summary,
        "error_digest": digest_text or error.body,
        "digest_status": "ok" if digest_text else "passthrough",
    }


def cli() -> None:
    """Console entry point — `build-digester` script and `python -m build_digester`."""
    mcp.run()
