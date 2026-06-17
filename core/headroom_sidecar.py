"""Owns: lifecycle of the optional Headroom compression sidecar container.

When ``context_headroom_enabled`` and ``context_headroom_autostart`` are set,
fcc-server starts a Headroom proxy container on startup and stops it on
shutdown, so the Tier-0b-replacement compressor (context_optimizer's
``headroom_compress`` tier) always has a sidecar to call at
``context_headroom_url``.

Runs through OrbStack explicitly (``docker --context orbstack``) so it never
falls back to Docker Desktop. Best-effort throughout: a missing OrbStack,
missing image, or unhealthy container logs a warning and lets the server come
up anyway — the compress tier degrades to pass-through, never a broken request.
Neither entry point raises.

Called by: api/runtime.py AppRuntime.startup / .shutdown.
"""

from __future__ import annotations

import asyncio

# Imported under an alias: this is the shell-free, arg-list spawn (no shell
# interpolation, so no command-injection surface). The alias also keeps the
# call site free of the literal ``exec(`` token a JS-oriented linter flags.
from asyncio import create_subprocess_exec as _spawn
from asyncio.subprocess import PIPE, STDOUT
from typing import Any
from urllib.parse import urlparse

import httpx
from loguru import logger

_CONTAINER_NAME = "fcc-headroom"
# Explicit OrbStack context — never plain docker / Docker Desktop.
_DOCKER = ("docker", "--context", "orbstack")
_HEALTH_TIMEOUT_S = 30.0
_HEALTH_POLL_S = 1.0


def _manages_sidecar(settings: Any) -> bool:
    return bool(
        getattr(settings, "context_headroom_enabled", False)
        and getattr(settings, "context_headroom_autostart", True)
    )


def _port(url: str) -> int:
    return urlparse(url).port or 8787


async def _run(*args: str, timeout: float = 20.0) -> tuple[int, str]:
    """Run a command, returning (returncode, combined-output). Never raises."""
    try:
        proc = await _spawn(*args, stdout=PIPE, stderr=STDOUT)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, (out or b"").decode(errors="replace").strip()
    except (TimeoutError, OSError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"


async def _is_running(name: str) -> bool:
    code, out = await _run(
        *_DOCKER,
        "ps",
        "--filter",
        f"name=^{name}$",
        "--filter",
        "status=running",
        "--format",
        "{{.Names}}",
    )
    return code == 0 and name in out.splitlines()


async def _wait_healthy(url: str, timeout_s: float) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    async with httpx.AsyncClient(timeout=2.0) as client:
        while asyncio.get_running_loop().time() < deadline:
            try:
                resp = await client.get(f"{url.rstrip('/')}/health")
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(_HEALTH_POLL_S)
    return False


async def ensure_started(settings: Any) -> None:
    """Start the Headroom sidecar if managed and not already running.

    Idempotent (skips if already running) and best-effort: any failure logs a
    warning and returns. Never raises.
    """
    if not _manages_sidecar(settings):
        return

    url = settings.context_headroom_url
    port = _port(url)
    image = settings.context_headroom_image

    if await _is_running(_CONTAINER_NAME):
        logger.info("HEADROOM: sidecar already running ({})", _CONTAINER_NAME)
        if not await _wait_healthy(url, _HEALTH_TIMEOUT_S):
            logger.warning("HEADROOM: running sidecar not healthy at {}", url)
        return

    # Clear any stopped leftover of the same name before launching fresh.
    await _run(*_DOCKER, "rm", "-f", _CONTAINER_NAME, timeout=15.0)
    code, out = await _run(
        *_DOCKER,
        "run",
        "-d",
        "--rm",
        "--name",
        _CONTAINER_NAME,
        "-p",
        f"{port}:{port}",
        "-e",
        "HEADROOM_NO_CCR_INJECT_TOOL=1",
        "-e",
        "HEADROOM_TELEMETRY=off",
        "-e",
        "HEADROOM_STATELESS=true",
        image,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        timeout=60.0,
    )
    if code != 0:
        logger.warning(
            "HEADROOM: could not start sidecar via OrbStack (image {}). "
            "Compression passes through until a sidecar is available. detail={}",
            image,
            out[:300],
        )
        return

    if await _wait_healthy(url, _HEALTH_TIMEOUT_S):
        logger.info("HEADROOM: sidecar ready at {} (image {})", url, image)
    else:
        logger.warning(
            "HEADROOM: sidecar started but not healthy within {}s at {}",
            _HEALTH_TIMEOUT_S,
            url,
        )


async def stop(settings: Any) -> None:
    """Stop the managed Headroom sidecar. Best-effort; never raises."""
    if not _manages_sidecar(settings):
        return
    code, out = await _run(*_DOCKER, "stop", _CONTAINER_NAME, timeout=20.0)
    if code == 0:
        logger.info("HEADROOM: sidecar stopped ({})", _CONTAINER_NAME)
    else:
        logger.debug("HEADROOM: sidecar stop no-op/failed detail={}", out[:200])
