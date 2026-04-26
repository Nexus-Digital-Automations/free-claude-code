"""Ollama daemon and model-warm supervisor for Tier 2a autocompaction.

Owns: ensuring the local Ollama HTTP daemon is reachable and the configured
model is preloaded, so background compaction never silently degrades to the
provider just because Ollama happened to be down. Also owns the
ready-state cache and failure cooldown that keep the optimizer hot path
cheap.

Does NOT own: the compaction call itself (providers/common/context_optimizer.py
owns _do_ollama_call), pulling missing models (out of scope; the user runs
`ollama pull` manually if a model is missing), shutting the daemon down
(it intentionally outlives the proxy so the model stays warm across
restarts).

Called by:
  - api/app.py:lifespan — fire-and-forget startup warm-up
  - providers/common/context_optimizer.py — gate before each Ollama call

Calls:
  - subprocess.Popen(["ollama", "serve"]) when daemon is unreachable
  - GET  {ollama_root}/api/tags     — health probe
  - POST {ollama_root}/api/generate — model warm-up with keep_alive

State diagram (per-process, single asyncio loop):

    unknown ──ensure_ready─┬──> ready    (cached for _READY_TTL_SECONDS)
                           └──> failed   (cooldown _FAILURE_COOLDOWN_SECONDS)
    ready  ──TTL expires──> unknown
    failed ──cooldown expires──> unknown

# @stable — api/app.py and providers/common/context_optimizer.py call ensure_ready.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from typing import Any, ClassVar

import httpx
from loguru import logger


class OllamaSupervisor:
    """Ensures Ollama is reachable and the configured model is warm.

    Class-level state is intentional: there is one supervisor per proxy
    process, mirroring ContextOptimizer's pattern. Single asyncio loop
    means access to the floats below is race-free without explicit
    locking; _lock only serialises the side-effectful refresh path so
    concurrent callers coalesce on a single check.
    """

    _READY_TTL_SECONDS: ClassVar[float] = 30.0
    _FAILURE_COOLDOWN_SECONDS: ClassVar[float] = 60.0
    _DAEMON_BOOT_TIMEOUT_SECONDS: ClassVar[float] = 30.0
    _DAEMON_POLL_INTERVAL_SECONDS: ClassVar[float] = 0.5
    _HEALTH_CHECK_TIMEOUT_SECONDS: ClassVar[float] = 2.0
    _WARMUP_TIMEOUT_SECONDS: ClassVar[float] = 60.0
    _WARMUP_KEEP_ALIVE: ClassVar[str] = "30m"

    _ready_until: ClassVar[float] = 0.0
    _cooldown_until: ClassVar[float] = 0.0
    _lock: ClassVar[asyncio.Lock | None] = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    async def ensure_ready(cls, settings: Any) -> bool:
        """True iff Ollama is reachable AND the configured model is loaded.

        Never raises. Falls back to False on any failure so the caller can
        route to a different compaction path (provider fallback in
        context_optimizer). Cached for _READY_TTL_SECONDS on success and
        held off via _FAILURE_COOLDOWN_SECONDS on failure.
        """
        now = time.monotonic()
        if now < cls._ready_until:
            return True
        if now < cls._cooldown_until:
            return False

        async with cls._get_lock():
            # Re-check inside the lock — a concurrent caller may have just
            # refreshed the state while we were waiting.
            now = time.monotonic()
            if now < cls._ready_until:
                return True
            if now < cls._cooldown_until:
                return False
            return await cls._refresh(settings)

    @classmethod
    async def _refresh(cls, settings: Any) -> bool:
        api_root = _api_root(settings.ollama_base_url)
        model = settings.ollama_model
        started = time.monotonic()

        if await cls._health_check(api_root):
            logger.info("OLLAMA: health_check ok url={}", api_root)
        else:
            logger.info("OLLAMA: health_check failed url={}", api_root)
            if not cls._spawn_daemon():
                cls._mark_failed()
                return False
            if not await cls._wait_for_health(api_root):
                logger.warning(
                    "OLLAMA: daemon spawned but never became reachable url={}", api_root
                )
                cls._mark_failed()
                return False

        if not await cls._warm_model(api_root, model):
            cls._mark_failed()
            return False

        cls._ready_until = time.monotonic() + cls._READY_TTL_SECONDS
        cls._cooldown_until = 0.0
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info("OLLAMA: ready model={} elapsed_ms={}", model, elapsed_ms)
        return True

    @classmethod
    def _mark_failed(cls) -> None:
        cls._cooldown_until = time.monotonic() + cls._FAILURE_COOLDOWN_SECONDS
        cls._ready_until = 0.0
        logger.warning(
            "OLLAMA: failed cooldown_seconds={}", int(cls._FAILURE_COOLDOWN_SECONDS)
        )

    @classmethod
    async def _health_check(cls, api_root: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=cls._HEALTH_CHECK_TIMEOUT_SECONDS) as client:
                resp = await client.get(f"{api_root}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _spawn_daemon() -> bool:
        """Spawn `ollama serve` detached. Returns False only if the binary is missing.

        WHY detached: the user expects Ollama to outlive the proxy so the
        model stays warm across proxy restarts. start_new_session=True
        puts it in its own process group so SIGTERM to the proxy does
        not propagate.
        """
        logger.info("OLLAMA: spawning daemon (ollama serve)")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except FileNotFoundError:
            logger.error(
                "OLLAMA: 'ollama' binary not found on PATH. "
                "Install from https://ollama.com and ensure `ollama` is callable."
            )
            return False
        except Exception as exc:
            logger.error(
                "OLLAMA: failed to spawn daemon {}: {}", type(exc).__name__, exc
            )
            return False

    @classmethod
    async def _wait_for_health(cls, api_root: str) -> bool:
        started = time.monotonic()
        deadline = started + cls._DAEMON_BOOT_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if await cls._health_check(api_root):
                elapsed_ms = int((time.monotonic() - started) * 1000)
                logger.info("OLLAMA: daemon ready elapsed_ms={}", elapsed_ms)
                return True
            await asyncio.sleep(cls._DAEMON_POLL_INTERVAL_SECONDS)
        return False

    @classmethod
    async def _warm_model(cls, api_root: str, model: str) -> bool:
        """Preload the model into Ollama memory with a long keep_alive.

        WHY POST /api/generate over `ollama run`: `run` is interactive and
        not script-friendly. /api/generate gives a status code we can act
        on, and `keep_alive: 30m` keeps the model resident long enough to
        cover normal compaction cadence.
        """
        logger.info("OLLAMA: warming model={}", model)
        try:
            async with httpx.AsyncClient(timeout=cls._WARMUP_TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    f"{api_root}/api/generate",
                    json={
                        "model": model,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": cls._WARMUP_KEEP_ALIVE,
                    },
                )
        except Exception as exc:
            logger.warning(
                "OLLAMA: warm-up call failed {}: {}", type(exc).__name__, exc
            )
            return False

        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            logger.error(
                "OLLAMA: model {!r} is not pulled. Run: ollama pull {}", model, model
            )
            return False
        logger.warning(
            "OLLAMA: warm-up returned status={} body={!r}",
            resp.status_code, resp.text[:200],
        )
        return False

    @classmethod
    def _reset_for_test(cls) -> None:
        # @internal — only tests/providers/test_ollama_supervisor.py calls this
        # to clear class state between cases.
        cls._ready_until = 0.0
        cls._cooldown_until = 0.0
        cls._lock = None


def _api_root(base_url: str) -> str:
    """Strip an OpenAI-compat suffix to get the Ollama native API root.

    Settings store ollama_base_url as the OpenAI-compatible endpoint
    (e.g. http://localhost:11434/v1). Ollama's native /api/tags and
    /api/generate live one level above.
    """
    return base_url.rstrip("/").removesuffix("/v1")
