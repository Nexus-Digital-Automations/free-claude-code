"""Owns: Ollama daemon health-check, auto-start, and model warm-up.

This is the same supervisor extracted from the proxy into the package so
any project using context-optimizer gets automatic Ollama management.

Counterpart: proxy's providers/common/ollama_supervisor.py re-exports this.

Does NOT own: the compaction call itself, pulling missing models (user
runs `ollama pull` manually), shutting the daemon down.

Called by: tiers/tier2.py (before each Ollama call), and optionally by
host-app startup code (fire-and-forget warm-up).
Calls: subprocess.Popen(["ollama", "serve"]), httpx GET /api/tags,
       httpx POST /api/generate.

State diagram (per-process, single asyncio loop):

    unknown ──ensure_ready──> ready    (cached _READY_TTL_SECONDS)
                           \\-> failed  (cooldown _FAILURE_COOLDOWN_SECONDS)
    ready  ──TTL expires──> unknown
    failed ──cooldown expires──> unknown

# @stable — tiers/tier2.py and host apps call ensure_ready.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
from loguru import logger

if TYPE_CHECKING:
    pass


class OllamaSupervisor:
    _READY_TTL_SECONDS: ClassVar[float] = 30.0
    _FAILURE_COOLDOWN_SECONDS: ClassVar[float] = 60.0
    _DAEMON_BOOT_TIMEOUT_SECONDS: ClassVar[float] = 30.0
    _DAEMON_POLL_INTERVAL_SECONDS: ClassVar[float] = 0.5
    _HEALTH_CHECK_TIMEOUT_SECONDS: ClassVar[float] = 2.0
    _WARMUP_TIMEOUT_SECONDS: ClassVar[float] = 60.0

    _ready_until: ClassVar[float] = 0.0
    _cooldown_until: ClassVar[float] = 0.0
    # Created at class-definition time so concurrent first callers cannot
    # each create a distinct Lock and bypass coalescing. asyncio.Lock since
    # 3.10 binds to the running loop lazily on first await, so this is safe.
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def ensure_ready(cls, settings: Any) -> bool:
        """True iff Ollama daemon is up and configured model is loaded. Never raises."""
        now = time.monotonic()
        if now < cls._ready_until:
            return True
        if now < cls._cooldown_until:
            return False
        async with cls._lock:
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
                logger.warning("OLLAMA: daemon spawned but never became reachable url={}", api_root)
                cls._mark_failed()
                return False

        if not await cls._warm_model(api_root, model, settings.context_compaction_keep_alive):
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
        logger.warning("OLLAMA: failed cooldown_seconds={}", int(cls._FAILURE_COOLDOWN_SECONDS))

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
            logger.error("OLLAMA: failed to spawn daemon {}: {}", type(exc).__name__, exc)
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
    async def _warm_model(cls, api_root: str, model: str, keep_alive: str) -> bool:
        logger.info("OLLAMA: warming model={}", model)
        try:
            async with httpx.AsyncClient(timeout=cls._WARMUP_TIMEOUT_SECONDS) as client:
                resp = await client.post(
                    f"{api_root}/api/generate",
                    json={"model": model, "prompt": "", "stream": False, "keep_alive": keep_alive},
                )
        except Exception as exc:
            logger.warning("OLLAMA: warm-up call failed {}: {}", type(exc).__name__, exc)
            return False
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            logger.error("OLLAMA: model {!r} not pulled. Run: ollama pull {}", model, model)
            return False
        logger.warning("OLLAMA: warm-up status={} body={!r}", resp.status_code, resp.text[:200])
        return False

    @classmethod
    def _reset_for_test(cls) -> None:
        # @internal — tests only. Replace the lock so a leftover acquired
        # state from a prior test event loop can't deadlock the next one.
        cls._ready_until = 0.0
        cls._cooldown_until = 0.0
        cls._lock = asyncio.Lock()


def _api_root(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")
