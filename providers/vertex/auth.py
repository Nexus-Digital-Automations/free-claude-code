"""Vertex AI access-token provider with ADC-preferred / static-token fallback.

Owner: providers/vertex package. State machine:

    [unfetched] --get_token()--> _refresh() ---success---> [fresh]
                                            \\---fail-----> AuthenticationError

    [fresh] --get_token()--> token if not within 60s of expiry
    [fresh] --get_token() within 60s of expiry--> _refresh()

Concurrency: ``_lock`` serialises refreshes so concurrent requests share
one ADC round-trip per cache miss. The token cache is per-process; the
proxy is single-process so this is sufficient.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from providers.exceptions import AuthenticationError

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

_REFRESH_LEEWAY_SECONDS = 60.0
_STATIC_TOKEN_TTL_SECONDS = 50 * 60.0


@dataclass(slots=True)
class _CachedToken:
    value: str
    expires_at_monotonic: float


class VertexAuthProvider:
    """Mints Vertex AI access tokens via ADC, with static-token fallback.

    Public entry point: :meth:`get_token` (async). Failure modes:

    * ``google-auth`` not installed AND ``static_token`` not set →
      :class:`AuthenticationError` with install instructions.
    * ADC chain returns no credentials AND ``static_token`` not set →
      :class:`AuthenticationError`.
    * ADC refresh raises a transport error AND ``static_token`` not set →
      :class:`AuthenticationError` chained from the transport error.

    @internal — instantiated only by :class:`VertexProvider`.
    """

    def __init__(
        self,
        *,
        static_token: str = "",
        credentials_file: str = "",
    ) -> None:
        self._static_token = static_token.strip()
        self._credentials_file = credentials_file.strip()
        self._lock = asyncio.Lock()
        self._cached: _CachedToken | None = None
        self._adc_credentials: Credentials | None = None

    async def get_token(self) -> str:
        """Return a fresh access token. Refreshes if within the leeway window."""
        if self._is_fresh():
            assert self._cached is not None
            return self._cached.value
        async with self._lock:
            if self._is_fresh():
                assert self._cached is not None
                return self._cached.value
            return await self._refresh()

    def _is_fresh(self) -> bool:
        cached = self._cached
        if cached is None:
            return False
        return time.monotonic() < cached.expires_at_monotonic - _REFRESH_LEEWAY_SECONDS

    async def _refresh(self) -> str:
        adc_error: Exception | None = None
        try:
            return await self._refresh_via_adc()
        except _AdcUnavailable as exc:
            adc_error = exc
            logger.info(
                "VERTEX_AUTH: adc_unavailable reason={} static_token_set={}",
                exc.reason,
                bool(self._static_token),
            )

        if self._static_token:
            self._cached = _CachedToken(
                value=self._static_token,
                expires_at_monotonic=time.monotonic() + _STATIC_TOKEN_TTL_SECONDS,
            )
            logger.warning(
                "VERTEX_AUTH: using_static_token ttl_s={:.0f} (refresh manually before expiry)",
                _STATIC_TOKEN_TTL_SECONDS,
            )
            return self._static_token

        message = (
            "Vertex AI auth unavailable: ADC failed and VERTEX_ACCESS_TOKEN is not set. "
            f"Reason: {adc_error}. Install with `pip install google-auth` and configure "
            "GOOGLE_APPLICATION_CREDENTIALS, or set VERTEX_ACCESS_TOKEN."
        )
        logger.error("VERTEX_AUTH: refresh_failed {}", message)
        raise AuthenticationError(message)

    async def _refresh_via_adc(self) -> str:
        if self._credentials_file and not os.path.exists(self._credentials_file):
            raise _AdcUnavailable(
                f"VERTEX_CREDENTIALS_FILE points to a missing path: "
                f"{self._credentials_file}"
            )

        env_overrides: dict[str, str] = {}
        if self._credentials_file:
            env_overrides["GOOGLE_APPLICATION_CREDENTIALS"] = self._credentials_file

        creds, token, expiry_monotonic = await asyncio.to_thread(
            _load_and_refresh_adc, env_overrides, self._adc_credentials
        )
        self._adc_credentials = creds
        self._cached = _CachedToken(value=token, expires_at_monotonic=expiry_monotonic)
        logger.info(
            "VERTEX_AUTH: adc_refreshed expires_in_s={:.0f}",
            expiry_monotonic - time.monotonic(),
        )
        return token


class _AdcUnavailable(Exception):
    """Sentinel for ADC-not-usable; caller decides whether to fall back."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _load_and_refresh_adc(
    env_overrides: dict[str, str],
    cached_credentials: Credentials | None,
) -> tuple[Credentials, str, float]:
    """Synchronous ADC load + refresh; runs on a worker thread.

    Returns (credentials, expires_at_monotonic). Raises _AdcUnavailable
    on any failure path so the caller can route to the static fallback.
    """
    try:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError
        from google.auth.transport.requests import Request
    except ImportError as exc:
        raise _AdcUnavailable(f"google-auth not installed ({exc})") from exc

    saved_env = {k: os.environ.get(k) for k in env_overrides}
    for k, v in env_overrides.items():
        os.environ[k] = v
    try:
        if cached_credentials is None:
            try:
                credentials, _project = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            except DefaultCredentialsError as exc:
                raise _AdcUnavailable(f"ADC default() failed: {exc}") from exc
        else:
            credentials = cached_credentials

        try:
            credentials.refresh(Request())
        except Exception as exc:
            raise _AdcUnavailable(f"ADC refresh() failed: {exc}") from exc

        token = credentials.token
        if not token:
            raise _AdcUnavailable("ADC refresh returned no token")

        expiry = credentials.expiry
        if expiry is None:
            expires_at_monotonic = time.monotonic() + _STATIC_TOKEN_TTL_SECONDS
        else:
            from datetime import datetime, timezone

            now_utc = datetime.now(timezone.utc)
            expiry_utc = (
                expiry if expiry.tzinfo else expiry.replace(tzinfo=timezone.utc)
            )
            seconds_to_expiry = max(0.0, (expiry_utc - now_utc).total_seconds())
            expires_at_monotonic = time.monotonic() + seconds_to_expiry

        return credentials, token, expires_at_monotonic
    finally:
        for k, prior in saved_env.items():
            if prior is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prior
