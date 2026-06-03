"""Vertex AI Model Garden provider for self-deployed Gemma endpoints.

Owner: providers/vertex package. Builds the regional OpenAI-compat URL
from project/region/endpoint settings and delegates streaming to
:class:`OpenAICompatibleProvider`. Auth (ADC-preferred, static-token
fallback) is injected per request via an httpx request event hook so
that ``AsyncOpenAI``'s constructor-time ``api_key`` is never the
authoritative credential.
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI

from config.settings import Settings
from providers.base import ProviderConfig
from providers.exceptions import AuthenticationError
from providers.openai_compat import OpenAIChatTransport

from .auth import VertexAuthProvider
from .request import build_request_body

_PROVIDER_NAME = "VERTEX"


class VertexProvider(OpenAIChatTransport):
    """Vertex AI Gemma provider with per-request token refresh.

    Failure modes:

    * Missing ``VERTEX_PROJECT`` / ``VERTEX_REGION`` / ``VERTEX_ENDPOINT_ID`` →
      :class:`AuthenticationError` at construction time.
    * Auth failure mid-stream surfaces via the OpenAI SDK as a 401/403; the
      shared ``OpenAIChatTransport`` translates it to an Anthropic SSE
      error event.

    @stable — instantiated by :func:`providers.registry._create_vertex`.
    """

    def __init__(self, config: ProviderConfig, *, settings: Settings) -> None:
        base_url = _build_base_url(settings)
        super().__init__(
            config,
            provider_name=_PROVIDER_NAME,
            base_url=base_url,
            api_key="placeholder",
        )
        self._auth = VertexAuthProvider(
            static_token=settings.vertex_access_token,
            credentials_file=settings.vertex_credentials_file,
        )
        self._models = _parse_models(settings.vertex_models)
        self._http_client = _build_http_client(config, self._auth)
        # Replace the SDK-owned client. The api_key here is unused — the
        # event hook on _http_client injects a fresh Authorization header
        # per request, overriding whatever the SDK set.
        self._client = AsyncOpenAI(
            api_key="placeholder",
            base_url=self._base_url,
            max_retries=0,
            http_client=self._http_client,
        )
        logger.info(
            "VERTEX_INIT: base_url={} models={} static_fallback={}",
            self._base_url,
            sorted(self._models),
            bool(settings.vertex_access_token),
        )

    async def cleanup(self) -> None:
        await super().cleanup()
        await self._http_client.aclose()

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    async def list_model_ids(self) -> frozenset[str]:
        """Return statically-configured model IDs (Vertex has no /models endpoint)."""
        return self._models


def _build_base_url(settings: Settings) -> str:
    """Compose the Vertex regional OpenAI-compat base URL.

    Returned URL stops at ``/endpoints/{id}``; the OpenAI SDK appends
    ``/chat/completions`` per request. For the Garden openapi shim,
    set ``VERTEX_ENDPOINT_ID=openapi``.
    """
    project = settings.vertex_project.strip()
    region = settings.vertex_region.strip()
    endpoint_id = settings.vertex_endpoint_id.strip()
    if not project or not region or not endpoint_id:
        raise AuthenticationError(
            "Vertex provider requires VERTEX_PROJECT, VERTEX_REGION, and "
            "VERTEX_ENDPOINT_ID. Set them in your .env file."
        )
    return (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{region}/endpoints/{endpoint_id}"
    )


def _parse_models(raw: str) -> frozenset[str]:
    return frozenset(name.strip() for name in raw.split(",") if name.strip())


def _build_http_client(
    config: ProviderConfig, auth: VertexAuthProvider
) -> httpx.AsyncClient:
    timeout = httpx.Timeout(
        config.http_read_timeout,
        connect=config.http_connect_timeout,
        read=config.http_read_timeout,
        write=config.http_write_timeout,
    )

    async def _attach_token(request: httpx.Request) -> None:
        token = await auth.get_token()
        request.headers["Authorization"] = f"Bearer {token}"

    return httpx.AsyncClient(
        timeout=timeout,
        proxy=config.proxy or None,
        event_hooks={"request": [_attach_token]},
    )
