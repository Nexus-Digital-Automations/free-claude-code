"""Tests for the Vertex provider auth + URL surface.

These cover security-relevant paths: ADC token caching, refresh leeway,
static-token fallback, and the missing-credentials error. Provider
dispatch wiring is exercised via api/dependencies tests; the provider's
streaming behaviour rides on OpenAICompatibleProvider's existing
coverage.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from providers.exceptions import AuthenticationError
from providers.vertex import auth as vertex_auth
from providers.vertex.auth import VertexAuthProvider, _AdcUnavailable
from providers.vertex.client import _build_base_url, _parse_models
from providers.vertex.request import _normalize_model_id


def test_parse_models_strips_and_filters_empty() -> None:
    """Empty entries and surrounding whitespace are dropped."""
    assert _parse_models("a, b ,c, ") == frozenset({"a", "b", "c"})
    assert _parse_models("") == frozenset()
    assert _parse_models(",,") == frozenset()


def test_normalize_model_id_prefixes_gemma_for_openapi_shim() -> None:
    """Bare gemma IDs gain the google/ namespace prefix; others pass through."""
    assert _normalize_model_id("gemma-3-9b-it") == "google/gemma-3-9b-it"
    assert _normalize_model_id("google/gemma-3-9b-it") == "google/gemma-3-9b-it"
    assert _normalize_model_id("custom/my-model") == "custom/my-model"
    assert _normalize_model_id("") == ""


def test_build_base_url_rejects_missing_required_settings() -> None:
    """Missing project/region/endpoint must raise — not silently produce a bad URL."""
    from config.settings import Settings

    settings = Settings()
    with pytest.raises(AuthenticationError) as exc:
        _build_base_url(settings)
    assert "VERTEX_PROJECT" in str(exc.value)


def test_build_base_url_composes_regional_endpoint(monkeypatch) -> None:
    from config.settings import Settings

    # Settings fields use env-alias population (no populate_by_name); set the
    # VERTEX_* env vars rather than passing field/alias kwargs.
    monkeypatch.setenv("VERTEX_PROJECT", "my-proj")
    monkeypatch.setenv("VERTEX_REGION", "us-central1")
    monkeypatch.setenv("VERTEX_ENDPOINT_ID", "1234567890")
    settings = Settings()
    url = _build_base_url(settings)
    assert url == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj"
        "/locations/us-central1/endpoints/1234567890"
    )


def _mock_adc_unavailable(reason: str = "google-auth not installed"):
    """Patch the thread-bound ADC loader to raise as if google-auth is missing."""
    return patch.object(
        vertex_auth,
        "_load_and_refresh_adc",
        side_effect=_AdcUnavailable(reason),
    )


@pytest.mark.asyncio
async def test_auth_static_token_fallback_when_adc_unavailable() -> None:
    """When google-auth is missing, static token is returned."""
    auth = VertexAuthProvider(static_token="static-abc")  # pragma: allowlist secret
    with _mock_adc_unavailable():
        token = await auth.get_token()
    assert token == "static-abc"


@pytest.mark.asyncio
async def test_auth_raises_when_no_creds_and_no_static_token() -> None:
    """Missing both ADC and static token must surface a clear AuthenticationError."""
    auth = VertexAuthProvider(static_token="")
    with _mock_adc_unavailable("ADC default() failed: no credentials"):
        with pytest.raises(AuthenticationError) as exc:
            await auth.get_token()
    msg = str(exc.value)
    assert "VERTEX_ACCESS_TOKEN" in msg
    assert "google-auth" in msg or "ADC" in msg


@pytest.mark.asyncio
async def test_auth_caches_static_token_within_leeway() -> None:
    """A second call within the cache window must not refresh."""
    auth = VertexAuthProvider(static_token="cached-token")  # pragma: allowlist secret
    with _mock_adc_unavailable():
        first = await auth.get_token()
        # Mutate static_token — if cache is honoured, get_token returns the prior value.
        auth._static_token = "SHOULD-NOT-APPEAR"
        second = await auth.get_token()
    assert first == second == "cached-token"


@pytest.mark.asyncio
async def test_auth_concurrent_callers_share_one_refresh() -> None:
    """Concurrent get_token() calls must serialise on the lock; one refresh."""
    auth = VertexAuthProvider(static_token="shared")  # pragma: allowlist secret
    with _mock_adc_unavailable() as mocked:
        results = await asyncio.gather(*[auth.get_token() for _ in range(8)])
    assert results == ["shared"] * 8
    # Only one refresh attempt; cached on first hit, rest short-circuit.
    assert mocked.call_count == 1


@pytest.mark.asyncio
async def test_auth_refresh_after_expiry_window() -> None:
    """When the cached token is past its leeway window, a new refresh fires."""
    auth = VertexAuthProvider(static_token="first")  # pragma: allowlist secret
    with _mock_adc_unavailable():
        await auth.get_token()
        cached = auth._cached
        assert cached is not None
        cached.expires_at_monotonic = time.monotonic() - 1.0
        auth._static_token = "second"  # pragma: allowlist secret
        token = await auth.get_token()
    assert token == "second"
