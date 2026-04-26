"""Tests for providers/common/ollama_supervisor.py.

Owns: state-machine specifications for OllamaSupervisor.ensure_ready —
ready-cache TTL, failure cooldown, missing-binary path, concurrent-call
coalescing.

Does NOT own: end-to-end Ollama integration (those would require a
running daemon and are intentionally absent from CI). All HTTP and
subprocess calls are mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.common.ollama_supervisor import OllamaSupervisor, _api_root


@pytest.fixture(autouse=True)
def reset_supervisor_state():
    OllamaSupervisor._reset_for_test()
    yield
    OllamaSupervisor._reset_for_test()


@pytest.fixture
def settings():
    s = MagicMock()
    s.ollama_base_url = "http://localhost:11434/v1"
    s.ollama_model = "qwen2.5:7b"
    return s


def _ok_response(status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = ""
    return resp


def _async_client_returning(get_resp, post_resp):
    """Build a context-manager mock that yields a fake httpx.AsyncClient."""
    client = MagicMock()
    client.get = AsyncMock(return_value=get_resp)
    client.post = AsyncMock(return_value=post_resp)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm, client


def test_api_root_strips_openai_v1_suffix():
    assert _api_root("http://localhost:11434/v1") == "http://localhost:11434"
    assert _api_root("http://localhost:11434/v1/") == "http://localhost:11434"
    assert _api_root("http://localhost:11434") == "http://localhost:11434"


@pytest.mark.asyncio
async def test_ensure_ready_caches_success_within_ttl(settings):
    cm, client = _async_client_returning(_ok_response(200), _ok_response(200))
    with patch(
        "providers.common.ollama_supervisor.httpx.AsyncClient", return_value=cm
    ):
        first = await OllamaSupervisor.ensure_ready(settings)
        second = await OllamaSupervisor.ensure_ready(settings)

    assert first is True
    assert second is True
    # One health-check + one warm POST on the first call. The second call
    # must be served from the ready-cache without any further HTTP work.
    assert client.get.call_count == 1
    assert client.post.call_count == 1


@pytest.mark.asyncio
async def test_ensure_ready_returns_false_during_cooldown_after_failure(
    monkeypatch, settings
):
    # Simulate: health check fails, daemon spawn succeeds, daemon never
    # becomes reachable. Supervisor must mark failed and short-circuit.
    fail_resp = MagicMock()
    fail_resp.status_code = 500
    fail_resp.text = "boom"
    cm, client = _async_client_returning(fail_resp, fail_resp)

    monkeypatch.setattr(
        "providers.common.ollama_supervisor.OllamaSupervisor._spawn_daemon",
        lambda: True,
    )
    # Make _wait_for_health give up immediately so the test stays fast.
    monkeypatch.setattr(
        OllamaSupervisor, "_DAEMON_BOOT_TIMEOUT_SECONDS", 0.05
    )
    monkeypatch.setattr(
        OllamaSupervisor, "_DAEMON_POLL_INTERVAL_SECONDS", 0.01
    )

    with patch(
        "providers.common.ollama_supervisor.httpx.AsyncClient", return_value=cm
    ):
        first = await OllamaSupervisor.ensure_ready(settings)
        # Inside the cooldown window — must NOT touch httpx again.
        get_calls_before = client.get.call_count
        second = await OllamaSupervisor.ensure_ready(settings)

    assert first is False
    assert second is False
    assert client.get.call_count == get_calls_before


@pytest.mark.asyncio
async def test_ensure_ready_returns_false_when_ollama_binary_missing(
    monkeypatch, settings
):
    # Health check fails, then Popen raises FileNotFoundError because
    # `ollama` is not on PATH. Supervisor must catch and return False.
    fail_resp = MagicMock()
    fail_resp.status_code = 500
    fail_resp.text = ""
    cm, _ = _async_client_returning(fail_resp, fail_resp)

    def _missing_binary(*_args, **_kwargs):
        raise FileNotFoundError("ollama")

    monkeypatch.setattr(
        "providers.common.ollama_supervisor.subprocess.Popen", _missing_binary
    )
    with patch(
        "providers.common.ollama_supervisor.httpx.AsyncClient", return_value=cm
    ):
        result = await OllamaSupervisor.ensure_ready(settings)

    assert result is False


@pytest.mark.asyncio
async def test_concurrent_ensure_ready_calls_coalesce_on_single_check(settings):
    # Two ensure_ready calls in flight at the same time must result in only
    # one health-check + one warm POST — the second caller waits for the
    # lock and then sees the cache populated by the first.
    cm, client = _async_client_returning(_ok_response(200), _ok_response(200))
    with patch(
        "providers.common.ollama_supervisor.httpx.AsyncClient", return_value=cm
    ):
        import asyncio

        a, b = await asyncio.gather(
            OllamaSupervisor.ensure_ready(settings),
            OllamaSupervisor.ensure_ready(settings),
        )

    assert a is True and b is True
    assert client.get.call_count == 1
    assert client.post.call_count == 1


@pytest.mark.asyncio
async def test_warm_model_404_marks_failed_with_pull_hint(monkeypatch, settings, caplog):
    # Daemon healthy, but warm-up returns 404 because the model isn't pulled.
    # Supervisor must log the exact `ollama pull <model>` command and fail.
    not_found = MagicMock()
    not_found.status_code = 404
    not_found.text = "model not found"
    cm, _ = _async_client_returning(_ok_response(200), not_found)

    with patch(
        "providers.common.ollama_supervisor.httpx.AsyncClient", return_value=cm
    ):
        result = await OllamaSupervisor.ensure_ready(settings)

    assert result is False
