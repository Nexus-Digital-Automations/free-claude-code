"""Owns: headroom_compress tier behavioural tests.

Counterpart: src/context_optimizer/tiers/headroom_compress.py.

Two layers:
  - Pure-unit (mocked httpx): the disabled short-circuit and every
    graceful-degradation path (non-200, connection error, malformed body)
    must return the original messages and never raise.
  - Live smoke (skipped unless a Headroom sidecar answers on 127.0.0.1:8787):
    real compression shrinks a tool_result while leaving user/system messages
    untouched.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from context_optimizer.settings import ContextOptimizerSettings
from context_optimizer.tiers import headroom_compress

_SIDECAR_HOST = "127.0.0.1"
_SIDECAR_PORT = 8787


def _tool_result_msgs(content: str) -> list[dict]:
    return [
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "run", "input": {"c": "x"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": content}
            ],
        },
    ]


class _FakeResp:
    def __init__(self, status_code: int, payload: object):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _FakeClient:
    """Stands in for httpx.AsyncClient: one POST returns _resp or raises _exc."""

    def __init__(self, *, resp: _FakeResp | None = None, exc: Exception | None = None):
        self._resp = resp
        self._exc = exc

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def post(self, url: str, json: dict) -> _FakeResp:
        if self._exc is not None:
            raise self._exc
        assert self._resp is not None
        return self._resp


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> None:
    monkeypatch.setattr(headroom_compress.httpx, "AsyncClient", lambda **_kw: client)


async def test_disabled_returns_same_object():
    settings = ContextOptimizerSettings(headroom_enabled=False)
    msgs = _tool_result_msgs("x" * 9000)
    result = await headroom_compress.apply(msgs, settings)
    assert result is msgs


async def test_returns_sidecar_messages_on_success(monkeypatch):
    compressed = _tool_result_msgs("compressed")
    _patch_client(
        monkeypatch,
        _FakeClient(
            resp=_FakeResp(200, {"messages": compressed, "transforms_applied": ["log"]})
        ),
    )
    settings = ContextOptimizerSettings(headroom_enabled=True)
    result = await headroom_compress.apply(_tool_result_msgs("x" * 9000), settings)
    assert result == compressed


async def test_non_200_passes_through(monkeypatch):
    _patch_client(monkeypatch, _FakeClient(resp=_FakeResp(503, {})))
    settings = ContextOptimizerSettings(headroom_enabled=True)
    msgs = _tool_result_msgs("x" * 9000)
    result = await headroom_compress.apply(msgs, settings)
    assert result is msgs


async def test_connection_error_passes_through(monkeypatch):
    _patch_client(
        monkeypatch, _FakeClient(exc=httpx.ConnectError("connection refused"))
    )
    settings = ContextOptimizerSettings(headroom_enabled=True)
    msgs = _tool_result_msgs("x" * 9000)
    result = await headroom_compress.apply(msgs, settings)
    assert result is msgs


async def test_malformed_body_passes_through(monkeypatch):
    _patch_client(
        monkeypatch, _FakeClient(resp=_FakeResp(200, {"messages": "not-a-list"}))
    )
    settings = ContextOptimizerSettings(headroom_enabled=True)
    msgs = _tool_result_msgs("x" * 9000)
    result = await headroom_compress.apply(msgs, settings)
    assert result is msgs


def _sidecar_up() -> bool:
    try:
        with socket.create_connection((_SIDECAR_HOST, _SIDECAR_PORT), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _sidecar_up(), reason="no Headroom sidecar on :8787")
async def test_live_sidecar_compresses_tool_result():
    # 600 repetitive log lines — Headroom's log compressor should collapse them.
    log = "\n".join(
        f"[{i:04d}] INFO collecting tests/test_{i % 40}.py ... ok" for i in range(600)
    )
    original = _tool_result_msgs(log)
    settings = ContextOptimizerSettings(
        headroom_enabled=True,
        headroom_url=f"http://{_SIDECAR_HOST}:{_SIDECAR_PORT}",
    )
    result = await headroom_compress.apply(original, settings)

    def _tool_result_text(msgs: list[dict]) -> str:
        return msgs[2]["content"][0]["content"]

    assert len(_tool_result_text(result)) < len(_tool_result_text(original))
    # User and system-style messages must be untouched.
    assert result[0] == original[0]


@pytest.mark.skipif(not _sidecar_up(), reason="no Headroom sidecar on :8787")
async def test_optimizer_routes_to_headroom_and_skips_tier0b(monkeypatch):
    """End-to-end through ContextOptimizer: with headroom_enabled the pipeline
    compresses tool output via the sidecar and the Ollama Tier 0b never runs."""
    import json

    from context_optimizer import ContextOptimizer
    from context_optimizer.tiers import tier0b

    async def _boom(*_a: object, **_k: object) -> list[dict]:
        raise AssertionError("tier0b must be skipped when headroom_enabled")

    monkeypatch.setattr(tier0b, "apply", _boom)

    # Single-line JSON: large in bytes, one line — so Tier 0's line-truncation
    # leaves it for the tool-output compressor (Headroom's SmartCrusher).
    records = [
        {
            "id": i,
            "path": f"src/f_{i}.py",
            "status": "ok" if i % 11 else "error",
            "msg": "matched reference" if i % 11 else f"ERROR import line {i}",
        }
        for i in range(400)
    ]
    big_json = json.dumps(records)
    msgs = _tool_result_msgs(big_json)
    settings = ContextOptimizerSettings(
        headroom_enabled=True,
        headroom_url=f"http://{_SIDECAR_HOST}:{_SIDECAR_PORT}",
        block_selection_mode="off",  # no Ollama block tower
        tier0c_digest_enabled=False,  # no Ollama tier0c
        tier0d_digest_enabled=False,  # no Ollama tier0d
    )
    out_msgs, _sys, _tokens = await ContextOptimizer.optimize(
        messages=msgs, system="sys", settings=settings
    )
    out_text = out_msgs[2]["content"][0]["content"]
    assert len(out_text) < len(big_json)
