"""Tests for the X-Free-Claude-Project header → resolution wiring.

Critical-path coverage: ensures untrusted header values cannot redirect
the proxy at filesystem locations outside the user's home, and that the
resolution path through Settings.resolve_model honors per-project
overrides ahead of env-var fallbacks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import Request

from api.context import current_project_cwd
from api.dependencies import (
    PROJECT_HEADER,
    _project_cwd_from_request,
    get_project_cwd_from_header,
)
from config.project_settings import clear_cache
from config.settings import Settings


@pytest.fixture(autouse=True)
def _isolate_cache():
    clear_cache()
    yield
    clear_cache()


def _request_with_header(value: str | None) -> Request:
    headers = []
    if value is not None:
        headers.append((PROJECT_HEADER.encode(), value.encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/messages",
        "headers": headers,
    }
    return Request(scope)


def test_missing_header_yields_none():
    request = _request_with_header(None)
    assert _project_cwd_from_request(request) is None


def test_relative_path_rejected():
    request = _request_with_header("relative/path")
    assert _project_cwd_from_request(request) is None


def test_path_outside_home_rejected(tmp_path, monkeypatch):
    # /etc must never be accepted regardless of HOME setting.
    request = _request_with_header("/etc")
    assert _project_cwd_from_request(request) is None


def test_nonexistent_path_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    request = _request_with_header(str(tmp_path / "does-not-exist"))
    assert _project_cwd_from_request(request) is None


def test_valid_path_under_home_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    project = tmp_path / "myproj"
    project.mkdir()
    request = _request_with_header(str(project))
    assert _project_cwd_from_request(request) == project.resolve()


@pytest.mark.asyncio
async def test_dependency_sets_and_clears_contextvar(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    project = tmp_path / "myproj"
    project.mkdir()
    request = _request_with_header(str(project))

    assert current_project_cwd.get() is None
    gen = get_project_cwd_from_header(request)
    yielded = await gen.__anext__()
    assert yielded == project.resolve()
    assert current_project_cwd.get() == project.resolve()

    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()
    assert current_project_cwd.get() is None


def test_resolve_model_uses_project_override_over_env(tmp_path, monkeypatch):
    project = tmp_path / "p"
    (project / ".claude").mkdir(parents=True)
    (project / ".claude" / "settings.json").write_text(
        json.dumps({"freeClaudeCode": {"models": {"sonnet": "deepseek/proj-flash"}}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("MODEL", "deepseek/global-fallback")
    monkeypatch.setenv("MODEL_SONNET", "deepseek/global-sonnet")
    settings = Settings()
    assert settings.resolve_model("claude-sonnet-4-6") == "deepseek/global-sonnet"
    assert (
        settings.resolve_model("claude-sonnet-4-6", project_cwd=project)
        == "deepseek/proj-flash"
    )


def test_resolve_model_falls_through_when_project_block_absent(tmp_path, monkeypatch):
    project = tmp_path / "p"
    project.mkdir()
    monkeypatch.setenv("MODEL", "deepseek/global-fallback")
    monkeypatch.setenv("MODEL_SONNET", "deepseek/global-sonnet")
    settings = Settings()
    assert (
        settings.resolve_model("claude-sonnet-4-6", project_cwd=project)
        == "deepseek/global-sonnet"
    )


def test_resolve_model_default_used_when_tier_missing(tmp_path, monkeypatch):
    project = tmp_path / "p"
    (project / ".claude").mkdir(parents=True)
    (project / ".claude" / "settings.json").write_text(
        json.dumps({"freeClaudeCode": {"models": {"default": "lmstudio/qwen"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL", "deepseek/global")
    settings = Settings()
    assert (
        settings.resolve_model("claude-opus-4-7", project_cwd=project)
        == "lmstudio/qwen"
    )
