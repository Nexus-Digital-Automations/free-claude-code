"""Tests for per-project settings loader (config/project_settings.py).

Critical-path coverage: this file gates which upstream model handles a
given project's traffic, including security/auth-sensitive routing
(e.g. an Opus-class agent could be misrouted to a cheaper or more
permissive backend if loader selection is wrong).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.project_settings import (
    ProjectSettings,
    clear_cache,
    load_project_settings,
)


@pytest.fixture(autouse=True)
def _isolate_cache():
    clear_cache()
    yield
    clear_cache()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_returns_none_when_no_settings_files_exist(tmp_path):
    assert load_project_settings(tmp_path) is None


def test_returns_none_when_freeClaudeCode_block_missing(tmp_path):
    _write(tmp_path / ".claude" / "settings.json", {"env": {"FOO": "bar"}})
    assert load_project_settings(tmp_path) is None


def test_loads_block_from_settings_json(tmp_path):
    _write(
        tmp_path / ".claude" / "settings.json",
        {"freeClaudeCode": {"models": {"sonnet": "deepseek/v4-flash"}}},
    )
    parsed = load_project_settings(tmp_path)
    assert parsed is not None
    assert isinstance(parsed, ProjectSettings)
    assert parsed.models.sonnet == "deepseek/v4-flash"
    assert parsed.models.opus is None


def test_local_overrides_base(tmp_path):
    _write(
        tmp_path / ".claude" / "settings.json",
        {"freeClaudeCode": {"models": {"sonnet": "deepseek/base"}}},
    )
    _write(
        tmp_path / ".claude" / "settings.local.json",
        {"freeClaudeCode": {"models": {"sonnet": "deepseek/local-wins"}}},
    )
    parsed = load_project_settings(tmp_path)
    assert parsed is not None
    assert parsed.models.sonnet == "deepseek/local-wins"


def test_local_merges_with_base_keys(tmp_path):
    _write(
        tmp_path / ".claude" / "settings.json",
        {"freeClaudeCode": {"models": {"sonnet": "deepseek/base"}}},
    )
    _write(
        tmp_path / ".claude" / "settings.local.json",
        {"freeClaudeCode": {"models": {"opus": "vertex/gemini-pro"}}},
    )
    parsed = load_project_settings(tmp_path)
    assert parsed is not None
    assert parsed.models.sonnet == "deepseek/base"
    assert parsed.models.opus == "vertex/gemini-pro"


def test_malformed_json_falls_back_to_none(tmp_path, caplog):
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{ this is not json", encoding="utf-8")
    assert load_project_settings(tmp_path) is None


def test_invalid_provider_model_shape_falls_back_to_none(tmp_path):
    _write(
        tmp_path / ".claude" / "settings.json",
        {"freeClaudeCode": {"models": {"sonnet": "no-slash-here"}}},
    )
    assert load_project_settings(tmp_path) is None


def test_cache_returns_same_instance_when_unchanged(tmp_path):
    _write(
        tmp_path / ".claude" / "settings.json",
        {"freeClaudeCode": {"models": {"sonnet": "deepseek/v1"}}},
    )
    a = load_project_settings(tmp_path)
    b = load_project_settings(tmp_path)
    assert a is b


def test_cache_invalidates_when_mtime_advances(tmp_path):
    settings_path = tmp_path / ".claude" / "settings.json"
    _write(settings_path, {"freeClaudeCode": {"models": {"sonnet": "deepseek/v1"}}})
    first = load_project_settings(tmp_path)
    assert first is not None
    assert first.models.sonnet == "deepseek/v1"

    # Bump mtime forward to force re-read; some filesystems have 1s mtime
    # resolution, so set explicitly rather than relying on wall-clock.
    import os

    new_mtime = settings_path.stat().st_mtime + 5.0
    _write(settings_path, {"freeClaudeCode": {"models": {"sonnet": "deepseek/v2"}}})
    os.utime(settings_path, (new_mtime, new_mtime))

    second = load_project_settings(tmp_path)
    assert second is not None
    assert second.models.sonnet == "deepseek/v2"
    assert second is not first


@pytest.mark.parametrize(
    "incoming, expected",
    [
        ("claude-opus-4-7", "vertex/gemini-pro"),
        ("claude-sonnet-4-6", "deepseek/v4-flash"),
        ("claude-haiku-4-5-20251001", "lmstudio/qwen-mini"),
        ("claude-3-5-sonnet-20241022", "deepseek/v4-flash"),
    ],
)
def test_model_for_classifies_by_substring(tmp_path, incoming, expected):
    _write(
        tmp_path / ".claude" / "settings.json",
        {
            "freeClaudeCode": {
                "models": {
                    "opus": "vertex/gemini-pro",
                    "sonnet": "deepseek/v4-flash",
                    "haiku": "lmstudio/qwen-mini",
                }
            }
        },
    )
    parsed = load_project_settings(tmp_path)
    assert parsed is not None
    assert parsed.model_for(incoming) == expected


def test_model_for_falls_back_to_default(tmp_path):
    _write(
        tmp_path / ".claude" / "settings.json",
        {"freeClaudeCode": {"models": {"default": "deepseek/v4-flash"}}},
    )
    parsed = load_project_settings(tmp_path)
    assert parsed is not None
    # No tier-specific override → use default for any incoming model.
    assert parsed.model_for("claude-opus-4-7") == "deepseek/v4-flash"
    assert parsed.model_for("claude-sonnet-4-6") == "deepseek/v4-flash"


def test_model_for_returns_none_when_no_overrides(tmp_path):
    _write(tmp_path / ".claude" / "settings.json", {"freeClaudeCode": {"models": {}}})
    parsed = load_project_settings(tmp_path)
    assert parsed is not None
    assert parsed.model_for("claude-opus-4-7") is None
