"""Token-savings rows written per optimized request (api/token_savings.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api import context_optimization
from api.context_optimization import ContextOptimizer
from api.models.anthropic import Message, MessagesRequest
from api.token_savings import record_optimization
from config.settings import Settings


@pytest.fixture
def savings_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "token-savings.jsonl"
    monkeypatch.setenv("CLAUDE_TOKEN_SAVINGS_LOG", str(path))
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_record_writes_the_shared_schema(savings_log: Path) -> None:
    record_optimization(1000, 400, "abc")
    (row,) = _rows(savings_log)
    assert row["event"] == "token_savings.record"
    assert row["context"] == {
        "system": "context-optimizer",
        "tool": "optimize",
        "session_id": None,
        "session_key": "abc",
        "kind": "savings",
        "tokens_before": 1000,
        "tokens_after": 400,
        "baseline_method": "raw_request",
        "estimator": "tokenizer",
        "cap_tokens": None,
    }


def test_unwritable_log_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("CLAUDE_TOKEN_SAVINGS_LOG", str(blocker / "x.jsonl"))
    record_optimization(1, 1, "k")


@pytest.mark.asyncio
async def test_optimize_records_pre_and_post_token_counts(
    savings_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_text = "tool output line\n" * 2000

    async def fake_optimize(*, messages, system, settings, tools):
        # Rewrite in place, like the real tiers may: the baseline must
        # already have been counted from the untouched payload.
        messages[0]["content"] = "short"
        return messages, system, 7

    monkeypatch.setattr(context_optimization._PkgOptimizer, "optimize", fake_optimize)
    request = MessagesRequest(
        model="claude-sonnet-4-5",
        max_tokens=10,
        messages=[Message(role="user", content=long_text)],
    )
    optimized, tokens = await ContextOptimizer.optimize(request, Settings())

    assert tokens == 7
    assert optimized.messages[0].content == "short"
    (row,) = _rows(savings_log)
    assert row["context"]["tokens_after"] == 7
    assert row["context"]["tokens_before"] > 1000
    assert len(row["context"]["session_key"]) == 16
