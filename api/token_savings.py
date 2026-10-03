"""Token-savings rows for context-optimizer requests.

Owns: one row per optimized request into the shared log that
~/.claude/scripts/token_savings_report.py reads. The schema is owned by
~/.claude/hooks/utils/savings_log.py; this proxy runs from its own venv, so
the copy here must be kept in step by hand.

The baseline is the request as Claude Code sent it, counted with the same
tokenizer the optimizer uses for its own result. No Claude Code session id
reaches the proxy, so rows carry the optimizer's derived session key instead.
Every request resends the whole conversation, so these totals are per-request
input savings, not unique tokens. cap_tokens stays null: there is no harness
output limit on this surface.

Called by: api/context_optimization.py ContextOptimizer.optimize.
Never raises: telemetry must not fail a request.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


def log_path() -> Path:
    default = Path.home() / ".claude" / "logs" / "token-savings.jsonl"
    return Path(os.environ.get("CLAUDE_TOKEN_SAVINGS_LOG", default))


def record_optimization(
    tokens_before: int, tokens_after: int, session_key: str
) -> None:
    context = {
        "system": "context-optimizer",
        "tool": "optimize",
        "session_id": None,
        "session_key": session_key,
        "kind": "savings",
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "baseline_method": "raw_request",
        "estimator": "tokenizer",
        "cap_tokens": None,
    }
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": "INFO",
        "event": "token_savings.record",
        "correlation_id": str(uuid.uuid4()),
        "context": context,
    }
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError as exc:
        logger.warning("TOKEN_SAVINGS: write_failed error={}", exc)
