"""Loguru-based structured logging configuration.

All logs are written to server.log as JSON lines for full traceability.
Stdlib logging is intercepted and funneled to loguru.
Context vars (request_id, node_id, chat_id) from contextualize() are
included at top level for easy grep/filter.
"""

import json
import logging
from pathlib import Path

from loguru import logger

_configured = False
_configured_path: str | None = None

# Context keys we promote to top-level JSON for traceability
_CONTEXT_KEYS = ("request_id", "node_id", "chat_id")


def _serialize_with_context(record) -> str:
    """Format record as JSON with context vars at top level.
    Returns a format template; we inject _json into record for output.
    """
    extra = record.get("extra", {})
    out = {
        "time": str(record["time"]),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    for key in _CONTEXT_KEYS:
        if key in extra and extra[key] is not None:
            out[key] = extra[key]
    record["_json"] = json.dumps(out, default=str)
    return "{_json}\n"


class InterceptHandler(logging.Handler):
    """Redirect stdlib logging to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging(log_file: str, *, force: bool = False) -> None:
    """Configure loguru with JSON output to log_file and intercept stdlib logging.

    Idempotent: skips if already configured at the same path. Use force=True
    only when the log path actually changes (e.g. tests with a different file).

    WHY the path-equality short-circuit even under force=True: re-running
    logger.remove() / logger.add() while another thread is mid-emit
    deadlocks loguru's internal lock. If the destination is the same, there
    is nothing to do — skipping is always safe.
    """
    global _configured, _configured_path
    if _configured and _configured_path == log_file:
        return
    if _configured and not force:
        return
    _configured = True
    _configured_path = log_file

    # Remove default loguru handler (writes to stderr)
    logger.remove()

    # Ensure logs/ (or any parent dir) exists before writing.
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    # Add file sink: JSON lines, DEBUG level, context vars at top level
    logger.add(
        log_file,
        level="DEBUG",
        format=_serialize_with_context,
        encoding="utf-8",
        mode="a",
        rotation="50 MB",
    )

    # Intercept stdlib logging: route all root logger output to loguru
    intercept = InterceptHandler()
    logging.root.handlers = [intercept]
    logging.root.setLevel(logging.DEBUG)
