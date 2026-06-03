"""Per-request context shared across FastAPI dependency, validator, and route.

Owns: the `current_project_cwd` ContextVar that carries the
`X-Free-Claude-Project` header value from the request-scoped dependency
into the Pydantic model_validator on `MessagesRequest`. The validator
cannot read request headers directly, so a contextvar bridges the gap.

Does NOT own: header parsing or validation (that's
`api/dependencies.py:get_project_cwd_from_header`), nor the resolution
logic itself (`config/settings.py:Settings.resolve_model`).

Concurrency: ContextVars are per-asyncio-task, so concurrent requests
get isolated values without locks. The dependency that sets the value
is responsible for clearing it on request exit (via FastAPI's `yield`
pattern) to avoid bleed across reused worker tasks.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path

current_project_cwd: ContextVar[Path | None] = ContextVar(
    "current_project_cwd",
    default=None,
)
