"""Per-project model overrides loaded from `<cwd>/.claude/settings.json`.

Owns: parsing and caching of the `freeClaudeCode.models` block from a
project's Claude Code settings files. Merges `settings.json` with
`settings.local.json` (local wins). Cache entries are invalidated when
either source file's `st_mtime` advances.

Does NOT own: HTTP request handling, header validation, env-var
fallback, or how the project cwd is discovered. The caller passes a
trusted absolute path; this module only reads files at that path.

Called by: `Settings.resolve_model()` in `config/settings.py`.

Concurrency: the cache is guarded by a `threading.Lock`. The loader is
synchronous; FastAPI calls it from a thread when invoked under
`asyncio.to_thread`-style dependencies, so a non-reentrant lock is
sufficient.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator

Tier = Literal["default", "opus", "sonnet", "haiku"]

_CACHE_CAPACITY = 64
_SETTINGS_FILENAME = "settings.json"
_LOCAL_SETTINGS_FILENAME = "settings.local.json"


class ProjectModels(BaseModel):
    """The `freeClaudeCode.models` block.

    All fields optional; missing fields fall through to the global env-var
    mapping (`MODEL`, `MODEL_OPUS`, `MODEL_SONNET`, `MODEL_HAIKU`).
    """

    default: str | None = None
    opus: str | None = None
    sonnet: str | None = None
    haiku: str | None = None

    model_config = {"extra": "ignore"}

    @field_validator("default", "opus", "sonnet", "haiku")
    @classmethod
    def _validate_provider_model_shape(cls, value: str | None) -> str | None:
        # `<provider>/<model>` shape — match Settings.parse_provider_type's
        # downstream split. Empty halves would produce a bogus provider lookup.
        if value is None:
            return None
        if "/" not in value:
            raise ValueError(f"expected '<provider>/<model>', got '{value}'")
        provider, model = value.split("/", 1)
        if not provider or not model:
            raise ValueError(f"provider and model must be non-empty (got '{value}')")
        return value


class ProjectSettings(BaseModel):
    """The subset of `.claude/settings.json` this proxy cares about."""

    models: ProjectModels = Field(default_factory=ProjectModels)

    def model_for(self, claude_model_name: str) -> str | None:
        """Return the project's override for an incoming Claude model name.

        Tier-specific override wins over the project default. Returns None
        when no project-level override applies — the caller falls back to
        the global env-var mapping.
        """
        name_lower = claude_model_name.lower()
        if "opus" in name_lower and self.models.opus is not None:
            return self.models.opus
        if "haiku" in name_lower and self.models.haiku is not None:
            return self.models.haiku
        if "sonnet" in name_lower and self.models.sonnet is not None:
            return self.models.sonnet
        return self.models.default


# Cache key: resolved absolute project cwd. Value: (mtime_signature, parsed).
# mtime_signature = (settings.json mtime or 0.0, settings.local.json mtime or 0.0)
_CacheKey = Path
_MtimeSig = tuple[float, float]
_cache: OrderedDict[_CacheKey, tuple[_MtimeSig, ProjectSettings | None]] = OrderedDict()
_cache_lock = threading.Lock()


def load_project_settings(cwd: Path) -> ProjectSettings | None:
    """Load and validate per-project settings, with mtime-aware caching.

    Returns None when neither settings file exists, when neither defines a
    `freeClaudeCode` block, or when both files are unreadable. Malformed
    JSON or schema validation errors are logged at WARNING and treated as
    "no project override" — the caller falls back to global config rather
    than failing the request.
    """
    cwd_resolved = cwd.resolve()
    settings_path = cwd_resolved / ".claude" / _SETTINGS_FILENAME
    local_path = cwd_resolved / ".claude" / _LOCAL_SETTINGS_FILENAME
    sig = (_mtime_or_zero(settings_path), _mtime_or_zero(local_path))

    with _cache_lock:
        cached = _cache.get(cwd_resolved)
        if cached is not None and cached[0] == sig:
            _cache.move_to_end(cwd_resolved)
            return cached[1]

    parsed = _read_and_parse(settings_path, local_path)

    with _cache_lock:
        _cache[cwd_resolved] = (sig, parsed)
        _cache.move_to_end(cwd_resolved)
        while len(_cache) > _CACHE_CAPACITY:
            _cache.popitem(last=False)
    return parsed


def clear_cache() -> None:
    """Drop all cached project settings. Test-only entry point."""
    with _cache_lock:
        _cache.clear()


def _mtime_or_zero(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0
    except OSError as exc:
        logger.warning(
            "PROJECT_SETTINGS: stat_failed path={} error={}",
            path,
            exc,
        )
        return 0.0


def _read_and_parse(
    settings_path: Path,
    local_path: Path,
) -> ProjectSettings | None:
    base = _read_json(settings_path)
    local = _read_json(local_path)
    if base is None and local is None:
        return None

    merged = _deep_merge(base or {}, local or {})
    block = merged.get("freeClaudeCode")
    if not isinstance(block, dict):
        return None

    try:
        return ProjectSettings.model_validate(block)
    except ValidationError as exc:
        logger.warning(
            "PROJECT_SETTINGS: schema_invalid base={} local={} errors={}",
            settings_path,
            local_path,
            exc.errors(),
        )
        return None


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            parsed = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "PROJECT_SETTINGS: read_failed path={} error={}",
            path,
            exc,
        )
        return None
    if not isinstance(parsed, dict):
        logger.warning(
            "PROJECT_SETTINGS: not_an_object path={} type={}",
            path,
            type(parsed).__name__,
        )
        return None
    return parsed


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge — override wins; non-dict values replace wholesale."""
    out = dict(base)
    for key, value in override.items():
        existing = out.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            out[key] = _deep_merge(existing, value)
        else:
            out[key] = value
    return out
