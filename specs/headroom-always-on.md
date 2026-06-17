---
title: Headroom sidecar always-on — fcc-server auto-manages the OrbStack container
status: active
created: 2026-06-16
---

## Vision

Make the Headroom compression tier (shipped in 2.2.0) **always available** without
manual steps: whenever `fcc-server` starts, it brings up the Headroom proxy as an
**OrbStack** container, and tears it down on shutdown. The `CONTEXT_HEADROOM_ENABLED`
flag is persisted in the user's `~/.fcc/.env` so the Tier-0b-replacement compressor
is on by default for this instance.

## Decisions (locked via grilling)

- fcc-server **auto-manages** the sidecar: start on startup, **stop on shutdown**
  (container lifecycle == server lifecycle; no restart policy).
- Flag persisted in **`~/.fcc/.env`** (`CONTEXT_HEADROOM_ENABLED=1`); the project
  default stays `False` so tests/CI/other users never depend on a sidecar.
- Container runs through **OrbStack explicitly** (`docker --context orbstack`),
  never plain Docker / Docker Desktop.
- **Best-effort**: a missing OrbStack, missing image, or unhealthy container logs
  a warning and lets the server start anyway — the compress tier degrades to
  pass-through. Sidecar problems never block or break startup.

## Requirements

- New `core/headroom_sidecar.py`: `async ensure_started(settings)` and
  `async stop(settings)`. Idempotent (skip if already running), bounded health
  wait, never raises.
- `AppRuntime.startup()` calls `ensure_started`; `AppRuntime.shutdown()` calls
  `stop` via the existing `best_effort` helper.
- `config/settings.py`: add `context_headroom_autostart` (default True) and
  `context_headroom_image` (default `ghcr.io/chopratejas/headroom:latest`).
- `.env.example` documents the `CONTEXT_HEADROOM_*` vars.
- `~/.fcc/.env` gains `CONTEXT_HEADROOM_ENABLED=1` (idempotent).

## Acceptance Criteria

- [x] `core/headroom_sidecar.py` exposes async `ensure_started(settings)` and `stop(settings)`, and neither raises on subprocess/health failure.
- [x] `ensure_started` is a no-op when `context_headroom_enabled` is False or `context_headroom_autostart` is False.
- [x] The sidecar is launched through OrbStack explicitly — the docker invocation includes `--context orbstack`.
- [x] `AppRuntime.startup` calls `ensure_started` and `AppRuntime.shutdown` calls `stop` (best-effort, never blocking server lifecycle).
- [x] `config/settings.py` exposes `context_headroom_autostart` (default True) and `context_headroom_image`.
- [x] `~/.fcc/.env` contains a `CONTEXT_HEADROOM_ENABLED=1` line.
- [x] Unit tests in `tests/core/test_headroom_sidecar.py` cover: disabled no-op, already-running skip, start-when-absent, and graceful degradation on launch failure.
- [x] `uv run ruff check` and `uv run ty check` pass on the changed files; the context-optimizer suite and the new test pass.
- [x] `pyproject.toml` version bumped (MINOR) and `uv.lock` updated in the same commit.
