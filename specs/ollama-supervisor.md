---
title: Ollama Auto-Supervisor for Autocompaction
status: active
created: 2026-04-26
---

## Vision

Background Tier 2a compaction depends on a local Ollama daemon plus a warm
model. Today the optimizer calls Ollama optimistically and falls back to
the configured provider on failure — meaning when Ollama is down the user
silently pays for provider compactions instead. This spec makes Ollama
availability **deterministic**: the proxy ensures the daemon is up and the
configured model is warm at boot, and re-verifies before every compaction.
On unrecoverable failure, the proxy logs loudly and falls back rather than
blocking the user.

## Requirements

### R1 — Boot-time start
On FastAPI startup (`api/app.py:lifespan`), the proxy verifies Ollama is
reachable. If not, it spawns `ollama serve` as a detached subprocess. Once
the daemon is reachable, it warms the configured model via the Ollama HTTP
API with `keep_alive: 30m`.

### R2 — Per-compaction recheck
Before every Tier 2a Ollama call (`_compact_via_ollama`, `_do_ollama_call`),
the optimizer calls `OllamaSupervisor.ensure_ready(settings)`. Result is
cached for 30 seconds to keep this cheap on the hot path.

### R3 — Cooldown on failure
On supervisor failure (binary missing, daemon won't start, model warm-up
errors), set a 60-second cooldown during which `ensure_ready` returns False
immediately without retrying. Lets compactions fall through to the provider
without each request paying spawn-attempt cost.

### R4 — Detached subprocess
`ollama serve` is spawned with `start_new_session=True` and stdio routed to
DEVNULL so the daemon outlives the proxy and doesn't pollute proxy logs.

### R5 — Concurrency safety
A single `asyncio.Lock` serialises supervisor checks; concurrent callers
coalesce on the same in-flight check rather than each spawning their own
daemon.

### R6 — Loud logging, soft failure
All supervisor activity logs at INFO/WARNING with prefix `OLLAMA:`. A
failed startup or warm-up never raises — it returns False and lets the
optimizer fall back to the provider compaction path.

## Acceptance Criteria

- [x] New module `providers/common/ollama_supervisor.py` defines
      `OllamaSupervisor.ensure_ready(settings) -> bool`.
- [x] `api/app.py:lifespan` schedules `ensure_ready` at startup
      (fire-and-forget background task — does not block server boot).
- [x] `_compact_via_ollama` and `_do_ollama_call` short-circuit and return
      False if `ensure_ready` returns False (without an HTTP attempt).
- [x] Health-check uses `GET {ollama_base_url}/api/tags`; warm-up uses
      `POST /api/generate` with `prompt: ""`, `stream: false`,
      `keep_alive: "30m"`.
- [x] Repeated `ensure_ready` calls within 30 seconds reuse the cached
      "ready" result. Verified by `test_ensure_ready_caches_success_within_ttl`
      (1 GET + 1 POST across two ensure_ready calls).
- [x] After a failure, `ensure_ready` returns False without further work
      for 60 seconds. Verified by
      `test_ensure_ready_returns_false_during_cooldown_after_failure`.
- [x] Loud logging: `OLLAMA: health_check ok|failed`,
      `OLLAMA: spawning daemon`, `OLLAMA: daemon ready elapsed_ms=N`,
      `OLLAMA: warming model=X`, `OLLAMA: ready elapsed_ms=N`, and a
      WARNING line on each failure mode.
- [x] `tests/providers/test_ollama_supervisor.py`: 6 tests covering
      cache hit, cooldown after failure, missing-binary path,
      concurrent-coalescing, 404 model-not-pulled hint, URL parsing.
- [x] Full lint + test suite green: 930 passed, ruff clean.

## Technical Decisions

- **HTTP `/api/generate` warm-up over `ollama run`.** `ollama run` is
  interactive and not script-friendly. `/api/generate` with empty prompt
  and `keep_alive: 30m` is the documented warm-up path and gives us a
  status code to check.
- **Class-level state, not instance.** Matches the existing pattern in
  `ContextOptimizer` (single-process, single-event-loop). Avoids dependency
  injection for a globally-singleton supervisor.
- **30s ready-cache + 60s failure-cooldown.** Ready-cache cheap enough to
  call before every compaction; failure-cooldown wide enough to avoid
  hammering when Ollama is genuinely broken.
- **No auto-pull of missing model.** `ollama pull qwen2.5:7b` is multi-GB
  and long-running. If the model isn't pulled, we log loudly with the
  exact `ollama pull <model>` command the user should run, and return
  False. Out of scope for this spec.

## Files

| File | Change |
|------|--------|
| `providers/common/ollama_supervisor.py` | NEW |
| `api/app.py` | startup hook: schedule `ensure_ready` task |
| `providers/common/context_optimizer.py` | gate `_compact_via_ollama` and `_do_ollama_call` on `ensure_ready` |
| `tests/providers/test_ollama_supervisor.py` | NEW |

## Progress

- [x] R1 — boot-time start
- [x] R2 — per-compaction recheck
- [x] R3 — failure cooldown
- [x] R4 — detached subprocess
- [x] R5 — concurrency lock
- [x] R6 — logging + soft fail
