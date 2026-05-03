---
title: Autocompaction + Observability Improvements (Tier 2)
status: active
created: 2026-05-03
---

## Vision

Build on the completed Tier 1 work (`specs/autocompaction-improvements.md`). Address the bugs, cost leaks, reliability gaps, and observability holes uncovered in the 2026-05-03 audit (`plans/please-see-what-improvements-warm-moth.md`). Companion to `specs/comprehensive-logging.md`.

## Requirements

### Correctness (P0)
- AsyncOpenAI client closes on every code path in Tier 2a, including under exceptions raised before `client.close()`.
- `OllamaSupervisor` lock is constructed exactly once across concurrent first callers (no lazy-init race).

### Cost (P2)
- Tier 0 head/tail line counts honor `ContextOptimizerSettings.tier0_head_lines` / `tier0_tail_lines` instead of being hardcoded.
- Prefix-cache lookup runs before Tier 0/1 cleanup, eliminating wasted dedup/strip work on cache hits.
- Proxy adapter (`providers/common/context_optimizer.py`) threads every relevant `Settings` field into `ContextOptimizerSettings`, not just six.

### Reliability (P3)
- `configure_logging(force=True)` cannot deadlock loguru when called from inside a log handler; idempotent fast-path used unless `log_file` actually changed.
- Ollama warm-up at app startup awaits readiness with a bounded timeout, then continues regardless.
- Tier 2 fallback log line distinguishes the failure mode: `parse_error | network | timeout | model_missing | busy`.

### Observability (P1)
This block tracks the open items from `specs/comprehensive-logging.md`. Current proxy state already has: `log_file` default of `logs/server.log`, mkdir-parents in `configure_logging`, `CLI_SESSION: start|resume|exit|session_id_captured`, `PROVIDER: done`. What is missing:
- `REQUEST: completed` entry on every request (success or failure) with `request_id`, `duration_ms`, `output_tokens`, `finish_reason`, `error`.
- `PROVIDER:` start entry (rename existing `*_STREAM` to keep grep prefix consistent) and `finish_reason` promoted to INFO.
- `CLI_SESSION:` stderr capped and structured instead of unbounded f-string.
- `REQUEST: provider_selected` log on every request with `provider`/`model`/`reason`.
- `request_id` propagation: created before the `try:` block in `create_message`, threaded via `logger.contextualize` for the full handler, echoed back as `X-Request-ID` response header.
- f-string warns in `api/app.py`, `cli/session.py`, `providers/rate_limit.py` converted to prefix-tagged structured entries.

### Metrics (P4)
- New optional `/metrics` endpoint exposing Prometheus exposition format, opt-in via `METRICS_ENABLED=1`.
- Counters: `proxy_request_total{provider, model, outcome}`, `compaction_invocation_total{tier, outcome}`, `prefix_cache_hit_total`, `prefix_cache_miss_total`.
- Histograms: `proxy_request_duration_seconds{provider, model}`, `compaction_tokens_saved`.
- Gauge: `ollama_supervisor_state{state}` set at supervisor state transitions.

## Acceptance Criteria

### P0 — bugs
- [ ] Tier 2a `_do_ollama_call` uses `async with AsyncOpenAI(...) as client:` so closure happens on exception.
- [ ] `OllamaSupervisor._lock` is created at class-definition time (not lazy), or initialized via a thread-/coroutine-safe primitive that can never produce two distinct locks.

### P2 — cost
- [ ] `tier0._truncate_long_outputs` accepts `head_lines`/`tail_lines` arguments; `tier0.apply` accepts them and `optimizer.optimize` passes the values from `ContextOptimizerSettings`.
- [ ] `optimizer.optimize` checks the prefix cache against the raw input messages first; only on miss does it run Tier 0 + Tier 1.
- [ ] Adapter constructs `ContextOptimizerSettings` with all of: `prefix_cache_max_entries`, `tier0_max_lines`, `tier0_head_lines`, `tier0_tail_lines`, `render_preview_chars`, `compaction_max_tokens`, `compaction_temperature`, `compaction_keep_alive` — sourced from proxy `Settings` (with new fields added there as needed).

### P3 — reliability
- [ ] `configure_logging` becomes a noop when called with the same `log_file` it was last configured with, even if `force=True`.
- [ ] `api/app.py` lifespan awaits `OllamaSupervisor.ensure_ready` with `asyncio.wait_for(..., timeout=settings.ollama_warmup_max_wait_s)`; `ollama_warmup_max_wait_s` defaults to 8s; on timeout proxy startup continues with a single `OLLAMA: warmup_timeout` warning.
- [ ] Every `compact_sync` / `_do_ollama_call` / `_compact_for_cache` failure path logs a `reason=...` with one of the documented enum values.

### P1 — observability
- [ ] `api/routes.py:create_message` emits `REQUEST: completed` on stream end (success and failure), with `duration_ms`, `output_tokens`, `finish_reason`, `error`.
- [ ] `api/routes.py:create_message` emits `REQUEST: provider_selected` immediately after provider resolution, with `provider`, `model`, `reason`.
- [ ] `request_id` is created at the top of `create_message` before any work; the entire handler runs inside `logger.contextualize(request_id=...)`; the response sets `X-Request-ID: <id>`.
- [ ] `providers/openai_compat.py` promotes `finish_reason` log to INFO; rebadges the existing `*_STREAM` start log to use the `PROVIDER:` prefix for grep consistency.
- [ ] `cli/session.py` caps the stderr dump (last 4 KiB) and emits it as `CLI_SESSION: stderr session_id=... bytes=... tail=...`.
- [ ] `api/app.py` shutdown step logs use `SHUTDOWN:` prefix; messaging warns use `MESSAGING:`; `providers/rate_limit.py` reactive limit uses `RATE_LIMIT:`.

### P4 — metrics
- [ ] `prometheus_client` added as an optional dependency.
- [ ] `GET /metrics` returns Prometheus exposition (text/plain, version 0.0.4) when `METRICS_ENABLED=1`; 404 otherwise.
- [ ] All counters and histograms documented above are populated by the request handler and the optimizer hooks.

### Validation
- [ ] `pytest` clean (no new failures from baseline).
- [ ] `ruff check .` clean.
- [ ] One end-to-end smoke run — fire a single message via `cc` in proxy mode, then `tail -f logs/server.log | jq` and verify a single request shows the full new prefix family: `REQUEST: start`, `REQUEST: provider_selected`, optional `CONTEXT_OPT: ...`, `PROVIDER: stream` (start), `PROVIDER: done`, `REQUEST: completed`.

## Technical Decisions

- Cache-before-cleanup (P2.2) changes the cache-key semantics: previously hashed post-cleanup messages, now hashed raw. Existing in-memory cache is bumped on rollout (one-time cold cache). Acceptable because the cache is in-process and never persisted.
- Prometheus chosen over OpenTelemetry: smaller dep, simpler exposition, fine for this single-process proxy. OTel can be layered later behind the same metric surface.
- `ollama_warmup_max_wait_s=8s`: long enough that `ollama serve` boot + small-model warm completes on a warm box, short enough that we don't block proxy startup if Ollama is missing.

## Progress

Tracked via `plans/please-see-what-improvements-warm-moth.md` and the in-session task list.
