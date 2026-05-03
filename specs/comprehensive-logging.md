---
title: Comprehensive Structured Logging
status: completed
created: 2026-04-25
---
## Vision
Every observable decision made by the proxy — request routing, context optimization tiers, provider calls, CLI session events — is captured as a structured JSON INFO entry in `logs/server-<timestamp>.log`. Operators can tail the log and immediately see whether the system is working as intended.

## Requirements
1. Log file lives in `logs/` not at project root
2. Context optimizer logs each tier that fires, token counts before/after, cache hit/miss, Ollama availability
3. Request pipeline logs: request received (model, provider, token count), optimization result, provider response time, response token count
4. Provider layer logs: call start, latency, finish reason, errors with full context
5. CLI session logs: session start/resume, task completion, exit code, stderr when present

## Acceptance Criteria
- [x] `settings.log_file` defaults to `logs/server.log`; `logging_config.configure_logging` creates `logs/` if absent (config/settings.py:227, config/logging_config.py:80)
- [x] `logs/` directory listed in `.gitignore` (already present) — no log files committed
- [x] `routes.py` logs request received (request_id, model, provider, input_tokens) at INFO (REQUEST: start)
- [x] `routes.py` logs request completed (request_id, duration_ms) at INFO (REQUEST: completed via _instrumented_stream)
- [x] `context_optimizer.py` logs per-tier decisions: Tier 0 bytes_saved, Tier 1 blocks_stripped, cache hit with k value, Tier 2a scheduled/skipped, Tier 2b triggered — all at INFO
- [x] `openai_compat.py` logs provider call completion (latency_ms, finish_reason, output_tokens estimate) at INFO (PROVIDER: stream_start, finish_reason, done)
- [x] `cli/session.py` logs session_start/session_resume with workspace and session_id at INFO, exit at INFO with code (CLI_SESSION: start|resume|session_id_captured|exit|stderr|stopping)
- [x] Running `tail -f logs/server.log | jq` shows human-readable trace of a request flowing through the system
- [x] `FULL_PAYLOAD` (the entire request body sent by Claude Code: system prompt, tools, messages) emitted at INFO so it appears in default-level scrapes — opt-out via `LOG_FULL_PAYLOAD=0` because it is verbose (api/routes.py, config/settings.py)
- [x] `PROVIDER: chunk` per-event DEBUG log records each SSE delta the upstream sends back, enabling reverse-engineering of streaming behavior when `LOGURU_LEVEL=DEBUG` (providers/openai_compat.py)

## Technical Decisions
- Reuse existing `loguru` + `_serialize_with_context` infrastructure — no new dependencies
- Keep all new log lines at INFO (not DEBUG) so they appear in the default log level
- Prefix context-optimizer entries with `CONTEXT_OPT:` (already established convention)
- Prefix request-pipeline entries with `REQUEST:` for easy grep
- Prefix provider entries with `PROVIDER:` (rebadged from `*_STREAM:` in this round)
- Prefix CLI-session entries with `CLI_SESSION:` (already partially done)
- Prefix shutdown / messaging / rate-limit entries with `SHUTDOWN:`, `MESSAGING:`, `RATE_LIMIT:` (added in this round to convert noisy f-strings)

## Progress
- [x] settings.py: change log_file default
- [x] logging_config.py: mkdir parents before creating file
- [x] routes.py: request received + completed log points
- [x] context_optimizer.py: per-tier INFO entries (enhance existing)
- [x] openai_compat.py: completion summary log entry + finish_reason promoted to INFO
- [x] cli/session.py: session_start/resume/exit structured entries + stderr capped and structured
- [x] api/app.py: structured shutdown/messaging entries
- [x] providers/rate_limit.py: structured RATE_LIMIT: entries
