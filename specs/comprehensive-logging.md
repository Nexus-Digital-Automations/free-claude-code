---
title: Comprehensive Structured Logging
status: active
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
- [ ] `settings.log_file` defaults to `logs/server.log`; `logging_config.configure_logging` creates `logs/` if absent
- [ ] `logs/` directory listed in `.gitignore` (already present) — no log files committed
- [ ] `routes.py` logs request received (request_id, model, provider, input_tokens) at INFO
- [ ] `routes.py` logs request completed (request_id, duration_ms) at INFO
- [x] `context_optimizer.py` logs per-tier decisions: Tier 0 bytes_saved, Tier 1 blocks_stripped, cache hit with k value, Tier 2a scheduled/skipped, Tier 2b triggered — all at INFO
- [ ] `openai_compat.py` logs provider call completion (latency_ms, finish_reason, output_tokens estimate) at INFO
- [ ] `cli/session.py` logs session_start/session_resume with workspace and session_id at INFO, exit at INFO with code
- [ ] Running `tail -f logs/server.log | python3 -c "import sys,json; [print(json.loads(l)['message']) for l in sys.stdin]"` shows human-readable trace of a request flowing through the system

## Technical Decisions
- Reuse existing `loguru` + `_serialize_with_context` infrastructure — no new dependencies
- Keep all new log lines at INFO (not DEBUG) so they appear in the default log level
- Prefix context-optimizer entries with `CONTEXT_OPT:` (already established convention)
- Prefix request-pipeline entries with `REQUEST:` for easy grep
- Prefix provider entries with `PROVIDER:` (already partially done)
- Prefix CLI-session entries with `CLI_SESSION:` (already partially done)

## Progress
- [ ] settings.py: change log_file default
- [ ] logging_config.py: mkdir parents before creating file
- [ ] routes.py: request received + completed log points
- [ ] context_optimizer.py: per-tier INFO entries (enhance existing)
- [ ] openai_compat.py: completion summary log entry
- [ ] cli/session.py: session_start/resume/exit structured entries
</content>
</invoke>