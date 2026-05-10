---
title: Per-project proxy model overrides via .claude/settings.json
status: in-progress
created: 2026-05-10
---

## Vision

Each project that points at the free-claude-code proxy can override its
model mapping (default / opus / sonnet / haiku) by adding a
`freeClaudeCode.models` block to its own `.claude/settings.json`. The
proxy daemon — which is shared across every project that runs Claude
Code on this machine — distinguishes requests by an
`X-Free-Claude-Project: <abs path>` HTTP header that Claude Code emits
via its native `ANTHROPIC_CUSTOM_HEADERS` mechanism. When the header is
absent or invalid, behavior is unchanged: the proxy falls back to the
existing global env-var mapping.

A second, smaller change: promote the model-mapping log line from DEBUG
to INFO and include the original (unresolved) model name plus the
project path in the existing `provider_selected` line, so "why did this
request hit the opus tier" becomes a single grep against
`logs/server.log`.

## Requirements

- Per-project config lives in `<cwd>/.claude/settings.json` (and
  `<cwd>/.claude/settings.local.json` for local overrides), under a new
  top-level key `freeClaudeCode.models` with optional sub-keys
  `default`, `opus`, `sonnet`, `haiku`.
- `.claude/settings.local.json` deep-merges over `.claude/settings.json`
  (local wins), matching Claude Code's own precedence.
- The proxy reads project settings only when an
  `X-Free-Claude-Project` header is present and the value resolves to
  an existing directory under the user's home directory.
- Project settings cache: keyed by absolute path; entry invalidated
  when either source file's mtime advances.
- Resolution precedence at request time, highest first:
  1. `freeClaudeCode.models.<tier>` from the project's settings (if the
     incoming model classifies as that tier)
  2. `freeClaudeCode.models.default` from the project's settings
  3. Existing global env-var fallback (`MODEL_OPUS` / `MODEL_SONNET` /
     `MODEL_HAIKU` / `MODEL`).
- `cc-provider --project init|set|show` writes / reads
  `<cwd>/.claude/settings.json` idempotently using `jq`.
- The mapping log (`api/models/anthropic.py` model_validator) emits at
  INFO when the resolved model differs from the original.
- The `provider_selected` log line in `api/routes.py` includes
  `original_model=...` and `project=...` fields.

## Acceptance Criteria


<!-- AUTO-CRITERIA START - managed by hooks/utils/quality_standards.py -->
- [x] Lint passes (zero errors)
- [ ] Unit tests pass
- [ ] Integration tests pass
- [x] Security scan clean; no secrets in diff
- [x] No stray output/, *.log, *.tmp, *.bak at project root
- [ ] All tracked changes committed and pushed
- [x] Branch synced with upstream
<!-- agentic-e2e: flip to [x] after exercising the feature in a live run -->
- [x] Direct agentic end-to-end: feature exercised in a live run
<!-- AUTO-CRITERIA END -->

- [x] `config/project_settings.py` exposes
  `load_project_settings(cwd: Path) -> ProjectSettings | None` that
  deep-merges `.claude/settings.json` and `.claude/settings.local.json`
  and validates the `freeClaudeCode.models` block.
- [x] Loader caches by absolute path; cache entry is invalidated when
  either source file's mtime advances.
- [x] `Settings.resolve_model(claude_model_name, project_cwd=None)`
  consults project settings first and falls back to env-var logic.
- [x] `api/dependencies.py:get_project_cwd_from_header` reads the
  `X-Free-Claude-Project` header, validates the path, sets the
  contextvar, and clears it on request exit (refuses paths outside
  `Path.home()` or paths that don't exist).
- [x] `/v1/messages` and `/v1/messages/count_tokens` apply per-project
  overrides when the header is present.
- [x] `api/routes.py` `provider_selected` log line includes
  `original_model` and `project` fields.
- [x] `scripts/cc-provider.sh --project init|set|show` writes the
  `freeClaudeCode.models` block and `ANTHROPIC_CUSTOM_HEADERS` env entry
  to `<cwd>/.claude/settings.json` idempotently.
- [x] New unit tests in `tests/config/test_project_settings.py` and
  `tests/api/test_project_header.py` cover loader caching, deep-merge,
  malformed JSON, header validation, and end-to-end resolution.
- [x] `pytest` green; `ruff check` clean; existing tests still pass.

## Technical Decisions

- **HTTP header instead of per-port proxies.** Single shared daemon is
  simpler operationally and matches existing deployment.
- **Contextvar instead of moving validation.** `MessagesRequest.map_model`
  (a Pydantic model_validator) can't read FastAPI request headers
  directly. Setting a contextvar in the request dependency before
  validation runs avoids restructuring two endpoints' request flow.
- **Header path must be under `Path.home()`.** Defense-in-depth: a
  request with a malicious `X-Free-Claude-Project: /etc` header should
  not be able to point the loader at arbitrary filesystem locations.
  The proxy is local-only, but this gate costs nothing.
- **`freeClaudeCode.models` block, not raw env-var keys.** Keeps proxy
  config separate from Claude Code's own settings schema, leaves room
  for future proxy-specific keys without polluting the `env` block.

## Progress

- 2026-05-10: spec drafted; plan approved
