---
title: Autocompaction Improvements (Tier 1)
status: active
created: 2026-04-26
---

## Vision

Targeted improvements to `providers/common/context_optimizer.py` across four
axes: cost, latency, quality, reliability. Source plan:
`~/.claude/plans/please-look-into-improvements-starry-puddle.md`. This spec
covers Tier 1 items only — Tier 2/3 will get separate specs if and when shipped.

## Requirements

### R1.1 — Eliminate double `get_token_count` on the request path
The proxy currently calls `get_token_count` twice per request (full tiktoken
pass each). After this change, it is called at most once outside `optimize()`.
`ContextOptimizer.optimize` returns the final token count alongside the
mutated `request_data`.

### R1.2 — Reframe summary injection
Compaction summary moves from a synthetic `<conversation_summary>` user
message into the system prompt as an "Earlier conversation (compacted): …"
block. This reduces the chance Claude treats the summary as a new user
instruction.

### R1.3 — Sharpen the compaction prompt
`_build_prompt` adds an explicit "preserve verbatim" enumeration covering
file paths, function names, error messages, user-stated goals, and open
questions, plus one worked example showing a good split.

### R1.4 — Unit tests for ContextOptimizer
Add `tests/providers/test_context_optimizer.py` with one happy-path and one
failure-mode test for each tier (Tier 0 ANSI/dedup, Tier 1 strip, prefix
cache, response parse, Ollama network error).

## Acceptance Criteria

- [x] `optimize()` returns `tuple[MessagesRequest, int]`; no breaking caller
      changes outside `api/routes.py:create_message`.
- [x] `api/routes.py` calls `get_token_count` exactly once before optimize
      and uses the optimizer's returned count for `REQUEST: optimized` log.
- [x] Sync compaction places the summary in `system` (string or list); the
      first message after compaction is no longer a synthetic user message
      with `<conversation_summary>` tags.
- [x] `_build_prompt` output contains the preserve-verbatim enumeration and
      one example split.
- [x] `tests/providers/test_context_optimizer.py` exists with ≥5 tests, all
      passing under `uv run pytest tests/providers/test_context_optimizer.py -v`.
      (10 tests, all green.)
- [x] Full test suite (`uv run pytest tests/ -x -q`) and lint
      (`uv run ruff check .`) pass. (924 passed, ruff clean.)

## Technical Decisions

- **Return tuple, not in-place mutation.** `optimize()` already returns a
  copied `request_data`; adding the int alongside is the minimum-disruption
  change. Rejected: storing token count on `request_data` itself (pollutes
  the model with derived state).
- **System-block prepending, not replacement.** Existing `request_data.system`
  is preserved; the summary is inserted as a leading block. Avoids destroying
  user-supplied system prompts.
- **Tests use mocked AsyncOpenAI client.** No real network calls. Mock
  responses include both happy path and parse-failure cases to lock the
  failure-fallback contract.

## Files Modified

| File | Change |
|------|--------|
| `providers/common/context_optimizer.py` | `optimize` returns tuple; `_apply_summary` takes/returns `system`; `_build_prompt` text update |
| `api/routes.py` | call-site update for new `optimize` signature |
| `tests/providers/test_context_optimizer.py` | NEW — unit tests |

## Progress

- [x] R1.1 — token-count single-pass
- [x] R1.2 — summary-as-system
- [x] R1.3 — prompt sharpening
- [x] R1.4 — unit tests
