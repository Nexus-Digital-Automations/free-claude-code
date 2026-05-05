---
title: Encourage parallel tool calls on DeepSeek via a system-prompt nudge
status: planning
created: 2026-05-05
---

## Context

The proxy already sets `parallel_tool_calls=True` on every DeepSeek request that carries tools (shipped in commit `451b88d`). Empirically, ~38% of tool-bearing responses pack 2+ parallel `tool_calls` (max observed: 9). The remaining ~37% emit exactly one tool call, and a meaningful fraction of those are likely parallelisable misses (independent `Read`/`Grep`/`Glob` calls issued sequentially).

Each saved round-trip = one less full-history shipped to DeepSeek. With Claude Code sessions averaging ~50K input tokens per turn, halving the round-trip count on a multi-tool task is the largest single lever for further DeepSeek-bill reduction.

OpenAI's `parallel_tool_calls` *permits* multi-tool emission; it doesn't *encourage* it. The model decides per-turn based on its trained policy + the system prompt. Adding a short, factual nudge to DeepSeek's system message tells the model that batching independent calls is preferred — the same mechanism Anthropic uses in Claude Code's own prompt.

User selections (from this planning round): **DeepSeek only**, **settings flag with default ON**.

## Files to modify

1. **`config/settings.py`** — add a single field:
   ```python
   deepseek_parallel_tool_call_nudge: bool = Field(
       default=True,
       validation_alias="DEEPSEEK_PARALLEL_TOOL_CALL_NUDGE",
   )
   ```
   Disambiguated name (vs the OpenAI knob `parallel_tool_calls` — separate concept). Flag is a kill-switch in case the nudge introduces parallel-call regressions.

2. **`providers/deepseek/client.py`** — `DeepSeekProvider.__init__` accepts a new keyword-only `parallel_tool_call_nudge: bool = True`, stored as `self._nudge_enabled`. `_build_request_body` passes it to `build_request_body`.

3. **`api/dependencies.py:80-97`** (the `provider_type == "deepseek"` block) — pass `parallel_tool_call_nudge=settings.deepseek_parallel_tool_call_nudge` to `DeepSeekProvider`.

4. **`providers/deepseek/request.py`** — define a fixed nudge constant and append it to `body["messages"][0]["content"]` when:
   - `parallel_tool_call_nudge` is True, AND
   - `body.get("tools")` is non-empty (same gate as `parallel_tool_calls=True`), AND
   - `body["messages"]` has a system message at index 0.

   Nudge string (~70 tokens, kept deliberately short to limit prefix overhead):
   ```
   Tool-call parallelism: when multiple tool calls in a turn are independent
   (no data dependency between them), emit them all in a single response so
   they can run in parallel. Examples of independent calls: reading several
   unrelated files, running multiple greps with different patterns, looking
   up several pieces of context. Sequential calls are only required when one
   tool's output is needed as input to the next.
   ```

   Append it (with a leading `\n\n` separator) to the existing system content. If `messages[0]` isn't a system message, **skip** — never inject a new system message that wasn't already there (would mutate the prefix shape in a way the user didn't ask for).

## Cache-stability note

The nudge becomes part of the system prefix. Because the gate uses the same condition as `parallel_tool_calls=True` (tools present), the nudge fires consistently for every real Claude Code request — tool-less title-gen / suggestion calls are already mocked-out by the proxy, so they never reach DeepSeek. The prefix is therefore byte-stable across all requests of the same session — no new cache invalidation.

## Why DeepSeek-only

NIM/OpenRouter/local providers don't have a measured baseline for multi-tool rate. Diff stays small, rollback is clean. If the nudge proves out on DeepSeek, the same constant + flag pattern can be lifted to other providers in a follow-up.

## Acceptance Criteria

- [ ] **AC1**: `DeepSeekProvider` accepts `parallel_tool_call_nudge: bool = True` as a keyword-only init argument; stored on the instance.
- [ ] **AC2**: `api/dependencies.py` passes `settings.deepseek_parallel_tool_call_nudge` through to the provider constructor when `provider_type == "deepseek"`.
- [ ] **AC3**: With `parallel_tool_call_nudge=True` and tools in the request, `build_request_body()` produces a body whose `messages[0]["content"]` ends with the nudge string (verified via `endswith` on the trailing chunk).
- [ ] **AC4**: With `parallel_tool_call_nudge=True` and **no** tools, the system content is unchanged (no nudge).
- [ ] **AC5**: With `parallel_tool_call_nudge=False` and tools present, the system content is unchanged (no nudge).
- [ ] **AC6**: With tools present but **no** system message at `messages[0]`, no system message is inserted; body shape unchanged.
- [ ] **AC7**: Existing 8 DeepSeek tests stay green.
- [ ] **AC8**: `ruff check providers/deepseek/ config/settings.py api/dependencies.py tests/providers/test_deepseek.py` exits 0.
- [ ] **AC9**: Setting `DEEPSEEK_PARALLEL_TOOL_CALL_NUDGE=false` in env disables the nudge end-to-end (verifiable by reading the resolved Settings instance).

## Verification

1. **Unit tests**: `pytest tests/providers/test_deepseek.py -q -o addopts=""` — 12/12 pass (8 prior + 4 new for AC3/AC4/AC5/AC6).
2. **Lint**: see AC8 above.
3. **Empirical baseline check** (manual, post-merge): drive a real Claude Code session for ~15 min on exploration-heavy work (lots of `Read`/`Grep`/`Glob` against unrelated files — these are the highest-yield parallel-call opportunities). Run the same per-`request_id` histogram script from the prior log analysis on the new logs:
   ```
   regex: request_id=(req_[a-f0-9]+) provider=DEEPSEEK
   regex: ChoiceDeltaToolCall(index=(\d+)
   bucket: max_index per request_id, +1 = tool_calls per response
   ```
   Compare distributions before/after. Expected: multi-tool fraction rises from 38% to 50–60% on exploration-heavy work; round-trip count per task drops correspondingly.
4. **Rollback**: `DEEPSEEK_PARALLEL_TOOL_CALL_NUDGE=false` env var disables the nudge without a code change.

## What this plan is NOT

- Not a NIM/OpenRouter/LM Studio/llamacpp change (DeepSeek-scoped per your selection).
- Not a `tool_choice` / `seed` / `n` change — those don't influence parallel emission directly.
- Not a system-prompt rewrite — purely additive, ~70 tokens at the tail of the existing system content.
- Not a tier 0/0b/0c/0d / block-tower / repo-index change. Disjoint from the compaction system.
- Not measurement automation — the verification step is a manual log re-run for now. If the nudge proves out, an automated A/B in CI is a separate spec.
