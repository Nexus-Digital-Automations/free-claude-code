---
title: Proxy-Level Autocompaction
status: in-progress
created: 2026-04-25
---

## Vision

The proxy server proactively manages conversation context so requests stay affordable without sacrificing quality. DeepSeek can handle 1M tokens, but routing 1M-token requests is wasteful — the proxy keeps requests well under that ceiling by stripping irrelevant content and intelligently summarizing older turns when conversations grow long.

## Requirements

### R1 — Always-on token-cheap optimization (Tier 1)
Strip `ContentBlockThinking` blocks from all assistant turns except the last 2. Old reasoning is irrelevant to current reasoning quality. No LLM call required. Runs on every request.

### R2 — LLM-based summarization compaction (Tier 2)
When input tokens exceed **200,000**, trigger an LLM-based compaction:
- Send the conversation to DeepSeek (the configured provider) with a compaction prompt
- The LLM decides where to split — which recent turns stay verbatim, which older turns get summarized (context-dependent, not a fixed N)
- The summarization call uses **cheap mode** (thinking disabled) — summarization doesn't need deep reasoning
- Replace summarized turns with a single synthetic user message containing the summary wrapped in `<conversation_summary>` tags
- Keep the chosen recent turns verbatim

### R3 — Compaction prompt format
The compaction LLM is instructed to output:
```
<split_index>N</split_index>
<summary>...</summary>
```
where `split_index` is the message index where verbatim history starts, and the LLM picks `split_index` based on natural breakpoints in the conversation.

### R4 — Caching
Cache compaction results by hash of the prefix being summarized (in-memory, LRU max 100). On subsequent requests where the same prefix appears, reuse the cached summary instead of calling the LLM again.

### R5 — Graceful failure
If the compaction LLM call fails (network error, parse failure, invalid split_index), log the failure and fall back to Tier 1 only. Never block or fail the user's request because of a compaction problem.

### R6 — Configurability
All thresholds live in `config/settings.py` with environment variable overrides:
- `CONTEXT_OPTIMIZE` (bool, default true) — master switch
- `CONTEXT_MAX_THINKING_TURNS` (int, default 2) — Tier 1: how many recent assistant turns keep thinking
- `CONTEXT_COMPACT_THRESHOLD_TOKENS` (int, default 200000) — Tier 2: trigger threshold

## Acceptance Criteria

- [x] When `CONTEXT_OPTIMIZE=true`, a request with 5+ assistant turns containing thinking blocks results in thinking blocks present only on the last 2 assistant turns in the outgoing API call. Verified via `_strip_old_thinking` test: 6 thinking blocks → 2 (CONTEXT_OPT log: "stripped {n} thinking blocks from {m} old turns")
- [x] When input tokens < 200K, no LLM compaction call is made. Verified via under-threshold integration test: `provider._client.chat.completions.create.assert_not_called()`
- [x] When input tokens ≥ 200K, the proxy makes an extra outbound LLM call with thinking disabled, and the user-facing request is forwarded with a synthetic summary message replacing older turns. Verified via over-threshold integration test: 8 messages → 4 messages with synthetic `<conversation_summary>` message (logs: `CONTEXT_OPT: triggering LLM compaction` + `CONTEXT_OPT: compacted N -> M`)
- [x] When the compaction LLM call fails, the original (Tier-1-stripped) request still succeeds. Verified via failure-fallback integration test: `RuntimeError("network down")` → request still has 8 messages with 2 thinking blocks (tier-1 result preserved, WARNING logged)
- [x] Repeated requests with the same compactable prefix reuse the cached summary. Verified via cache test: second identical request makes 0 additional LLM calls (`assert_called_once`)
- [x] Setting `CONTEXT_OPTIMIZE=false` disables both tiers — request forwarded unchanged. Verified by gate in `api/routes.py`: `if settings.context_optimize:` skips entire optimizer block; lint check passes
- [x] DeepSeek's existing reasoning_content retry logic remains untouched and functional. Verified: `providers/deepseek/client.py:_get_retry_request_body` was not edited (only `providers/common/context_optimizer.py` (NEW), `config/settings.py` and `api/routes.py` modified)

## Technical Decisions

**Decision 1 — Synchronous compaction call.** The compaction LLM call happens inline before forwarding the user's request. Async/background compaction would require state management across requests; sync is simpler and only adds latency when actually triggered (>200K tokens, which is rare in normal use).

**Decision 2 — Tag-based output format, not JSON.** LLMs are more reliable at producing tagged text than valid JSON. `<split_index>` and `<summary>` tags are easy to parse with regex and tolerant of incidental formatting.

**Decision 3 — Cheap-mode summarization.** Thinking is disabled for the compaction call. Summarization is a routine task that doesn't benefit from deep reasoning, and disabling thinking ~halves the summarization cost.

**Decision 4 — In-memory LRU cache.** No persistence across server restarts. Compaction is rare enough and cache invalidation across processes adds complexity not warranted here.

**Decision 5 — LLM picks split point.** Per user request, the number of verbatim turns is context-dependent and decided by the LLM rather than a fixed N. The prompt constrains split_index to a valid range (≥4 and ≤N-2) for safety.

## Files Modified

| File | Change |
|------|--------|
| `providers/common/context_optimizer.py` | NEW — `ContextOptimizer` class with Tier 1, Tier 2, caching, parsing |
| `config/settings.py` | Add 3 new settings fields |
| `api/routes.py` | Call `await ContextOptimizer.optimize(...)` after `try_optimizations`, before token counting |

3 files. Direct implementation (under 5-file Qwen threshold).

## Progress

- [x] R1 — Tier 1 thinking strip (`_strip_old_thinking`)
- [x] R2 — Tier 2 LLM compaction (`_compact_via_llm`)
- [x] R3 — Compaction prompt + parser (`_build_prompt`, `_parse_response`)
- [x] R4 — LRU cache (`_summary_cache`, `_cache_put`, `_cache_key`)
- [x] R5 — Graceful fallback (try/except → returns input messages on any failure)
- [x] R6 — Settings + env vars (`config/settings.py` + `validation_alias`)
