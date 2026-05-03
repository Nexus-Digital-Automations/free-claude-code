---
title: DeepSeek V4 Support
status: completed
created: 2026-04-24
---

## Vision

Update the proxy to support DeepSeek V4 models (deepseek-v4-pro, deepseek-v4-flash), which launched April 24, 2026 with 1M context windows. Remove references to the retiring models (deepseek-chat, deepseek-reasoner retire July 24, 2026).

## Requirements

1. README config examples for the DeepSeek direct API provider use v4 model names
2. DeepSeek model list in README updated to v4 model names
3. `SUPPORTED_CLAUDE_MODELS` in routes.py includes current Claude 4.x model IDs (4.5, 4.6, 4.7) so Claude Code clients that send those IDs are handled
4. Thinking logic in `providers/deepseek/request.py` uses an extensible pattern for built-in reasoner models so adding future reasoner variants requires one-line changes

## Acceptance Criteria

- [x] README DeepSeek direct API example shows `deepseek-v4-pro` for Opus and `deepseek-v4-flash` for Sonnet/Haiku/fallback — no old model names remain (README:108-113)
- [x] README DeepSeek model list section lists `deepseek-v4-pro` and `deepseek-v4-flash` with a deprecation note for old names (README:382-385)
- [x] `SUPPORTED_CLAUDE_MODELS` includes `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` (api/routes.py:34-44)
- [x] `request.py` thinking check uses `not in _BUILTIN_REASONER_MODELS` frozenset instead of `!= "deepseek-reasoner"` (providers/deepseek/request.py:11,31)
- [x] `ruff` passes with no errors

## Technical Decisions

- Old model names (deepseek-chat, deepseek-reasoner) fully removed from examples per user preference — they retire July 24 anyway
- OpenRouter DeepSeek references (e.g. `open_router/deepseek/deepseek-r1-0528:free`) are unchanged — those are OpenRouter's model IDs, not DeepSeek direct API
- `.env.example` requires no changes — its MODEL_* vars already use nvidia_nim/open_router examples
