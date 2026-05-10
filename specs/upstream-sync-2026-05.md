---
title: Upstream sync 2026-05 — cherry-pick triage
status: completed
created: 2026-05-10
---

## Vision

Reconcile this fork (Nexus-Digital-Automations/free-claude-code) with `upstream/main`
(Alishahryar1/free-claude-code) without disturbing the heavy custom work
(autocompaction, ollama-supervisor, repo-index, MCP digesters, tier0c-f filters).
A naive `git merge upstream/main` was attempted previously and produced 18
conflicts in load-bearing files; that merge was aborted in favor of cherry-pick
triage on a `upstream-sync-2026-05` branch.

## Outcome

**17 of 66 upstream commits cherry-picked cleanly.** Tests: 948 passing
(`pytest -o addopts='' --ignore=tests/providers/test_context_optimizer.py`).
Lint: clean (`ruff check`). One pre-existing collection error
(`tests/providers/test_context_optimizer.py` — `context_optimizer.cache` package
not installed in dev env) is unrelated to this sync.

After sync: `git rev-list --left-right --count upstream/main...HEAD` reports
**66 behind, 75 ahead**. The "behind" count is unchanged because git counts by
commit sha, not patch-id; `git cherry HEAD upstream/main` would show fewer
upstream-only patches.

## Cherry-picked (clean apply)

Chronological order, original SHA → new SHA on `upstream-sync-2026-05`:

| Upstream | Local | Subject |
|----------|-------|---------|
| 180c942 | 9261c54 | docs: fix quick-start uv installation instructions (#191) |
| 20a4d8c | 7220392 | Update fastapi[standard] requirement (#175) |
| 837374a | 30a9509 | Update httpx[socks] requirement (#174) |
| c7fe66d | a033ed8 | Bump astral-sh/setup-uv 7.3.0 → 8.1.0 (#172) |
| 4b89183 | 4eeb0b0 | Raise default http connect timeout to 10s |
| d2db1bd | d2eaaa8 | Treat empty model overrides as fallback |
| 751694a | aa0ddd4 | Refactor smoke testing framework and enhance provider configurations |
| ffa8237 | 73b7109 | Refactor native Anthropic messages providers (#147) |
| efa9f36 | 18b43a1 | Revert "Refactor native Anthropic messages providers (#147)" |
| 2df42c1 | ae392a8 | Strengthen validation logic in NimSettings (#180) |
| 080ebef | 57651ef | fix: detect Claude Code 2.1+ session title requests |
| 19ce656 | d655279 | fix: handle env-prefixed commands in filepath extraction (#207) |
| 0cca569 | b1538ec | fix(messaging): reuse parent CLI session for Telegram (#233) |
| 51112a4 | 6584b39 | fix: only strip valid env assignments in command parsing (#229) |
| 0294e04 | 93c08cd | Update .env.example |
| b8f8508 | 0f46bc7 | fix(security): constant-time comparison for ANTHROPIC_AUTH_TOKEN (#262) |
| 1dfa54a | 62f7ad6 | Update .env.example |

**Notable inclusions:**
- **#262 constant-time auth-token compare** — security fix for timing attack.
- **NimSettings validation** — better startup errors for misconfigured NIM.
- **Telegram session reuse, env-prefixed command parsing** — messaging stability.
- **fastapi/httpx/setup-uv bumps** — keep deps current with upstream.

The `ffa8237 → efa9f36` refactor+revert pair adds two no-op commits to our
history. Harmless but not load-bearing; could be squashed if desired.

## Skipped — architecture divergence (43 commits)

These all conflict because the fork branched **before** upstream's
`26b8a29 Architecture refactor: core anthropic, runtime, smoke tiers, remove
providers.common`. We don't have `core/anthropic/`, `api/runtime.py`,
`api/web_tools/`, or the post-refactor `providers/{kimi,wafer,ollama}/`
directories. Custom autocompaction + ollama-supervisor + repo-index code is
built on the *pre-refactor* file shape, so any upstream commit that touches
the new layout fights our custom work.

| SHA | Subject | Conflict surface |
|-----|---------|------------------|
| 7d80cc3 | Add star chart to readme | README.md |
| 053beff | Updated README | README.md |
| 9996a3d | Update contribution guidelines in README | README.md |
| 89d6a79 | Update README | README.md |
| 43a3cc9 | Update README | README.md |
| 5706d00 | Remove redundant powershell command in README (#354) | README.md |
| 4102373 | Update installation instructions in README | README.md |
| 97a88af | Updated README | README.md |
| b9ed704 | Added Ollama to README | README.md |
| 6b8c697 | add kimi to readme (#361) | README.md |
| b0f5a49 | docs: link Wafer provider site | README.md |
| d78869d | Removed cached models list | README.md |
| 15415d5 | Bump minor-and-patch group with 17 updates (#173) | uv.lock |
| 29c3150 | Bump minor-and-patch group with 3 updates (#314) | pyproject.toml |
| 07b30aa | Bump minor-and-patch group with 5 updates (#389) | pyproject.toml |
| 72b34ad | Added claude-code native model picker | README.md (image add) |
| 07bc9a3 | Updated model picker image | png deleted in HEAD |
| 3e9d1eb | updated model picket image again | png deleted in HEAD |
| 3bde98a | Raise default timeouts for write and connect | .env.example |
| c521589 | feat: add Kimi (Moonshot) provider (#335) | .env.example |
| 5294661 | feat: add Wafer provider | .env.example, settings.py, smoke/ |
| 8687fb3 | fix: accept betas body field (#360) | api/models/anthropic.py |
| 19e08f2 | fix(deepseek): document blocks and tool_result content (#358) | api/models/anthropic.py, deepseek/request.py |
| 847916b | fixed deepseek issue | providers/deepseek/request.py |
| 2d2bf3d | fix: replay reasoning_content for DeepSeek/NIM | providers/common/, deepseek/, nvidia_nim/ |
| 6297b48 | feat(deepseek): use native Anthropic Messages transport | providers/deepseek/, openai_compat.py |
| 36d236b | fix(206): defer post-tool assistant content for OpenAI conversion | providers/common/sse_builder.py, openai_compat.py |
| f96f541 | fix(smoke): accept reasoning-only streams | smoke/test_provider_live.py |
| abae61d | Fix null usage in SSE for OpenAI-compat (#209, #123) | providers/common/sse_builder.py |
| eb5516e | Validate configured models at startup | config/settings.py, providers/__init__.py |
| 85232a3 | Log startup model validation failures clearly | tests/api/test_app_lifespan_and_errors.py |
| d9040ce | Report startup validation failures without tracebacks | api/app.py, server.py |
| db3c952 | Add no-thinking model picker variants | api/routes.py, README.md |
| d3a3b37 | Filter OpenRouter model variants by thinking support | api/routes.py, providers/base.py, open_router/client.py |
| f29e693 | Add per-model thinking toggles | config/env.example, providers/base.py, deepseek/client.py |
| 2e4a4fe | fix(smoke): avoid nested uv run on Windows | smoke/README.md |
| 2283772 | Removed PLAN.md | (we have a custom PLAN.md) |
| 7f1e860 | Use root env example for fcc init | config/env.example (we deleted this file) |
| 40951c1 | refactor: drop legacy title-generation detection copy | smoke/product/test_api_product_live.py (deleted in HEAD) |
| b525217 | [feat] ollama method support (#129) | config/settings.py |
| 07497c7 | Add NVIDIA NIM CLI smoke matrix and tool schema aliasing | .env.example, providers/nvidia_nim/ |
| de8e902 | Add Claude CLI smoke matrices | .env.example, smoke/ |
| 26b8a29 | **Architecture refactor: core anthropic, runtime, smoke tiers** | 12 files including settings.py, deepseek/, openai_compat.py — root cause of all conflicts below |
| 66ef230 | Refactor provider routing and smoke coverage | api/routes.py, providers/openai_compat.py |
| 0e3b2c2 | refactor: remove OpenRouter rollback, shims, redundant layers | 17 files — depends on 26b8a29 |
| b926f60 | feat: Anthropic web server tools, provider metadata | 18 files — depends on 26b8a29 |
| f3a7528 | Major refactor: API, providers, messaging, Anthropic protocol | 38 files — depends on 26b8a29 |

**Why these aren't worth resolving:** The architecture refactors (26b8a29,
f3a7528, 0e3b2c2, b926f60, 66ef230) collectively rewrite ~50 files we've
heavily customized. Resolving them would mean re-porting autocompaction,
ollama-supervisor, repo-index, and MCP digesters onto the new layout —
days of work, with high risk of silent regressions. The provider/feature
commits that depend on the refactor (Wafer, Kimi, native deepseek-anthropic,
per-model thinking, web server tools) are blocked by the same.

## Open follow-ups

- **Provider parity** — we lack the post-refactor `kimi`, `wafer`, and
  `ollama` providers from upstream. If those are needed, the work is
  re-implementation against our pre-refactor shape, not a sync.
- **Architecture rebase** — at some point the fork either rebases onto
  upstream's post-refactor architecture or formally diverges. Postponing.
- **Deps drift** — pyproject.toml deps bumps (#173, #314, #389) skipped
  due to local additions; our resolver re-derives. Run `uv lock --upgrade`
  if explicit refresh is needed.
- **Pre-existing test collection error** —
  `tests/providers/test_context_optimizer.py` imports
  `context_optimizer.cache` which isn't installed via the project's dev
  env. Pre-dates this sync; tracked separately.

## Verification

```bash
# Run on the upstream-sync-2026-05 branch:
git rev-list --left-right --count upstream/main...HEAD   # 66 75
ruff check api config providers messaging cli smoke      # 0 findings
uv run pytest -o addopts='' \
  --ignore=tests/providers/test_context_optimizer.py -q  # 948 passed
```
