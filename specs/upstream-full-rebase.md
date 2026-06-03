# Full Rebase Plan — Re-platform the fork onto upstream's post-`26b8a29` architecture

## Context

The fork (`Nexus-Digital-Automations/free-claude-code`) branched from upstream
(`Alishahryar1/free-claude-code`) **before** upstream's architecture refactor (`26b8a29`
"core anthropic, runtime, smoke tiers, remove providers.common", plus follow-ons `66ef230`,
`0e3b2c2`, `f3a7528`). The `2026-05` and `2026-06` cherry-pick syncs have now absorbed every
**layout-independent** upstream commit; everything of value left (tiktoken #382, structured
TRACE logging, 5xx retry, SSE fixes, CLI teardown, and a dozen new providers + admin UI) is
welded to the new layout. Incremental cherry-picking is exhausted (see
`specs/upstream-sync-2026-06-triage.md`). This plan re-platforms the fork onto upstream's
current architecture so future syncs become trivial again.

**Intended outcome:** the fork's custom features run on upstream's `core/anthropic` +
`api/runtime`/`api/services` layout; `main` ends up a small, legible delta over `upstream/main`
(≈10–15 feature commits instead of 98 divergent ones); whole-repo gate stays green.

## Strategy decision — re-platform, not `git rebase`

A literal `git rebase upstream/main` replays ~98 fork commits across the refactor boundary →
conflicts on nearly every commit, most touching files that no longer exist. **Rejected.**

**Chosen:** branch `rebase/onto-upstream` from `upstream/main` (a known-good tip), then
re-apply the fork's custom features as a **fresh, logical commit series** (one commit per
feature area), validating after each. The old `main` is preserved as a tag for fallback. Cut
over only when the new branch is green. This treats the custom work as a feature set to port,
not a history to replay — which is what it actually is.

## What the fork gains from upstream

Adopting the new layout pulls in, for free: `core/anthropic/` (sse, thinking, tokens,
conversion, **stream_recovery** mid-stream retry + tool-JSON repair, native-messages request
builder), `core/trace.py` structured tracing, `core/rate_limit.py`, `api/runtime.py` +
`api/services.py` clean lifecycle/dispatch, modular `api/web_tools/`, the **admin UI**, the
tiktoken special-token fix, 5xx/503 retry, and ~10 new providers (mistral, fireworks, zai,
groq, gemini, cerebras, codestral, opencode, wafer, kimi).

## Custom-feature inventory → new home

Effort is re-port effort onto the new layout. "Copy" = directory has no proxy-internal imports.

| # | Feature | Current location | New home on upstream layout | Effort |
|---|---|---|---|---|
| 1 | **context-optimizer package** | `packages/context-optimizer/` | copy verbatim (self-contained) | Copy |
| 2 | context-optimizer **adapter** | `providers/common/context_optimizer.py` | **new** `core/context/optimizer_adapter.py` (providers.common is gone) | Med |
| 3 | **ollama supervisor** shim | `providers/common/ollama_supervisor.py` | re-export from new `core/context/` | Low |
| 4 | repo-index / block-tower | inside pkg (1) | copy verbatim | Copy |
| 5 | MCP digesters | `packages/{build,lint,test}-digester/`, `lsp-bridge`, `memory-recall` | copy verbatim | Copy |
| 6 | **Vertex provider** | `providers/vertex/` (ADC auth) | port to new provider pattern (`OpenAIChatTransport` subclass + `registry.py` factory entry); reconcile with upstream's new `gemini/` provider | Med |
| 7 | **per-project model overrides** | `config/project_settings.py`, `api/context.py`, deps + `resolve_model()` | re-wire into `api/model_router.py` / `api/dependencies.py` (resolution moved out of routes) | Med-High |
| 8 | cc-provider / cc-status scripts | `scripts/` | copy; refresh provider-list validation | Low |
| 9 | request optimizations | `api/optimization_handlers.py` | **reconcile** — upstream has its own; merge fork-only handlers (quota mock, title/suggestion skip, filepath mock) | Med |
| 10 | **messaging** (trees, platforms) | `messaging/` (+ `api/app.py` lifespan) | re-wire init into `api/runtime.py` `AppRuntime` (highest risk) | **High** |
| 11 | `context_*` settings (30+) | `config/settings.py` | merge into upstream `config/settings.py`, exact field names | Low |
| 12 | per-project / title-skip / auth fixes | scattered in routes/deps | fold into new `services.py`/`dependencies.py` | Low |

Total ≈ 7–9 hours of focused work; risk concentrated in #7, #9, #10.

## Key old→new mapping (the parts the port must respect)

- `providers/common/*` → **deleted**. Split into `core/anthropic/{sse,thinking,tokens,conversion,tools,utils,content}.py` + `providers/error_mapping.py`. Any fork code importing `providers.common.*` must repoint.
- `api/request_utils.py` (fork's token-count + routing home) → split into `api/services.py` (`ClaudeProxyService.create_message`), `api/model_router.py`, `api/dependencies.py`. **The fork's compaction call moves here** — `ContextOptimizer.optimize()` now belongs inside `ClaudeProxyService.create_message`, *before* `provider.stream_response`, with token counting via `core/anthropic/tokens.get_token_count`.
- `api/app.py` inline lifespan/init → `api/runtime.py` `AppRuntime`. **Messaging + Ollama warm-up re-wire here.**
- provider instantiation (fork's `api/dependencies.py` if/elif branches) → `providers/registry.py` `PROVIDER_FACTORIES` + `config/provider_catalog.py`. **Vertex registers here.**
- `api/metrics.py` (fork's COMPACTION_* counters) → upstream uses `core/trace.py`; port compaction metrics onto the trace module or keep a thin metrics shim.

## Phased execution

Each phase ends with `ruff check` + targeted `pytest` green before the next begins.

1. **Prep & fallback.** Tag current `main` as `pre-rebase-2026-06`. Branch `rebase/onto-upstream` from `upstream/main`. Confirm upstream tip + record SHA.
2. **Copy-only packages (#1,4,5).** Drop in `packages/context-optimizer/` and the digesters; wire into root `pyproject.toml` workspace. Run the package's own test suite in isolation (it has no proxy deps).
3. **Settings merge (#11).** Add all `context_*`, `ollama_*`, `vertex_*` fields into upstream `config/settings.py`; resolve any name collisions; add `[vertex]`/`[ml]` optional extras.
4. **Context-optimizer adapter + Ollama (#2,3).** Create `core/context/optimizer_adapter.py` (port the 103-line shim, repointing `providers.common` imports to `core.anthropic`). Wire `ContextOptimizer.optimize()` into `ClaudeProxyService.create_message`; wire Ollama warm-up into `AppRuntime`. Port compaction metrics onto `core/trace.py`.
5. **Vertex provider (#6).** Port `providers/vertex/` to the new transport base + register in `registry.py`/`provider_catalog.py`. Decide overlap with upstream `gemini/` (keep both; vertex = Model-Garden Gemma via ADC, gemini = native). Run `tests/providers/test_vertex.py`.
6. **Per-project overrides (#7).** Port `config/project_settings.py` + `api/context.py`; move resolution into `api/model_router.py`; wire the `X-Free-Claude-Project` header dep into the new route signatures. Run override tests.
7. **Request optimizations (#9).** Diff fork vs upstream `optimization_handlers.py`; merge the fork-only handlers; wire into `services.py`.
8. **Messaging (#10) — highest risk.** Copy `messaging/`; re-implement the lifespan wiring inside `AppRuntime` (platform factory, `CLISessionManager`, tree-state restore, graceful shutdown). **Contingency:** if integration is incompatible, gate behind `MESSAGING_ENABLED=false` and land it in a follow-up rather than block the rebase.
9. **Scripts (#8) + docs.** Copy cc-provider/cc-status; refresh provider list. Write `MIGRATION.md` noting moved paths.
10. **Whole-repo validation.** `ruff check api config providers messaging cli smoke core`; `uv run pytest -o addopts='' -q` (target parity with current 981+); smoke matrix for deepseek/vertex/openrouter/nim/lmstudio.
11. **Cutover.** Fast-forward/replace `main` with `rebase/onto-upstream` (merge or reset-and-force with the `pre-rebase-2026-06` tag as the safety net), push `origin main`.

## Risk register

- **Messaging lifespan re-wire (HIGH):** deeply coupled to old `api/app.py`. Mitigation: `MESSAGING_ENABLED` flag + defer to follow-up if needed.
- **Per-project override resolution (MED):** resolution moved from routes → `model_router`; the ContextVar header bridge must survive. Mitigation: port the bridge as-is, add the header dep in `routes.py` only.
- **`optimization_handlers.py` divergence (MED):** both sides have one. Mitigation: 3-way diff, keep fork-only handlers, drop duplicates upstream already covers.
- **Config field collisions (LOW):** upstream may have added overlapping settings. Mitigation: explicit collision scan in Phase 3.
- **Compaction metrics (LOW):** `api/metrics.py` gone. Mitigation: port onto `core/trace.py`.

## Rollback

`pre-rebase-2026-06` tag preserves the exact current `main`. Cutover is the only destructive
step; until then all work is on `rebase/onto-upstream`. If validation can't go green within the
budget, abandon the branch — `main` is untouched.

## Resolved decisions (confirmed 2026-06-02)

1. **Vertex + Gemini — keep both.** Port fork `providers/vertex/` AND adopt upstream `providers/gemini/`.
2. **Admin UI — adopt** upstream's `api/admin_*.py` + static assets as-is.
3. **Messaging — hard gate.** No `MESSAGING_ENABLED` flag escape; messaging (Discord/Telegram + conversation trees) must be fully wired into `AppRuntime` and green before cutover. Risk register updated: messaging blocks cutover.
4. **Cutover — reset `main` + tag.** Tag current `main` as `pre-rebase-2026-06`, reset `main` to the rebased branch (clean linear delta). Force-push confirmed with the user immediately before it runs.

## Acceptance criteria

- [ ] `rebase/onto-upstream` is based on `upstream/main` (`544008a` or newer) and contains the fork's custom feature set as a clean per-feature commit series.
- [ ] No `providers.common.*` imports remain anywhere in fork code (all repointed to `core.anthropic.*` / `core.context.*`).
- [ ] Context optimizer fires from inside `ClaudeProxyService.create_message` before provider dispatch; compaction metrics emitted via `core/trace.py`; all `context_*` settings present with identical names/defaults.
- [ ] Both `vertex` and `gemini` providers resolve and stream; `tests/providers/test_vertex.py` green.
- [ ] Per-project model overrides resolve via the new `model_router` path; `X-Free-Claude-Project` header honored; project + header tests green.
- [ ] Messaging starts under `AppRuntime` (platform factory, CLI session manager, tree-state restore, graceful shutdown); a Discord/Telegram round-trip works; tree restore verified.
- [ ] Admin UI reachable; cc-provider/cc-status scripts work against the new provider list.
- [ ] Whole-repo gate green on the branch: `ruff check` (excluding vendored `docs/headroom/`), `uv run pytest -o addopts='' -q` at parity with current 981+ passing, provider smoke matrix for deepseek/vertex/gemini/openrouter/nim/lmstudio.
- [ ] Working tree clean; `pre-rebase-2026-06` tag preserves rollback point.

## Execution progress & findings (2026-06-02)

Branch `rebase/onto-upstream` (from `upstream/main` `544008a`), rollback tag
`pre-rebase-2026-06` @ `4a203f4`. `main` untouched.

- **P1 done** — prep, tag, branch.
- **P2 done** (`d77ed5f`) — 6 fork packages vendored; `context-optimizer` wired as
  editable dependency; `uv lock`/`sync` clean; `symbol_graph` suite (18) green.
- **P3 done** (`5dad384`) — 36 fork settings merged; `Settings` loads; ruff clean.

Findings that change the remaining work:

1. **`ollama_base_url` collision** — upstream now ships its own native ollama provider
   with `ollama_base_url` (bare host). The optimizer needs the `/v1` OpenAI-compatible
   surface. Resolution: keep upstream's bare URL; the P4 adapter appends `/v1`.
   Only `ollama_model` + `ollama_warmup_max_wait_s` were added.
2. **Upstream already has `messaging/trees/`** — P8 is a 3-way reconciliation, not a
   blind copy. Compare fork trees vs upstream trees before porting.
3. **`create_message` is SYNC on upstream** (`api/services.py:102`); the fork optimizer
   is async (Ollama httpx). P4 wiring: make `ClaudeProxyService.create_message` async and
   `await service.create_message(...)` at `api/routes.py:173`. Insert the optimizer call
   after model routing and before token counting (two token-count sites: the OpenAI-chat
   branch at ~line 119 and the main path at ~line 184). `count_tokens` stays sync.
4. **Adapter is cleanly decoupled** — the fork `providers/common/context_optimizer.py`
   imports only `api.models.anthropic` (Message/SystemContent — both present upstream) and
   the `context_optimizer` package. Port it verbatim to `core/context/optimizer_adapter.py`;
   no `providers.common` repointing needed inside the adapter itself.

### Blockers requiring the user's environment (cannot be satisfied autonomously)

- **Messaging round-trip** (hard cutover gate) needs Discord/Telegram bot credentials.
- **Provider smoke matrix** needs API keys (deepseek/vertex/gemini/nim/openrouter/lmstudio).
- **Cutover** is a force-push of `main`; requires explicit user confirmation.

## Progress update 2 (P4 done, P5 scoped)

- **P4 done** (`22f06aa`) — context-optimizer fully wired into the async request path
  (`api/context_optimization.py` + `ClaudeProxyService.create_message` async + `AppRuntime`
  ollama warm-up + `core/trace` compaction events). **Full suite: 1481 passed; ruff + ty clean.**
  Note: the adapter had to live in `api/` not `core/` — the `core/` neutrality contract
  (`tests/contracts/test_import_boundaries.py`) forbids importing `api.models`.
- Proxy unit suite now defaults `CONTEXT_OPTIMIZE=0` (`tests/conftest.py`); compaction has no
  live Ollama in CI and the package owns its tier tests.

### P5 (Vertex) — exact remaining steps (parked clean; files restorable from tag)

The fork's `providers/vertex/` + `tests/providers/test_vertex.py` were removed from the branch
to keep it green; re-restore with `git checkout pre-rebase-2026-06 -- providers/vertex tests/providers/test_vertex.py`. Then:

1. `providers/vertex/request.py`: `from providers.common.message_converter import build_base_request_body`
   → `from core.anthropic.conversion import build_base_request_body, ReasoningReplayMode`; replace
   `build_base_request_body(request, include_thinking=thinking_enabled)` with
   `build_base_request_body(request, reasoning_replay=ReasoningReplayMode.THINK_TAGS if thinking_enabled else ReasoningReplayMode.DISABLED)`
   (Gemma is a plain OpenAI-compat endpoint, no `reasoning_content` — THINK_TAGS, not REASONING_CONTENT).
2. `providers/vertex/client.py`: `OpenAICompatibleProvider` → `OpenAIChatTransport` (import + base);
   change `_build_request_body(self, request)` → `_build_request_body(self, request, thinking_enabled=None)`
   passing `thinking_enabled=self._is_thinking_enabled(request, thinking_enabled)` (mirror
   `providers/nvidia_nim/client.py:36`). Hooks `self._client`, `self._base_url`, `cleanup` all exist on the new base.
3. `config/provider_catalog.py`: add a `"vertex"` `ProviderDescriptor` — `transport_type=<OPENAI_CHAT enum>`,
   `capabilities=(…)` (mirror nvidia_nim), `credential_env=None` (ADC, keyless → skips `_require_credential`),
   `static_credential="placeholder"`, `proxy_attr="vertex_proxy"`, `default_base_url=None` (VertexProvider
   computes its regional URL from settings).
4. `config/provider_ids.py`: add `"vertex"` to `SUPPORTED_PROVIDER_IDS`.
5. `providers/registry.py`: add `_create_vertex(config, settings)` returning `VertexProvider(config, settings=settings)`
   and `PROVIDER_FACTORIES["vertex"]` (the descriptors/factories/ids sync-assertion will enforce all three).
6. `api/services.py`: add `"vertex"` to `_OPENAI_CHAT_UPSTREAM_IDS` (vertex is /chat/completions, not native Messages).
7. `.env.example`: document `VERTEX_*` vars.
8. Validate: re-restored `tests/providers/test_vertex.py` (its `_build_base_url`/`_parse_models`/auth imports are
   unaffected; only a docstring mentions the old base name). ruff + ty + full suite.

### Still pending: P6 (per-project overrides → model_router), P7 (optimization_handlers reconcile),
P8 (messaging, hard gate), P9 (scripts + MIGRATION), version bump, cutover.
