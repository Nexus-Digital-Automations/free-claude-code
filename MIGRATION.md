# Migration: fork re-platformed onto upstream's post-`26b8a29` architecture

This fork was re-platformed from its pre-refactor layout onto upstream
(`Alishahryar1/free-claude-code`) after the `26b8a29` "core anthropic / runtime"
refactor. The fork's custom capabilities were ported, not the old file layout.
Full plan and per-phase log: `specs/upstream-full-rebase.md`. Rollback point: the
`pre-rebase-2026-06` git tag.

## Where the fork's custom code now lives

| Capability | Old location (pre-refactor) | New location |
|---|---|---|
| Anthropic protocol helpers | `providers/common/*` | `core/anthropic/*` (+ `providers/error_mapping.py`) — `providers/common/` is **deleted** |
| Context-optimizer adapter | `providers/common/context_optimizer.py` | `api/context_optimization.py` (binds to `api.models`, so it lives in `api/`, not `core/`) |
| Ollama supervisor | `providers/common/ollama_supervisor.py` (shim) | imported directly from `context_optimizer.ollama_supervisor`; warm-up wired in `api/runtime.py` `AppRuntime` |
| Context-optimizer package | `packages/context-optimizer/` | unchanged (editable `[tool.uv.sources]` dependency) |
| Compaction call site | `api/routes.py` | `api/services.py` `ClaudeProxyService.create_message` (now `async`) |
| Compaction metrics | `api/metrics.py` (`COMPACTION_*`) | `core/trace.py` events (`api.compaction.applied` / `.error`) |
| Vertex provider | `providers/vertex/` (on `OpenAICompatibleProvider`) | `providers/vertex/` on `OpenAIChatTransport`; registered in `config/provider_catalog.py`, `providers/registry.py`, admin manifest |
| Per-project model overrides | `Settings.resolve_model` reading a contextvar | `Settings.resolve_model(project_cwd=...)` param (config stays neutral) + `api/model_router.py` reads `api/context.py:current_project_cwd` |
| Optimization handlers | `api/optimization_handlers.py` | upstream already provides identical handlers — no port needed |

## Behavioral notes

- **`OLLAMA_BASE_URL`** is shared: upstream's native ollama provider uses the bare
  host; the context-optimizer adapter appends `/v1` for the OpenAI-compatible surface.
- **Context compaction** defaults on in production but is disabled in the proxy unit
  test suite (`tests/conftest.py: CONTEXT_OPTIMIZE=0`) since it needs a live Ollama;
  the `context-optimizer` package owns its own tier tests.
- **Vertex** auth is ADC-preferred (install the `vertex` extra: `google-auth`) with a
  `VERTEX_ACCESS_TOKEN` static fallback. It keeps upstream's native `gemini` provider.
- **Conventions adopted from upstream:** Python 3.14, `uv run`, `ruff format` (py314),
  `ty` type-checking (no `# type: ignore`), and a semver bump in `pyproject.toml` for
  any production-file change.

## Still pending at time of writing

Messaging (`messaging/`) reconcile + `AppRuntime` wiring, and the final cutover, are
tracked in `specs/upstream-full-rebase.md`.
