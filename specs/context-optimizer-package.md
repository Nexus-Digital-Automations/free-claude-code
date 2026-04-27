---
title: context-optimizer — Standalone Pip-Installable Package
status: planning
created: 2026-04-26
---

## Vision

The autocompaction system in this proxy is valuable on its own. Extracting it
into a standalone pip-installable package (`context-optimizer`) lets any
Python codebase reduce LLM conversation token costs by calling a single async
function. The proxy becomes one consumer of this package rather than owning
the implementation.

## Requirements

### R1 — Standalone pip package
A new Python package lives at `packages/context-optimizer/` in this repo with
its own `pyproject.toml`. It is installable via:
```
pip install git+https://github.com/Nexus-Digital-Automations/free-claude-code.git#subdirectory=packages/context-optimizer
```
or as a local editable install:
```
pip install -e ./packages/context-optimizer
```
It has no dependency on any module from the proxy's `api.*`, `config.*`, or
`providers.*` namespaces.

### R2 — Full tier system
All tiers are present in the package:
- **Tier 0** — ANSI strip, tool-result content-hash dedup, long-output truncation
- **Tier 1** — Strip ContentBlockThinking from old assistant turns (keep last N)
- **Prefix cache** — In-memory LRU; applies pre-computed summaries before tier 2
- **Tier 2a** — Background Ollama compaction at a soft token threshold
- **Tier 2b** — Blocking LLM compaction at a hard token threshold (via async callable)
- **Ollama supervisor** — Boot-time daemon start + model warm-up (moved from proxy)

### R3 — Plain-dict public API
The public entry point accepts and returns plain Python dicts so callers have
no Pydantic dependency:

```python
from context_optimizer import ContextOptimizer, ContextOptimizerSettings

settings = ContextOptimizerSettings(
    compact_threshold_tokens=200_000,
    compact_soft_threshold_tokens=80_000,
    ollama_base_url="http://localhost:11434/v1",
    ollama_model="qwen2.5:7b",
)

async def my_llm(prompt: str) -> str:
    resp = await openai_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return resp.choices[0].message.content

messages_in = [
    {"role": "user", "content": "Fix the bug in api/routes.py"},
    {"role": "assistant", "content": [{"type": "text", "text": "Sure..."}]},
]

new_messages, new_system, token_count = await ContextOptimizer.optimize(
    messages=messages_in,
    system="You are a coding assistant.",
    token_count=95_000,   # pre-computed by caller; avoids extra tiktoken pass
    settings=settings,
    llm_provider=my_llm,  # used for Tier 2b and background Ollama fallback
)
```

Internally the package converts dicts to typed models and back.

### R4 — Proxy adopts the package
The proxy's `providers/common/context_optimizer.py` is replaced with a thin
adapter that imports `ContextOptimizer` from the package, converts the
proxy's Pydantic `Message` / `MessagesRequest` objects to plain dicts, calls
`optimize`, and converts the output back. The proxy's
`providers/common/ollama_supervisor.py` is replaced by a shim that re-exports
`OllamaSupervisor` from the package.

### R5 — ContextOptimizerSettings dataclass
All thresholds and knobs live in a plain `dataclasses.dataclass` (not Pydantic)
so any project can construct it without a Pydantic dependency:

```python
@dataclass
class ContextOptimizerSettings:
    compact_threshold_tokens: int = 200_000
    compact_soft_threshold_tokens: int = 80_000
    compact_deepseek_fallback_threshold_tokens: int = 150_000
    max_thinking_turns: int = 2
    prefix_cache_max_entries: int = 100
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"
    tier0_max_lines: int = 200
    render_preview_chars: int = 2_000
```

### R6 — Token counting bundled
The package ships its own `token_count(messages, system, tools) -> int` using
tiktoken `cl100k_base`. Callers may pass a pre-computed count to skip the
tiktoken pass.

## Acceptance Criteria

- [ ] `packages/context-optimizer/pyproject.toml` defines the package with
      `name = "context-optimizer"` and pins its own deps (pydantic, tiktoken,
      httpx, openai, loguru).
- [ ] `pip install -e ./packages/context-optimizer` works from the repo root.
- [ ] `from context_optimizer import ContextOptimizer, ContextOptimizerSettings`
      works in a fresh Python environment with no proxy modules on sys.path.
- [ ] `await ContextOptimizer.optimize(messages=..., system=..., token_count=..., settings=..., llm_provider=...)` returns `(list[dict], str|list|None, int)`.
- [ ] All four tiers plus prefix cache are exercised in the package's own
      `tests/` (Tier 0 ANSI+dedup, Tier 1 strip, prefix cache hit, Tier 2b
      via mocked llm_provider callable, Ollama supervisor mock).
- [ ] `providers/common/context_optimizer.py` in the proxy is replaced by a
      ≤50-line adapter; proxy's test suite remains fully green (930+ tests).
- [ ] `providers/common/ollama_supervisor.py` in the proxy re-exports from the
      package so existing callers (`api/app.py`) are unaffected.
- [ ] `uv run ruff check .` clean in both the proxy and the package.
- [ ] `uv run pytest tests/ -x -q` green in the proxy after the refactor.

## Package layout

```
packages/context-optimizer/
  pyproject.toml
  README.md
  src/
    context_optimizer/
      __init__.py           # public surface: ContextOptimizer, ContextOptimizerSettings, OllamaSupervisor
      optimizer.py          # ContextOptimizer.optimize() — orchestrates tiers
      settings.py           # ContextOptimizerSettings dataclass
      token_counter.py      # get_token_count(messages, system, tools) -> int
      cache.py              # PrefixCache (LRU, keyed by content hash)
      prompts.py            # _build_prompt, _parse_response
      ollama_supervisor.py  # OllamaSupervisor (moved from proxy)
      tiers/
        __init__.py
        tier0.py            # NLP cleanup (ANSI, dedup, truncation)
        tier1.py            # Thinking-block strip
        tier2.py            # LLM compaction (Ollama + llm_provider callable)
  tests/
    test_tier0.py
    test_tier1.py
    test_cache.py
    test_tier2.py
    test_ollama_supervisor.py
    test_optimizer_integration.py
```

## Proxy adapter layout (after refactor)

```
providers/common/context_optimizer.py   ← 30-50 line adapter
providers/common/ollama_supervisor.py   ← 5-line re-export shim
```

## Technical Decisions

- **`dataclass` settings, not Pydantic.** Pydantic is a dep of the package,
  but forcing callers to use Pydantic-style field validation in settings would
  add friction. A plain dataclass is constructable with keyword arguments,
  works with mypy, and has zero magic.

- **`llm_provider: Callable[[str], Awaitable[str]]` for Tier 2b.** Decouples
  the package from any specific provider SDK. The caller controls the model,
  auth, and base URL. Background Ollama still uses its own HTTP client (no
  llm_provider involvement).

- **Flat internal dict format, not OpenAI SDK objects.** Internally messages
  are `list[dict]` with the Anthropic schema. The package defines lightweight
  TypedDicts (`InternalMessage`, etc.) for type hints without imposing external
  Pydantic dependency on callers.

- **Proxy adapter converts Pydantic ↔ dicts.** This is cheap and isolates the
  schema-conversion concern to a single file rather than permeating the package.

- **Monorepo, not separate repo.** The package lives in `packages/` within this
  repo so changes to both proxy and package can land in a single commit and the
  same CI run validates the full stack.

## Files created / modified

| File | Change |
|------|--------|
| `packages/context-optimizer/pyproject.toml` | NEW |
| `packages/context-optimizer/src/context_optimizer/__init__.py` | NEW |
| `packages/context-optimizer/src/context_optimizer/optimizer.py` | NEW |
| `packages/context-optimizer/src/context_optimizer/settings.py` | NEW |
| `packages/context-optimizer/src/context_optimizer/token_counter.py` | NEW |
| `packages/context-optimizer/src/context_optimizer/cache.py` | NEW |
| `packages/context-optimizer/src/context_optimizer/prompts.py` | NEW |
| `packages/context-optimizer/src/context_optimizer/ollama_supervisor.py` | NEW (moved) |
| `packages/context-optimizer/src/context_optimizer/tiers/__init__.py` | NEW |
| `packages/context-optimizer/src/context_optimizer/tiers/tier0.py` | NEW |
| `packages/context-optimizer/src/context_optimizer/tiers/tier1.py` | NEW |
| `packages/context-optimizer/src/context_optimizer/tiers/tier2.py` | NEW |
| `packages/context-optimizer/tests/` (6 files) | NEW |
| `packages/context-optimizer/README.md` | NEW |
| `providers/common/context_optimizer.py` | REPLACE with adapter |
| `providers/common/ollama_supervisor.py` | REPLACE with re-export shim |
| `pyproject.toml` | ADD local dep on `./packages/context-optimizer` |

17 files. Delegated to Qwen per the 5-file backend threshold.

## Progress

- [ ] R1 — pip package scaffold (pyproject.toml, layout)
- [ ] R2 — tier implementations in package
- [ ] R3 — plain-dict public API
- [ ] R4 — proxy adapter + shims
- [ ] R5 — ContextOptimizerSettings dataclass
- [ ] R6 — bundled token counter
