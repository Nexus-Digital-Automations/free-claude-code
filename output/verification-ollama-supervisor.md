# Verification: Ollama Supervisor (commit 7f6e204)

## Original user request (verbatim from session)

> "okay now please make sure it forces the ollama model to be running. I don't
> want it to be something happenstancial where the ollama model is down and
> that doesn't work. it should have means to start up the model and whatnot
> if it's not already running"

User then answered four AskUserQuestion choices:
- "Daemon up + model warm" (Recommended)
- "Both: start at boot + recheck before each compaction"
- "Spawn `ollama serve` if down + `ollama run` to warm" (Recommended)
- "Log loudly, fall back to provider" (Recommended)

User reply on the spec presentation: **"approved"**.

## Source-of-truth spec

`specs/ollama-supervisor.md` (committed in 9baef2e, criteria checked off in 7f6e204).

## Acceptance-criteria → diff mapping

| Criterion | Implementation | File:line |
|-----------|---------------|-----------|
| `OllamaSupervisor.ensure_ready(settings) -> bool` | Class method with cache + cooldown | `providers/common/ollama_supervisor.py:78-101` |
| Boot-time fire-and-forget | `asyncio.create_task` in lifespan | `api/app.py:60-67` |
| `_compact_via_ollama` short-circuits | `if not await ensure_ready: return False` | `providers/common/context_optimizer.py:333-345` |
| `_do_ollama_call` short-circuits | same gate | `providers/common/context_optimizer.py:348-357` |
| Health check via `GET /api/tags` | `_health_check` | `providers/common/ollama_supervisor.py:139-145` |
| Warm via `POST /api/generate` keep_alive=30m | `_warm_model` | `providers/common/ollama_supervisor.py:177-211` |
| 30s ready cache | `_READY_TTL_SECONDS = 30.0` | `providers/common/ollama_supervisor.py:60` |
| 60s failure cooldown | `_FAILURE_COOLDOWN_SECONDS = 60.0` | `providers/common/ollama_supervisor.py:61` |
| OLLAMA: prefixed logs | every transition | throughout `ollama_supervisor.py` |
| 6 supervisor tests | full coverage | `tests/providers/test_ollama_supervisor.py` |

## Test output

```
$ uv run pytest tests/providers/test_ollama_supervisor.py -v
test_api_root_strips_openai_v1_suffix                                 PASSED
test_ensure_ready_caches_success_within_ttl                           PASSED
test_ensure_ready_returns_false_during_cooldown_after_failure         PASSED
test_ensure_ready_returns_false_when_ollama_binary_missing            PASSED
test_concurrent_ensure_ready_calls_coalesce_on_single_check           PASSED
test_warm_model_404_marks_failed_with_pull_hint                       PASSED
====== 6 passed ======

$ uv run pytest tests/ -x -q
====== 930 passed in 9.30s ======

$ uv run ruff check .
All checks passed!
```

## Commits in this task

- `9baef2e` — spec only (planning, awaiting approval)
- `7f6e204` — full implementation + 6 tests (post-approval)
