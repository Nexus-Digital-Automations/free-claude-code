# Upstream Sync 2026-06 — Triage Report

**Branch:** `upstream-sync-2026-06` (from `main` @ `3acc975`)
**Upstream:** `Alishahryar1/free-claude-code` @ `upstream/main` `544008a`
**Process:** cherry-pick triage, mirroring `specs/upstream-sync-2026-05.md`.

## Summary

- Divergence at start: `main` was **80 behind / 169 ahead** of `upstream/main`.
- Of the 80 upstream commits, `git cherry -v main upstream/main` shows **13 already
  patch-applied** on `main` (absorbed during the May sync — httpx/setup-uv bumps, smoke
  refactor, NimSettings validation, Telegram session reuse, env-prefixed command parsing,
  10s connect timeout, Claude Code 2.1 title-skip, docs, and the `#262` constant-time
  auth-token fix).
- Of the remaining ~67 unapplied commits: **2 cherry-picked cleanly**, the rest **SKIPPED**.

| Category | Count |
|---|---|
| TAKE (clean cherry-pick) | 2 |
| Already applied (May sync, by patch-id) | 13 |
| SKIP — coupled to `26b8a29` refactor | ~30 |
| SKIP — README / asset / docs churn | ~28 |
| SKIP — bulk dep-group / lock bumps | 3 |

**Validation (green):**
```
ruff check api config providers messaging cli smoke   → 0 findings
uv run pytest -o addopts='' --ignore=tests/providers/test_context_optimizer.py -q  → 981 passed
```

## TAKEN (2)

| Upstream SHA | New SHA | Subject | Reason |
|---|---|---|---|
| `3bde98a` | `faee9e6` | Raise default timeouts for write and connect | `.env.example` only; layout-independent. |
| `2637824` | `7ba63fb` | fix(openai): close async client with supported method | Real fix in `providers/openai_compat.py` + test; applied clean. |

## Key finding — the cherry-pick well has run dry

The May sync already absorbed every **layout-independent** fix. What remains divides into:

1. **Refactor-coupled (~30, SKIP):** every "isolated-looking" fix now conflicts via
   **modify/delete on post-`26b8a29` files** the fork never adopted — `core/anthropic/{conversion,tokens}.py`,
   `providers/anthropic_messages.py`, `providers/error_mapping.py`, `api/services.py`,
   `providers/kimi/`, `providers/wafer/`. Cherry-picking these is no longer a cherry-pick; it
   is the manual re-port that the "full architecture rebase" option entails.
2. **README/asset/docs churn (~28, SKIP):** model-picker images, star chart, logos, section
   renumbering, etc.
3. **Bulk dependency-group / lock bumps (3, SKIP):** `#173`, `#314`, `#389` — conflict on
   `uv.lock`; our resolver re-derives.

### High-value fixes locked behind the refactor (would need manual porting)

These are genuinely worth having but each requires porting upstream's logic into the fork's
pre-refactor layout (they touch files we deleted/renamed in the refactor split):

| Upstream SHA | Subject | Why it conflicts |
|---|---|---|
| `1e97dff` | fix: handle disallowed special tokens in tiktoken encoder (#382) | logic lives in `core/anthropic/tokens.py` (post-refactor); ours is `api/request_utils.py`. |
| `29e7714` | feat(logging): structured TRACE events + request correlation | spans `api/app.py`, `api/routes.py`, deleted `api/services.py`. |
| `21ff213` + `41f2bc7` | retry upstream 503 / all 5xx like 429 | logic in deleted `providers/error_mapping.py` + `providers/anthropic_messages.py`. |
| `abae61d` | Fix null usage in SSE for OpenAI-compatible streams (#209,#123) | conflicts in `providers/common/sse_builder.py`. |
| `36d236b` | fix(206): defer post-tool assistant content for OpenAI chat conversion | needs `core/anthropic/conversion.py`. |
| `ca2cf6a` + `2569507` | fix(cli): terminate process trees / clean interrupt exit | conflicts in `cli/entrypoints.py`, `cli/session.py`. |

**Recommendation:** the structural choice is now binary — either invest in porting the handful
of fixes above into the pre-refactor layout (the realistic value left), or undertake the full
rebase onto upstream's post-`26b8a29` architecture. Pure cherry-pick triage is exhausted.
