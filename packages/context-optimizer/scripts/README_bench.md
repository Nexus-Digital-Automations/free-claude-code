# bench_ollama_digest

Owns: Comparing local Ollama models on the five digest-prompt shapes used in
this stack (proxy tier0b compaction + diff_summary + test/build/lint
digesters). Picks a recommended default; does NOT change any default
itself — that's a follow-up task after reviewing the report.

## Run

```bash
python packages/context-optimizer/scripts/bench_ollama_digest.py \
  --models qwen2.5:7b,qwen2.5-coder:7b,qwen3-coder:8b \
  --runs 2 \
  --judge-model claude-opus-4-7
```

Defaults: 3 models (qwen2.5:7b, qwen2.5-coder:7b, qwen3-coder:8b), 2 runs
per (model, fixture), judge=claude-opus-4-7, output to
`~/.claude/mcp-repo-tools/benchmarks/<today>/`.

## Requirements

- Ollama daemon running and the models locally pulled (the harness
  auto-pulls missing ones via `ollama pull`).
- `ANTHROPIC_API_KEY` exported. Without it the harness skips the judge
  and the quality column shows blanks; latency + stability columns still
  populate.

## Fixtures

22 cells across 5 prompt shapes in `bench_fixtures/`:

- `diff_map/` — 5 per-file git diffs of varied size (847 B – 13 KB)
- `diff_reduce/` — 2 per-file digest aggregations for the reduce phase
- `test_failures/` — 5 pytest tracebacks
- `build_errors/` — 5 mypy / tsc / rustc error blobs
- `lint/` — 5 ruff / eslint single-rule findings

To add a new prompt shape, append to `_PROMPT_REGISTRY` in
`bench_ollama_digest.py` and drop fixtures in a matching subdir.

## Output

- `raw.json` — one record per (model, fixture, run) with latency, output
  text, status. Source of truth for the report.
- `report.md` — ranked composite table + per-model flags + recommended
  default + the formula and reproduce command.
- `judge_cache.json` — keyed on `(rubric_version, judge_model,
  candidate_model, category, fixture, output_hash)` so re-runs of the same
  cohort don't burn API spend.

## Scoring

`composite = 0.5 * quality + 0.3 * (1 - latency_norm) + 0.2 * stability`

- quality: judge rubric mean (faithfulness + conciseness + risk_callout)/3,
  rescaled 1..5 → 0..1
- latency_norm: p95 latency, min-max normalised across this run's cohort
- stability: fraction of (model, fixture) pairs whose two runs were
  byte-identical at temp=0.0. **A model with stability < 1.0 is
  disqualified for the hot path** because tier0b's prefix cache breaks
  economically without it.

## Re-running

Idempotent at temp=0.0. Re-running with the same models + fixtures should
produce byte-identical raw.json (latency aside) and zero new judge API
calls (cache hits). `_content_hash` logged at the end confirms this — if
it changes between runs, a model drifted at temp=0.0, which is itself a
finding.

## Expected runtime

22 fixtures × 3 models × 2 runs ≈ 132 Ollama calls. At 3–5 s/call on
M-series silicon, plan ~10–15 min for the Ollama side. Judge calls add
~22 × N_models Anthropic calls (cached after first run).
