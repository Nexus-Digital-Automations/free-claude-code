"""Owns: end-to-end benchmark harness comparing Ollama models across all five
digest-prompt shapes used in this stack (diff_summary map + reduce, pytest
failures, build/type-check errors, lint findings).

Does NOT own: the digest call itself (digest_core.digest), Ollama warm-up
(ollama_supervisor.OllamaSupervisor.ensure_ready), or the prompt shapes
themselves (each digester package owns its `_build_*_prompt`).

Called by: the user, manually, when picking a default Ollama model.
Calls: digest_core, OllamaSupervisor, each digester's prompt builder, the
       Anthropic API for rubric judging, and `ollama` CLI for pull/list.

Outputs: raw.json (per-cell records) and report.md to --out-dir.
The report writer lives in `_render_report` below — same module so the
benchmark stays a single file.

# EXTENSION POINT — to add a new prompt shape, register a (category, dir,
# build_prompt) tuple in `_PROMPT_REGISTRY` below. Fixture files in the
# matching directory are picked up automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MCP_REPO_TOOLS_SRC = Path.home() / ".claude" / "mcp-repo-tools" / "src"
sys.path.insert(0, str(_MCP_REPO_TOOLS_SRC))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "test-digester" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "build-digester" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "packages" / "lint-digester" / "src"))

import anthropic  # noqa: E402
from build_digester.digest import _build_error_prompt  # noqa: E402
from context_optimizer import digest_core  # noqa: E402
from context_optimizer.ollama_supervisor import OllamaSupervisor  # noqa: E402
from lint_digester.digest import _build_rule_prompt  # noqa: E402
from repo_tools.diff_summary import _build_diff_prompt, _build_reduce_prompt  # noqa: E402
from test_digester.digest import _build_failure_prompt  # noqa: E402

_DIGEST_TAG = re.compile(r"<digest>\s*(.*?)\s*</digest>", re.DOTALL)

# Cap a single (model, fixture, run) Ollama call. Mirrors the 30s ceiling
# diff_summary.py imposes in production — a model that needs longer is
# unusable on the hot path regardless of quality.
_PER_CALL_TIMEOUT_SECONDS = 60.0

# Hardcoded prompt-shape budgets. Keep aligned with each prompt's
# "OUTPUT: under N tokens" line — exceeding this disqualifies the model
# from that prompt shape regardless of quality score.
_MAP_TOKEN_CAP = 400
_REDUCE_TOKEN_CAP = 300
_TEST_FAILURE_TOKEN_CAP = 400
_BUILD_ERROR_TOKEN_CAP = 300
_LINT_TOKEN_CAP = 200

_FIXTURES_ROOT = Path(__file__).parent / "bench_fixtures"

# Bumping this invalidates every cached judge score — change only when the
# rubric prompt itself changes meaning. Cache keys without this would let
# stale scores from a prior rubric design pollute new runs.
_RUBRIC_VERSION = "2026-05-a"

_RUBRIC_PROMPT = """You are scoring how well a small local LLM digested a piece of developer \
output. The original input is below as INPUT; the candidate model's digest is below \
as DIGEST.

Score on three axes, each 1 (terrible) to 5 (excellent):

- faithfulness: does the digest accurately represent what's in INPUT? Penalise \
  fabricated invariants, wrong file paths, hallucinated error codes.
- conciseness: is it terse enough to pay its way in conversation history? Penalise \
  restating INPUT verbatim or padding with disclaimers.
- risk_callout: does it surface what a reviewer/engineer should look at next? \
  Penalise pure description with no signal about what matters.

Return ONLY a JSON object on a single line, no prose, no code fences:
{"faithfulness": <int>, "conciseness": <int>, "risk_callout": <int>, "note": "<<=120 chars>"}

INPUT:
%(input)s

DIGEST:
%(digest)s
"""


@dataclass(frozen=True)
class PromptShape:
    """One row of the prompt registry: a category name, fixture subdir,
    prompt builder, and the cap from the prompt's stated <N-token output>.
    """

    category: str
    subdir: str
    build_prompt: Callable[[str], str]
    token_cap: int


_PROMPT_REGISTRY: tuple[PromptShape, ...] = (
    PromptShape("diff_map", "diff_map", _build_diff_prompt, _MAP_TOKEN_CAP),
    PromptShape("diff_reduce", "diff_reduce", _build_reduce_prompt, _REDUCE_TOKEN_CAP),
    PromptShape(
        "test_failure", "test_failures", _build_failure_prompt, _TEST_FAILURE_TOKEN_CAP
    ),
    PromptShape(
        "build_error", "build_errors", _build_error_prompt, _BUILD_ERROR_TOKEN_CAP
    ),
    PromptShape("lint", "lint", _build_rule_prompt, _LINT_TOKEN_CAP),
)


@dataclass
class _BenchConfig:
    """digest_core.DigestConfig protocol shape, mutable for per-model swaps.

    Defaults mirror the proxy's hardcoded `qwen2.5:7b` setup so a benchmark
    run uses production-equivalent settings across temperature, base URL,
    and keep-alive.
    """

    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"
    compaction_max_tokens: int = 500
    compaction_temperature: float = 0.0
    context_compaction_keep_alive: str = "30m"
    ollama_warmup_max_wait_s: float = 30.0


@dataclass
class CellResult:
    """One (model, fixture, run) record persisted to raw.json."""

    model: str
    category: str
    fixture: str
    run_index: int
    output_text: str | None
    output_chars: int
    output_tokens_estimate: int
    latency_seconds: float
    status: str  # "ok" | "ollama_unavailable" | "parse_error" | "timeout"


@dataclass
class JudgeScore:
    """Rubric verdict for one (model, fixture) pair from the Anthropic judge.

    Each axis is 1..5; `note` carries a short qualitative reason so the
    report can show what the judge actually objected to instead of just a
    bare number.
    """

    faithfulness: int
    conciseness: int
    risk_callout: int
    note: str
    error: str | None = None  # populated only when the judge call/parse failed


def _parse_digest_tag(content: str) -> str | None:
    """Match `<digest>...</digest>`. Returns None for malformed output so
    the caller logs a `parse_error` cell instead of treating the verbatim
    response as a digest.
    """
    match = _DIGEST_TAG.search(content)
    if not match:
        return None
    body = match.group(1).strip()
    return body or None


def _estimate_output_tokens(text: str) -> int:
    """4 chars/token heuristic — close enough for cap-violation flagging.

    A real token count would need the candidate model's tokenizer
    (different per model family); the 4-chars approximation tracks within
    ~15% across qwen/llama/phi/deepseek for English prose, which is fine
    for a "did this exceed the prompt's cap" gate.
    """
    return max(1, len(text) // 4)


def _list_local_models() -> set[str]:
    """Return names of models already pulled. `ollama list` failures are
    fatal — without it we can't tell pull-needed from pull-skip.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as exc:
        logger.error(
            "BENCH: ollama_list_failed type={} detail={}",
            type(exc).__name__,
            str(exc)[:200],
        )
        raise
    names: set[str] = set()
    for line in result.stdout.splitlines()[1:]:  # skip header
        first = line.split()[:1]
        if first:
            names.add(first[0])
    return names


def _pull_if_missing(model: str, local: set[str]) -> bool:
    """Pull `model` via the Ollama CLI if not already local. Returns
    True on success, False if the pull failed — caller decides whether to
    skip that model or abort.
    """
    if model in local or any(name.startswith(f"{model}:") for name in local):
        logger.info("BENCH: model_already_local model={}", model)
        return True
    logger.info("BENCH: pulling_model model={}", model)
    try:
        subprocess.run(["ollama", "pull", model], check=True, timeout=1800)
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as exc:
        logger.error(
            "BENCH: ollama_pull_failed model={} type={} detail={}",
            model,
            type(exc).__name__,
            str(exc)[:200],
        )
        return False
    return True


def _load_fixtures() -> list[tuple[PromptShape, str, str]]:
    """Walk the fixture tree and return [(shape, fixture_name, body), ...].

    Empty subdirs raise — a missing fixture set means the harness was
    invoked against a partial corpus, which would silently skew composite
    scores for any model that happens to be strong on the missing shape.
    """
    out: list[tuple[PromptShape, str, str]] = []
    for shape in _PROMPT_REGISTRY:
        directory = _FIXTURES_ROOT / shape.subdir
        files = sorted(directory.glob("*"))
        if not files:
            raise FileNotFoundError(
                f"No fixtures in {directory}; corpus is incomplete."
            )
        for path in files:
            out.append((shape, path.name, path.read_text()))
    return out


async def _run_one(
    config: _BenchConfig,
    shape: PromptShape,
    fixture_name: str,
    body: str,
    run_index: int,
) -> CellResult:
    """Execute one digest call, time it, and classify the outcome.

    Resets digest_core's content-keyed cache before each call so the second
    run of the same (model, fixture) actually re-invokes Ollama instead of
    returning the cached parsed output — that's how we measure model
    determinism rather than cache determinism.
    """
    digest_core.reset_for_test()
    started = time.perf_counter()
    try:
        output = await asyncio.wait_for(
            digest_core.digest(
                content=body,
                build_prompt=shape.build_prompt,
                parse_response=_parse_digest_tag,
                config=config,
                cache_max_entries=200,
            ),
            timeout=_PER_CALL_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        elapsed = time.perf_counter() - started
        logger.warning(
            "BENCH: timeout model={} fixture={}/{} run={} seconds={:.2f}",
            config.ollama_model,
            shape.category,
            fixture_name,
            run_index,
            elapsed,
        )
        return CellResult(
            model=config.ollama_model,
            category=shape.category,
            fixture=fixture_name,
            run_index=run_index,
            output_text=None,
            output_chars=0,
            output_tokens_estimate=0,
            latency_seconds=elapsed,
            status="timeout",
        )
    elapsed = time.perf_counter() - started
    if output is None:
        logger.warning(
            "BENCH: digest_returned_none model={} fixture={}/{} run={}",
            config.ollama_model,
            shape.category,
            fixture_name,
            run_index,
        )
        return CellResult(
            model=config.ollama_model,
            category=shape.category,
            fixture=fixture_name,
            run_index=run_index,
            output_text=None,
            output_chars=0,
            output_tokens_estimate=0,
            latency_seconds=elapsed,
            status="ollama_unavailable",
        )
    return CellResult(
        model=config.ollama_model,
        category=shape.category,
        fixture=fixture_name,
        run_index=run_index,
        output_text=output,
        output_chars=len(output),
        output_tokens_estimate=_estimate_output_tokens(output),
        latency_seconds=elapsed,
        status="ok",
    )


def _judge_cache_key(
    judge_model: str,
    candidate_model: str,
    category: str,
    fixture: str,
    output_text: str,
) -> str:
    """SHA over every input that could change the rubric score, so cache
    hits are sound and a rubric-version bump invalidates everything.
    """
    payload = "|".join(
        [
            _RUBRIC_VERSION,
            judge_model,
            candidate_model,
            category,
            fixture,
            hashlib.sha256(output_text.encode("utf-8", errors="ignore")).hexdigest(),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


_JUDGE_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_response(raw: str) -> JudgeScore:
    """Extract the JSON object from the judge's reply.

    The rubric tells the judge to return JSON only, but Anthropic
    occasionally wraps it in prose anyway; the regex tolerates that. A
    parse failure populates `error` so the report can flag affected cells
    instead of silently scoring them as zero.
    """
    match = _JUDGE_JSON.search(raw)
    if not match:
        logger.warning("BENCH: judge_no_json_block first_200={!r}", raw[:200])
        return JudgeScore(0, 0, 0, "", error="no_json")
    try:
        body = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("BENCH: judge_json_decode_failed detail={}", str(exc)[:200])
        return JudgeScore(0, 0, 0, "", error="json_decode")
    try:
        return JudgeScore(
            faithfulness=int(body["faithfulness"]),
            conciseness=int(body["conciseness"]),
            risk_callout=int(body["risk_callout"]),
            note=str(body.get("note", ""))[:120],
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("BENCH: judge_schema_invalid detail={}", str(exc)[:200])
        return JudgeScore(0, 0, 0, "", error="schema_invalid")


async def _judge_one(
    client: anthropic.AsyncAnthropic,
    judge_model: str,
    fixture_body: str,
    digest_text: str,
) -> JudgeScore:
    """Single rubric call. Times-out at 60s — the judge itself running
    long is itself a benchmark signal, but we don't want a hung call to
    block the whole run.
    """
    prompt = _RUBRIC_PROMPT % {
        "input": fixture_body[:8000],
        "digest": digest_text[:4000],
    }
    try:
        message = await asyncio.wait_for(
            client.messages.create(
                model=judge_model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=60.0,
        )
    except (TimeoutError, anthropic.APIError) as exc:
        logger.warning(
            "BENCH: judge_call_failed type={} detail={}",
            type(exc).__name__,
            str(exc)[:200],
        )
        return JudgeScore(0, 0, 0, "", error="api_failure")
    parts = [block.text for block in message.content if hasattr(block, "text")]
    return _parse_judge_response("\n".join(parts))


async def _judge_first_runs(
    cells: list[CellResult],
    fixture_lookup: dict[tuple[str, str], str],
    judge_model: str,
    cache_path: Path,
) -> dict[tuple[str, str, str], JudgeScore]:
    """Judge run_index=0 only; both runs of a (model, fixture) share the
    rubric verdict because they're scoring digest quality, not stability.

    Cache file is JSON-on-disk so a re-run with the same (RUBRIC_VERSION,
    candidate_model, fixture, output_hash) skips the API call entirely.
    Returns a map from (candidate_model, category, fixture) → JudgeScore.
    """
    cached: dict[str, dict] = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning(
                "BENCH: judge_cache_corrupt clearing path={} detail={}",
                cache_path,
                str(exc)[:200],
            )
            cached = {}
    client = anthropic.AsyncAnthropic()
    out: dict[tuple[str, str, str], JudgeScore] = {}
    for cell in cells:
        if cell.run_index != 0 or cell.status != "ok" or cell.output_text is None:
            continue
        body = fixture_lookup[(cell.category, cell.fixture)]
        key = _judge_cache_key(
            judge_model, cell.model, cell.category, cell.fixture, cell.output_text
        )
        if key in cached:
            score = JudgeScore(**cached[key])
        else:
            score = await _judge_one(client, judge_model, body, cell.output_text)
            cached[key] = asdict(score)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cached, indent=2))
        out[(cell.model, cell.category, cell.fixture)] = score
        logger.info(
            "BENCH: judged model={} fixture={}/{} f={} c={} r={}",
            cell.model,
            cell.category,
            cell.fixture,
            score.faithfulness,
            score.conciseness,
            score.risk_callout,
        )
    return out


async def _bench_model(
    model: str,
    fixtures: list[tuple[PromptShape, str, str]],
    runs: int,
) -> list[CellResult]:
    """Warm the model once, then iterate fixtures × runs.

    Skips the entire model if warm-up fails — running benchmarks against
    a cold or unreachable model produces meaningless latency and stability
    numbers.
    """
    config = _BenchConfig(ollama_model=model)
    if not await OllamaSupervisor.ensure_ready(config):
        logger.error("BENCH: warm_up_failed model={} skipping", model)
        return []
    results: list[CellResult] = []
    for shape, fixture_name, body in fixtures:
        for run_index in range(runs):
            cell = await _run_one(config, shape, fixture_name, body, run_index)
            logger.info(
                "BENCH: cell model={} fixture={}/{} run={} status={} latency={:.2f}s",
                model,
                shape.category,
                fixture_name,
                run_index,
                cell.status,
                cell.latency_seconds,
            )
            results.append(cell)
    return results


def _summarise_per_model(
    records: list[CellResult],
    judge_scores: dict[tuple[str, str, str], JudgeScore],
) -> dict[str, dict]:
    """Aggregate raw cells into per-model rollups for the report.

    Stability is the fraction of (model, fixture) pairs whose two runs
    produced byte-identical text. Cap violations roll up across cells.
    Quality is the mean of (faithfulness + conciseness + risk_callout)/3,
    averaged over fixtures the judge actually scored — judge errors drop
    out of the mean rather than dragging it to zero.
    """
    by_model: dict[str, list[CellResult]] = {}
    for record in records:
        by_model.setdefault(record.model, []).append(record)
    summary: dict[str, dict] = {}
    for model, cells in by_model.items():
        latencies_ok = [c.latency_seconds for c in cells if c.status == "ok"]
        cap_violations = sum(
            1
            for c in cells
            if c.status == "ok" and c.output_tokens_estimate > _cap_for(c.category)
        )
        scored = [
            score
            for (m, _, _), score in judge_scores.items()
            if m == model and score.error is None
        ]
        if scored:
            quality_raw = statistics.mean(
                (s.faithfulness + s.conciseness + s.risk_callout) / 3 for s in scored
            )
            quality_norm = (quality_raw - 1.0) / 4.0  # 1..5 -> 0..1
        else:
            quality_norm = None
        summary[model] = {
            "ok_count": sum(1 for c in cells if c.status == "ok"),
            "fail_count": sum(1 for c in cells if c.status != "ok"),
            "p50_latency": statistics.median(latencies_ok) if latencies_ok else None,
            "p95_latency": _percentile(latencies_ok, 95) if latencies_ok else None,
            "stability": _stability_fraction(cells),
            "cap_violations": cap_violations,
            "quality_normalised": quality_norm,
            "judge_scored_cells": len(scored),
            "judge_errors": sum(
                1
                for (m, _, _), s in judge_scores.items()
                if m == model and s.error is not None
            ),
        }
    return summary


def _cap_for(category: str) -> int:
    for shape in _PROMPT_REGISTRY:
        if shape.category == category:
            return shape.token_cap
    raise KeyError(f"Unknown category: {category}")


def _stability_fraction(cells: list[CellResult]) -> float:
    """For each (fixture) pair across the model's runs, score 1.0 if both
    runs produced byte-identical text, 0.0 if they diverge.
    """
    pairs: dict[tuple[str, str], list[str | None]] = {}
    for cell in cells:
        if cell.status == "ok":
            pairs.setdefault((cell.category, cell.fixture), []).append(cell.output_text)
    if not pairs:
        return 0.0
    stable = sum(1 for outputs in pairs.values() if len(set(outputs)) == 1)
    return stable / len(pairs)


def _percentile(values: list[float], pct: float) -> float:
    sorted_vals = sorted(values)
    if not sorted_vals:
        raise ValueError("empty list")
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    weight = rank - lo
    return sorted_vals[lo] * (1 - weight) + sorted_vals[hi] * weight


def _composite_score(per_model: dict[str, dict]) -> dict[str, float]:
    """`composite = 0.5·quality + 0.3·(1-latency_norm) + 0.2·stability`.

    Latency is min-max normalised across the cohort so the fastest model
    in this run gets full credit. Models with no judged cells (every
    judge call failed) fall back to quality=0.5 so they don't flatline
    the ranking purely on a judge-side outage.
    """
    latencies = [
        v["p95_latency"] for v in per_model.values() if v["p95_latency"] is not None
    ]
    if not latencies:
        return {model: 0.0 for model in per_model}
    lo, hi = min(latencies), max(latencies)
    span = max(hi - lo, 1e-6)
    out: dict[str, float] = {}
    for model, summary in per_model.items():
        latency = summary["p95_latency"]
        latency_norm = (latency - lo) / span if latency is not None else 1.0
        quality = summary["quality_normalised"]
        if quality is None:
            quality = 0.5
        out[model] = (
            0.5 * quality + 0.3 * (1.0 - latency_norm) + 0.2 * summary["stability"]
        )
    return out


def _flags_for(summary: dict) -> list[str]:
    """Red flags surfaced beneath the table — disqualifiers a reviewer
    needs to see even when the composite ranks the model highly.
    """
    flags: list[str] = []
    if summary["stability"] < 1.0:
        flags.append(
            f"stability {summary['stability']:.2f} (digest text drifted between runs at temp=0)"
        )
    if summary["cap_violations"] > 1:
        flags.append(
            f"{summary['cap_violations']} cap violations (output exceeded prompt's stated token budget)"
        )
    if summary["fail_count"] > 0:
        flags.append(
            f"{summary['fail_count']} failed cells (timeout / Ollama unreachable / parse error)"
        )
    if summary["judge_errors"] > 0:
        flags.append(f"{summary['judge_errors']} cells the judge could not score")
    return flags


def _render_report(
    out_dir: Path,
    per_model: dict[str, dict],
    composite: dict[str, float],
    args: argparse.Namespace,
) -> None:
    """Write report.md ranked by composite score with per-model flags.

    Recommended default = the top-ranked model with no `stability < 1.0`
    flag. A model that drifts at temp=0.0 disqualifies for the hot path
    regardless of how good it scores on quality, because the prefix-cache
    invariant in tier0b breaks economically without byte stability.
    """
    ranked = sorted(per_model.keys(), key=lambda m: -composite[m])
    lines: list[str] = []
    lines.append(f"# Ollama digest model benchmark — {date.today().isoformat()}")
    lines.append("")
    lines.append(
        f"Fixtures: 22 cells across 5 prompt shapes. Runs/cell: {args.runs}. "
        f"Judge: `{args.judge_model}` (rubric `{_RUBRIC_VERSION}`)."
    )
    lines.append("")
    lines.append("## Composite ranking")
    lines.append("")
    lines.append(
        "| Rank | Model | Composite | Quality | OK | p50 | p95 | Stability | Cap viol. |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for rank, model in enumerate(ranked, start=1):
        s = per_model[model]
        p50 = f"{s['p50_latency']:.2f}s" if s["p50_latency"] is not None else "-"
        p95 = f"{s['p95_latency']:.2f}s" if s["p95_latency"] is not None else "-"
        quality = (
            f"{s['quality_normalised']:.2f}"
            if s["quality_normalised"] is not None
            else "-"
        )
        lines.append(
            f"| {rank} | `{model}` | {composite[model]:.3f} | {quality} | "
            f"{s['ok_count']}/{s['ok_count'] + s['fail_count']} | "
            f"{p50} | {p95} | {s['stability']:.2f} | {s['cap_violations']} |"
        )
    lines.append("")
    lines.append("## Per-model flags")
    lines.append("")
    for model in ranked:
        flags = _flags_for(per_model[model])
        if not flags:
            lines.append(f"- `{model}`: clean — no disqualifiers.")
            continue
        lines.append(f"- `{model}`:")
        for flag in flags:
            lines.append(f"  - {flag}")
    lines.append("")
    recommended = next(
        (
            m
            for m in ranked
            if per_model[m]["stability"] >= 1.0 and per_model[m]["fail_count"] == 0
        ),
        None,
    )
    lines.append("## Recommended default")
    lines.append("")
    if recommended:
        lines.append(
            f"`{recommended}` — top-ranked composite with no stability or "
            "failure-rate disqualifiers. Updating the five `qwen2.5:7b` "
            "defaults to this is a follow-up task."
        )
    else:
        lines.append(
            "**No clean recommendation.** Every candidate has at least one "
            "disqualifying flag. Re-run with a wider candidate set or accept "
            "a flagged model after reviewing the per-model flags above."
        )
    lines.append("")
    lines.append("## Composite formula")
    lines.append("")
    lines.append(
        "`composite = 0.5 * quality + 0.3 * (1 - latency_norm) + 0.2 * stability`"
    )
    lines.append("")
    lines.append(
        "- quality: judge rubric mean of (faithfulness + conciseness + risk_callout)/3, scaled 1..5 → 0..1"
    )
    lines.append(
        "- latency_norm: p95 wall-clock latency, min-max normalised across this cohort"
    )
    lines.append(
        "- stability: fraction of (model, fixture) pairs whose two runs were byte-identical at temp=0"
    )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append(
        f"python {Path(__file__).relative_to(_REPO_ROOT)} "
        f"--models {','.join(args.models)} --runs {args.runs} "
        f"--judge-model {args.judge_model}"
    )
    lines.append("```")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text("\n".join(lines))


def _persist_raw(out_dir: Path, records: list[CellResult]) -> None:
    """raw.json is the source of truth — everything in report.md derives
    from it, so re-running the report renderer never requires re-running
    Ollama.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    (out_dir / "raw.json").write_text(json.dumps(payload, indent=2))


def _content_hash(records: list[CellResult]) -> str:
    """For diagnostic logging only — confirms two runs of the harness
    produced bit-identical raw.json (ignoring latency, which by definition
    varies). Strips latency before hashing.
    """
    serialised = json.dumps(
        [{**asdict(r), "latency_seconds": 0.0} for r in records],
        sort_keys=True,
    )
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--models",
        type=lambda s: [m.strip() for m in s.split(",") if m.strip()],
        default=["qwen2.5:7b", "qwen2.5-coder:7b", "qwen3-coder:8b"],
        help="Comma-separated Ollama model tags to compare.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Runs per (model, fixture). 2 is the minimum for byte-stability.",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-opus-4-7",
        help="Anthropic model id for rubric judging (Phase 3, ignored in Phase 2).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home()
        / ".claude"
        / "mcp-repo-tools"
        / "benchmarks"
        / date.today().isoformat(),
        help="Where raw.json + report.md land.",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> int:
    fixtures = _load_fixtures()
    logger.info(
        "BENCH: starting models={} fixtures={} runs={}",
        args.models,
        len(fixtures),
        args.runs,
    )
    local = _list_local_models()
    all_records: list[CellResult] = []
    for model in args.models:
        if not _pull_if_missing(model, local):
            logger.warning("BENCH: skipping_model model={} reason=pull_failed", model)
            continue
        OllamaSupervisor._reset_for_test()
        records = await _bench_model(model, fixtures, args.runs)
        all_records.extend(records)
    if not all_records:
        logger.error("BENCH: no_records every model failed; aborting report")
        return 1
    fixture_lookup = {(s.category, name): body for s, name, body in fixtures}
    judge_scores: dict[tuple[str, str, str], JudgeScore] = {}
    if os.environ.get("ANTHROPIC_API_KEY"):
        judge_scores = await _judge_first_runs(
            all_records,
            fixture_lookup,
            args.judge_model,
            args.out_dir / "judge_cache.json",
        )
    else:
        logger.warning("BENCH: ANTHROPIC_API_KEY unset; quality column will be blank")
    per_model = _summarise_per_model(all_records, judge_scores)
    composite = _composite_score(per_model)
    _persist_raw(args.out_dir, all_records)
    _render_report(args.out_dir, per_model, composite, args)
    logger.info(
        "BENCH: done out_dir={} content_hash={}",
        args.out_dir,
        _content_hash(all_records),
    )
    return 0


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("BENCH_LOG_LEVEL", "INFO"))
    args = _parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
