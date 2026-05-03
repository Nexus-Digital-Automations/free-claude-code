"""Owns: Prometheus metric definitions and the /metrics route handler.

Defines the counter / histogram / gauge surfaces used by the rest of the
proxy. The objects themselves are module-level singletons so callers just
import them and call .inc() / .observe() / .set() — exactly the pattern
prometheus_client expects.

Does NOT own: scraping (Prometheus does that), labels for non-proxy
metrics (the Ollama supervisor's state is mirrored from log events),
or autocompaction-package internals.

Called by: api/routes.py (request lifecycle, /metrics endpoint),
providers/common/context_optimizer.py (compaction outcomes).
Calls: prometheus_client only.

# @stable — external scrapers depend on metric names and label sets.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Dedicated registry so importing prometheus_client elsewhere can't
# cross-pollute. Tests can also reset by re-importing this module.
REGISTRY = CollectorRegistry()

REQUEST_TOTAL = Counter(
    "proxy_request_total",
    "Total /v1/messages requests by outcome.",
    labelnames=("provider", "model", "outcome"),
    registry=REGISTRY,
)

REQUEST_DURATION_SECONDS = Histogram(
    "proxy_request_duration_seconds",
    "End-to-end /v1/messages duration (server-side).",
    labelnames=("provider", "model"),
    registry=REGISTRY,
)

COMPACTION_INVOCATION_TOTAL = Counter(
    "compaction_invocation_total",
    "Compaction attempts by tier and outcome.",
    labelnames=("tier", "outcome"),
    registry=REGISTRY,
)

COMPACTION_TOKENS_SAVED = Histogram(
    "compaction_tokens_saved",
    "Tokens removed from a request by the optimizer (raw - final).",
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 200000),
    registry=REGISTRY,
)

PREFIX_CACHE_HIT_TOTAL = Counter(
    "prefix_cache_hit_total",
    "Prefix cache lookups that hit a cached summary.",
    registry=REGISTRY,
)

PREFIX_CACHE_MISS_TOTAL = Counter(
    "prefix_cache_miss_total",
    "Prefix cache lookups that did not hit a cached summary.",
    registry=REGISTRY,
)

# Ollama supervisor state. 1.0 means the supervisor most recently observed
# the labelled state; the others are 0.0. Caller is the proxy adapter that
# mirrors OLLAMA: log lines into state transitions.
OLLAMA_SUPERVISOR_STATE = Gauge(
    "ollama_supervisor_state",
    "1.0 for the most recently observed Ollama supervisor state.",
    labelnames=("state",),
    registry=REGISTRY,
)


def render_exposition() -> tuple[bytes, str]:
    """Return (body, content_type) for /metrics. Never raises."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def set_ollama_state(state: str) -> None:
    """Mark `state` as the current state, others as 0. Idempotent."""
    for s in ("ready", "failed", "unknown"):
        OLLAMA_SUPERVISOR_STATE.labels(state=s).set(1.0 if s == state else 0.0)
