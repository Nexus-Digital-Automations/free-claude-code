"""Owns: ContextOptimizerSettings — configuration for all optimizer tiers.

Does NOT own: defaults validation (caller's responsibility), provider
configuration (caller constructs llm_provider callable separately).

Called by: optimizer.py, tiers/tier2.py, ollama_supervisor.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextOptimizerSettings:
    """Plain-dataclass configuration so callers need no Pydantic dependency.

    All thresholds are in approximate token counts (tiktoken cl100k_base).
    """

    # ---- Tier 2 thresholds ----
    # Lowered from the original 65/50/25 anchor on observation that real proxy
    # traffic clusters bimodally (5-25K and 98-102K). Combined with the
    # tier2_keep_recent_turns floor below, lower thresholds catch sessions
    # earlier without risking recent-turn loss in the resulting summary.
    compact_threshold_tokens: int = 55_000
    """Hard limit — sync blocking compaction via llm_provider."""

    compact_soft_threshold_tokens: int = 18_000
    """Soft limit — schedule background Ollama compaction. Earlier trigger
    gives the warm Ollama call (~3-5s) time to finish before the next request
    races the hard limit."""

    compact_deepseek_fallback_threshold_tokens: int = 40_000
    """Mid-point — near hard limit; Ollama fallback to llm_provider if busy."""

    tier2_keep_recent_turns: int = 8
    """Quality floor — Tier 2 must preserve at least this many trailing
    messages verbatim. Clamps the LLM-chosen split_index so a too-aggressive
    summary cannot collapse the most recent context the next turn depends on."""

    # ---- Tier 1 ----
    max_thinking_turns: int = 0
    """How many recent assistant turns keep their thinking blocks. Default 0
    strips all thinking — both saves tokens and keeps the prefix byte-stable
    across consecutive requests, which maximises DeepSeek prefix-cache hits.
    A relative 'keep last N' rule would mutate the prefix every time a new
    turn arrived (the position of 'last N' shifts). Set higher only if
    downstream tooling needs to inspect intermediate reasoning."""

    # ---- Prefix cache ----
    prefix_cache_max_entries: int = 100

    # ---- Ollama ----
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"

    # ---- Tier 0 ----
    tier0_max_lines: int = 120
    """Tool results longer than this get head+tail truncation."""

    tier0_head_lines: int = 60
    tier0_tail_lines: int = 60

    # ---- Tier 0b (Ollama tool-result digester) ----
    tier0b_digest_enabled: bool = True
    """Run Ollama-based content-aware digest on tool_results above the
    byte threshold. Falls back to Tier 0's mechanical truncation on any
    Ollama failure."""

    tier0b_digest_min_bytes: int = 8000
    """Tool results smaller than this skip the digest tier entirely.
    Mechanical head+tail (Tier 0) handles them adequately and the Ollama
    round-trip would cost more latency than its savings are worth."""

    tier0b_digest_timeout_seconds: float = 5.0
    """Per-batch timeout for the asyncio.gather of digest calls.
    Long enough for a warm 7B model on a typical machine; short enough
    that a cold or stuck Ollama doesn't stall the user's request."""

    tier0b_digest_cache_max_entries: int = 500
    """LRU bound for the digest cache. Each entry stores one digest
    text keyed by SHA-256 of the original tool_result content."""

    # ---- Compaction prompt ----
    render_preview_chars: int = 2_000
    """Max chars shown per message in the compaction prompt preview."""

    compaction_max_tokens: int = 4_000
    compaction_temperature: float = 0.3
    context_compaction_keep_alive: str = "30m"
    """Ollama keep_alive value for model warm-up calls."""

    # ---- Tokenizer ----
    tokenizer_name: str = "cl100k_base"
    """tiktoken encoding name (e.g. 'cl100k_base') or HuggingFace model ID
    (e.g. 'deepseek-ai/DeepSeek-V3') for accurate per-provider token counting.
    Names containing '/' are treated as HuggingFace model IDs and loaded via
    the `tokenizers` library; all other names are passed to tiktoken."""
