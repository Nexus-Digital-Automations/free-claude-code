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
    compact_threshold_tokens: int = 200_000
    """Hard limit — sync blocking compaction via llm_provider."""

    compact_soft_threshold_tokens: int = 80_000
    """Soft limit — schedule background Ollama compaction."""

    compact_deepseek_fallback_threshold_tokens: int = 150_000
    """Mid-point — near hard limit; Ollama fallback to llm_provider if busy."""

    # ---- Tier 1 ----
    max_thinking_turns: int = 2
    """How many recent assistant turns keep their thinking blocks."""

    # ---- Prefix cache ----
    prefix_cache_max_entries: int = 100

    # ---- Ollama ----
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"

    # ---- Tier 0 ----
    tier0_max_lines: int = 200
    """Tool results longer than this get head+tail truncation."""

    tier0_head_lines: int = 50
    tier0_tail_lines: int = 50

    # ---- Compaction prompt ----
    render_preview_chars: int = 2_000
    """Max chars shown per message in the compaction prompt preview."""

    compaction_max_tokens: int = 4_000
    compaction_temperature: float = 0.3
    compaction_keep_alive: str = "30m"
    """Ollama keep_alive value for model warm-up calls."""
