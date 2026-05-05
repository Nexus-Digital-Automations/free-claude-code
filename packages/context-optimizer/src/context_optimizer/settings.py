"""Owns: ContextOptimizerSettings — configuration for all optimizer tiers.

Does NOT own: defaults validation (caller's responsibility), provider
configuration (caller constructs llm_provider callable separately).

Called by: optimizer.py, tiers/tier2.py, ollama_supervisor.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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

    context_cache_dir: str | None = None
    """Directory for the persisted prefix-cache file.  ``None`` auto-resolves
    from the current working directory (git root → ``.claude/data/``, or
    ``cwd/.claude/data/`` if not a git repo).  Set to an explicit path to
    override location.  The cache file is always named ``context-cache.json``.
    Set to ``""`` to disable persistence entirely."""

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

    # ---- Tier 0c (Ollama tool_use input digester) ----
    tier0c_digest_enabled: bool = True
    """Run Ollama digest on tool_use input blocks above the byte threshold,
    skipping the last `tier0c_keep_recent_calls` tool_use blocks so the model
    can still reference its most recent calls verbatim."""

    tier0c_digest_min_bytes: int = 4000
    """Tool_use blocks whose serialised input is below this byte size skip the
    digest tier. Most tool calls (Read, Bash, simple Edit) are small; Edit/Write
    of large bodies and MultiEdit are the realistic candidates."""

    tier0c_keep_recent_calls: int = 3
    """How many trailing tool_use blocks are kept verbatim. Protects against
    the model re-using its most recent call args ('let me redo that edit
    with X')."""

    # ---- Tier 0d (Ollama long-user-paste digester) ----
    tier0d_digest_enabled: bool = True
    """Run Ollama digest on long user-message text blocks. Always skips the
    LAST user message (the active request) — only historical user pastes
    are eligible. Length-gated by tier0d_digest_min_bytes."""

    tier0d_digest_min_bytes: int = 16_000
    """User-text blocks smaller than this size pass through unchanged. Set
    high (16K = ~4K tokens) so we only digest genuinely large pastes —
    typical conversational prompts must never be touched."""

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

    # ---- Repo index (Layer -1 stable prefix) ----
    # Disabled by default so existing callers see no behaviour change.
    # Enable by setting repo_index_enabled=True; auto-detects git root from cwd.
    repo_index_enabled: bool = False
    repo_index_root: str | None = None
    """Absolute path to the git repo root to index. None = auto-detect from cwd."""

    repo_index_context_dir: str | None = None
    """Directory for .context/ files. None = <git_root>/.context/."""

    repo_index_top_n: int = 0
    """Hard upper bound on files passed to the mass selector.
    0 = auto: min(20 + n_tracked_files // 10, 100) — scales with repo size so the mass
    selector has room to work without an arbitrary cap silently overriding it.
    Set explicitly to pin the ceiling (e.g. 30 for a tightly-scoped prefix).
    Counterpart: index._compute_effective_top_n."""

    repo_index_chunk_size_tokens: int = 200
    repo_index_chunk_overlap_tokens: int = 20

    repo_index_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    """HuggingFace model ID for chunk embeddings (sentence-transformers compatible)."""

    repo_index_query_top_k: int = 10
    repo_index_repomix_timeout: float = 120.0
    repo_index_poll_interval_seconds: float = 30.0
    repo_index_watch_enabled: bool = False

    repo_index_pagerank_mass_target: float = 0.80
    """Fraction of total PageRank score mass to cover when selecting files.
    The mass selector stops adding files once their cumulative score reaches this fraction
    of the total graph score. Lower values (0.60) suit monoliths where a few hub files
    dominate; higher values (0.90) suit flat utility-heavy repos. Bounded by repo_index_top_n."""

    repo_index_max_prefix_tokens: int = 0
    """Hard token ceiling for the stable prefix.
    0 = auto: min(8_000 + sqrt(n_tracked_files) * 1_200, 56_000).
    Set explicitly to override. Counterpart: index._compute_token_ceiling."""

    repo_index_repomix_extra_args: list[str] = field(default_factory=list)
    """Additional CLI arguments passed verbatim to repomix."""
