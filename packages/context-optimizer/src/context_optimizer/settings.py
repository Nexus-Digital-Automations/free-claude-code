"""Owns: ContextOptimizerSettings — configuration for all optimizer layers.

Does NOT own: defaults validation (caller's responsibility) or provider
configuration (the package owns its own Ollama client; no llm_provider
needed by callers any more).

Called by: optimizer.py, ollama_supervisor.py, block_tower/*.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContextOptimizerSettings:
    """Plain-dataclass configuration so callers need no Pydantic dependency.

    All thresholds are in approximate token counts (tiktoken cl100k_base).
    """

    # ---- Cold-start emergency seal ----
    # Repurposed from the old Tier 2 hard threshold. When the very first
    # request of a session arrives already over this size and no blocks
    # have been sealed yet, the block tower runs a synchronous seal_sync
    # (bounded by sealer._SYNC_SEAL_TIMEOUT_SECONDS) before the request
    # is forwarded. Counterpart: block_tower/sealer.py:seal_sync.
    compact_threshold_tokens: int = 55_000
    """Tail-token threshold above which a cold-start emergency seal fires."""

    # ---- Tier 1 ----
    max_thinking_turns: int = 0
    """How many recent assistant turns keep their thinking blocks. Default 0
    strips all thinking — both saves tokens and keeps the prefix byte-stable
    across consecutive requests, which maximises DeepSeek prefix-cache hits.
    A relative 'keep last N' rule would mutate the prefix every time a new
    turn arrived (the position of 'last N' shifts). Set higher only if
    downstream tooling needs to inspect intermediate reasoning."""

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

    tier0c_digest_timeout_seconds: float = 5.0
    """Per-batch timeout for tier0c's asyncio.gather of digest calls. Defaults
    to the tier0b value because the same Ollama model handles both, but kept
    independent so tool_use prompts (often longer JSON inputs) can be tuned
    without affecting tier0b's tool_result digest budget."""

    tier0c_digest_cache_max_entries: int = 500
    """LRU bound for tier0c's digest cache. Independent of tier0b/0d so each
    tier's hit-rate can be tuned to its own input distribution."""

    # ---- Tier 0d (Ollama long-user-paste digester) ----
    tier0d_digest_enabled: bool = True
    """Run Ollama digest on long user-message text blocks. Always skips the
    LAST user message (the active request) — only historical user pastes
    are eligible. Length-gated by tier0d_digest_min_bytes."""

    tier0d_digest_min_bytes: int = 16_000
    """User-text blocks smaller than this size pass through unchanged. Set
    high (16K = ~4K tokens) so we only digest genuinely large pastes —
    typical conversational prompts must never be touched."""

    tier0d_digest_timeout_seconds: float = 5.0
    """Per-batch timeout for tier0d's asyncio.gather of digest calls. Independent
    of tier0b so user-paste digests (typically larger inputs than tool_results)
    can be given a longer budget without slowing the tool_result tier."""

    tier0d_digest_cache_max_entries: int = 500
    """LRU bound for tier0d's digest cache. Independent of tier0b/0c so each
    tier's hit-rate can be tuned to its own input distribution."""

    # ---- Tier 0e (error-aware tool-call filter) ----
    tier0e_enabled: bool = True
    """When True, drop the input payload from every assistant tool_use whose
    paired tool_result is NOT classified as an error. Tool_result content is
    preserved verbatim — the codebase signal the model needs survives; only
    the command/args wrapper is shed.

    Errored pairs pass through unchanged. Three error signals (any one
    triggers retention): the canonical `is_error` flag on tool_result, a
    Bash-specific non-zero exit-code pattern in result text, and a keyword
    scan scoped to noisy-tool names (Bash/WebFetch/WebSearch) where stderr-
    shaped output reliably means failure. Read/Edit/Write/Grep/Glob are
    excluded from the keyword scan to avoid false positives when source
    files legitimately contain words like "error".

    Default ON. Counterpart: tiers/tier0e.py."""

    # ---- Tier 0f (span-level Rabin-Karp dedup) ----
    tier0f_enabled: bool = True
    """Drop repeated text spans >= tier0f_min_tokens that appear more than
    once across messages. First occurrence kept verbatim; later occurrences
    deleted with no replacement marker.

    System prompt and last user message act as read-only definer sources:
    their content can never be deleted (system protects DeepSeek's prefix
    cache; last user message protects the active query) but spans inside
    them register as "originals" so messages duplicating them get cleaned.

    Default ON. Counterpart: tiers/tier0f.py."""

    tier0f_min_tokens: int = 70
    """Minimum K-gram length for a duplicate span to be eligible for deletion.
    At 70 cl100k tokens (~280 chars / ~50 words / 3 sentences), real-world
    duplicates are dominated by repeated structural blocks (file outlines,
    error traces, code) that begin/end at newlines, so deletions land on
    whitespace boundaries — mid-text fusion is rare. Lowering to 50 captures
    ~10% more savings but raises fusion risk noticeably; raising to 100 is
    safest but starts missing recurring boilerplate."""

    tier0f_skip_system: bool = True
    """When True, the system prompt is treated as a read-only definer source.
    Its content can be matched against, but never deleted from. Load-bearing:
    a mutated system prompt invalidates DeepSeek's prefix cache for every
    subsequent request, costing far more than dedup saves."""

    tier0f_skip_last_user: bool = True
    """When True, the most recent user message is treated as a read-only
    definer source. Protects the active request from accidental content loss
    when it quotes earlier history."""

    render_preview_chars: int = 2_000
    """Max chars shown per message in the block-tower seal prompt preview."""

    compaction_max_tokens: int = 4_000
    """Output token cap for tier0b/0c/0d Ollama digest calls. Block-tower
    seals compute their own cap from block_target_summary_tokens."""

    compaction_temperature: float = 0.0
    """Temperature for the block-tower seal Ollama call and tier0b/0c/0d
    digest calls. 0.0 enforces greedy decoding so identical inputs produce
    byte-identical outputs — load-bearing for DeepSeek prefix cache stability
    across daemon restarts and concurrent cache misses on the same input."""

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

    # ---- Block tower (Layer 0 — sole conversation-level compaction path) ----
    # Counterpart: block_tower/ module. The tower seals immutable blocks
    # from the uncompacted tail, prunes irrelevant blocks per-request via
    # an Ollama selector, and runs a synchronous emergency seal if tokens
    # exceed compact_threshold_tokens before any block is sealed.
    block_selection_mode: str = "selective"
    """One of {"all", "selective", "off"}.
    "off"        — Layer 0 disabled entirely (no tower load, no seal, no
                   prefix prepend). Only the per-message tiers run.
    "all"        — load tower, include every block (no Ollama selector call).
    "selective"  — Ollama scores each block against the current message and
                   omits blocks deemed irrelevant. Falls back to "all" on any
                   Ollama failure (preserves request reliability)."""

    block_seal_min_tail_tokens: int = 3_000
    """Tail token count below which sealing a new block is mathematically
    unprofitable (the one-time write cost dominates the recurring token
    savings). Counterpart: block_tower/sealer.should_seal."""

    block_seal_min_requests: int = 4
    """Minimum requests since the last seal (or session start) before a new
    block may be sealed. Protects short sessions from a one-shot compaction
    whose cost would never be amortised over future requests."""

    block_target_summary_tokens: int = 500
    """Target body size for a sealed block. Passed to the sealing prompt as
    a budget — actual size will vary ± ~20%."""

    block_storage_dir: str | None = None
    """Directory under which `<session_key>/block-NNNN.txt` files live.
    None = <repo_root>/.context/blocks/ (auto-derived from cwd's git root)."""
