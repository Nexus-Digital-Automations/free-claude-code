"""Centralized configuration using Pydantic Settings."""

import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .nim import NimSettings


def _env_files() -> tuple[Path, ...]:
    """Return env file paths in priority order (later overrides earlier)."""
    files: list[Path] = [
        Path.home() / ".config" / "free-claude-code" / ".env",
        Path(".env"),
    ]
    if explicit := os.environ.get("FCC_ENV_FILE"):
        files.append(Path(explicit))
    return tuple(files)


def _configured_env_files(model_config: Mapping[str, Any]) -> tuple[Path, ...]:
    """Return the currently configured env files for Settings."""
    configured = model_config.get("env_file")
    if configured is None:
        return ()
    if isinstance(configured, str | Path):
        return (Path(configured),)
    return tuple(Path(item) for item in configured)


def _env_file_contains_key(path: Path, key: str) -> bool:
    """Check whether a dotenv-style file defines the given key."""
    return _env_file_value(path, key) is not None


def _env_file_value(path: Path, key: str) -> str | None:
    """Return a dotenv value when the file explicitly defines the key."""
    if not path.is_file():
        return None

    try:
        values = dotenv_values(path)
    except OSError:
        return None

    if key not in values:
        return None
    value = values[key]
    return "" if value is None else value


def _env_file_override(model_config: Mapping[str, Any], key: str) -> str | None:
    """Return the last configured dotenv value that explicitly defines a key."""
    configured_value: str | None = None
    for env_file in _configured_env_files(model_config):
        value = _env_file_value(env_file, key)
        if value is not None:
            configured_value = value
    return configured_value


def _removed_env_var_message(model_config: Mapping[str, Any]) -> str | None:
    """Return a migration error for removed env vars, if present."""
    removed_key = "NIM_ENABLE_THINKING"
    replacement = "ENABLE_THINKING"

    if removed_key in os.environ:
        return (
            f"{removed_key} has been removed in this release. "
            f"Rename it to {replacement}."
        )

    for env_file in _configured_env_files(model_config):
        if _env_file_contains_key(env_file, removed_key):
            return (
                f"{removed_key} has been removed in this release. "
                f"Rename it to {replacement}. Found in {env_file}."
            )

    return None


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==================== OpenRouter Config ====================
    open_router_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")

    # ==================== DeepSeek Config ====================
    deepseek_api_key: str = Field(default="", validation_alias="DEEPSEEK_API_KEY")

    # ==================== Messaging Platform Selection ====================
    # Valid: "telegram" | "discord"
    messaging_platform: str = Field(
        default="discord", validation_alias="MESSAGING_PLATFORM"
    )

    # ==================== NVIDIA NIM Config ====================
    nvidia_nim_api_key: str = ""

    # ==================== LM Studio Config ====================
    lm_studio_base_url: str = Field(
        default="http://localhost:1234/v1",
        validation_alias="LM_STUDIO_BASE_URL",
    )

    # ==================== Llama.cpp Config ====================
    llamacpp_base_url: str = Field(
        default="http://localhost:8080/v1",
        validation_alias="LLAMACPP_BASE_URL",
    )

    # ==================== Model ====================
    # All Claude model requests are mapped to this single model (fallback)
    # Format: provider_type/model/name
    model: str = "nvidia_nim/stepfun-ai/step-3.5-flash"

    # Per-model overrides (optional, falls back to MODEL)
    # Each can use a different provider
    model_opus: str | None = Field(default=None, validation_alias="MODEL_OPUS")
    model_sonnet: str | None = Field(default=None, validation_alias="MODEL_SONNET")
    model_haiku: str | None = Field(default=None, validation_alias="MODEL_HAIKU")

    # ==================== Per-Provider Proxy ====================
    nvidia_nim_proxy: str = Field(default="", validation_alias="NVIDIA_NIM_PROXY")
    open_router_proxy: str = Field(default="", validation_alias="OPENROUTER_PROXY")
    lmstudio_proxy: str = Field(default="", validation_alias="LMSTUDIO_PROXY")
    llamacpp_proxy: str = Field(default="", validation_alias="LLAMACPP_PROXY")

    # ==================== Provider Rate Limiting ====================
    provider_rate_limit: int = Field(default=40, validation_alias="PROVIDER_RATE_LIMIT")
    provider_rate_window: int = Field(
        default=60, validation_alias="PROVIDER_RATE_WINDOW"
    )
    provider_max_concurrency: int = Field(
        default=5, validation_alias="PROVIDER_MAX_CONCURRENCY"
    )
    enable_thinking: bool = Field(default=True, validation_alias="ENABLE_THINKING")

    # ==================== HTTP Client Timeouts ====================
    http_read_timeout: float = Field(
        default=120.0, validation_alias="HTTP_READ_TIMEOUT"
    )
    http_write_timeout: float = Field(
        default=10.0, validation_alias="HTTP_WRITE_TIMEOUT"
    )
    http_connect_timeout: float = Field(
        default=2.0, validation_alias="HTTP_CONNECT_TIMEOUT"
    )

    # ==================== Fast Prefix Detection ====================
    fast_prefix_detection: bool = True

    # ==================== Optimizations ====================
    enable_network_probe_mock: bool = True
    enable_title_generation_skip: bool = True
    enable_suggestion_mode_skip: bool = True
    enable_filepath_extraction_mock: bool = True

    # ==================== Context Autocompaction ====================
    # Tier 0: deterministic NLP strip (ANSI, dedup, truncate). Always-on.
    # Tier 0b/0c/0d: Ollama digests for tool results, tool_use inputs, long pastes.
    # Tier 1: strip thinking blocks from old turns. Always-on.
    # Layer 0 (block tower): immutable per-session compaction blocks plus an
    #   Ollama relevance selector that prunes irrelevant blocks per request.
    #   Cold-start emergency seal fires when tokens cross compact_threshold_tokens.
    # See providers/common/context_optimizer.py and packages/context-optimizer/
    # for full layer documentation.
    context_optimize: bool = Field(default=True, validation_alias="CONTEXT_OPTIMIZE")
    context_max_thinking_turns: int = Field(
        default=1, validation_alias="CONTEXT_MAX_THINKING_TURNS"
    )
    # Cold-start emergency seal threshold. When the very first request of a
    # session arrives already over this size and no blocks have been sealed
    # yet, the block tower runs a synchronous seal (bounded ~12s) before
    # forwarding so the upstream payload stays under budget.
    context_compact_threshold_tokens: int = Field(
        default=65000, validation_alias="CONTEXT_COMPACT_THRESHOLD_TOKENS"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="qwen2.5:7b", validation_alias="OLLAMA_MODEL")
    # Bounded await for the supervisor warm-up at app startup. Long enough that
    # `ollama serve` boot + small-model warm completes on a warm box; short
    # enough that we don't block proxy startup if Ollama is missing.
    # Counterpart: api/app.py:lifespan, ollama_supervisor.OllamaSupervisor.
    ollama_warmup_max_wait_s: float = Field(
        default=8.0, validation_alias="OLLAMA_WARMUP_MAX_WAIT_S"
    )

    # Context-optimizer package config that the adapter forwards verbatim.
    # The package exposes these as ContextOptimizerSettings dataclass fields.
    context_tier0_max_lines: int = Field(
        default=200, validation_alias="CONTEXT_TIER0_MAX_LINES"
    )
    context_tier0_head_lines: int = Field(
        default=50, validation_alias="CONTEXT_TIER0_HEAD_LINES"
    )
    context_tier0_tail_lines: int = Field(
        default=50, validation_alias="CONTEXT_TIER0_TAIL_LINES"
    )
    context_render_preview_chars: int = Field(
        default=2000, validation_alias="CONTEXT_RENDER_PREVIEW_CHARS"
    )
    context_compaction_max_tokens: int = Field(
        default=4000, validation_alias="CONTEXT_COMPACTION_MAX_TOKENS"
    )
    context_compaction_temperature: float = Field(
        default=0.3, validation_alias="CONTEXT_COMPACTION_TEMPERATURE"
    )
    context_compaction_keep_alive: str = Field(
        default="30m", validation_alias="CONTEXT_COMPACTION_KEEP_ALIVE"
    )
    context_tokenizer_model: str = Field(
        default="deepseek-ai/DeepSeek-V3",
        validation_alias="CONTEXT_TOKENIZER_MODEL",
        description=(
            "HuggingFace model ID (e.g. 'deepseek-ai/DeepSeek-V3') or tiktoken encoding name "
            "(e.g. 'cl100k_base') used for token counting across both the compaction optimizer "
            "and request logging. Names containing '/' are loaded via the `tokenizers` library "
            "with automatic fallback to cl100k_base on download failure."
        ),
    )
    context_tier0b_digest_enabled: bool = Field(
        default=True,
        validation_alias="CONTEXT_TIER0B_DIGEST_ENABLED",
        description=(
            "Run Ollama-based content-aware digest on long tool_result blocks before "
            "they enter conversation history. Reduces cumulative input tokens billed by "
            "the upstream provider on every subsequent request that carries the same "
            "tool_result. Falls back to Tier 0's mechanical head/tail truncation on any "
            "Ollama failure."
        ),
    )
    context_tier0b_digest_min_bytes: int = Field(
        default=8000,
        validation_alias="CONTEXT_TIER0B_DIGEST_MIN_BYTES",
        description=(
            "Tool results smaller than this byte threshold skip the digest tier. Below "
            "this size the Ollama round-trip costs more latency than its savings are "
            "worth."
        ),
    )
    context_tier0b_digest_timeout_seconds: float = Field(
        default=5.0,
        validation_alias="CONTEXT_TIER0B_DIGEST_TIMEOUT_SECONDS",
        description=(
            "Per-batch timeout for the asyncio.gather of Ollama digest calls. On "
            "timeout the affected tool_results pass through with Tier 0's mechanical "
            "truncation already applied."
        ),
    )
    context_tier0c_digest_enabled: bool = Field(
        default=True,
        validation_alias="CONTEXT_TIER0C_DIGEST_ENABLED",
        description=(
            "Run Ollama digest on old assistant tool_use input dicts (Edit, Write, "
            "MultiEdit) when serialised input exceeds the byte threshold. Skips the "
            "most recent N tool_use blocks so the model can still reference its "
            "latest call args."
        ),
    )
    context_tier0c_digest_min_bytes: int = Field(
        default=4000,
        validation_alias="CONTEXT_TIER0C_DIGEST_MIN_BYTES",
        description=(
            "Tool_use blocks whose serialised input is below this byte size skip the "
            "tier0c digest pass."
        ),
    )
    context_tier0c_keep_recent_calls: int = Field(
        default=3,
        validation_alias="CONTEXT_TIER0C_KEEP_RECENT_CALLS",
        description=(
            "How many trailing tool_use blocks tier0c keeps verbatim. Protects "
            "against the model re-using its most recent call args."
        ),
    )
    context_tier0d_digest_enabled: bool = Field(
        default=True,
        validation_alias="CONTEXT_TIER0D_DIGEST_ENABLED",
        description=(
            "Run Ollama digest on long historical user-text blocks. Always skips "
            "the LAST user message (the active request). Length-gated by "
            "CONTEXT_TIER0D_DIGEST_MIN_BYTES."
        ),
    )
    context_tier0d_digest_min_bytes: int = Field(
        default=16_000,
        validation_alias="CONTEXT_TIER0D_DIGEST_MIN_BYTES",
        description=(
            "User-text blocks smaller than this byte size pass through unchanged. "
            "The default (16 KB ~= 4K tokens) is high to ensure typical "
            "conversational prompts are never digested."
        ),
    )

    # ---- Block tower (Layer 0 — sole conversation-level compaction path) ----
    # Counterpart: packages/context-optimizer/.../block_tower/. Always-on; the
    # tower seals immutable per-session blocks and prunes irrelevant ones per
    # request via the Ollama selector. Set context_block_selection_mode='off'
    # to disable Layer 0 entirely (only the per-message tiers run).
    context_block_selection_mode: str = Field(
        default="selective",
        validation_alias="CONTEXT_BLOCK_SELECTION_MODE",
        description=(
            "One of {'selective', 'all', 'off'}. 'selective' = Ollama picks "
            "relevant blocks per request; 'all' = always include every block "
            "(skips selector call); 'off' = disable Layer 0 entirely."
        ),
    )
    context_block_seal_min_tail_tokens: int = Field(
        default=3_000,
        validation_alias="CONTEXT_BLOCK_SEAL_MIN_TAIL_TOKENS",
        description=(
            "Minimum uncompacted-tail tokens before sealing is mathematically "
            "profitable. Below this, the one-time write cost dominates the "
            "recurring savings."
        ),
    )
    context_block_seal_min_requests: int = Field(
        default=4,
        validation_alias="CONTEXT_BLOCK_SEAL_MIN_REQUESTS",
        description=(
            "Minimum requests since the last seal before a new block may be "
            "sealed. Protects short sessions from one-shot compactions whose "
            "cost would never amortise."
        ),
    )
    context_block_target_summary_tokens: int = Field(
        default=500,
        validation_alias="CONTEXT_BLOCK_TARGET_SUMMARY_TOKENS",
        description="Target body size for a sealed block, passed to the seal prompt.",
    )
    context_block_storage_dir: str | None = Field(
        default=None,
        validation_alias="CONTEXT_BLOCK_STORAGE_DIR",
        description=(
            "Directory under which <session_key>/block-NNNN.txt files live. "
            "None = <repo_root>/.context/blocks/ auto-derived from cwd."
        ),
    )

    preflight_token_count: bool = Field(
        default=False,
        validation_alias="PREFLIGHT_TOKEN_COUNT",
        description=(
            "Make a max_tokens=1 non-streaming call before each stream to obtain the "
            "provider's actual prompt_tokens for message_start.usage.input_tokens. "
            "Without this, Claude Code's TUI displays cl100k_base estimates that "
            "diverge from DeepSeek/upstream tokenizers by 1.65-2.35x. "
            "Disable with PREFLIGHT_TOKEN_COUNT=0."
        ),
    )

    # ==================== NIM Settings ====================
    nim: NimSettings = Field(default_factory=NimSettings)

    # ==================== Voice Note Transcription ====================
    voice_note_enabled: bool = Field(
        default=True, validation_alias="VOICE_NOTE_ENABLED"
    )
    # Device: "cpu" | "cuda" | "nvidia_nim"
    # - "cpu"/"cuda": local Whisper (requires voice_local extra: uv sync --extra voice_local)
    # - "nvidia_nim": NVIDIA NIM Whisper API (requires voice extra: uv sync --extra voice)
    whisper_device: str = Field(default="cpu", validation_alias="WHISPER_DEVICE")
    # Whisper model ID or short name (for local Whisper) or NVIDIA NIM model (for nvidia_nim)
    # Local Whisper: "tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"
    # NVIDIA NIM: "nvidia/parakeet-ctc-1.1b-asr", "openai/whisper-large-v3", etc.
    whisper_model: str = Field(default="base", validation_alias="WHISPER_MODEL")
    # Hugging Face token for faster model downloads (optional, for local Whisper)
    hf_token: str = Field(default="", validation_alias="HF_TOKEN")

    # ==================== Bot Wrapper Config ====================
    telegram_bot_token: str | None = None
    allowed_telegram_user_id: str | None = None
    discord_bot_token: str | None = Field(
        default=None, validation_alias="DISCORD_BOT_TOKEN"
    )
    allowed_discord_channels: str | None = Field(
        default=None, validation_alias="ALLOWED_DISCORD_CHANNELS"
    )
    claude_workspace: str = "./agent_workspace"
    allowed_dir: str = ""

    # ==================== Server ====================
    host: str = "0.0.0.0"
    port: int = 8082
    log_file: str = "logs/server.log"
    # Opt-in Prometheus exposition at /metrics. Off by default to keep the
    # endpoint surface minimal in solo deployments. Counterpart: api/metrics.py.
    metrics_enabled: bool = Field(default=False, validation_alias="METRICS_ENABLED")
    # When true, the full request body Claude Code sends (system prompt,
    # tools, messages) is logged at INFO. Highest-signal line for
    # reverse-engineering Claude Code behavior; opt-out is per-deploy
    # (set LOG_FULL_PAYLOAD=0) because it is verbose.
    log_full_payload: bool = Field(
        default=True, validation_alias="LOG_FULL_PAYLOAD",
    )
    # Optional server API key to protect endpoints (Anthropic-style)
    # Set via env `ANTHROPIC_AUTH_TOKEN`. When empty, no auth is required.
    anthropic_auth_token: str = Field(
        default="", validation_alias="ANTHROPIC_AUTH_TOKEN"
    )

    @model_validator(mode="before")
    @classmethod
    def reject_removed_env_vars(cls, data: Any) -> Any:
        """Fail fast when removed environment variables are still configured."""
        if message := _removed_env_var_message(cls.model_config):
            raise ValueError(message)
        return data

    # Handle empty strings for optional string fields
    @field_validator(
        "telegram_bot_token",
        "allowed_telegram_user_id",
        "discord_bot_token",
        "allowed_discord_channels",
        mode="before",
    )
    @classmethod
    def parse_optional_str(cls, v: Any) -> Any:
        if v == "":
            return None
        return v

    @field_validator("whisper_device")
    @classmethod
    def validate_whisper_device(cls, v: str) -> str:
        if v not in ("cpu", "cuda", "nvidia_nim"):
            raise ValueError(
                f"whisper_device must be 'cpu', 'cuda', or 'nvidia_nim', got {v!r}"
            )
        return v

    @field_validator("model", "model_opus", "model_sonnet", "model_haiku")
    @classmethod
    def validate_model_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        valid_providers = (
            "nvidia_nim",
            "open_router",
            "deepseek",
            "lmstudio",
            "llamacpp",
        )
        if "/" not in v:
            raise ValueError(
                f"Model must be prefixed with provider type. "
                f"Valid providers: {', '.join(valid_providers)}. "
                f"Format: provider_type/model/name"
            )
        provider = v.split("/", 1)[0]
        if provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: '{provider}'. "
                f"Supported: 'nvidia_nim', 'open_router', 'deepseek', 'lmstudio', 'llamacpp'"
            )
        return v

    @model_validator(mode="after")
    def check_nvidia_nim_api_key(self) -> Settings:
        if (
            self.voice_note_enabled
            and self.whisper_device == "nvidia_nim"
            and not self.nvidia_nim_api_key.strip()
        ):
            raise ValueError(
                "NVIDIA_NIM_API_KEY is required when WHISPER_DEVICE is 'nvidia_nim'. "
                "Set it in your .env file."
            )
        return self

    @model_validator(mode="after")
    def prefer_dotenv_anthropic_auth_token(self) -> Settings:
        """Let explicit .env auth config override stale shell/client tokens."""
        dotenv_value = _env_file_override(self.model_config, "ANTHROPIC_AUTH_TOKEN")
        if dotenv_value is not None:
            self.anthropic_auth_token = dotenv_value
        return self

    def uses_process_anthropic_auth_token(self) -> bool:
        """Return whether proxy auth came from process env, not dotenv config."""
        if _env_file_override(self.model_config, "ANTHROPIC_AUTH_TOKEN") is not None:
            return False
        return bool(os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    @property
    def provider_type(self) -> str:
        """Extract provider type from the default model string."""
        return self.model.split("/", 1)[0]

    @property
    def model_name(self) -> str:
        """Extract the actual model name from the default model string."""
        return self.model.split("/", 1)[1]

    def resolve_model(self, claude_model_name: str) -> str:
        """Resolve a Claude model name to the configured provider/model string.

        Classifies the incoming Claude model (opus/sonnet/haiku) and
        returns the model-specific override if configured, otherwise the fallback MODEL.
        """
        name_lower = claude_model_name.lower()
        if "opus" in name_lower and self.model_opus is not None:
            return self.model_opus
        if "haiku" in name_lower and self.model_haiku is not None:
            return self.model_haiku
        if "sonnet" in name_lower and self.model_sonnet is not None:
            return self.model_sonnet
        return self.model

    @staticmethod
    def parse_provider_type(model_string: str) -> str:
        """Extract provider type from any 'provider/model' string."""
        return model_string.split("/", 1)[0]

    @staticmethod
    def parse_model_name(model_string: str) -> str:
        """Extract model name from any 'provider/model' string."""
        return model_string.split("/", 1)[1]

    model_config = SettingsConfigDict(
        env_file=_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
