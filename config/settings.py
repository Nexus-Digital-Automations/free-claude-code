"""Centralized configuration using Pydantic Settings."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import HTTP_CONNECT_TIMEOUT_DEFAULT
from .nim import NimSettings
from .paths import default_claude_workspace_path, managed_env_path
from .provider_ids import SUPPORTED_PROVIDER_IDS


@dataclass(frozen=True, slots=True)
class ConfiguredChatModelRef:
    """A unique configured chat model reference and the env keys that set it."""

    model_ref: str
    provider_id: str
    model_id: str
    sources: tuple[str, ...]


def _env_files() -> tuple[Path, ...]:
    """Return env file paths in priority order (later overrides earlier)."""
    files: list[Path] = [
        Path(".env"),
        managed_env_path(),
    ]
    if explicit := os.environ.get("FCC_ENV_FILE"):
        files.append(Path(explicit))
    return tuple(files)


def _configured_env_files(model_config: Mapping[str, Any]) -> tuple[Path, ...]:
    """Return the currently configured env files for Settings."""
    configured = model_config.get("env_file")
    if configured is None:
        return ()
    if isinstance(configured, (str, Path)):
        return (Path(configured),)
    return tuple(Path(item) for item in configured)


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


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==================== OpenRouter Config ====================
    open_router_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")

    # ==================== Mistral La Plateforme ====================
    mistral_api_key: str = Field(default="", validation_alias="MISTRAL_API_KEY")

    # ==================== Mistral Codestral (codestral.mistral.ai) ====================
    codestral_api_key: str = Field(default="", validation_alias="CODESTRAL_API_KEY")

    # ==================== DeepSeek Config ====================
    deepseek_api_key: str = Field(default="", validation_alias="DEEPSEEK_API_KEY")

    # ==================== Kimi Config ====================
    kimi_api_key: str = Field(default="", validation_alias="KIMI_API_KEY")

    # ==================== Wafer Config ====================
    wafer_api_key: str = Field(default="", validation_alias="WAFER_API_KEY")

    # ==================== OpenCode Zen / OpenCode Go ====================
    # Same key from opencode.ai/auth; zen uses prefix ``opencode/``, Go uses ``opencode_go/``.
    opencode_api_key: str = Field(default="", validation_alias="OPENCODE_API_KEY")

    # ==================== Z.ai Config ====================
    zai_api_key: str = Field(default="", validation_alias="ZAI_API_KEY")

    # ==================== Fireworks AI Config ====================
    fireworks_api_key: str = Field(default="", validation_alias="FIREWORKS_API_KEY")

    # ==================== Google Gemini (Google AI Studio) ====================
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")

    # ==================== Groq (OpenAI-compatible) ====================
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")

    # ==================== Cerebras Inference (OpenAI-compatible) ====================
    cerebras_api_key: str = Field(default="", validation_alias="CEREBRAS_API_KEY")

    # ==================== Messaging Platform Selection ====================
    # Valid: "telegram" | "discord" | "none"
    messaging_platform: str = Field(
        default="discord", validation_alias="MESSAGING_PLATFORM"
    )
    messaging_rate_limit: int = Field(
        default=1, validation_alias="MESSAGING_RATE_LIMIT"
    )
    messaging_rate_window: float = Field(
        default=1.0, validation_alias="MESSAGING_RATE_WINDOW"
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

    # ==================== Ollama Config ====================
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    # Context-optimizer reaches Ollama over the OpenAI-compatible /v1 surface;
    # its adapter appends /v1 to ollama_base_url rather than duplicating the URL.
    ollama_model: str = Field(default="qwen2.5:7b", validation_alias="OLLAMA_MODEL")
    # Bounded await for the supervisor warm-up at app startup: long enough that
    # `ollama serve` boot + small-model warm completes on a warm box, short
    # enough that we don't block proxy startup if Ollama is missing.
    # Counterpart: api/runtime.py AppRuntime, context_optimizer.OllamaSupervisor.
    ollama_warmup_max_wait_s: float = Field(
        default=8.0, validation_alias="OLLAMA_WARMUP_MAX_WAIT_S"
    )

    # ==================== Model ====================
    # All Claude model requests are mapped to this single model (fallback)
    # Format: provider_type/model/name
    model: str = "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"

    # Per-model overrides (optional, falls back to MODEL)
    # Each can use a different provider
    model_opus: str | None = Field(default=None, validation_alias="MODEL_OPUS")
    model_sonnet: str | None = Field(default=None, validation_alias="MODEL_SONNET")
    model_haiku: str | None = Field(default=None, validation_alias="MODEL_HAIKU")

    # ==================== Vertex AI (fork) ====================
    vertex_project: str = Field(default="", validation_alias="VERTEX_PROJECT")
    vertex_region: str = Field(default="us-central1", validation_alias="VERTEX_REGION")
    vertex_endpoint_id: str = Field(default="", validation_alias="VERTEX_ENDPOINT_ID")
    vertex_access_token: str = Field(default="", validation_alias="VERTEX_ACCESS_TOKEN")
    vertex_credentials_file: str = Field(
        default="", validation_alias="VERTEX_CREDENTIALS_FILE"
    )
    # Comma-separated; Vertex endpoints have no /models discovery, so we
    # accept the inventory statically (e.g. "gemma-3-9b-it,gemma-3-27b-it").
    vertex_models: str = Field(default="", validation_alias="VERTEX_MODELS")
    vertex_proxy: str = Field(default="", validation_alias="VERTEX_PROXY")

    # ==================== Context optimization (fork) ====================
    # Counterpart: packages/context-optimizer + api/context_optimization.py.
    context_optimize: bool = Field(default=True, validation_alias="CONTEXT_OPTIMIZE")
    context_max_thinking_turns: int = Field(
        default=1, validation_alias="CONTEXT_MAX_THINKING_TURNS"
    )
    # Cold-start emergency seal threshold. When the first request of a session
    # arrives already over this size and no blocks have been sealed yet, the
    # block tower runs a synchronous seal (bounded ~12s) before forwarding.
    context_compact_threshold_tokens: int = Field(
        default=65000, validation_alias="CONTEXT_COMPACT_THRESHOLD_TOKENS"
    )
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
            "HuggingFace model ID or tiktoken encoding name used for token "
            "counting across the compaction optimizer and request logging. "
            "Names containing '/' load via the `tokenizers` library with "
            "automatic fallback to cl100k_base on download failure."
        ),
    )
    context_tier0b_digest_enabled: bool = Field(
        default=True, validation_alias="CONTEXT_TIER0B_DIGEST_ENABLED"
    )
    context_tier0b_digest_min_bytes: int = Field(
        default=8000, validation_alias="CONTEXT_TIER0B_DIGEST_MIN_BYTES"
    )
    context_tier0b_digest_timeout_seconds: float = Field(
        default=5.0, validation_alias="CONTEXT_TIER0B_DIGEST_TIMEOUT_SECONDS"
    )
    context_tier0c_digest_enabled: bool = Field(
        default=True, validation_alias="CONTEXT_TIER0C_DIGEST_ENABLED"
    )
    context_tier0c_digest_min_bytes: int = Field(
        default=4000, validation_alias="CONTEXT_TIER0C_DIGEST_MIN_BYTES"
    )
    context_tier0c_keep_recent_calls: int = Field(
        default=3, validation_alias="CONTEXT_TIER0C_KEEP_RECENT_CALLS"
    )
    context_tier0d_digest_enabled: bool = Field(
        default=True, validation_alias="CONTEXT_TIER0D_DIGEST_ENABLED"
    )
    context_tier0d_digest_min_bytes: int = Field(
        default=16_000, validation_alias="CONTEXT_TIER0D_DIGEST_MIN_BYTES"
    )
    context_tier0e_enabled: bool = Field(
        default=True, validation_alias="CONTEXT_TIER0E_ENABLED"
    )
    # ---- Block tower (Layer 0 — sole conversation-level compaction path) ----
    context_block_selection_mode: str = Field(
        default="selective",
        validation_alias="CONTEXT_BLOCK_SELECTION_MODE",
        description=(
            "One of {'selective','all','off'}. 'selective' = Ollama picks "
            "relevant blocks per request; 'all' = include every block; "
            "'off' = disable Layer 0 entirely."
        ),
    )
    context_block_seal_min_tail_tokens: int = Field(
        default=3_000, validation_alias="CONTEXT_BLOCK_SEAL_MIN_TAIL_TOKENS"
    )
    context_block_seal_min_requests: int = Field(
        default=4, validation_alias="CONTEXT_BLOCK_SEAL_MIN_REQUESTS"
    )
    context_block_target_summary_tokens: int = Field(
        default=500, validation_alias="CONTEXT_BLOCK_TARGET_SUMMARY_TOKENS"
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
            "Make a max_tokens=1 non-streaming call before each stream to obtain "
            "the provider's actual prompt_tokens for message_start usage. Without "
            "this, Claude Code's TUI shows cl100k_base estimates that diverge from "
            "DeepSeek/upstream tokenizers by 1.65-2.35x."
        ),
    )

    # ==================== Per-Provider Proxy ====================
    nvidia_nim_proxy: str = Field(default="", validation_alias="NVIDIA_NIM_PROXY")
    open_router_proxy: str = Field(default="", validation_alias="OPENROUTER_PROXY")
    mistral_proxy: str = Field(default="", validation_alias="MISTRAL_PROXY")
    codestral_proxy: str = Field(default="", validation_alias="CODESTRAL_PROXY")
    lmstudio_proxy: str = Field(default="", validation_alias="LMSTUDIO_PROXY")
    llamacpp_proxy: str = Field(default="", validation_alias="LLAMACPP_PROXY")
    kimi_proxy: str = Field(default="", validation_alias="KIMI_PROXY")
    wafer_proxy: str = Field(default="", validation_alias="WAFER_PROXY")
    opencode_proxy: str = Field(default="", validation_alias="OPENCODE_PROXY")
    opencode_go_proxy: str = Field(default="", validation_alias="OPENCODE_GO_PROXY")
    zai_proxy: str = Field(default="", validation_alias="ZAI_PROXY")
    fireworks_proxy: str = Field(default="", validation_alias="FIREWORKS_PROXY")
    gemini_proxy: str = Field(default="", validation_alias="GEMINI_PROXY")
    groq_proxy: str = Field(default="", validation_alias="GROQ_PROXY")
    cerebras_proxy: str = Field(default="", validation_alias="CEREBRAS_PROXY")

    # ==================== Provider Rate Limiting ====================
    provider_rate_limit: int = Field(default=40, validation_alias="PROVIDER_RATE_LIMIT")
    provider_rate_window: int = Field(
        default=60, validation_alias="PROVIDER_RATE_WINDOW"
    )
    provider_max_concurrency: int = Field(
        default=5, validation_alias="PROVIDER_MAX_CONCURRENCY"
    )
    enable_model_thinking: bool = Field(
        default=True, validation_alias="ENABLE_MODEL_THINKING"
    )
    enable_opus_thinking: bool | None = Field(
        default=None, validation_alias="ENABLE_OPUS_THINKING"
    )
    enable_sonnet_thinking: bool | None = Field(
        default=None, validation_alias="ENABLE_SONNET_THINKING"
    )
    enable_haiku_thinking: bool | None = Field(
        default=None, validation_alias="ENABLE_HAIKU_THINKING"
    )

    # ==================== HTTP Client Timeouts ====================
    http_read_timeout: float = Field(
        default=120.0, validation_alias="HTTP_READ_TIMEOUT"
    )
    http_write_timeout: float = Field(
        default=10.0, validation_alias="HTTP_WRITE_TIMEOUT"
    )
    http_connect_timeout: float = Field(
        default=HTTP_CONNECT_TIMEOUT_DEFAULT,
        validation_alias="HTTP_CONNECT_TIMEOUT",
    )

    # ==================== Fast Prefix Detection ====================
    fast_prefix_detection: bool = True

    # ==================== Optimizations ====================
    enable_network_probe_mock: bool = True
    enable_title_generation_skip: bool = True
    enable_suggestion_mode_skip: bool = True
    enable_filepath_extraction_mock: bool = True

    # ==================== Local web server tools (web_search / web_fetch) ====================
    # Off by default: these tools perform outbound HTTP from the proxy (SSRF risk).
    enable_web_server_tools: bool = Field(
        default=False, validation_alias="ENABLE_WEB_SERVER_TOOLS"
    )
    # Comma-separated URL schemes allowed for web_fetch (default: http,https).
    web_fetch_allowed_schemes: str = Field(
        default="http,https", validation_alias="WEB_FETCH_ALLOWED_SCHEMES"
    )
    # When true, skip private/loopback/link-local IP blocking for web_fetch (lab only).
    web_fetch_allow_private_networks: bool = Field(
        default=False, validation_alias="WEB_FETCH_ALLOW_PRIVATE_NETWORKS"
    )

    # ==================== Debug / diagnostic logging (avoid sensitive content) ====================
    # When false (default), API and SSE helpers log only metadata (counts, lengths, ids).
    log_raw_api_payloads: bool = Field(
        default=False, validation_alias="LOG_RAW_API_PAYLOADS"
    )
    log_raw_sse_events: bool = Field(
        default=False, validation_alias="LOG_RAW_SSE_EVENTS"
    )
    # When false (default), unhandled exceptions log only type + route metadata (no message/traceback).
    log_api_error_tracebacks: bool = Field(
        default=False, validation_alias="LOG_API_ERROR_TRACEBACKS"
    )
    # When false (default), messaging logs omit text/transcription previews (metadata only).
    log_raw_messaging_content: bool = Field(
        default=False, validation_alias="LOG_RAW_MESSAGING_CONTENT"
    )
    # When true, log full Claude CLI stderr, non-JSON lines, and parser error text.
    log_raw_cli_diagnostics: bool = Field(
        default=False, validation_alias="LOG_RAW_CLI_DIAGNOSTICS"
    )
    # When true, log exception text / CLI error strings in messaging (may leak user content).
    log_messaging_error_details: bool = Field(
        default=False, validation_alias="LOG_MESSAGING_ERROR_DETAILS"
    )
    debug_platform_edits: bool = Field(
        default=False, validation_alias="DEBUG_PLATFORM_EDITS"
    )
    debug_subagent_stack: bool = Field(
        default=False, validation_alias="DEBUG_SUBAGENT_STACK"
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
    allowed_dir: str = ""
    max_message_log_entries_per_chat: int | None = Field(
        default=None, validation_alias="MAX_MESSAGE_LOG_ENTRIES_PER_CHAT"
    )

    # ==================== Server ====================
    host: str = "0.0.0.0"
    port: int = 8082
    # Optional server API key to protect endpoints (Anthropic-style)
    # Set via env `ANTHROPIC_AUTH_TOKEN`. When empty, no auth is required.
    anthropic_auth_token: str = Field(
        default="", validation_alias="ANTHROPIC_AUTH_TOKEN"
    )

    # Handle empty strings for optional string fields
    @field_validator(
        "telegram_bot_token",
        "allowed_telegram_user_id",
        "discord_bot_token",
        "allowed_discord_channels",
        "model_opus",
        "model_sonnet",
        "model_haiku",
        "enable_opus_thinking",
        "enable_sonnet_thinking",
        "enable_haiku_thinking",
        mode="before",
    )
    @classmethod
    def parse_optional_str(cls, v: Any) -> Any:
        if v == "":
            return None
        return v

    @field_validator("max_message_log_entries_per_chat", mode="before")
    @classmethod
    def parse_optional_log_cap(cls, v: Any) -> Any:
        if v == "" or v is None:
            return None
        return v

    @property
    def claude_workspace(self) -> str:
        """Return the fixed Claude data workspace path."""

        return str(default_claude_workspace_path())

    @property
    def claude_cli_bin(self) -> str:
        """Return the fixed Claude Code binary name."""

        return "claude"

    @field_validator("whisper_device")
    @classmethod
    def validate_whisper_device(cls, v: str) -> str:
        if v not in ("cpu", "cuda", "nvidia_nim"):
            raise ValueError(
                f"whisper_device must be 'cpu', 'cuda', or 'nvidia_nim', got {v!r}"
            )
        return v

    @field_validator("messaging_platform")
    @classmethod
    def validate_messaging_platform(cls, v: str) -> str:
        if v not in ("telegram", "discord", "none"):
            raise ValueError(
                f"messaging_platform must be 'telegram', 'discord', or 'none', got {v!r}"
            )
        return v

    @field_validator("messaging_rate_limit")
    @classmethod
    def validate_messaging_rate_limit(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("messaging_rate_limit must be > 0")
        return v

    @field_validator("messaging_rate_window")
    @classmethod
    def validate_messaging_rate_window(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("messaging_rate_window must be > 0")
        return float(v)

    @field_validator("web_fetch_allowed_schemes")
    @classmethod
    def validate_web_fetch_allowed_schemes(cls, v: str) -> str:
        schemes = [part.strip().lower() for part in v.split(",") if part.strip()]
        if not schemes:
            raise ValueError("web_fetch_allowed_schemes must list at least one scheme")
        for scheme in schemes:
            if not scheme.isascii() or not scheme.isalpha():
                raise ValueError(
                    f"Invalid URL scheme in web_fetch_allowed_schemes: {scheme!r}"
                )
        return ",".join(schemes)

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, v: str) -> str:
        if v.rstrip("/").endswith("/v1"):
            raise ValueError(
                "OLLAMA_BASE_URL must be the Ollama root URL for native Anthropic "
                "messages, e.g. http://localhost:11434 (without /v1)."
            )
        return v

    @field_validator("model", "model_opus", "model_sonnet", "model_haiku")
    @classmethod
    def validate_model_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if "/" not in v:
            raise ValueError(
                f"Model must be prefixed with provider type. "
                f"Valid providers: {', '.join(SUPPORTED_PROVIDER_IDS)}. "
                f"Format: provider_type/model/name"
            )
        provider = v.split("/", 1)[0]
        if provider not in SUPPORTED_PROVIDER_IDS:
            supported = ", ".join(f"'{p}'" for p in SUPPORTED_PROVIDER_IDS)
            raise ValueError(f"Invalid provider: '{provider}'. Supported: {supported}")
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
        return Settings.parse_provider_type(self.model)

    @property
    def model_name(self) -> str:
        """Extract the actual model name from the default model string."""
        return Settings.parse_model_name(self.model)

    def resolve_model(
        self, claude_model_name: str, project_cwd: Path | None = None
    ) -> str:
        """Resolve a Claude model name to the configured provider/model string.

        Resolution order, highest precedence first:
          1. Per-project override from `<project_cwd>/.claude/settings.json`
             (and `.local.json`), under `freeClaudeCode.models.<tier>` or
             `freeClaudeCode.models.default`.
          2. Tier-specific env override (MODEL_OPUS / MODEL_SONNET / MODEL_HAIKU).
          3. Global MODEL fallback.

        `project_cwd` is None when the request carried no valid
        `X-Free-Claude-Project` header — behavior then matches the base impl.
        """
        if project_cwd is not None:
            # Lazy import: project_settings is pure-config (imports nothing from
            # settings.py); keep it local to avoid any early-import surprises.
            from .project_settings import load_project_settings

            project = load_project_settings(project_cwd)
            if project is not None:
                override = project.model_for(claude_model_name)
                if override is not None:
                    return override

        name_lower = claude_model_name.lower()
        if "opus" in name_lower and self.model_opus is not None:
            return self.model_opus
        if "haiku" in name_lower and self.model_haiku is not None:
            return self.model_haiku
        if "sonnet" in name_lower and self.model_sonnet is not None:
            return self.model_sonnet
        return self.model

    def configured_chat_model_refs(self) -> tuple[ConfiguredChatModelRef, ...]:
        """Return unique configured chat provider/model refs with source env keys."""
        candidates = (
            ("MODEL", self.model),
            ("MODEL_OPUS", self.model_opus),
            ("MODEL_SONNET", self.model_sonnet),
            ("MODEL_HAIKU", self.model_haiku),
        )
        sources_by_ref: dict[str, list[str]] = {}
        for source, model_ref in candidates:
            if model_ref is None:
                continue
            sources_by_ref.setdefault(model_ref, []).append(source)

        return tuple(
            ConfiguredChatModelRef(
                model_ref=model_ref,
                provider_id=Settings.parse_provider_type(model_ref),
                model_id=Settings.parse_model_name(model_ref),
                sources=tuple(sources),
            )
            for model_ref, sources in sources_by_ref.items()
        )

    def resolve_thinking(self, claude_model_name: str) -> bool:
        """Resolve whether thinking is enabled for an incoming Claude model name."""
        name_lower = claude_model_name.lower()
        if "opus" in name_lower and self.enable_opus_thinking is not None:
            return self.enable_opus_thinking
        if "haiku" in name_lower and self.enable_haiku_thinking is not None:
            return self.enable_haiku_thinking
        if "sonnet" in name_lower and self.enable_sonnet_thinking is not None:
            return self.enable_sonnet_thinking
        return self.enable_model_thinking

    def web_fetch_allowed_scheme_set(self) -> frozenset[str]:
        """Return normalized schemes allowed for web_fetch."""
        return frozenset(
            part.strip().lower()
            for part in self.web_fetch_allowed_schemes.split(",")
            if part.strip()
        )

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
