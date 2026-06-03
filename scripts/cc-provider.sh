#!/usr/bin/env bash
# Owns: switching the proxy backend by writing MODEL / MODEL_OPUS / MODEL_SONNET /
#       MODEL_HAIKU into the project .env, and mirroring those values into the env
#       block of ~/.claude/settings.json for visibility.
# Does NOT own: launching claude (cc), reporting status (cc-status), starting the
#               proxy server (start.sh), or storing credentials (setup-cc-switcher.sh).
# Called by:    /cc-provider slash command and the user from the shell.
# Calls:        jq (settings.json edits), awk (env upsert).
#
# Usage:
#   cc-provider <provider>/<model>                set default MODEL
#   cc-provider --tier opus|sonnet|haiku <p>/<m>  set MODEL_OPUS / MODEL_SONNET / MODEL_HAIKU
#   cc-provider --clear-tier opus|sonnet|haiku    remove a tier override
#   cc-provider --project init                    write X-Free-Claude-Project header
#                                                 into $PWD/.claude/settings.json
#   cc-provider --project set <tier> <p>/<m>      tier ∈ {default,opus,sonnet,haiku}
#                                                 writes freeClaudeCode.models.<tier>
#                                                 into $PWD/.claude/settings.json
#   cc-provider --project show                    print merged per-project config
#   cc-provider list                              print supported providers + current selection
#   cc-provider                                   same as list

set -euo pipefail

readonly SETTINGS="$HOME/.claude/settings.json"

# Keep in sync with config/provider_catalog.py (SUPPORTED_PROVIDER_IDS).
readonly SUPPORTED_PROVIDERS=(nvidia_nim open_router gemini deepseek mistral mistral_codestral opencode opencode_go wafer kimi cerebras groq fireworks zai lmstudio llamacpp ollama vertex)

repo_root() {
    local source="${BASH_SOURCE[0]}"
    while [[ -L "$source" ]]; do
        local dir
        dir="$(cd -P "$(dirname "$source")" && pwd)"
        source="$(readlink "$source")"
        [[ "$source" != /* ]] && source="$dir/$source"
    done
    cd -P "$(dirname "$source")/.." && pwd
}

require_jq() {
    if ! command -v jq &>/dev/null; then
        echo "cc-provider: jq is required but not installed" >&2
        exit 1
    fi
}

is_supported_provider() {
    local needle="$1"
    local p
    for p in "${SUPPORTED_PROVIDERS[@]}"; do
        [[ "$p" == "$needle" ]] && return 0
    done
    return 1
}

validate_value() {
    # Reject anything that's not "<provider>/<model>" with both halves non-empty.
    local value="$1" provider model
    if [[ "$value" != */* ]]; then
        echo "cc-provider: '$value' must be in the form <provider>/<model>" >&2
        echo "             example: deepseek/deepseek-v4-pro" >&2
        return 1
    fi
    provider="${value%%/*}"
    model="${value#*/}"
    if [[ -z "$provider" || -z "$model" ]]; then
        echo "cc-provider: provider and model must both be non-empty (got '$value')" >&2
        return 1
    fi
    if ! is_supported_provider "$provider"; then
        echo "cc-provider: unknown provider '$provider'." >&2
        echo "             supported: ${SUPPORTED_PROVIDERS[*]}" >&2
        return 1
    fi
}

env_upsert() {
    # Replace the line for KEY in $env_file, or append it if absent. Uses awk
    # so comments and ordering are preserved. Values are written quoted to
    # match the existing .env style.
    local env_file="$1" key="$2" value="$3"
    local tmp
    tmp="$(mktemp)"
    KEY="$key" VAL="$value" awk '
        BEGIN { found = 0; key = ENVIRON["KEY"]; val = ENVIRON["VAL"] }
        $0 ~ "^[[:space:]]*" key "=" { print key "=\"" val "\""; found = 1; next }
        { print }
        END { if (!found) print key "=\"" val "\"" }
    ' "$env_file" > "$tmp"
    mv "$tmp" "$env_file"
}

env_remove() {
    local env_file="$1" key="$2"
    local tmp
    tmp="$(mktemp)"
    KEY="$key" awk '
        BEGIN { key = ENVIRON["KEY"] }
        $0 ~ "^[[:space:]]*" key "=" { next }
        { print }
    ' "$env_file" > "$tmp"
    mv "$tmp" "$env_file"
}

env_read() {
    # Echo the unquoted value of KEY in env_file, or empty if absent.
    local env_file="$1" key="$2"
    [[ -f "$env_file" ]] || { printf ''; return; }
    local line
    line="$(grep -E "^[[:space:]]*${key}=" "$env_file" | tail -n1 || true)"
    [[ -z "$line" ]] && { printf ''; return; }
    local raw="${line#*=}"
    raw="${raw%\"}"
    raw="${raw#\"}"
    raw="${raw%\'}"
    raw="${raw#\'}"
    printf '%s' "$raw"
}

settings_set() {
    local key="$1" value="$2"
    [[ -f "$SETTINGS" ]] || echo '{}' > "$SETTINGS"
    local tmp
    tmp="$(mktemp)"
    jq --arg k "$key" --arg v "$value" '.env[$k] = $v' "$SETTINGS" > "$tmp"
    mv "$tmp" "$SETTINGS"
}

settings_unset() {
    local key="$1"
    [[ -f "$SETTINGS" ]] || return 0
    local tmp
    tmp="$(mktemp)"
    jq --arg k "$key" 'del(.env[$k])' "$SETTINGS" > "$tmp"
    mv "$tmp" "$SETTINGS"
}

project_settings_path() {
    printf '%s/.claude/settings.json' "$PWD"
}

project_ensure_settings_file() {
    local path="$1"
    mkdir -p "$(dirname "$path")"
    [[ -f "$path" ]] || echo '{}' > "$path"
}

project_init_header() {
    # Idempotently appends `X-Free-Claude-Project: $PWD` to ANTHROPIC_CUSTOM_HEADERS.
    # Claude Code reads ANTHROPIC_CUSTOM_HEADERS as the literal HTTP header
    # block; comma-joining preserves any caller-defined headers.
    local path
    path="$(project_settings_path)"
    project_ensure_settings_file "$path"
    local header_value="X-Free-Claude-Project: $PWD"
    local tmp existing
    existing="$(jq -r '.env.ANTHROPIC_CUSTOM_HEADERS // ""' "$path")"
    if [[ "$existing" == *"X-Free-Claude-Project:"* ]]; then
        local replaced
        replaced="$(printf '%s\n' "$existing" | sed -E "s|X-Free-Claude-Project: [^,]*|${header_value}|")"
        tmp="$(mktemp)"
        jq --arg v "$replaced" '.env.ANTHROPIC_CUSTOM_HEADERS = $v' "$path" > "$tmp"
        mv "$tmp" "$path"
        echo "Updated X-Free-Claude-Project in $path"
        return 0
    fi
    local new_value="$header_value"
    [[ -n "$existing" ]] && new_value="$existing, $header_value"
    tmp="$(mktemp)"
    jq --arg v "$new_value" '.env.ANTHROPIC_CUSTOM_HEADERS = $v' "$path" > "$tmp"
    mv "$tmp" "$path"
    echo "Wrote ANTHROPIC_CUSTOM_HEADERS to $path"
    echo "  X-Free-Claude-Project: $PWD"
}

is_supported_project_tier() {
    case "$1" in default|opus|sonnet|haiku) return 0 ;; *) return 1 ;; esac
}

project_set_tier() {
    local tier="$1" value="$2" path tmp
    if ! is_supported_project_tier "$tier"; then
        echo "cc-provider: unknown project tier '$tier' (default,opus,sonnet,haiku)" >&2
        return 1
    fi
    validate_value "$value"
    path="$(project_settings_path)"
    project_ensure_settings_file "$path"
    tmp="$(mktemp)"
    jq --arg t "$tier" --arg v "$value" \
        '.freeClaudeCode.models[$t] = $v' "$path" > "$tmp"
    mv "$tmp" "$path"
    echo "Set freeClaudeCode.models.$tier=$value in $path"
}

project_show() {
    local path local_path
    path="$(project_settings_path)"
    local_path="${path%.json}.local.json"
    [[ -f "$path" || -f "$local_path" ]] || {
        echo "No per-project settings at $path"; return 0;
    }
    echo "Project: $PWD"
    if [[ -f "$path" ]]; then
        echo "--- $path"
        jq '.freeClaudeCode // {}' "$path"
    fi
    if [[ -f "$local_path" ]]; then
        echo "--- $local_path"
        jq '.freeClaudeCode // {}' "$local_path"
    fi
}

cmd_project() {
    local sub="${1:-}"
    case "$sub" in
        init) project_init_header ;;
        set)
            [[ $# -ge 3 ]] || { echo "cc-provider --project set <tier> <provider/model>" >&2; exit 1; }
            project_set_tier "$2" "$3"
            ;;
        show) project_show ;;
        *)
            echo "cc-provider: --project requires init|set|show (got '${sub:-<empty>}')" >&2
            exit 1
            ;;
    esac
}

tier_to_env_key() {
    case "$1" in
        opus)   printf 'MODEL_OPUS'   ;;
        sonnet) printf 'MODEL_SONNET' ;;
        haiku)  printf 'MODEL_HAIKU'  ;;
        *)
            echo "cc-provider: unknown tier '$1' (expected: opus, sonnet, haiku)" >&2
            return 1
            ;;
    esac
}

credentials_marker() {
    # Mirrors smoke/lib/config.py:has_provider_configuration — checks whichever
    # env var that file consults to decide if a provider is usable.
    local provider="$1" var=""
    case "$provider" in
        nvidia_nim)        var="NVIDIA_NIM_API_KEY" ;;
        open_router)       var="OPENROUTER_API_KEY" ;;
        gemini)            var="GEMINI_API_KEY" ;;
        deepseek)          var="DEEPSEEK_API_KEY" ;;
        mistral)           var="MISTRAL_API_KEY" ;;
        mistral_codestral) var="CODESTRAL_API_KEY" ;;
        opencode|opencode_go) var="OPENCODE_API_KEY" ;;
        wafer)             var="WAFER_API_KEY" ;;
        kimi)              var="KIMI_API_KEY" ;;
        cerebras)          var="CEREBRAS_API_KEY" ;;
        groq)              var="GROQ_API_KEY" ;;
        fireworks)         var="FIREWORKS_API_KEY" ;;
        zai)               var="ZAI_API_KEY" ;;
        lmstudio)          var="LM_STUDIO_BASE_URL" ;;
        llamacpp)          var="LLAMACPP_BASE_URL" ;;
        ollama)            var="OLLAMA_BASE_URL" ;;
        vertex)            var="VERTEX_PROJECT" ;;
    esac
    local val
    val="$(env_read "$REPO_ROOT_DIR/.env" "$var")"
    if [[ -n "$val" ]]; then printf '✓'; else printf '✗'; fi
}

print_listing() {
    local env_file="$REPO_ROOT_DIR/.env"
    local current_default current_opus current_sonnet current_haiku
    current_default="$(env_read "$env_file" MODEL)"
    current_opus="$(env_read "$env_file" MODEL_OPUS)"
    current_sonnet="$(env_read "$env_file" MODEL_SONNET)"
    current_haiku="$(env_read "$env_file" MODEL_HAIKU)"

    echo "Supported providers (✓ = credentials present in .env):"
    local p
    for p in "${SUPPORTED_PROVIDERS[@]}"; do
        printf '  %s %s\n' "$(credentials_marker "$p")" "$p"
    done
    echo
    echo "Current selection (from $env_file):"
    printf '  Default (MODEL):   %s\n' "${current_default:-(unset)}"
    printf '  Opus override:     %s\n' "${current_opus:-(unset → uses default)}"
    printf '  Sonnet override:   %s\n' "${current_sonnet:-(unset → uses default)}"
    printf '  Haiku override:    %s\n' "${current_haiku:-(unset → uses default)}"
    echo
    echo "Examples:"
    echo "  cc-provider deepseek/deepseek-v4-pro"
    echo "  cc-provider --tier opus nvidia_nim/glm-4.7"
    echo "  cc-provider --clear-tier opus"
}

print_restart_hint() {
    cat >&2 <<'EOF'

Restart the proxy to pick up the change:
  - kill the running proxy (Ctrl+C in its terminal, or `pkill -f scripts/start.sh`)
  - then re-run `bash scripts/start.sh` (or just run `cc` to auto-start it)
EOF
}

cmd_set_default() {
    local value="$1"
    validate_value "$value"
    env_upsert "$REPO_ROOT_DIR/.env" MODEL "$value"
    settings_set MODEL "$value"
    echo "Set MODEL=$value in $REPO_ROOT_DIR/.env (mirrored in $SETTINGS)"
    print_restart_hint
}

cmd_set_tier() {
    local tier="$1" value="$2" env_key
    env_key="$(tier_to_env_key "$tier")"
    validate_value "$value"
    env_upsert "$REPO_ROOT_DIR/.env" "$env_key" "$value"
    settings_set "$env_key" "$value"
    echo "Set $env_key=$value in $REPO_ROOT_DIR/.env (mirrored in $SETTINGS)"
    print_restart_hint
}

cmd_clear_tier() {
    local tier="$1" env_key
    env_key="$(tier_to_env_key "$tier")"
    env_remove "$REPO_ROOT_DIR/.env" "$env_key"
    settings_unset "$env_key"
    echo "Removed $env_key from $REPO_ROOT_DIR/.env (and $SETTINGS)"
    print_restart_hint
}

main() {
    require_jq
    REPO_ROOT_DIR="$(repo_root)"
    readonly REPO_ROOT_DIR

    if [[ $# -eq 0 ]] || [[ "${1:-}" == "list" ]]; then
        print_listing
        return 0
    fi

    case "${1:-}" in
        --tier)
            [[ $# -ge 3 ]] || { echo "cc-provider: --tier requires <tier> <provider/model>" >&2; exit 1; }
            cmd_set_tier "$2" "$3"
            ;;
        --clear-tier)
            [[ $# -ge 2 ]] || { echo "cc-provider: --clear-tier requires <tier>" >&2; exit 1; }
            cmd_clear_tier "$2"
            ;;
        --project)
            shift
            cmd_project "$@"
            ;;
        -h|--help)
            sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            ;;
        *)
            cmd_set_default "$1"
            ;;
    esac
}

main "$@"
