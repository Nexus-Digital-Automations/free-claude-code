#!/usr/bin/env bash
# Owns: one-time wiring for the cc switcher — Keychain entries,
#       and symlinks into ~/.local/bin and ~/.claude/commands.
# Does NOT own: launching Claude Code (cc) or changing modes (cc-set-api-mode.sh).
# Called by: the user, once per machine.
# Calls:     `security` (Keychain), ln, mkdir, read.
#
# Re-running is safe: `security add-generic-password -U` updates in place,
# and `ln -sf` replaces existing symlinks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly REPO_ROOT
readonly BIN_DIR="$HOME/.local/bin"
readonly COMMANDS_DIR="$HOME/.claude/commands"
readonly KEYCHAIN_DIRECT="free-claude-code/anthropic-direct"
readonly KEYCHAIN_PROXY="free-claude-code/proxy-auth"

require_macos() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        echo "setup: Keychain storage requires macOS. Detected $(uname -s)." >&2
        exit 1
    fi
}

prompt_secret() {
    # Prints the secret to stdout; reads silently from /dev/tty so piping
    # this script into bash still gets interactive input.
    local label="$1"
    local secret
    printf '%s: ' "$label" >&2
    IFS= read -rs secret </dev/tty
    printf '\n' >&2
    printf '%s' "$secret"
}

store_in_keychain() {
    local service="$1" secret="$2"
    if [[ -z "$secret" ]]; then
        echo "setup: refusing to store empty value for '$service'." >&2
        exit 1
    fi
    security add-generic-password -a "$USER" -s "$service" -w "$secret" -U
}

link() {
    local src="$1" dst="$2"
    mkdir -p "$(dirname "$dst")"
    ln -sf "$src" "$dst"
    echo "setup: linked $dst -> $src"
}

install_symlinks() {
    link "$REPO_ROOT/scripts/cc"          "$BIN_DIR/cc"
    link "$REPO_ROOT/scripts/cc-status"   "$BIN_DIR/cc-status"
}

# Slash commands are written inline (not symlinked) because `.claude/` is in
# .gitignore — a symlink to a gitignored source would break on fresh clones.
# Each command shells out via the absolute repo path.
write_slash_command() {
    local name="$1" description="$2" cmd="$3"
    local dst="$COMMANDS_DIR/$name.md"
    mkdir -p "$COMMANDS_DIR"
    cat > "$dst" <<EOF
---
description: $description
allowed-tools: Bash
---

Run this exact command and show the output verbatim. Do not add commentary:

\`$cmd\`
EOF
    echo "setup: wrote $dst"
}

install_slash_commands() {
    local set_api_mode="$REPO_ROOT/scripts/cc-set-api-mode.sh"
    local status="$REPO_ROOT/scripts/cc-status"
    write_slash_command "use-proxy"  "Switch to proxy mode — sets ANTHROPIC_BASE_URL in settings.json" "bash \"$set_api_mode\" proxy"
    write_slash_command "use-direct" "Switch to direct mode — removes ANTHROPIC_BASE_URL from settings.json" "bash \"$set_api_mode\" direct"
    write_slash_command "cc-status"  "Show current cc mode, base URL, and proxy server health"                  "\"$status\""
}

warn_if_path_missing() {
    case ":$PATH:" in
        *":$BIN_DIR:"*) ;;
        *) echo "setup: WARNING — $BIN_DIR is not on your PATH. Add it to your shell rc." >&2 ;;
    esac
}

main() {
    require_macos

    chmod +x "$REPO_ROOT/scripts/cc" \
             "$REPO_ROOT/scripts/cc-set-api-mode.sh" \
             "$REPO_ROOT/scripts/cc-status"

    echo "Storing your real Anthropic API key (used in direct mode)."
    echo "Get one at https://console.anthropic.com/ — input is hidden."
    direct_key="$(prompt_secret "ANTHROPIC_API_KEY (direct mode)")"
    store_in_keychain "$KEYCHAIN_DIRECT" "$direct_key"

    echo
    echo "Storing the token sent to your local proxy. If the proxy's"
    echo "ANTHROPIC_AUTH_TOKEN setting is empty, just enter: dummy"
    proxy_token="$(prompt_secret "ANTHROPIC_API_KEY (proxy mode)")"
    store_in_keychain "$KEYCHAIN_PROXY" "$proxy_token"

    install_symlinks
    install_slash_commands
    warn_if_path_missing

    cat <<'EOF'

Done. Next steps:
  - Run `cc` to launch Claude Code in the saved mode.
  - Inside any session: `/use-proxy`, `/use-direct`, or `/cc-status`.
  - After switching modes, exit (Ctrl+D) and run `cc` again.
EOF
}

main "$@"
