#!/usr/bin/env bash
# Owns: writing ANTHROPIC_BASE_URL into ~/.claude/settings.json env block.
# Does NOT own: launching claude (cc), checking status (cc-status), or
#               storing credentials (setup-cc-switcher.sh handles Keychain).
# Called by: /use-proxy and /use-direct slash commands.
# Calls:     jq (edits settings.json).
#
# Usage: cc-set-api-mode.sh proxy|direct

set -euo pipefail

MODE="${1:-}"
SETTINGS="$HOME/.claude/settings.json"

if [[ "$MODE" != "proxy" && "$MODE" != "direct" ]]; then
  echo "Usage: cc-set-api-mode.sh proxy|direct" >&2
  exit 1
fi

if ! command -v jq &>/dev/null; then
  echo "Error: jq is required but not installed" >&2
  exit 1
fi

if [[ "$MODE" == "proxy" ]]; then
  jq '.env.ANTHROPIC_BASE_URL = "http://127.0.0.1:8082"' "$SETTINGS" \
    > /tmp/_settings_tmp.json && mv /tmp/_settings_tmp.json "$SETTINGS"
  echo "Set ANTHROPIC_BASE_URL=http://127.0.0.1:8082 in settings.json"
else
  jq 'del(.env.ANTHROPIC_BASE_URL)' "$SETTINGS" \
    > /tmp/_settings_tmp.json && mv /tmp/_settings_tmp.json "$SETTINGS"
  echo "Removed ANTHROPIC_BASE_URL from settings.json (direct to api.anthropic.com)"
fi

echo "env block is now:"
jq '.env' "$SETTINGS"
