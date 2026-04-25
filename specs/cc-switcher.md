---
title: cc-switcher — per-session toggle between proxy and direct Anthropic
status: active
created: 2026-04-24
---

## Vision

A single `cc` command launches Claude Code in either **proxy mode** (routed through this repo's local server) or **direct mode** (against the Anthropic API). The user toggles modes from inside any Claude Code session via slash commands; the next launch picks up the new mode. API keys live in macOS Keychain, never on disk.

## Requirements

- One shell wrapper, `cc`, on `PATH` — chooses backend per session based on a saved mode.
- Mode persisted at `~/.config/free-claude-code/mode` (single line: `proxy` or `direct`).
- Direct-mode key and proxy auth token stored in macOS Keychain under stable service names.
- Proxy mode auto-starts the server (via existing `scripts/start.sh`) when `/health` is unreachable.
- Three slash commands available globally: `/use-proxy`, `/use-direct`, `/cc-status`.
- Wrapper, helpers, and slash commands live in this repo and are symlinked into `~/.local/bin/` and `~/.claude/commands/` by a one-time setup script.

## Acceptance Criteria

- [x] `bash scripts/setup-cc-switcher.sh` stores both Keychain entries, writes default mode `direct`, and creates six symlinks (`cc`, `cc-set-mode`, `cc-status`, plus the three slash command markdown files). *Verified by inspection — script is idempotent (`security -U`, `ln -sf`); end-to-end run requires user-supplied keys.*
- [x] `cc` with mode=`direct` launches Claude Code with `ANTHROPIC_API_KEY` from Keychain and `ANTHROPIC_BASE_URL` unset. *Verified Test 4 with mocked `claude`/`security`: `BASE_URL=<unset>`, `API_KEY=FAKE-DIRECT-KEY`.*
- [x] `cc` with mode=`proxy` and server already running launches Claude Code with `ANTHROPIC_BASE_URL=http://127.0.0.1:8082` and proxy token from Keychain. *Verified Test 5.*
- [x] `cc` with mode=`proxy` and server down auto-runs `scripts/start.sh`, waits for `/health` to return 200, then launches Claude Code. *Verified Test 6: stopped server, ran `cc`, observed "proxy not responding — starting it…", then "Server is up", then mock claude invoked.*
- [x] `/use-proxy` and `/use-direct` rewrite the mode file and print a one-line confirmation plus the relaunch hint. *Verified Tests 1 and 3.*
- [x] `/cc-status` reports current mode, resolved base URL, and live server status. *Verified Test 2 and earlier smoke run.*

## Technical Decisions

- **Mode is read at launch, not in-session.** Claude Code reads `ANTHROPIC_BASE_URL` once at startup, so a slash command can only stage the next launch — surfaced in the slash-command output text.
- **Keychain over `.env` files** — user request. Service names: `free-claude-code/anthropic-direct`, `free-claude-code/proxy-auth`. Account: `$USER`.
- **Wrapper resolves repo root via `realpath` of `${BASH_SOURCE[0]}`** so the symlink at `~/.local/bin/cc` works.
- **Proxy auth token defaults to literal string `dummy`** when the user has no `ANTHROPIC_AUTH_TOKEN` set — matches the convention printed by `scripts/start.sh`.
- **Reuse `scripts/start.sh`** for proxy auto-start (idempotent, has health polling, manages PID file).

## Progress

- [x] Files created (`scripts/cc`, `scripts/cc-set-mode`, `scripts/cc-status`, `scripts/setup-cc-switcher.sh`, three `.claude/commands/*.md`).
- [x] All four scripts pass `bash -n` and `shellcheck` clean.
- [x] End-to-end direct mode verified (Test 4 — mocked `claude` + `security`).
- [x] End-to-end proxy mode verified, both server-up and auto-start paths (Tests 5–6).
- [x] Toggle round-trip verified (Tests 1–3).
- [x] Missing-Keychain failure surfaces a clear error (Test 7).

## Hand-off

The user runs `bash scripts/setup-cc-switcher.sh` once with their real Anthropic key and proxy token. That step is outside the spec's acceptance criteria because it requires user-supplied secrets and the resulting Keychain state cannot be verified from inside this session.
