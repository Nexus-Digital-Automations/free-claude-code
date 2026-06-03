# cc-provider

Switch which AI backend the local proxy routes to, from a single command.

The proxy at `http://127.0.0.1:8082` reads `MODEL` (and the optional per-tier
overrides `MODEL_OPUS` / `MODEL_SONNET` / `MODEL_HAIKU`) from the project
`.env`. `cc-provider` writes those values for you and mirrors them into
`~/.claude/settings.json` for visibility. **The proxy must be restarted
after each switch** — same UX as `/use-proxy` and `/use-direct`.

The script lives at `scripts/cc-provider.sh`. The slash-command form is
`/cc-provider` (installed by `scripts/setup-cc-switcher.sh`).

---

## Quick reference

```sh
cc-provider                                 # show current selection + supported providers
cc-provider list                            # same
cc-provider <provider>/<model>              # set the default MODEL
cc-provider --tier opus|sonnet|haiku <p>/<m>   # set a per-tier override
cc-provider --clear-tier opus|sonnet|haiku     # remove a tier override
```

The model string is always `<provider>/<model_name>`. Supported providers:
`nvidia_nim`, `open_router`, `deepseek`, `lmstudio`, `llamacpp`, `vertex`.

---

## Switching between DeepSeek V4 Flash and Vertex Gemma 4

These are the two cases you'll hit most often.

### Use DeepSeek V4 Flash for everything

```sh
bash scripts/cc-provider.sh deepseek/deepseek-v4-flash
```

Or from inside Claude Code:

```
/cc-provider deepseek/deepseek-v4-flash
```

After running, restart the proxy:

```sh
pkill -f scripts/start.sh        # kill the running proxy if any
bash scripts/start.sh            # or just run `cc` and it auto-starts
```

Confirm the change:

```sh
bash scripts/cc-status
```

### Use Vertex Gemma 4 for everything

```sh
bash scripts/cc-provider.sh vertex/gemma-4
```

Or:

```
/cc-provider vertex/gemma-4
```

> **Heads up.** The `gemma-4` model name on the right of the slash must match
> a model ID registered in your `VERTEX_MODELS` env var. Vertex has no
> `/models` endpoint, so the proxy can't discover them automatically. Verify
> with `bash scripts/cc-provider.sh list` and check `.env` if `vertex` shows
> `✗` (no credentials). Required env vars: `VERTEX_PROJECT`, `VERTEX_REGION`,
> `VERTEX_ENDPOINT_ID`, `VERTEX_MODELS`.

### Mix them — Gemma 4 for Opus, Flash for Sonnet/Haiku

If you want Claude Code's Opus tier to hit Gemma 4 (slower, higher quality)
while Sonnet and Haiku stay on DeepSeek Flash (cheap and fast):

```sh
bash scripts/cc-provider.sh deepseek/deepseek-v4-flash         # baseline for all tiers
bash scripts/cc-provider.sh --tier opus vertex/gemma-4         # promote Opus to Gemma 4
```

Then restart the proxy. `cc-status` will show:

```
Active provider (proxy reads from .../.env):
  Default (MODEL):   deepseek/deepseek-v4-flash
  Opus override:     vertex/gemma-4
  Sonnet override:   (unset → uses default)
  Haiku override:    (unset → uses default)
```

To go back to a uniform setup, clear the override:

```sh
bash scripts/cc-provider.sh --clear-tier opus
```

---

## Toggling back and forth

There's no built-in alias, but these one-liners work:

```sh
# Flash everywhere
bash scripts/cc-provider.sh deepseek/deepseek-v4-flash && pkill -f scripts/start.sh ; bash scripts/start.sh

# Gemma 4 everywhere
bash scripts/cc-provider.sh vertex/gemma-4 && pkill -f scripts/start.sh ; bash scripts/start.sh
```

Add shell aliases if you flip often:

```sh
alias use-flash='bash "$REPO_ROOT/scripts/cc-provider.sh" deepseek/deepseek-v4-flash'
alias use-gemma='bash "$REPO_ROOT/scripts/cc-provider.sh" vertex/gemma-4'
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `unknown provider 'X'` | Typo or unsupported provider | `cc-provider list` shows the valid set |
| `must be in the form <provider>/<model>` | Forgot the `/` | Use `<provider>/<model>`, e.g. `deepseek/deepseek-v4-flash` |
| `cc-status` shows the new value but the proxy still routes to the old one | Proxy not restarted | `pkill -f scripts/start.sh` then `bash scripts/start.sh` |
| `cc-provider list` shows `✗` next to your provider | Credentials env var missing in `.env` | DeepSeek needs `DEEPSEEK_API_KEY`; Vertex needs `VERTEX_PROJECT`, `VERTEX_REGION`, `VERTEX_ENDPOINT_ID`, `VERTEX_MODELS` |
| Proxy log says `Unknown provider_type` | Provider name in `.env` doesn't match the canonical set | Run `cc-provider <p>/<m>` instead of hand-editing — the script validates |

---

## Where this writes

- `<repo>/.env` — `MODEL`, `MODEL_OPUS`, `MODEL_SONNET`, `MODEL_HAIKU` (the proxy's source of truth)
- `~/.claude/settings.json` env block — same values mirrored for `cc-status` to display

The two stay in sync because `cc-provider` always writes to both.
