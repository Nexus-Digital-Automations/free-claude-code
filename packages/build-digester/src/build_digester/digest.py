"""Owns: build-error prompt template + thin wrapper over digest_core.

Build/typecheck errors care about which type/import/call broke and a
one-line root-cause hypothesis, not a balanced shrink. The prompt biases
the digest toward that shape.

Does NOT own: the cache or Ollama call (digest_core), framework parsing
(runner.py).
Called by: server.handle_run.
Calls: digest_core.digest, ollama_supervisor.OllamaSupervisor.ensure_ready.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from context_optimizer import digest_core
from context_optimizer.ollama_supervisor import OllamaSupervisor

_DIGEST_TAG = re.compile(r"<digest>\s*(.*?)\s*</digest>", re.DOTALL)


@dataclass
class BuildDigestConfig:
    """Minimum config the digest call needs.

    Mirrors digest_core.DigestConfig protocol; defaults match the proxy's
    Ollama setup so users get sensible behaviour without configuring twice.
    """

    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"
    compaction_max_tokens: int = 400
    compaction_temperature: float = 0.0
    context_compaction_keep_alive: str = "30m"
    digest_cache_max_entries: int = 200


def _build_error_prompt(body: str) -> str:
    """Prompt biases the digest toward action-relevant detail.

    The agent reading this digest decides whether to investigate; we keep
    enough specificity (file:line, error code, symbol names, type info)
    that the agent never has to re-run the build with full verbosity just
    to know what broke.
    """
    return (
        "You are summarising a compiler/type-check error so an engineer can act on it "
        "without re-running with full verbosity.\n\n"
        "PRESERVE VERBATIM:\n"
        "- File path and line number\n"
        "- Error code (TS2345, error[E0382], [arg-type], etc.)\n"
        "- The exact message text\n"
        "- Any symbol/type names mentioned (functions, classes, traits, types)\n"
        "- Inferred-vs-expected types when shown\n\n"
        "ADD (one short line each):\n"
        "- A one-line root-cause hypothesis if it's obvious\n"
        "- The likely fix in one phrase if equally obvious\n\n"
        "DROP:\n"
        "- Stack frames inside the compiler / language server\n"
        "- Repeated identical context lines\n"
        "- ANSI escape sequences\n\n"
        "OUTPUT: under 300 tokens, plain text, wrapped in <digest>...</digest> tags.\n\n"
        "ERROR BODY:\n"
        f"{body}"
    )


def _parse_digest(content: str) -> str | None:
    """Extract digest body from tagged Ollama output. None on parse miss."""
    match = _DIGEST_TAG.search(content)
    if not match:
        return None
    body = match.group(1).strip()
    return body or None


async def digest_error_body(body: str, config: BuildDigestConfig) -> str | None:
    """Return a cached or freshly-digested summary, or None on Ollama failure.

    Caller falls back to passing through the original body unchanged on None.
    """
    if not await OllamaSupervisor.ensure_ready(config):
        return None
    return await digest_core.digest(
        content=body,
        build_prompt=_build_error_prompt,
        parse_response=_parse_digest,
        config=config,
        cache_max_entries=config.digest_cache_max_entries,
    )
