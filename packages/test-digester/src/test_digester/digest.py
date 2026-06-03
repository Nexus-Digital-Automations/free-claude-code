"""Owns: failure-text prompt template + thin wrapper over digest_core.

Test failures want a different summary shape than tool_result digests: we
care about which assertion failed and a one-line root-cause hypothesis,
not a balanced shrink of the whole output. The prompt here biases the
model toward that shape.

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
class FailureDigestConfig:
    """Minimum config the digest call needs.

    Mirrors digest_core.DigestConfig protocol; defaults match the proxy's
    Ollama setup so users get sensible behaviour without configuring twice.
    """

    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"
    compaction_max_tokens: int = 500
    compaction_temperature: float = 0.0
    context_compaction_keep_alive: str = "30m"
    digest_cache_max_entries: int = 200


def _build_failure_prompt(body: str) -> str:
    """Prompt biases the digest toward action-relevant detail.

    The agent reading this digest will decide whether to investigate; we
    keep enough specificity (file:line, assertion text, exception type)
    that the agent never has to re-run with -tb=long just to find what
    broke.
    """
    return (
        "You are summarising a Python test failure so an engineer can act on it "
        "without re-running with full traceback.\n\n"
        "PRESERVE VERBATIM:\n"
        "- File path and line number of the failure\n"
        "- The assertion or exception message exactly as written\n"
        "- The exception class name (AssertionError, ValueError, etc.)\n"
        "- Any unique identifier names (functions, fixtures, parametrized values)\n\n"
        "ADD (one short line each):\n"
        "- A one-line root-cause hypothesis if it's obvious from the trace\n"
        "- Whether this looks like setup/teardown vs. assertion-in-body\n\n"
        "DROP:\n"
        "- Stack frames inside pytest itself (_pytest/, _hookcaller, etc.)\n"
        "- Frames inside site-packages unless the exception originates there\n"
        "- Repeated identical lines\n\n"
        "OUTPUT: under 400 tokens, plain text, wrapped in <digest>...</digest> tags.\n\n"
        "FAILURE TRACE:\n"
        f"{body}"
    )


def _parse_digest(content: str) -> str | None:
    """Extract digest body from tagged Ollama output. None on parse miss."""
    match = _DIGEST_TAG.search(content)
    if not match:
        return None
    body = match.group(1).strip()
    return body or None


async def digest_failure_body(body: str, config: FailureDigestConfig) -> str | None:
    """Return a cached or freshly-digested summary, or None on Ollama failure.

    Caller falls back to passing through the original body unchanged on None.
    """
    if not await OllamaSupervisor.ensure_ready(config):
        return None
    return await digest_core.digest(
        content=body,
        build_prompt=_build_failure_prompt,
        parse_response=_parse_digest,
        config=config,
        cache_max_entries=config.digest_cache_max_entries,
    )
