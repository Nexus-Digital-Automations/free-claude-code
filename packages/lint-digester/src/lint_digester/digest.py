"""Owns: lint-rule prompt template + thin wrapper over digest_core.

Lint findings care about which rule and what's wrong, not the textual
spread across many files. We digest one representative example per rule
and trust the agent to extrapolate from that example to the file list.

Does NOT own: the cache or Ollama call (digest_core), linter parsing
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
class LintDigestConfig:
    """Minimum config the digest call needs.

    Mirrors digest_core.DigestConfig protocol; defaults match the proxy's
    Ollama setup.
    """

    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"
    compaction_max_tokens: int = 200
    compaction_temperature: float = 0.0
    context_compaction_keep_alive: str = "30m"
    digest_cache_max_entries: int = 200


def _build_rule_prompt(content: str) -> str:
    """Prompt biases toward a one-line "what does this rule mean" + fix.

    The agent already has the rule_id and one example finding; it doesn't
    need the digester to repeat them. The digester explains *why the rule
    exists* and what the fix shape is, so the agent can plan a change
    once and apply it across the file list verbatim.
    """
    return (
        "You are explaining a lint rule violation to an engineer. The agent "
        "already has the rule code and one example finding from the linter. "
        "Your job is to add value the linter didn't.\n\n"
        "OUTPUT TWO SHORT LINES inside <digest>...</digest> tags:\n"
        "  1. What the rule prevents in one phrase (a real bug class, not "
        "     the rule's title verbatim)\n"
        "  2. The shape of the fix in one phrase (e.g. 'remove unused import', "
        "     'add explicit return None', 'replace == None with is None')\n\n"
        "DO NOT repeat the file path, line number, or the literal rule message — "
        "the agent already has those.\n\n"
        "FINDING:\n"
        f"{content}"
    )


def _parse_digest(content: str) -> str | None:
    """Extract digest body from tagged Ollama output. None on parse miss."""
    match = _DIGEST_TAG.search(content)
    if not match:
        return None
    body = match.group(1).strip()
    return body or None


async def digest_rule(content: str, config: LintDigestConfig) -> str | None:
    """Return a cached or freshly-digested explanation, or None on failure.

    Caller falls back to the original content if None.
    """
    if not await OllamaSupervisor.ensure_ready(config):
        return None
    return await digest_core.digest(
        content=content,
        build_prompt=_build_rule_prompt,
        parse_response=_parse_digest,
        config=config,
        cache_max_entries=config.digest_cache_max_entries,
    )
