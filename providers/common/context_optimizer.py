"""Proxy adapter — wraps the context-optimizer package for this proxy's types.

Owns: converting proxy Pydantic models <-> plain dicts and calling the
standalone context_optimizer package.  Owns nothing else.

Does NOT own: tier logic, caching, Ollama management — all in the package.
Called by: api/routes.py:create_message.
Calls: context_optimizer.ContextOptimizer.optimize (the package).

# @stable — api/routes.py depends on ContextOptimizer.optimize() signature:
#   async (request_data, settings, provider) -> (request_data, int)
"""

from __future__ import annotations

from typing import Any

from context_optimizer import ContextOptimizer as _PkgOptimizer
from context_optimizer import ContextOptimizerSettings as _PkgSettings


class ContextOptimizer:
    """Thin adapter; delegates entirely to the context-optimizer package."""

    @classmethod
    async def optimize(
        cls, request_data: Any, settings: Any, provider: Any
    ) -> tuple[Any, int]:
        """Convert proxy types to dicts, run the package optimizer, convert back."""
        from api.models.anthropic import Message

        pkg_settings = _PkgSettings(
            compact_threshold_tokens=settings.context_compact_threshold_tokens,
            compact_soft_threshold_tokens=settings.context_compact_soft_threshold_tokens,
            compact_deepseek_fallback_threshold_tokens=(
                settings.context_compact_deepseek_fallback_threshold_tokens
            ),
            max_thinking_turns=settings.context_max_thinking_turns,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
        )

        dict_messages = [m.model_dump() for m in request_data.messages]
        dict_system = _system_to_dicts(request_data.system)
        llm_provider = _make_provider(provider, request_data.model)

        out_messages, out_system, token_count = await _PkgOptimizer.optimize(
            messages=dict_messages,
            system=dict_system,
            settings=pkg_settings,
            llm_provider=llm_provider,
            tools=None,
        )

        new_request = request_data.model_copy(update={
            "messages": [Message.model_validate(m) for m in out_messages],
            "system": _dicts_to_system(out_system),
        })
        return new_request, token_count


def _system_to_dicts(system: Any) -> Any:
    if system is None or isinstance(system, str):
        return system
    if isinstance(system, list):
        return [{"type": s.type, "text": s.text} for s in system]
    return system


def _dicts_to_system(system: Any) -> Any:
    from api.models.anthropic import SystemContent
    if system is None or isinstance(system, str):
        return system
    if isinstance(system, list):
        return [
            SystemContent(type=d.get("type", "text"), text=d.get("text", ""))
            if isinstance(d, dict) else d
            for d in system
        ]
    return system


def _make_provider(provider: Any, model: str):
    """Build an async (prompt: str) -> str callable from the proxy's provider._client."""
    async def call_llm(prompt: str) -> str:
        resp = await provider._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.3,
            stream=False,
        )
        return resp.choices[0].message.content or ""
    return call_llm
