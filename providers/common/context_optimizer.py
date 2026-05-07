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
        """Convert proxy types to dicts, run the package optimizer, convert back.

        `provider` is accepted for backward signature compatibility but no
        longer used — the package owns its own Ollama-driven block tower
        and does not call back into the upstream provider for compaction.
        """
        from api.models.anthropic import Message

        pkg_settings = _PkgSettings(
            compact_threshold_tokens=settings.context_compact_threshold_tokens,
            max_thinking_turns=settings.context_max_thinking_turns,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
            tier0_max_lines=settings.context_tier0_max_lines,
            tier0_head_lines=settings.context_tier0_head_lines,
            tier0_tail_lines=settings.context_tier0_tail_lines,
            render_preview_chars=settings.context_render_preview_chars,
            compaction_max_tokens=settings.context_compaction_max_tokens,
            compaction_temperature=settings.context_compaction_temperature,
            context_compaction_keep_alive=settings.context_compaction_keep_alive,
            tokenizer_name=settings.context_tokenizer_model,
            tier0b_digest_enabled=settings.context_tier0b_digest_enabled,
            tier0b_digest_min_bytes=settings.context_tier0b_digest_min_bytes,
            tier0b_digest_timeout_seconds=settings.context_tier0b_digest_timeout_seconds,
            tier0c_digest_enabled=settings.context_tier0c_digest_enabled,
            tier0c_digest_min_bytes=settings.context_tier0c_digest_min_bytes,
            tier0c_keep_recent_calls=settings.context_tier0c_keep_recent_calls,
            tier0d_digest_enabled=settings.context_tier0d_digest_enabled,
            tier0d_digest_min_bytes=settings.context_tier0d_digest_min_bytes,
            tier0e_enabled=getattr(settings, "context_tier0e_enabled", False),
            block_selection_mode=getattr(settings, "context_block_selection_mode", "selective"),
            block_seal_min_tail_tokens=getattr(settings, "context_block_seal_min_tail_tokens", 3_000),
            block_seal_min_requests=getattr(settings, "context_block_seal_min_requests", 4),
            block_target_summary_tokens=getattr(settings, "context_block_target_summary_tokens", 500),
            block_storage_dir=getattr(settings, "context_block_storage_dir", None),
        )

        dict_messages = [m.model_dump() for m in request_data.messages]
        dict_system = _system_to_dicts(request_data.system)
        dict_tools = [t.model_dump() for t in request_data.tools] if request_data.tools else None

        out_messages, out_system, token_count = await _PkgOptimizer.optimize(
            messages=dict_messages,
            system=dict_system,
            settings=pkg_settings,
            tools=dict_tools,
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


