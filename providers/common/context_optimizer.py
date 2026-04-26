"""Proxy-level autocompaction for long conversations.

Owns: token-cheap stripping of stale thinking blocks (Tier 1) and LLM-based
summarization of older turns when the request crosses a token threshold (Tier 2).

Does NOT own: format conversion (see message_converter.py), provider request
building (see providers/<name>/request.py), or the DeepSeek reasoning_content
retry path (see providers/deepseek/client.py — that remains the safety net
beneath this layer).

Called by: api/routes.py create_message(), once per inbound /v1/messages request.
Calls: api.request_utils.get_token_count for budget checks; the active
provider's existing AsyncOpenAI client for the compaction summarization call.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Any, ClassVar

from loguru import logger

# WHY lazy imports: api/__init__.py loads api.app which loads api.routes which
# imports this module — top-level `from api.* import` would cycle. The api
# layer is loaded by the time these methods run, so deferral is safe.

# Compaction call output limits — summary should be concise, prompt overhead
# stays low. Temperature kept low for deterministic, stable summaries (helps
# cache hit rate for identical prefixes).
_COMPACTION_MAX_TOKENS = 4000
_COMPACTION_TEMPERATURE = 0.3

# Per-message preview length when rendering history into the compaction prompt;
# WHY: full content of every message would itself blow the context budget on
# the compaction call. 2000 chars captures intent without sending every token.
_RENDER_PREVIEW_CHARS = 2000

# Cache budget. Process-local; concurrent misses are acceptable (compute is
# idempotent — two parallel requests with the same prefix may both compact,
# but the result will be the same and one entry will win the dict slot).
_CACHE_MAX_ENTRIES = 100

_SPLIT_TAG = re.compile(r"<split_index>\s*(\d+)\s*</split_index>")
_SUMMARY_TAG = re.compile(r"<summary>\s*(.*?)\s*</summary>", re.DOTALL)


class ContextOptimizer:
    """Two-tier conversation optimizer.

    States flow through optimize() in one direction:
      raw -> tier1_stripped -> (if over budget) tier2_compacted -> out

    On Tier 2 failure, the request falls back to tier1_stripped — the user
    request is never blocked or failed by an optimization problem.

    # @stable — called from api/routes.py.
    """

    # Cache key -> (split_index, summary_text). OrderedDict gives FIFO eviction.
    _summary_cache: ClassVar[OrderedDict[str, tuple[int, str]]] = OrderedDict()

    @classmethod
    async def optimize(cls, request_data: Any, settings: Any, provider: Any) -> Any:
        """Apply Tier 1 unconditionally; Tier 2 if still over budget.

        Returns a (possibly new) request_data with optimized messages. Never
        raises — internal failures degrade to the tier-1 result.
        """
        from api.request_utils import get_token_count

        original = list(request_data.messages)
        stripped = cls._strip_old_thinking(original, settings.context_max_thinking_turns)

        tokens = get_token_count(stripped, request_data.system, request_data.tools)
        if tokens < settings.context_compact_threshold_tokens:
            return cls._maybe_replace(request_data, original, stripped)

        logger.info(
            "CONTEXT_OPT: triggering LLM compaction tokens={} threshold={} msgs={}",
            tokens,
            settings.context_compact_threshold_tokens,
            len(stripped),
        )
        compacted = await cls._compact_via_llm(stripped, request_data, provider)
        if compacted is stripped:
            # Tier 2 unavailable — fall back to tier-1 result.
            return cls._maybe_replace(request_data, original, stripped)

        new_tokens = get_token_count(
            compacted, request_data.system, request_data.tools
        )
        logger.info(
            "CONTEXT_OPT: compacted {} -> {} messages, tokens {} -> {}",
            len(stripped),
            len(compacted),
            tokens,
            new_tokens,
        )
        return request_data.model_copy(update={"messages": compacted})

    @staticmethod
    def _maybe_replace(request_data: Any, original: list, current: list) -> Any:
        if current is original:
            return request_data
        return request_data.model_copy(update={"messages": current})

    @staticmethod
    def _strip_old_thinking(messages: list, keep_last_n: int) -> list:
        """Drop ContentBlockThinking from all but the last keep_last_n assistant turns.

        WHY: old reasoning has no bearing on current reasoning quality, but every
        thinking block costs tokens on every subsequent request.
        """
        from api.models.anthropic import ContentBlockThinking

        if keep_last_n <= 0:
            return messages
        assistant_idx = [i for i, m in enumerate(messages) if m.role == "assistant"]
        if len(assistant_idx) <= keep_last_n:
            return messages

        keep = set(assistant_idx[-keep_last_n:])
        result: list = []
        stripped_blocks = 0
        for i, msg in enumerate(messages):
            if msg.role == "assistant" and i not in keep and isinstance(msg.content, list):
                pruned = [b for b in msg.content if not isinstance(b, ContentBlockThinking)]
                if len(pruned) < len(msg.content):
                    stripped_blocks += len(msg.content) - len(pruned)
                    msg = msg.model_copy(update={"content": pruned or msg.content})
            result.append(msg)
        if stripped_blocks:
            logger.debug(
                "CONTEXT_OPT: stripped {} thinking blocks from {} old turns",
                stripped_blocks,
                len(assistant_idx) - keep_last_n,
            )
        return result

    @classmethod
    async def _compact_via_llm(cls, messages: list, request_data: Any, provider: Any) -> list:
        """Summarize old turns via the configured provider.

        Returns compacted message list on success, or the input `messages`
        unchanged on any failure (network, parse, invalid split). Caller treats
        identity-equality with input as the failure signal.
        """
        cache_key = cls._cache_key(messages)
        cached = cls._summary_cache.get(cache_key)
        if cached is not None:
            cls._summary_cache.move_to_end(cache_key)
            split_index, summary = cached
            logger.debug("CONTEXT_OPT: cache hit, reusing summary")
            return cls._apply_summary(messages, split_index, summary)

        prompt = cls._build_prompt(messages)
        try:
            # Counterpart: provider._client is set up in providers/openai_compat.py
            # via AsyncOpenAI. Reusing it avoids duplicating credentials/base_url
            # resolution logic for every provider type.
            resp = await provider._client.chat.completions.create(
                model=request_data.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_COMPACTION_MAX_TOKENS,
                temperature=_COMPACTION_TEMPERATURE,
                stream=False,
            )
            content = resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("CONTEXT_OPT: compaction call failed: {}: {}", type(exc).__name__, exc)
            return messages

        parsed = cls._parse_response(content, len(messages))
        if parsed is None:
            logger.warning("CONTEXT_OPT: compaction response parse failed; first 200 chars: {!r}", content[:200])
            return messages

        split_index, summary = parsed
        cls._cache_put(cache_key, (split_index, summary))
        return cls._apply_summary(messages, split_index, summary)

    @staticmethod
    def _build_prompt(messages: list) -> str:
        rendered_lines = []
        for i, msg in enumerate(messages):
            text = ContextOptimizer._render_content(msg.content)
            if len(text) > _RENDER_PREVIEW_CHARS:
                text = text[:_RENDER_PREVIEW_CHARS] + " ... [truncated]"
            rendered_lines.append(f"[{i}] {msg.role}: {text}")
        history = "\n\n".join(rendered_lines)
        n = len(messages)
        max_split = max(4, n - 2)
        return (
            "You are a conversation compactor. Below is a conversation history that needs "
            "to be compacted to reduce token usage while preserving the context the "
            "assistant needs to continue helping the user.\n\n"
            "Your task:\n"
            "1. Choose split_index (an integer): the message index where verbatim history "
            "should begin. Messages BEFORE split_index will be replaced by your summary; "
            "messages from split_index onward will be kept verbatim. Choose split_index based "
            "on natural breakpoints in the conversation, keeping enough recent context that the "
            "conversation can continue naturally. Pick a value where messages[split_index] is a "
            "user message (so the conversation alternates correctly).\n"
            "2. Write a concise summary of messages BEFORE split_index that captures: "
            "key decisions made, file paths discussed, code patterns and structures, "
            "unresolved tasks, important context the user shared, and tools/commands used.\n\n"
            "Output format (use these exact tags, nothing else):\n"
            "<split_index>NUMBER</split_index>\n"
            "<summary>SUMMARY TEXT</summary>\n\n"
            f"Conversation has {n} messages. Valid split_index is between 4 and {max_split}.\n\n"
            "CONVERSATION:\n"
            f"{history}"
        )

    @staticmethod
    def _parse_response(content: str, num_messages: int) -> tuple[int, str] | None:
        idx_match = _SPLIT_TAG.search(content)
        sum_match = _SUMMARY_TAG.search(content)
        if not idx_match or not sum_match:
            return None
        split_index = int(idx_match.group(1))
        if split_index < 4 or split_index >= num_messages - 1:
            return None
        summary = sum_match.group(1).strip()
        if not summary:
            return None
        return split_index, summary

    @staticmethod
    def _apply_summary(messages: list, split_index: int, summary: str) -> list:
        from api.models.anthropic import ContentBlockText, Message

        synthetic = Message(
            role="user",
            content=[
                ContentBlockText(
                    type="text",
                    text=(
                        "<conversation_summary>\n"
                        "Earlier conversation has been compacted. Summary:\n\n"
                        f"{summary}\n"
                        "</conversation_summary>"
                    ),
                )
            ],
        )
        return [synthetic, *messages[split_index:]]

    @staticmethod
    def _render_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)
        parts: list[str] = []
        for block in content:
            btype = getattr(block, "type", None)
            if btype == "text":
                parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                parts.append(f"[tool_use: {getattr(block, 'name', '?')}]")
            elif btype == "tool_result":
                result = getattr(block, "content", "")
                if isinstance(result, list):
                    result = " ".join(getattr(b, "text", "") for b in result if hasattr(b, "text"))
                parts.append(f"[tool_result: {str(result)[:500]}]")
            elif btype == "image":
                parts.append("[image]")
            # ContentBlockThinking deliberately skipped — we don't want to
            # spend prompt tokens on stale reasoning during compaction either.
        return " ".join(p for p in parts if p)

    @staticmethod
    def _cache_key(messages: list) -> str:
        h = hashlib.sha256()
        for msg in messages:
            h.update(msg.role.encode())
            h.update(b"\x1f")
            h.update(ContextOptimizer._render_content(msg.content).encode())
            h.update(b"\x1e")
        return h.hexdigest()

    @classmethod
    def _cache_put(cls, key: str, value: tuple[int, str]) -> None:
        if key in cls._summary_cache:
            cls._summary_cache.move_to_end(key)
            cls._summary_cache[key] = value
            return
        if len(cls._summary_cache) >= _CACHE_MAX_ENTRIES:
            cls._summary_cache.popitem(last=False)
        cls._summary_cache[key] = value
