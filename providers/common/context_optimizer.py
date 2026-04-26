"""Proxy-level autocompaction for long conversations.

Owns:
  Tier 0 — deterministic NLP preprocessing (ANSI strip, tool-result dedup,
            output truncation). Free, always-on.
  Tier 1 — strip stale thinking blocks from old assistant turns. Free, always-on.
  Tier 2a — background Ollama compaction at soft token threshold. Non-blocking.
  Tier 2b — sync provider compaction at hard token threshold. Blocking fallback.
  Prefix cache — in-memory LRU; applied before any tier to reuse earlier summaries.

Does NOT own: message format conversion (message_converter.py), provider request
building (providers/<name>/request.py), DeepSeek reasoning_content retry
(providers/deepseek/client.py — that is the safety net beneath this layer).

Called by: api/routes.py create_message().
Calls: api.request_utils.get_token_count; provider._client (openai_compat.py);
       Ollama at settings.ollama_base_url.

State flow per request:
  raw -> tier0 -> tier1 -> prefix_cache_check ->
    tokens >= hard_threshold:  await tier2b (blocking, then return)
    tokens >= soft_threshold:  fire tier2a in background, return immediately
    else:                      return as-is
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import OrderedDict
from typing import Any, ClassVar

from loguru import logger
from openai import AsyncOpenAI

# Compaction call parameters — low temperature for stable, reproducible summaries
# (improves cache hit rate when the same prefix is seen across requests).
_COMPACTION_MAX_TOKENS = 4000
_COMPACTION_TEMPERATURE = 0.3

# History preview length in the compaction prompt. Full content would blow the
# compaction call's own context; 2000 chars captures intent without the tokens.
_RENDER_PREVIEW_CHARS = 2000

_CACHE_MAX_ENTRIES = 100

# Tier 0 — ANSI regex covers both CSI sequences and single-char escape codes.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b[A-Z]")
_TIER0_MAX_LINES = 200
_TIER0_HEAD_LINES = 50
_TIER0_TAIL_LINES = 50
_DEDUPE_PLACEHOLDER = "[identical to earlier tool result — omitted]"

_SPLIT_TAG = re.compile(r"<split_index>\s*(\d+)\s*</split_index>")
_SUMMARY_TAG = re.compile(r"<summary>\s*(.*?)\s*</summary>", re.DOTALL)


class ContextOptimizer:
    """Four-tier conversation optimizer.

    Cache is keyed by the hash of the PREFIX being replaced (messages[:split_index]),
    not the full history. This lets _apply_prefix_cache find summaries computed in
    prior requests whose output is still a valid prefix of the current conversation.

    _inflight prevents duplicate background Ollama tasks for the same conversation
    state. _background_tasks holds strong references so asyncio does not GC tasks
    before they finish.

    # @stable — api/routes.py depends on the optimize() signature.
    # EXTENSION POINT: add new tiers inside optimize() between Tier 1 and Tier 2a.
    """

    _summary_cache: ClassVar[OrderedDict[str, tuple[int, str]]] = OrderedDict()
    _inflight: ClassVar[set[str]] = set()
    # Strong refs prevent GC of fire-and-forget tasks (Python asyncio requirement).
    _background_tasks: ClassVar[set[asyncio.Task]] = set()
    # Serialises all background Ollama calls so multiple simultaneous Claude Code
    # instances don't pile concurrent requests onto a single Ollama process.
    _ollama_semaphore: ClassVar[asyncio.Semaphore | None] = None

    @classmethod
    def _get_ollama_semaphore(cls) -> asyncio.Semaphore:
        if cls._ollama_semaphore is None:
            cls._ollama_semaphore = asyncio.Semaphore(1)
        return cls._ollama_semaphore

    @classmethod
    async def optimize(cls, request_data: Any, settings: Any, provider: Any) -> Any:
        """Apply all optimization tiers; return (possibly new) request_data.

        Never raises — failures degrade gracefully to the previous tier's result.
        # @stable
        """
        from api.request_utils import get_token_count

        messages = cls._apply_tier0(list(request_data.messages))
        messages = cls._strip_old_thinking(messages, settings.context_max_thinking_turns)
        messages = cls._apply_prefix_cache(messages)

        tokens = get_token_count(messages, request_data.system, request_data.tools)

        if tokens >= settings.context_compact_threshold_tokens:
            logger.info(
                "CONTEXT_OPT: triggering sync compaction tokens={} threshold={} msgs={}",
                tokens, settings.context_compact_threshold_tokens, len(messages),
            )
            compacted = await cls._compact_via_provider(messages, request_data, provider)
            if compacted is not messages:
                new_tokens = get_token_count(compacted, request_data.system, request_data.tools)
                logger.info(
                    "CONTEXT_OPT: compacted {} -> {} messages, tokens {} -> {}",
                    len(messages), len(compacted), tokens, new_tokens,
                )
                messages = compacted
        elif tokens >= settings.context_compact_soft_threshold_tokens:
            near_hard = tokens >= settings.context_compact_deepseek_fallback_threshold_tokens
            cls._schedule_background_compaction(
                messages, settings, provider, request_data,
                use_deepseek_fallback=near_hard,
            )

        return request_data.model_copy(update={"messages": messages})

    # ---- Tier 0: deterministic NLP ----

    @staticmethod
    def _apply_tier0(messages: list) -> list:
        before_bytes = sum(len(ContextOptimizer._render_content(m.content)) for m in messages)
        messages = ContextOptimizer._strip_ansi_from_messages(messages)
        messages = ContextOptimizer._dedupe_tool_results(messages)
        messages = ContextOptimizer._truncate_long_outputs(messages)
        after_bytes = sum(len(ContextOptimizer._render_content(m.content)) for m in messages)
        if before_bytes != after_bytes:
            logger.info(
                "CONTEXT_OPT: tier0 bytes_before={} bytes_after={} saved={}",
                before_bytes, after_bytes, before_bytes - after_bytes,
            )
        return messages

    @staticmethod
    def _strip_ansi_from_messages(messages: list) -> list:
        result = []
        changed = False
        for msg in messages:
            if isinstance(msg.content, str):
                cleaned = _ANSI_RE.sub("", msg.content)
                if cleaned != msg.content:
                    msg = msg.model_copy(update={"content": cleaned})
                    changed = True
            elif isinstance(msg.content, list):
                new_blocks, msg_changed = ContextOptimizer._strip_ansi_from_blocks(msg.content)
                if msg_changed:
                    msg = msg.model_copy(update={"content": new_blocks})
                    changed = True
            result.append(msg)
        return result if changed else messages

    @staticmethod
    def _strip_ansi_from_blocks(blocks: list) -> tuple[list, bool]:
        result = []
        changed = False
        for block in blocks:
            btype = getattr(block, "type", None)
            if btype == "text":
                cleaned = _ANSI_RE.sub("", block.text)
                if cleaned != block.text:
                    block = block.model_copy(update={"text": cleaned})
                    changed = True
            elif btype == "tool_result" and isinstance(getattr(block, "content", None), str):
                cleaned = _ANSI_RE.sub("", block.content)
                if cleaned != block.content:
                    block = block.model_copy(update={"content": cleaned})
                    changed = True
            result.append(block)
        return result, changed

    @staticmethod
    def _dedupe_tool_results(messages: list) -> list:
        """Replace duplicate tool-result content with a short placeholder.

        WHY: repeated file reads and identical command outputs are pure token
        waste. Content-hash dedup catches them regardless of which tool ran.
        """
        seen: set[str] = set()
        result = []
        changed = False
        for msg in messages:
            if isinstance(msg.content, list):
                new_blocks = []
                msg_changed = False
                for block in msg.content:
                    if getattr(block, "type", None) == "tool_result":
                        raw = getattr(block, "content", "") or ""
                        text = raw if isinstance(raw, str) else str(raw)
                        h = hashlib.sha256(text.encode()).hexdigest()
                        if text.strip() and h in seen:
                            block = block.model_copy(update={"content": _DEDUPE_PLACEHOLDER})
                            msg_changed = True
                        else:
                            seen.add(h)
                    new_blocks.append(block)
                if msg_changed:
                    msg = msg.model_copy(update={"content": new_blocks})
                    changed = True
            result.append(msg)
        return result if changed else messages

    @staticmethod
    def _truncate_long_outputs(messages: list) -> list:
        """Truncate tool results that exceed _TIER0_MAX_LINES lines.

        WHY: long bash output (find, grep -r, test runs) inflates token counts
        with noise; head+tail preserves both start and end of output.
        """
        result = []
        changed = False
        for msg in messages:
            if isinstance(msg.content, list):
                new_blocks = []
                msg_changed = False
                for block in msg.content:
                    if getattr(block, "type", None) == "tool_result":
                        raw = getattr(block, "content", "") or ""
                        if isinstance(raw, str):
                            lines = raw.split("\n")
                            if len(lines) > _TIER0_MAX_LINES:
                                omitted = len(lines) - _TIER0_HEAD_LINES - _TIER0_TAIL_LINES
                                truncated = (
                                    "\n".join(lines[:_TIER0_HEAD_LINES])
                                    + f"\n... [{omitted} lines omitted] ...\n"
                                    + "\n".join(lines[-_TIER0_TAIL_LINES:])
                                )
                                block = block.model_copy(update={"content": truncated})
                                msg_changed = True
                    new_blocks.append(block)
                if msg_changed:
                    msg = msg.model_copy(update={"content": new_blocks})
                    changed = True
            result.append(msg)
        return result if changed else messages

    # ---- Tier 1: thinking-block strip ----

    @staticmethod
    def _strip_old_thinking(messages: list, keep_last_n: int) -> list:
        """Drop ContentBlockThinking from all but the last keep_last_n assistant turns.

        WHY: old reasoning has no bearing on current quality, but every thinking
        block costs tokens on every subsequent request.
        """
        from api.models.anthropic import ContentBlockThinking

        if keep_last_n <= 0:
            return messages
        assistant_idx = [i for i, m in enumerate(messages) if m.role == "assistant"]
        if len(assistant_idx) <= keep_last_n:
            return messages

        keep = set(assistant_idx[-keep_last_n:])
        result = []
        stripped_blocks = 0
        for i, msg in enumerate(messages):
            if msg.role == "assistant" and i not in keep and isinstance(msg.content, list):
                pruned = [b for b in msg.content if not isinstance(b, ContentBlockThinking)]
                if len(pruned) < len(msg.content):
                    stripped_blocks += len(msg.content) - len(pruned)
                    msg = msg.model_copy(update={"content": pruned or msg.content})
            result.append(msg)
        if stripped_blocks:
            logger.info(
                "CONTEXT_OPT: tier1 stripped={} thinking_blocks old_turns={}",
                stripped_blocks, len(assistant_idx) - keep_last_n,
            )
        return result

    # ---- Prefix cache lookup ----

    @classmethod
    def _apply_prefix_cache(cls, messages: list) -> list:
        """Apply a cached summary for any stored prefix of the current messages.

        WHY: background Tier 2a stores summaries keyed by the prefix they replace.
        On the next request the same prefix is a leading substring of the longer
        history; we can apply the pre-computed summary without a new LLM call.
        """
        n = len(messages)
        candidates = sorted(
            {k for k in (n - 2, max(4, n // 2), max(4, n // 3), 4) if 4 <= k <= n - 2},
            reverse=True,
        )
        for k in candidates:
            key = cls._cache_key(messages[:k])
            entry = cls._summary_cache.get(key)
            if entry is not None:
                cls._summary_cache.move_to_end(key)
                _, summary = entry
                logger.info("CONTEXT_OPT: prefix_cache hit k={} msgs_replaced={}", k, k)
                return cls._apply_summary(messages, k, summary)
        return messages

    # ---- Tier 2a: background Ollama ----

    @classmethod
    def _schedule_background_compaction(
        cls,
        messages: list,
        settings: Any,
        provider: Any,
        request_data: Any,
        *,
        use_deepseek_fallback: bool = False,
    ) -> None:
        """Fire background compaction unless one is already in-flight for this prefix."""
        inflight_key = cls._cache_key(messages)
        if inflight_key in cls._inflight:
            logger.debug("CONTEXT_OPT: background compaction already in-flight, skipping")
            return
        cls._inflight.add(inflight_key)
        logger.info(
            "CONTEXT_OPT: scheduling background compaction msgs={} deepseek_fallback={}",
            len(messages), use_deepseek_fallback,
        )
        task = asyncio.create_task(
            cls._run_background_compaction(
                messages, settings, inflight_key,
                provider=provider, request_data=request_data,
                use_deepseek_fallback=use_deepseek_fallback,
            )
        )
        cls._background_tasks.add(task)
        task.add_done_callback(cls._background_tasks.discard)

    @classmethod
    async def _run_background_compaction(
        cls,
        messages: list,
        settings: Any,
        inflight_key: str,
        *,
        provider: Any,
        request_data: Any,
        use_deepseek_fallback: bool,
    ) -> None:
        try:
            if use_deepseek_fallback:
                # Near the hard token limit — try Ollama only if it's immediately available.
                # If it's busy or down, fall back to the provider rather than queue and risk
                # the conversation reaching 200K without any compaction having happened.
                # asyncio is single-threaded so locked() + acquire() is race-free (no await
                # between check and acquire when _value > 0).
                if not cls._get_ollama_semaphore().locked():
                    async with cls._get_ollama_semaphore():
                        compacted = await cls._do_ollama_call(messages, settings)
                    if compacted is messages:
                        logger.info("CONTEXT_OPT: Ollama failed near hard limit, using provider fallback")
                        compacted = await cls._compact_via_provider(messages, request_data, provider)
                else:
                    logger.info("CONTEXT_OPT: Ollama busy near hard limit, using provider fallback")
                    compacted = await cls._compact_via_provider(messages, request_data, provider)
            else:
                compacted = await cls._compact_via_ollama(messages, settings)

            if compacted is not messages:
                logger.info(
                    "CONTEXT_OPT: background compaction stored {} -> {} messages",
                    len(messages), len(compacted),
                )
        except Exception as exc:
            # Swallowing here is intentional — background task failures must not
            # propagate to the event loop as unhandled exceptions (asyncio noise).
            logger.warning(
                "CONTEXT_OPT: background compaction raised {}: {}", type(exc).__name__, exc
            )
        finally:
            cls._inflight.discard(inflight_key)

    @classmethod
    async def _compact_via_ollama(cls, messages: list, settings: Any) -> list:
        """Compact via local Ollama with semaphore serialisation. Returns compacted or original."""
        async with cls._get_ollama_semaphore():
            return await cls._do_ollama_call(messages, settings)

    @classmethod
    async def _do_ollama_call(cls, messages: list, settings: Any) -> list:
        """Raw Ollama HTTP call without semaphore. Caller must hold the semaphore.

        Stores in prefix cache on success; returns original messages on any failure.
        """
        prompt = cls._build_prompt(messages)
        try:
            # api_key="ollama" is required by the SDK but ignored by Ollama's server.
            client = AsyncOpenAI(api_key="ollama", base_url=settings.ollama_base_url)  # pragma: allowlist secret — Ollama ignores api_key
            resp = await client.chat.completions.create(
                model=settings.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_COMPACTION_MAX_TOKENS,
                temperature=_COMPACTION_TEMPERATURE,
                stream=False,
            )
            await client.close()
            content = resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("CONTEXT_OPT: Ollama call failed {}: {}", type(exc).__name__, exc)
            return messages

        parsed = cls._parse_response(content, len(messages))
        if parsed is None:
            logger.warning(
                "CONTEXT_OPT: Ollama parse failed; first 200 chars: {!r}", content[:200]
            )
            return messages

        split_index, summary = parsed
        # Keyed by the replaced prefix so _apply_prefix_cache can find it next turn.
        cls._cache_put(cls._cache_key(messages[:split_index]), (split_index, summary))
        logger.info(
            "CONTEXT_OPT: ollama compacted split_index={} msgs_before={} summary_chars={}",
            split_index, len(messages), len(summary),
        )
        return cls._apply_summary(messages, split_index, summary)

    # ---- Tier 2b: sync provider compaction ----

    @classmethod
    async def _compact_via_provider(
        cls, messages: list, request_data: Any, provider: Any
    ) -> list:
        """Compact via the active provider (blocking). Returns compacted or original.

        Counterpart: provider._client is AsyncOpenAI set up in openai_compat.py.
        Reusing it avoids duplicating credential/base_url resolution per provider.
        """
        prompt = cls._build_prompt(messages)
        try:
            resp = await provider._client.chat.completions.create(
                model=request_data.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_COMPACTION_MAX_TOKENS,
                temperature=_COMPACTION_TEMPERATURE,
                stream=False,
            )
            content = resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(
                "CONTEXT_OPT: provider compaction failed {}: {}", type(exc).__name__, exc
            )
            return messages

        parsed = cls._parse_response(content, len(messages))
        if parsed is None:
            logger.warning(
                "CONTEXT_OPT: provider parse failed; first 200 chars: {!r}", content[:200]
            )
            return messages

        split_index, summary = parsed
        cls._cache_put(cls._cache_key(messages[:split_index]), (split_index, summary))
        logger.info(
            "CONTEXT_OPT: provider compacted split_index={} msgs_before={} summary_chars={}",
            split_index, len(messages), len(summary),
        )
        return cls._apply_summary(messages, split_index, summary)

    # ---- Shared compaction helpers ----

    @staticmethod
    def _build_prompt(messages: list) -> str:
        lines = []
        for i, msg in enumerate(messages):
            text = ContextOptimizer._render_content(msg.content)
            if len(text) > _RENDER_PREVIEW_CHARS:
                text = text[:_RENDER_PREVIEW_CHARS] + " ... [truncated]"
            lines.append(f"[{i}] {msg.role}: {text}")
        n = len(messages)
        return (
            "You are a conversation compactor. Compact this conversation history to reduce "
            "token usage while preserving the context the assistant needs.\n\n"
            "1. Choose split_index: the message index where verbatim history begins. "
            "Messages BEFORE split_index are replaced by your summary; messages from "
            "split_index onward are kept verbatim. Choose a natural breakpoint where "
            "messages[split_index] is a user message.\n"
            "2. Summarize messages BEFORE split_index: key decisions, file paths, code "
            "patterns, unresolved tasks, important user context, tools used.\n\n"
            "Output ONLY these tags:\n"
            "<split_index>NUMBER</split_index>\n"
            "<summary>SUMMARY TEXT</summary>\n\n"
            f"Conversation has {n} messages. split_index must be between 4 and {max(4, n-2)}.\n\n"
            "CONVERSATION:\n" + "\n\n".join(lines)
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
        return (split_index, summary) if summary else None

    @staticmethod
    def _apply_summary(messages: list, split_index: int, summary: str) -> list:
        from api.models.anthropic import ContentBlockText, Message

        synthetic = Message(
            role="user",
            content=[ContentBlockText(
                type="text",
                text=(
                    "<conversation_summary>\n"
                    "Earlier conversation compacted. Summary:\n\n"
                    f"{summary}\n"
                    "</conversation_summary>"
                ),
            )],
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
                r = getattr(block, "content", "")
                if isinstance(r, list):
                    r = " ".join(getattr(b, "text", "") for b in r if hasattr(b, "text"))
                parts.append(f"[tool_result: {str(r)[:500]}]")
            elif btype == "image":
                parts.append("[image]")
            # ContentBlockThinking skipped — stale reasoning wastes prompt tokens.
        return " ".join(p for p in parts if p)

    # ---- Cache internals ----

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
