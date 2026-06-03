"""Owns: tier0f — span-level Rabin-Karp dedup before block-tower compaction.

Finds every exact substring >= settings.tier0f_min_tokens (default 70) that
appears more than once across in-scope text spans of the request. Keeps the
first occurrence verbatim; deletes every later occurrence with no replacement
marker — bare deletion, by design.

Read-only definer sources (their tokens act as "originals" but are NEVER
deleted, only used to dedup messages against):
  - The system prompt — DeepSeek prefix-cache anchor; mutating it would
    invalidate caching for every subsequent request, costing far more than
    dedup saves.
  - The last user message — the active query; deleting from it could
    silently drop content the user just typed.

Operates on text-bearing locations only:
  - msg["content"] when str
  - block["text"] for text blocks
  - block["content"] for tool_result blocks (str only; structured-list
    tool_results are coalesced upstream by the OpenAI converter).

Does NOT touch:
  - tool_use.input / .id / .name (structural identifiers; tier0c/0e own .input)
  - tool_result.tool_use_id (link)
  - thinking blocks (tier1 strips them next)

# Decision record: 70-token threshold + no replacement marker
# - 70 tokens (~280 chars / ~50 words / 3 sentences): real-world duplicates at
#   this size are dominated by repeated *structural* blocks (file outlines,
#   error traces, code) that begin/end at newlines, so deletions land on
#   whitespace by construction. Mid-text "word fusion" is rare in practice.
# - 50 tokens captures ~10% more savings but raises fusion risk noticeably.
# - 100 tokens is safest but starts missing recurring boilerplate (import
#   blocks, repeated short error messages).
# - No marker: any marker eats ~10 tokens per occurrence and partially
#   defeats the savings on smaller matches. Re-evaluate if model behaviour
#   degrades.

Called by: optimizer.optimize() between tier0e and tier1.
Calls: token_counter._get_encoder for the configured tokenizer.

# @stable
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..settings import ContextOptimizerSettings
from ..token_counter import _get_encoder

# Polynomial rolling hash — Mersenne prime fits in a Python int with no
# overflow concerns. Collision rate ~2^-61 per probe; every probe is
# verified by full token-tuple equality before deletion, so collisions
# never produce wrong output.
_HASH_MOD = (1 << 61) - 1
_HASH_BASE = 131

_Ref = tuple[Any, ...]


def apply(
    messages: list[dict],
    settings: ContextOptimizerSettings,
    *,
    system: str | list | None = None,
) -> list[dict]:
    """Drop repeated >= K-token spans from messages; return same identity if nothing changed.

    K = settings.tier0f_min_tokens. Greedy maximal extension: a 200-token
    repeat collapses to a single deletion of 200 tokens, not three of K.

    Failure modes:
      - Tokenizer load error: propagates from token_counter._get_encoder.
        The same encoder is used by count_tokens() immediately after this
        tier in optimizer.optimize(), so a broken encoder is a system-wide
        problem the orchestrator must surface.
    # @stable
    """
    if not settings.tier0f_enabled:
        return messages
    k = settings.tier0f_min_tokens
    if k < 2:
        return messages
    enc = _get_encoder(settings.tokenizer_name)
    return _dedupe(messages, system, enc, k, settings)


# ---- core ----

def _dedupe(
    messages: list[dict],
    system: str | list | None,
    enc: Any,
    k: int,
    settings: ContextOptimizerSettings,
) -> list[dict]:
    deletable = list(_collect_deletable_spans(messages, settings))
    if not deletable:
        return messages

    span_tokens: list[list[int]] = []
    seen: dict[int, list[tuple[int, int]]] = {}

    for _, text in _collect_definer_spans(messages, system, settings):
        toks = enc.encode(text)
        sid = len(span_tokens)
        span_tokens.append(toks)
        _register(toks, sid, seen, k)

    rewrites: dict[_Ref, str] = {}
    for ref, text in deletable:
        toks = enc.encode(text)
        deletions = _find_deletions(toks, span_tokens, seen, k)
        sid = len(span_tokens)
        span_tokens.append(toks)
        _register(toks, sid, seen, k)
        if deletions:
            rewrites[ref] = enc.decode(_apply_deletions(toks, deletions))

    if not rewrites:
        return messages
    return _splice(messages, rewrites)


def _register(
    tokens: list[int], sid: int, seen: dict[int, list[tuple[int, int]]], k: int,
) -> None:
    for i, h in _rolling_hashes(tokens, k):
        seen.setdefault(h, []).append((sid, i))


def _find_deletions(
    tokens: list[int],
    span_tokens: list[list[int]],
    seen: dict[int, list[tuple[int, int]]],
    k: int,
) -> list[tuple[int, int]]:
    """Return non-overlapping (start, end) deletion ranges in tokens, sorted."""
    n = len(tokens)
    if n < k:
        return []
    hashes = dict(_rolling_hashes(tokens, k))

    deletions: list[tuple[int, int]] = []
    i = 0
    while i + k <= n:
        match_len = _longest_match_at(tokens, i, span_tokens, seen, hashes, k, n)
        if match_len >= k:
            deletions.append((i, i + match_len))
            i += match_len
        else:
            i += 1
    return deletions


def _longest_match_at(
    tokens: list[int],
    i: int,
    span_tokens: list[list[int]],
    seen: dict[int, list[tuple[int, int]]],
    hashes: dict[int, int],
    k: int,
    n: int,
) -> int:
    """Length of the greedy-maximal definer match starting at tokens[i], or 0."""
    candidates = seen.get(hashes[i])
    if not candidates:
        return 0
    window = tokens[i:i + k]
    for sid, off in candidates:
        def_toks = span_tokens[sid]
        if def_toks[off:off + k] != window:
            continue
        length = k
        while (
            i + length < n
            and off + length < len(def_toks)
            and tokens[i + length] == def_toks[off + length]
        ):
            length += 1
        return length
    return 0


def _apply_deletions(
    tokens: list[int], deletions: list[tuple[int, int]],
) -> list[int]:
    kept: list[int] = []
    cursor = 0
    for start, end in deletions:
        kept.extend(tokens[cursor:start])
        cursor = end
    kept.extend(tokens[cursor:])
    return kept


def _rolling_hashes(tokens: list[int], k: int) -> Iterator[tuple[int, int]]:
    n = len(tokens)
    if n < k:
        return
    base_k = pow(_HASH_BASE, k, _HASH_MOD)
    h = 0
    for j in range(k):
        h = (h * _HASH_BASE + (tokens[j] + 1)) % _HASH_MOD
    yield 0, h
    for i in range(1, n - k + 1):
        out_tok = tokens[i - 1] + 1
        in_tok = tokens[i + k - 1] + 1
        h = (h * _HASH_BASE - out_tok * base_k + in_tok) % _HASH_MOD
        yield i, h


# ---- span collection ----

def _collect_deletable_spans(
    messages: list[dict], settings: ContextOptimizerSettings,
) -> Iterator[tuple[_Ref, str]]:
    """Yield (ref, text) for every text-bearing span eligible for deletion."""
    last_user_idx = (
        _last_user_message_index(messages) if settings.tier0f_skip_last_user else -1
    )
    for msg_idx, msg in enumerate(messages):
        if msg_idx == last_user_idx:
            continue
        yield from _spans_in_message(msg, msg_idx)


def _collect_definer_spans(
    messages: list[dict],
    system: str | list | None,
    settings: ContextOptimizerSettings,
) -> Iterator[tuple[_Ref, str]]:
    """Yield (ref, text) for read-only definer spans (system + last user)."""
    if settings.tier0f_skip_system and system:
        yield from _system_spans(system)
    if settings.tier0f_skip_last_user:
        idx = _last_user_message_index(messages)
        if idx >= 0:
            yield from _spans_in_message(messages[idx], idx)


def _system_spans(system: str | list) -> Iterator[tuple[_Ref, str]]:
    if isinstance(system, str):
        if system:
            yield ("system", 0), system
        return
    if not isinstance(system, list):
        return
    for i, block in enumerate(system):
        if isinstance(block, dict):
            text = block.get("text", "")
            if isinstance(text, str) and text:
                yield ("system", i), text


def _spans_in_message(
    msg: dict, msg_idx: int,
) -> Iterator[tuple[_Ref, str]]:
    content = msg.get("content")
    if isinstance(content, str):
        if content:
            yield (msg_idx, "content"), content
        return
    if not isinstance(content, list):
        return
    for block_idx, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if isinstance(text, str) and text:
                yield (msg_idx, block_idx, "text"), text
        elif btype == "tool_result":
            raw = block.get("content", "")
            if isinstance(raw, str) and raw:
                yield (msg_idx, block_idx, "tool_result"), raw


def _last_user_message_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


# ---- splicing ----

def _splice(messages: list[dict], rewrites: dict[_Ref, str]) -> list[dict]:
    return [_rewrite_message(msg, idx, rewrites) for idx, msg in enumerate(messages)]


def _rewrite_message(
    msg: dict, msg_idx: int, rewrites: dict[_Ref, str],
) -> dict:
    content = msg.get("content")
    if isinstance(content, str):
        new_text = rewrites.get((msg_idx, "content"))
        if new_text is None:
            return msg
        return {**msg, "content": new_text}
    if not isinstance(content, list):
        return msg
    new_blocks = [
        _rewrite_block(b, msg_idx, bi, rewrites) for bi, b in enumerate(content)
    ]
    if all(nb is ob for nb, ob in zip(new_blocks, content, strict=True)):
        return msg
    return {**msg, "content": new_blocks}


def _rewrite_block(
    block: Any, msg_idx: int, block_idx: int, rewrites: dict[_Ref, str],
) -> Any:
    if not isinstance(block, dict):
        return block
    btype = block.get("type")
    if btype == "text":
        new_text = rewrites.get((msg_idx, block_idx, "text"))
        if new_text is not None:
            return {**block, "text": new_text}
    elif btype == "tool_result":
        new_content = rewrites.get((msg_idx, block_idx, "tool_result"))
        if new_content is not None:
            return {**block, "content": new_content}
    return block
