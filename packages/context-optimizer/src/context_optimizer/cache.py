"""Owns: in-memory LRU prefix cache for compaction summaries.

The cache is keyed by the content-hash of the prefix being replaced. On
a subsequent request whose message list starts with that same prefix,
the cached summary is applied (inserting it into the system prompt) so
no new LLM call is needed.

Does NOT own: computing summaries (tier2 does that) or applying them
to the system prompt (delegates to _core.apply_summary).
Called by: optimizer.py.
Calls: _core.content_hash, _core.apply_summary.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from loguru import logger

from ._core import apply_summary, content_hash


class PrefixCache:
    """LRU cache mapping prefix-hash -> (split_index, summary).

    # State: entries: {hash: (split_index, summary)}, ordered by recency.
    # Max size enforced at put time; LRU eviction when full.
    """

    def __init__(self, max_entries: int = 100) -> None:
        self._store: OrderedDict[str, tuple[int, str]] = OrderedDict()
        self._max_entries = max_entries

    def lookup(
        self, messages: list[dict], system: Any
    ) -> tuple[list[dict], Any] | None:
        """Find and apply a cached summary for any stored prefix of messages.

        Checks candidate k values {n-2, n/2, n/3, 4} ∩ [4, n-2]. Returns
        (truncated_messages, updated_system) on hit, None on miss.

        WHY these candidates: they cover the most likely split points from
        prior compactions without exhaustive linear search.
        """
        n = len(messages)
        candidates = sorted(
            {k for k in (n - 2, max(4, n // 2), max(4, n // 3), 4) if 4 <= k <= n - 2},
            reverse=True,
        )
        for k in candidates:
            key = content_hash(messages[:k])
            entry = self._store.get(key)
            if entry is not None:
                self._store.move_to_end(key)
                _, summary = entry
                logger.info("CONTEXT_OPT: prefix_cache hit k={} msgs_replaced={}", k, k)
                return apply_summary(messages, k, summary, system)
        return None

    def store(self, prefix_messages: list[dict], split_index: int, summary: str) -> None:
        """Cache a (split_index, summary) pair keyed by the prefix hash."""
        key = content_hash(prefix_messages[:split_index])
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = (split_index, summary)
            return
        if len(self._store) >= self._max_entries:
            self._store.popitem(last=False)
        self._store[key] = (split_index, summary)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
