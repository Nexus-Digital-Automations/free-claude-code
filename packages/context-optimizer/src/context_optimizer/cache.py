"""Owns: in-memory LRU prefix cache for compaction summaries.

The cache is keyed by the content-hash of the prefix being replaced. On
a subsequent request whose message list starts with that same prefix,
the cached summary is applied (inserting it into the system prompt) so
no new LLM call is needed.

Persistence: when ``persist_path`` is provided the cache writes itself to
disk after every ``store()`` via an atomic temp+rename, and loads from
disk on construction.  Default ``persist_path=None`` — zero file I/O.

Does NOT own: computing summaries (tier2 does that) or applying them
to the system prompt (delegates to _core.apply_summary).
Called by: optimizer.py.
Calls: _core.content_hash, _core.apply_summary.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from loguru import logger

from ._core import apply_summary, content_hash

_CACHE_VERSION = 2


class PrefixCache:
    """LRU cache mapping prefix-hash -> (split_index, summary).

    # State:
    #   _store: dict {hash: (split_index, summary)}  — LRU ordered
    #   _persist_path: str | None  — disk path or None = ephemeral
    #   _dirty: bool              — true when _store changed since last write

    # EXTENSION POINT: subclasses can override _serialize / _deserialize for
    # alternative storage formats (e.g. compressed, sharded).

    # @stable
    """

    def __init__(
        self, max_entries: int = 100, persist_path: str | None = None
    ) -> None:
        self._store: OrderedDict[str, tuple[int, str]] = OrderedDict()
        self._max_entries = max_entries
        self._persist_path = persist_path
        self._dirty = False
        if persist_path is not None:
            self._load()

    def lookup(
        self, messages: list[dict], system: Any
    ) -> tuple[list[dict], Any] | None:
        """Find and apply a cached summary for any stored prefix of messages.

        Iterates every stored entry and checks whether its
        ``content_hash(messages[:entry.split_index])`` matches the entry's key.
        First match wins — entries are searched in MRU order so the most
        recently added/used summary is preferred. Returns (truncated_messages,
        updated_system) on hit, None on miss.

        WHY iterate vs. fixed candidates: Tier 2a stores at whatever
        split_index the LLM picked (a "natural breakpoint" — rarely n/2,
        n/3, or n-2). The previous fixed-candidate lookup missed those
        legitimate hits. O(cache_size) hashing is cheap (≤100 entries by
        default) compared to the multi-second LLM call this avoids.
        """
        n = len(messages)
        # Iterate MRU first — OrderedDict yields insertion order; reverse
        # gives most-recently-added/touched first.
        for key in reversed(self._store):
            split_index, summary = self._store[key]
            if split_index < 4 or split_index > n - 2:
                continue
            if content_hash(messages[:split_index]) != key:
                continue
            self._store.move_to_end(key)
            self._dirty = True
            logger.info(
                "CONTEXT_OPT: prefix_cache hit k={} msgs_replaced={}",
                split_index, split_index,
            )
            return apply_summary(messages, split_index, summary, system)
        return None

    def store(self, prefix_messages: list[dict], split_index: int, summary: str) -> None:
        """Cache a (split_index, summary) pair keyed by the prefix hash.

        Persists to disk if ``persist_path`` was set.
        """
        key = content_hash(prefix_messages[:split_index])
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = (split_index, summary)
        else:
            if len(self._store) >= self._max_entries:
                self._store.popitem(last=False)
            self._store[key] = (split_index, summary)
        self._dirty = True
        if self._persist_path is not None:
            self._save()

    def clear(self) -> None:
        """Reset the cache and remove the on-disk file if present."""
        self._store.clear()
        self._dirty = False
        if self._persist_path is not None:
            try:
                os.remove(self._persist_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    "CONTEXT_OPT: failed to remove cache file {}: {}",
                    self._persist_path, exc,
                )

    # ── Serialization helpers ──────────────────────────────────────────────

    def _serialize(self) -> dict:
        return {
            "version": _CACHE_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "entries": [
                {
                    "key": k,
                    "split_index": v[0],
                    "summary": v[1],
                }
                for k, v in self._store.items()
            ],
        }

    @staticmethod
    def _deserialize(data: dict) -> list[tuple[str, tuple[int, str]]]:
        """Return list of (key, (split_index, summary)) tuples in storage order."""
        version = data.get("version", 1)
        if version < _CACHE_VERSION:
            logger.info(
                "CONTEXT_OPT: upgrading cache v{} -> v{}",
                version, _CACHE_VERSION,
            )
        return [
            (e["key"], (e["split_index"], e["summary"]))
            for e in data.get("entries", [])
        ]

    def _save(self) -> None:
        """Atomic temp+rename write. Logs and swallows I/O errors."""
        if not self._dirty:
            return
        assert self._persist_path is not None  # guard
        tmp = self._persist_path + ".tmp"
        try:
            Path(self._persist_path).parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(self._serialize(), f, indent=2)
            os.rename(tmp, self._persist_path)
            self._dirty = False
        except OSError as exc:
            logger.warning(
                "CONTEXT_OPT: failed to persist cache to {}: {}",
                self._persist_path, exc,
            )
            try:
                os.remove(tmp)
            except (FileNotFoundError, OSError):
                pass

    def _load(self) -> None:
        """Load entries from disk. On any error start with an empty cache."""
        assert self._persist_path is not None  # guard
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            entries = self._deserialize(data)
            # Preserve insertion order so LRU eviction matches chronology
            for key, value in entries:
                self._store[key] = value
            logger.info(
                "CONTEXT_OPT: loaded {} cached entries from {}",
                len(self._store), self._persist_path,
            )
        except FileNotFoundError:
            logger.debug(
                "CONTEXT_OPT: no cache file at {} — starting empty",
                self._persist_path,
            )
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as exc:
            logger.warning(
                "CONTEXT_OPT: corrupt or unreadable cache {} ({}); starting empty",
                self._persist_path, exc,
            )

    def __len__(self) -> int:
        return len(self._store)
