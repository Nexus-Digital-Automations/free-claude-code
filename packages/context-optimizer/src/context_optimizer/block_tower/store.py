"""Owns: on-disk + in-memory block storage for one session's tower.

A "tower" is an ordered, append-only list of immutable blocks. Each block is
stored as two files on disk:

    <storage_dir>/<session_key>/block-NNNN.txt        (frozen body bytes)
    <storage_dir>/<session_key>/block-NNNN.meta.json  (range + header)

Once written, a block file is never modified or deleted by this module.
The in-memory `BlockStore` is a per-session singleton — multiple Layer 0
calls within a process share the same instance.

State diagram (per session_key, single asyncio loop):

    unloaded ──get_or_build──> loaded
    loaded   ──seal──────────> loaded   (new BlockHandle appended)
    loaded   ──reset_for_test─> unloaded

Does NOT own: the sealing prompt (sealer.py), selection logic (selector.py),
or session-key derivation (session_key.py).
Called by: optimizer.py Layer 0, sealer.py.
Calls: stdlib only (json, os, pathlib).

# @stable — optimizer.py and sealer.py depend on these names.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from loguru import logger

_BLOCK_FILE_RE = re.compile(r"^block-(\d{4})\.txt$")


@dataclass(frozen=True)
class BlockHandle:
    """One sealed, immutable block. All fields are read-only after creation.

    range_start / range_end are message-list indices: this block covers
    `messages[range_start:range_end]` from the original conversation. The
    next block's range_start equals this block's range_end (no gaps, no
    overlaps — invariant enforced by BlockStore.seal).

    # @stable
    """
    block_index: int
    range_start: int
    range_end: int
    header: str        # one-line summary used by selector.py
    body_path: Path
    body_bytes: int    # cached file size for cheap token estimation


class BlockStore:
    """Per-session, in-memory + on-disk view of the block tower.

    Singleton per session_key — concurrent Layer 0 calls in the same
    asyncio process share state through the class-level _by_session
    map. Single-threaded asyncio makes this safe without locks.

    Layout invariants (enforced by seal):
      • blocks[i].block_index == i + 1  (1-indexed, contiguous)
      • blocks[i].range_start == blocks[i-1].range_end  (no gaps)

    # @stable
    """

    _by_session: ClassVar[dict[str, "BlockStore"]] = {}

    def __init__(self, session_key: str, session_dir: Path) -> None:
        self.session_key = session_key
        self.session_dir = session_dir
        self.blocks: list[BlockHandle] = []
        self.requests_since_last_seal: int = 0

    @classmethod
    def get_or_build(cls, session_key: str, storage_dir: Path) -> "BlockStore":
        """Return the in-memory store for `session_key`, loading from disk on first use.

        Never raises — disk-read failures are logged and yield an empty store.
        """
        existing = cls._by_session.get(session_key)
        if existing is not None:
            return existing

        session_dir = storage_dir / session_key
        store = cls(session_key, session_dir)
        try:
            store._load_from_disk()
        except Exception as exc:
            logger.warning(
                "BLOCK_TOWER: load failed session={} reason={} {}",
                session_key[:7], type(exc).__name__, exc,
            )
        cls._by_session[session_key] = store
        return store

    def _load_from_disk(self) -> None:
        if not self.session_dir.is_dir():
            return
        # Race: session_dir can disappear between is_dir() and listdir() if a
        # parallel cleanup deletes it (e.g. test teardown). Treat as "no
        # blocks loaded" rather than crashing the request.
        try:
            entries = sorted(os.listdir(self.session_dir))
        except FileNotFoundError:
            return
        for entry in entries:
            match = _BLOCK_FILE_RE.match(entry)
            if not match:
                continue
            block_index = int(match.group(1))
            body_path = self.session_dir / entry
            meta_path = self.session_dir / f"block-{block_index:04d}.meta.json"
            if not meta_path.is_file():
                logger.warning(
                    "BLOCK_TOWER: orphan block body skipped path={}", body_path,
                )
                continue
            meta = json.loads(meta_path.read_text())
            self.blocks.append(BlockHandle(
                block_index=block_index,
                range_start=meta["range_start"],
                range_end=meta["range_end"],
                header=meta["header"],
                body_path=body_path,
                body_bytes=body_path.stat().st_size,
            ))
        if self.blocks:
            logger.info(
                "BLOCK_TOWER: loaded session={} blocks={}",
                self.session_key[:7], len(self.blocks),
            )

    def seal(self, range_start: int, range_end: int, body: str, header: str) -> BlockHandle:
        """Append a new block atomically. Raises ValueError on invalid range.

        Range invariants (raises ValueError if violated):
          • range_start == self.last_end()  (no gap, no overlap)
          • range_end > range_start         (block covers ≥1 message)
        Header is truncated to 200 chars defensively.

        Atomic write: body/meta are written to .tmp paths and renamed.
        On any IO failure the partial files are cleaned up and the
        exception is logged + re-raised — callers must treat this as a
        skipped seal, not a successful one.
        """
        expected_start = self.last_end()
        if range_start != expected_start:
            raise ValueError(
                f"block range_start={range_start} must equal previous range_end={expected_start}"
            )
        if range_end <= range_start:
            raise ValueError(
                f"block range_end={range_end} must exceed range_start={range_start}"
            )

        next_index = len(self.blocks) + 1
        self.session_dir.mkdir(parents=True, exist_ok=True)
        body_path = self.session_dir / f"block-{next_index:04d}.txt"
        meta_path = self.session_dir / f"block-{next_index:04d}.meta.json"
        body_tmp = body_path.with_suffix(".txt.tmp")
        meta_tmp = meta_path.with_suffix(".meta.json.tmp")

        truncated_header = header.strip().replace("\n", " ")[:200]
        meta_doc = {
            "block_index": next_index,
            "range_start": range_start,
            "range_end": range_end,
            "header": truncated_header,
        }
        try:
            body_tmp.write_text(body)
            meta_tmp.write_text(json.dumps(meta_doc, indent=2))
            os.replace(body_tmp, body_path)
            os.replace(meta_tmp, meta_path)
        except OSError as exc:
            logger.error(
                "BLOCK_TOWER: seal write failed session={} index={} reason={} {}",
                self.session_key[:7], next_index, type(exc).__name__, exc,
            )
            for tmp in (body_tmp, meta_tmp):
                tmp.unlink(missing_ok=True)
            raise

        handle = BlockHandle(
            block_index=next_index,
            range_start=range_start,
            range_end=range_end,
            header=truncated_header,
            body_path=body_path,
            body_bytes=body_path.stat().st_size,
        )
        self.blocks.append(handle)
        self.requests_since_last_seal = 0
        logger.info(
            "BLOCK_TOWER: sealed session={} index={} range=[{}:{}] body_bytes={}",
            self.session_key[:7], next_index, range_start, range_end, handle.body_bytes,
        )
        return handle

    def last_end(self) -> int:
        """Return the message-list index where the next block must start (0 if empty)."""
        return self.blocks[-1].range_end if self.blocks else 0

    def read_body(self, handle: BlockHandle) -> str:
        """Return the frozen body bytes for `handle`. Raises FileNotFoundError if deleted externally."""
        return handle.body_path.read_text()

    def increment_request_counter(self) -> None:
        """Bump the seal-eligibility counter. Called once per Layer 0 entry."""
        self.requests_since_last_seal += 1

    @classmethod
    def reset_for_test(cls) -> None:
        # @internal — test isolation only.
        cls._by_session.clear()


def resolve_storage_dir(repo_root: str, override: str | None) -> Path:
    """Pick the directory to store the per-session block tree.

    Override wins; otherwise <repo_root>/.context/blocks/ mirrors the
    convention used by repo_index. Counterpart: settings.block_storage_dir.
    """
    if override:
        return Path(override)
    return Path(repo_root) / ".context" / "blocks"
