"""Owns: Layer 0 immutable block tower — frozen compaction with selective inclusion.

Public exports are imported by optimizer.py; nothing else in the package
depends on this module's internals.

# @stable — optimizer.py depends on these names.
"""

from __future__ import annotations

from .selector import select_blocks
from .sealer import schedule_seal_if_due, should_seal
from .session_key import derive_session_key
from .store import BlockHandle, BlockStore

__all__ = [
    "BlockHandle",
    "BlockStore",
    "derive_session_key",
    "schedule_seal_if_due",
    "select_blocks",
    "should_seal",
]
