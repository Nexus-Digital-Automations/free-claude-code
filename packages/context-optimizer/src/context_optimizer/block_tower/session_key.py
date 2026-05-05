"""Owns: derivation of a stable session key from a message list.

The session key is the directory name under .context/blocks/ that scopes
one tower to one conversation. Two distinct conversations get distinct
towers; a resumed conversation rejoins its tower.

WHY first message hash: Claude Code does not propagate a stable session
ID through to the proxy, but the very first user message of a session is
verbatim-stable across resumes (it is what the user originally typed).
Hashing it gives a deterministic, conversation-scoped key without any
coordination with the host harness.

Does NOT own: the BlockStore (store.py), or the message list itself.
Called by: optimizer.py (Layer 0 entry).
Calls: _core.content_hash for the underlying SHA-256.
"""

from __future__ import annotations

from .._core import content_hash

_EMPTY_SESSION = "empty"


def derive_session_key(messages: list[dict]) -> str:
    """Return a 16-char prefix of sha256(first user message). Never raises.

    Returns "empty" when there are no user messages yet — the BlockStore
    treats that key as a no-op tower (zero blocks always).
    """
    for msg in messages:
        if msg.get("role") == "user":
            return content_hash([msg])[:16]
    return _EMPTY_SESSION
