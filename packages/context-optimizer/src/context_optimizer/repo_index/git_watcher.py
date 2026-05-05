"""Owns: git HEAD tracking — get_head_sha() pure function and GitWatcher async poller.

Does NOT own: repo indexing, file parsing, or LLM calls.
Called by: repo_index/index.py (get_head_sha), optimizer.py (optional GitWatcher start/stop).
Calls: subprocess (git), gitpython (optional), asyncio.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Awaitable, Callable

from loguru import logger


def get_head_sha(repo_root: str) -> str | None:
    """Return the full 40-char commit SHA of HEAD, or None if not a git repo or HEAD is unborn.

    Pure function — no caching, no side effects.
    Tries GitPython first (richer error info), falls back to subprocess git.
    Callers must NOT assume the result is stable — always call fresh before acting on it.
    """
    try:
        from git import InvalidGitRepositoryError, NoSuchPathError, Repo

        try:
            repo = Repo(repo_root, search_parent_directories=False)
            return repo.head.commit.hexsha
        except (InvalidGitRepositoryError, NoSuchPathError, ValueError):
            pass
        except Exception as exc:
            logger.debug("REPO_INDEX: gitpython head_sha failed root={} reason={}", repo_root, exc)
    except ImportError:
        pass

    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            return sha or None
    except Exception as exc:
        logger.debug("REPO_INDEX: subprocess head_sha failed root={} reason={}", repo_root, exc)

    return None


class GitWatcher:
    """Polls HEAD commit SHA at a configurable interval; fires a callback on change.

    States: stopped ──start()──> running ──stop()──> stopped
    The callback is awaited in the event loop — it should not block.
    One instance per repo_root; do not share across event loops.
    """

    def __init__(
        self,
        repo_root: str,
        on_commit_change: Callable[[str], Awaitable[None]],
        poll_interval_seconds: float = 30.0,
    ) -> None:
        self._repo_root = repo_root
        self._on_commit_change = on_commit_change
        self._poll_interval = poll_interval_seconds
        self._task: asyncio.Task | None = None
        self._last_sha: str | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._last_sha = get_head_sha(self._repo_root)
        self._task = asyncio.create_task(self._poll_loop(), name="git-watcher")
        logger.info(
            "REPO_INDEX: git_watcher started root={} sha={}",
            self._repo_root,
            self._last_sha and self._last_sha[:7],
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("REPO_INDEX: git_watcher stopped root={}", self._repo_root)

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                sha = get_head_sha(self._repo_root)
                if sha and sha != self._last_sha:
                    logger.info(
                        "REPO_INDEX: git_watcher commit_change old={} new={}",
                        self._last_sha and self._last_sha[:7],
                        sha[:7],
                    )
                    self._last_sha = sha
                    await self._on_commit_change(sha)
            except Exception as exc:
                logger.warning("REPO_INDEX: git_watcher poll_error reason={}", exc)
