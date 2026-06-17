"""Owns: core.headroom_sidecar lifecycle behaviour tests.

Counterpart: core/headroom_sidecar.py.

All docker/health I/O is mocked — these assert the control flow: the no-op
gates, idempotent skip when already running, fresh launch through OrbStack
when absent, and graceful degradation when a launch fails. No real container
is touched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core import headroom_sidecar


def _settings(*, enabled: bool = True, autostart: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        context_headroom_enabled=enabled,
        context_headroom_autostart=autostart,
        context_headroom_url="http://127.0.0.1:8787",
        context_headroom_image="ghcr.io/example/headroom:test",
    )


class _Runner:
    """Records every _run invocation and replies based on the subcommand."""

    def __init__(self, *, running: bool = False, run_rc: int = 0):
        self.calls: list[tuple[str, ...]] = []
        self._running = running
        self._run_rc = run_rc

    async def __call__(self, *args: str, timeout: float = 20.0) -> tuple[int, str]:
        self.calls.append(args)
        if "ps" in args:
            return (0, "fcc-headroom") if self._running else (0, "")
        if "run" in args:
            return (self._run_rc, "" if self._run_rc == 0 else "boom")
        return (0, "")

    def launched_container(self) -> bool:
        return any("run" in call for call in self.calls)

    def every_call_used_orbstack(self) -> bool:
        return all(
            call[:3] == ("docker", "--context", "orbstack") for call in self.calls
        )


def _patch(monkeypatch, runner: _Runner) -> None:
    async def _healthy(_url: str, _timeout: float) -> bool:
        return True

    monkeypatch.setattr(headroom_sidecar, "_run", runner)
    monkeypatch.setattr(headroom_sidecar, "_wait_healthy", _healthy)


@pytest.mark.asyncio
async def test_disabled_is_noop(monkeypatch):
    runner = _Runner()
    _patch(monkeypatch, runner)
    await headroom_sidecar.ensure_started(_settings(enabled=False))
    assert runner.calls == []


@pytest.mark.asyncio
async def test_autostart_false_is_noop(monkeypatch):
    runner = _Runner()
    _patch(monkeypatch, runner)
    await headroom_sidecar.ensure_started(_settings(enabled=True, autostart=False))
    assert runner.calls == []


@pytest.mark.asyncio
async def test_already_running_skips_launch(monkeypatch):
    runner = _Runner(running=True)
    _patch(monkeypatch, runner)
    await headroom_sidecar.ensure_started(_settings())
    assert not runner.launched_container()  # only a `ps` probe, no `run`


@pytest.mark.asyncio
async def test_starts_when_absent_via_orbstack(monkeypatch):
    runner = _Runner(running=False)
    _patch(monkeypatch, runner)
    await headroom_sidecar.ensure_started(_settings())
    assert runner.launched_container()
    assert runner.every_call_used_orbstack()  # criterion: explicit OrbStack context


@pytest.mark.asyncio
async def test_launch_failure_degrades_without_raising(monkeypatch):
    runner = _Runner(running=False, run_rc=1)
    _patch(monkeypatch, runner)
    # Must not raise even though `docker run` returned non-zero.
    await headroom_sidecar.ensure_started(_settings())
    assert runner.launched_container()


@pytest.mark.asyncio
async def test_stop_noop_when_unmanaged(monkeypatch):
    runner = _Runner()
    _patch(monkeypatch, runner)
    await headroom_sidecar.stop(_settings(enabled=False))
    assert runner.calls == []


@pytest.mark.asyncio
async def test_stop_invokes_docker_when_managed(monkeypatch):
    runner = _Runner()
    _patch(monkeypatch, runner)
    await headroom_sidecar.stop(_settings())
    assert any("stop" in call for call in runner.calls)
    assert runner.every_call_used_orbstack()
