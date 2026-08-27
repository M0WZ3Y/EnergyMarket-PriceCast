"""Tests for src/runtime.py — keep_awake() must fail loudly.

The defect these pin: keep_awake() returned None and printed at most a
platform note, so a caller could not tell whether it had been held awake,
and Modern Standby killed unattended runs while the function looked like it
had succeeded. A guard that cannot guarantee its guarantee must say so.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.runtime import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, KeepAwakeStatus, keep_awake


class _Kernel32:
    def __init__(self, result: int = 1) -> None:
        self.result = result
        self.calls: list[int] = []

    def SetThreadExecutionState(self, flags: int) -> int:  # noqa: N802
        self.calls.append(flags)
        return self.result


@pytest.fixture
def fake_ctypes(monkeypatch):
    """Install a fake ctypes.windll so these run on any platform."""

    def install(result: int = 1, explode: bool = False):
        mod = types.ModuleType("ctypes")
        if explode:
            # A non-Windows interpreter: ctypes imports fine, windll is
            # absent, so attribute access raises.
            monkeypatch.setitem(sys.modules, "ctypes", mod)
            return None
        kernel32 = _Kernel32(result)
        mod.windll = types.SimpleNamespace(kernel32=kernel32)
        monkeypatch.setitem(sys.modules, "ctypes", mod)
        return kernel32

    return install


def test_yields_a_status_rather_than_none(fake_ctypes, caplog):
    fake_ctypes(result=1)
    with caplog.at_level(logging.WARNING, logger="src.runtime"):
        with keep_awake() as status:
            assert isinstance(status, KeepAwakeStatus)
            assert status.requested is True
            # The whole point: never claims a guarantee.
            assert status.guaranteed is False


def test_warns_even_when_the_request_is_accepted(fake_ctypes, caplog):
    """The loud path is not an error path. The request succeeding still does
    not make the run safe, so it still warns and still names the cause."""
    fake_ctypes(result=1)
    with caplog.at_level(logging.WARNING, logger="src.runtime"):
        with keep_awake() as status:
            pass
    assert caplog.records, "no warning emitted at all"
    assert all(r.levelno >= logging.WARNING for r in caplog.records)
    assert "Modern Standby" in caplog.text
    assert status.guaranteed is False


def test_warns_loudly_when_the_platform_cannot_honour_the_request(fake_ctypes, caplog):
    """ctypes present but windll absent — a non-Windows interpreter."""
    fake_ctypes(explode=True)
    with caplog.at_level(logging.WARNING, logger="src.runtime"):
        with keep_awake() as status:
            pass
    assert status.requested is False
    assert status.guaranteed is False
    assert "not held awake" in caplog.text
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_warns_when_the_api_call_itself_fails(fake_ctypes, caplog):
    """SetThreadExecutionState returning 0 is failure. The old code discarded
    this, so a failed call was indistinguishable from a successful one."""
    kernel32 = fake_ctypes(result=0)
    with caplog.at_level(logging.WARNING, logger="src.runtime"):
        with keep_awake() as status:
            pass
    assert status.requested is False
    assert "failed" in caplog.text
    # And it must not try to "restore" a state it never set.
    assert kernel32.calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]


def test_releases_the_request_on_exit(fake_ctypes):
    kernel32 = fake_ctypes(result=1)
    with keep_awake():
        pass
    assert kernel32.calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]


def test_releases_the_request_even_when_the_body_raises(fake_ctypes):
    kernel32 = fake_ctypes(result=1)
    with pytest.raises(RuntimeError):
        with keep_awake():
            raise RuntimeError("boom")
    assert kernel32.calls[-1] == ES_CONTINUOUS


def test_docstring_states_the_limitation_in_its_first_line():
    """Buried caveats do not get read. T2 requires it first, and by name."""
    first = keep_awake.__doc__.strip().splitlines()[0]
    assert "ADVISORY ONLY" in first
    assert "Modern Standby" in first
