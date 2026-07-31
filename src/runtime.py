"""Process-level runtime helpers — src/runtime.py

Small utilities that long-running jobs need regardless of what they
compute. Kept out of src/evaluation/ because none of this is part of the
forecasting protocol.
"""

from __future__ import annotations

import contextlib

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_awake():
    """Hold Windows out of idle sleep for the duration of a long job.

    The job has to assert this itself. task_monitor.py makes the same
    call, but only while IT is tracking a job, so anything launched
    directly was unprotected -- which is how the 2026-07-30 and
    2026-07-31 walk-forward runs were both killed mid-flight by
    connected standby (see logs/decisions.md). This machine is Modern
    Standby (S0) with a 5-minute idle timeout on battery, so an
    unattended run on battery dies within minutes of the last keypress.

    Not a total guarantee: closing the lid still forces standby. Long
    runs should also stay on AC, where the idle timeout is "never".
    """
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception:
        kernel32 = None
        print("[keep-awake] unavailable on this platform", flush=True)
    try:
        yield
    finally:
        if kernel32 is not None:
            kernel32.SetThreadExecutionState(ES_CONTINUOUS)
