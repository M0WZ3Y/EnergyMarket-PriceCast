"""Process-level runtime helpers — src/runtime.py

Small utilities that long-running jobs need regardless of what they
compute. Kept out of src/evaluation/ because none of this is part of the
forecasting protocol.
"""

from __future__ import annotations

import contextlib
import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class KeepAwakeStatus(NamedTuple):
    """Outcome of a keep_awake() request.

    `guaranteed` is never True. It exists so callers cannot mistake "the
    request was accepted" for "the run is safe": the first is `requested`,
    the second is not something this function can deliver.
    """

    requested: bool
    guaranteed: bool
    detail: str


@contextlib.contextmanager
def keep_awake():
    """ADVISORY ONLY — cannot stop Windows Modern Standby from killing a run.

    Requests ES_SYSTEM_REQUIRED so the system idle timer does not fire. The
    OS may ignore it: on this machine, Modern Standby (S0) still suspends on
    lid close and under battery/DRIPS policy, and a user-initiated sleep
    always wins. Yields a KeepAwakeStatus and logs at WARNING every time,
    because the silence is what made this dangerous — the 2026-07-30 and
    2026-07-31 walk-forward runs were both killed mid-flight while this
    function returned as though it had prevented exactly that.

    Long unattended runs should stay on AC, where the idle timeout is
    "never". That is the actual mitigation; this call is not.
    """
    kernel32 = None
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # Returns the previous state, or 0 on failure. The old code threw
        # this away, so a failed call looked identical to a successful one.
        if kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
            status = KeepAwakeStatus(
                True, False, "requested, but advisory only: Modern Standby (S0) may still suspend"
            )
        else:
            kernel32 = None
            status = KeepAwakeStatus(False, False, "SetThreadExecutionState failed; not held awake")
    except Exception as exc:
        kernel32 = None
        status = KeepAwakeStatus(
            False, False, f"unavailable on this platform ({type(exc).__name__}); not held awake"
        )

    logger.warning("[keep-awake] %s", status.detail)
    try:
        yield status
    finally:
        if kernel32 is not None:
            kernel32.SetThreadExecutionState(ES_CONTINUOUS)
