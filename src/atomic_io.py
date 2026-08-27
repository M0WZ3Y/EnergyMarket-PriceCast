"""Atomic CSV writes — src/atomic_io.py

On 2026-08-02 concurrent writers corrupted validation_preds/lightgbm.csv.
The append path in run_full_baselines.run_one opens the file in mode="a"
once per origin, so two processes writing the same path interleave their
rows and can emit a second header mid-file. Downstream that is not a clean
failure: pivot() raises on duplicate (origin, hour) pairs, daily_baseload
raises, and the monitor's row count reports inflated progress.

Atomicity does NOT make concurrent writers safe in the sense of preserving
both. It converts an interleaved, unparseable file into a clean
last-writer-wins: every reader sees either the old file or one complete new
one, never a half-written one. That is the guarantee this project needs —
the runs are resumable, so a lost update costs a rerun, while a corrupt file
costs a silent wrong number.

Deliberately no locking library and no new dependency: os.replace is atomic
on the same filesystem on both POSIX and Windows, which is the whole
mechanism. The temporary file MUST be created in the destination directory —
a cross-filesystem rename is not atomic and silently degrades to copy.

Lives in src/ because that is the shared layer scripts already import;
scripts in this repo never import each other.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pandas as pd


def atomic_write_csv(
    frame: pd.DataFrame, path: str | Path, *, append: bool = False, **to_csv_kwargs
) -> None:
    """Write `frame` to `path` so readers never observe a partial file.

    With append=True the existing rows are read and concatenated first, so
    the whole file is rewritten each call. That is O(n) per call rather than
    O(1), but these frames are small (a few MB at most) and the callers are
    dominated by model fitting, so the cost is not measurable next to the
    corruption it removes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_csv_kwargs.setdefault("index", False)

    if append and path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            frame.to_csv(fh, **to_csv_kwargs)
            fh.flush()
            os.fsync(fh.fileno())
        _replace_with_retry(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _replace_with_retry(tmp: str, path: Path, attempts: int = 400) -> None:
    """os.replace, retried past Windows sharing violations.

    POSIX rename always succeeds here. Windows returns ERROR_ACCESS_DENIED
    when the destination is momentarily open — which is exactly the
    concurrent case this module exists for, so failing on it would defeat
    the purpose. The retry does not weaken atomicity: each attempt either
    replaces the whole file or does nothing.
    """
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.01)
