"""Tests for src/atomic_io.py — the 2026-08-02 corruption guard.

The defect these pin: run_full_baselines wrote per-origin rows with
mode="a", so two concurrent runs against the same path interleaved their
rows and could emit a second header mid-file. The file stayed *parseable*
often enough that the damage surfaced later, as duplicate (origin, hour)
pairs, rather than as a crash at write time.

The contract asserted here is deliberately NOT "both writers survive".
Atomic replace gives last-writer-wins: every reader sees one COMPLETE
version, old or new, never a spliced one.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.atomic_io import atomic_write_csv

N_ROWS = 400


def _frame(tag: str) -> pd.DataFrame:
    """A frame big enough that a non-atomic write is interruptible mid-file."""
    return pd.DataFrame(
        {
            "origin": [f"2016-01-{1 + i % 28:02d}" for i in range(N_ROWS)],
            "hour": [i % 24 for i in range(N_ROWS)],
            "y_true": [float(i) for i in range(N_ROWS)],
            "y_pred": [float(i) + 0.5 for i in range(N_ROWS)],
            "model": [tag] * N_ROWS,
        }
    )


def test_atomic_write_round_trips(tmp_path):
    path = tmp_path / "lightgbm.csv"
    frame = _frame("LightGBM")
    atomic_write_csv(frame, path)
    pd.testing.assert_frame_equal(pd.read_csv(path), frame)


def test_append_accumulates_and_leaves_one_header(tmp_path):
    path = tmp_path / "lightgbm.csv"
    atomic_write_csv(_frame("a"), path, append=True)
    atomic_write_csv(_frame("b"), path, append=True)
    out = pd.read_csv(path)
    assert len(out) == 2 * N_ROWS
    # A stray header row would survive read_csv as a literal "origin" value.
    assert "origin" not in set(out["origin"].astype(str))


def test_no_temp_files_survive_a_successful_write(tmp_path):
    path = tmp_path / "lightgbm.csv"
    atomic_write_csv(_frame("LightGBM"), path)
    assert [p.name for p in tmp_path.iterdir()] == ["lightgbm.csv"]


def test_failed_write_leaves_the_original_intact_and_no_debris(tmp_path):
    path = tmp_path / "lightgbm.csv"
    original = _frame("original")
    atomic_write_csv(original, path)

    class Exploding(pd.DataFrame):
        def to_csv(self, *a, **k):  # noqa: D102
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        atomic_write_csv(Exploding(_frame("new")), path)

    pd.testing.assert_frame_equal(pd.read_csv(path), original)
    assert [p.name for p in tmp_path.iterdir()] == ["lightgbm.csv"]


def _big_frame(tag: str, n: int = 20_000) -> pd.DataFrame:
    """Large enough that a non-atomic write is observably slow to complete.

    The corruption is a race, so the file has to take long enough to write
    that a concurrent reader can catch it half-written. At 400 rows a plain
    write finishes between two reads and the race is invisible -- which is
    exactly how the first version of this test passed against the very bug
    it was written to catch.
    """
    return pd.DataFrame(
        {
            "origin": [f"2016-01-{1 + i % 28:02d}" for i in range(n)],
            "hour": [i % 24 for i in range(n)],
            "y_true": [float(i) for i in range(n)],
            "y_pred": [float(i) + 0.5 for i in range(n)],
            "model": [tag] * n,
        }
    )


def test_a_reader_never_observes_a_partial_file(tmp_path):
    """The regression test for 2026-08-02.

    This asserts the guarantee that actually matters: while writers hammer
    one path, every state a reader can observe is a COMPLETE file. Asserting
    only the final state is not enough -- the last writer to finish leaves a
    whole file even without atomicity, so such a test passes against the
    original bug.

    Against a plain open-and-write this fails, either on a short read or on
    EmptyDataError from a freshly truncated file.
    """
    path = tmp_path / "lightgbm.csv"
    n = 20_000
    tags = [f"w{i}" for i in range(4)]
    frames = {t: _big_frame(t, n) for t in tags}
    atomic_write_csv(frames[tags[0]], path)

    stop = threading.Event()
    errors: list[BaseException] = []
    bad: list[str] = []
    reads = [0]

    def write(tag: str) -> None:
        try:
            for _ in range(15):
                atomic_write_csv(frames[tag], path)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    def read() -> None:
        while not stop.is_set():
            try:
                out = pd.read_csv(path)
            except (PermissionError, FileNotFoundError):
                # Windows only, and NOT the defect: while os.replace swaps
                # the directory entry, an opener can transiently be refused.
                # The file is never half-written, it is momentarily
                # unopenable. Retrying is what a real reader would do.
                time.sleep(0.001)
                continue
            except Exception as exc:
                # A parse failure IS the defect: it means bytes were read
                # from a file that was mid-write. EmptyDataError is what a
                # freshly truncated plain write looks like.
                bad.append(f"unreadable mid-write: {exc!r}")
                continue
            reads[0] += 1
            time.sleep(0.001)
            if len(out) != n:
                bad.append(f"partial file: {len(out)} rows, expected {n}")
            elif out["model"].nunique() != 1:
                bad.append(f"spliced writers: {set(out['model'].unique())}")

    reader = threading.Thread(target=read)
    reader.start()
    writers = [threading.Thread(target=write, args=(t,)) for t in tags]
    for t in writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    reader.join()

    assert not errors, f"writer raised: {errors[0]!r}"
    # Without this the test is vacuous: a reader that never got a read in
    # would report no bad observations and pass against anything.
    assert reads[0] > 0, "reader never observed the file at all"
    assert not bad, f"{len(bad)} bad observation(s), first: {bad[0]}"

    out = pd.read_csv(path)
    assert len(out) == n
    pd.testing.assert_frame_equal(out, frames[out["model"].iloc[0]])
