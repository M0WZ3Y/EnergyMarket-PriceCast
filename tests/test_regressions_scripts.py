"""Wave C/D regression reproductions — script-layer defects.

`scripts/` was almost entirely untested: of 12 scripts only
run_ood_stress.py had any coverage. These reproduce the defects found in
the 2026-08-05 sweep.

Every test here redirects output to tmp_path. Nothing under reports/,
data/ or models/ is ever written — those hold frozen, git-tagged results.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves annotations via
    # sys.modules[cls.__module__], which is None for an unregistered module.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


export_tables = _load_script("export_tables")
task_monitor = _load_script("task_monitor")
run_full_baselines = _load_script("run_full_baselines")
week5 = _load_script("week5_checkpoint")


# ==========================================================================
# scripts/export_tables.py
# ==========================================================================
def test_fmt_cell_does_not_turn_an_rmae_of_one_into_a_bound():
    """_fmt_cell's p-value rules are applied to METRIC columns too.

    The bound-printing rules exist because p-values underflow to exactly
    0.0 for hopelessly-beaten comparisons, and "0.0000" asserts an exact
    zero probability. Sound for p-values -- wrong for metrics. An rMAE of
    exactly 1.000 is the single most load-bearing number in the OOD story
    ("every trained model exceeds rMAE 1.0") and rendered as "$>$0.999",
    a bound rather than a value.
    """
    out = export_tables._fmt_cell(1.0, is_pvalue=False)
    assert out == "1.000", f"metric 1.0 rendered as {out!r}"


def test_fmt_cell_still_bounds_an_underflowed_p_value():
    """Guard: the p-value bound behaviour must survive for p-values."""
    assert export_tables._fmt_cell(0.0, is_pvalue=True) == r"$<$0.001"
    assert export_tables._fmt_cell(1.0, is_pvalue=True) == r"$>$0.999"


def test_fmt_cell_does_not_print_a_signed_zero():
    """The `0.0 <= v < 0.001` guard excludes negatives, so a small negative
    metric printed as "-0.000" -- a signed zero, which is meaningless in a
    thesis table.
    """
    assert export_tables._fmt_cell(-0.0004, is_pvalue=False) != "-0.000"


def test_export_handles_an_ood_table_carrying_exactly_the_metric_columns(tmp_path, monkeypatch):
    """export() dispatches on `df.columns.tolist() == METRICS` and then calls
    _bold_best, which reads a 'target' index level.

    Only build_canonical() produces a (model, target) MultiIndex; build_ood()
    returns a flat 'model' index. An ood_summary.csv lacking the 'MAE ratio'
    column collapses `keep` to exactly the four metric names and the whole
    export dies -- AFTER results_canonical and the DM tables have already
    been overwritten, leaving reports/tables/ half-updated.
    """
    monkeypatch.setattr(export_tables, "OUT_DIR", tmp_path)
    df = pd.DataFrame(
        {"MAE": [1.0, 2.0], "RMSE": [1.5, 2.5], "sMAPE": [10.0, 20.0], "rMAE": [1.0, 1.2]},
        index=pd.Index(["LSTM", "LightGBM"], name="model"),
    )
    assert df.columns.tolist() == export_tables.METRICS  # the crashing shape

    export_tables.export(df, "ood_probe", caption="c", label="l", dry_run=False)

    assert (tmp_path / "ood_probe.tex").exists()
    assert (tmp_path / "ood_probe.csv").exists()


def test_captions_report_the_data_actually_exported(tmp_path, monkeypatch):
    """Captions hardcode statistics the script computes but does not use:
    "728 origins", "$p=0.023$", "173 days", "Mean price 98.67".

    Re-running after any data or threshold change silently emits a table
    whose numbers contradict its own caption -- the worst failure mode for
    a thesis artifact, because the caption is what a reader trusts.

    Reproduced by exporting a 100-origin SUBSET of the real frozen frames
    (read-only) and asserting the caption follows the data.
    """
    src = REPO_ROOT / "data" / "processed" / "baselines"
    if not src.exists():
        pytest.skip("frozen baseline frames not present")

    sub_hourly = tmp_path / "hourly"
    sub_hourly.mkdir()
    keep = None
    for fname in export_tables.HOURLY_FILES.values():
        frame = pd.read_csv(src / fname, parse_dates=["origin"])
        if keep is None:
            keep = sorted(frame["origin"].unique())[:100]
        frame[frame["origin"].isin(keep)].to_csv(sub_hourly / fname, index=False)

    monkeypatch.setattr(export_tables, "HOURLY_DIR", sub_hourly)
    monkeypatch.setattr(export_tables, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(export_tables, "build_ood", lambda: None)

    export_tables.main(dry_run=False)

    caption = (tmp_path / "out" / "results_canonical.tex").read_text(encoding="utf-8")
    assert "728 origins" not in caption, "caption hardcodes the full-run origin count"
    assert "100 origins" in caption


def test_block_bootstrap_caption_constants_are_pinned():
    """The two block-bootstrap p ranges in the dm_regime_split caption come
    from scripts/run_dm_ensembles.py, which export_tables.py never runs, so
    they cannot be interpolated from the frames it holds.

    Pinning them here is what keeps them honest: if run_dm_ensembles.py's
    sensitivity analysis ever changes, this fails loudly instead of letting
    a stale range be printed under a freshly computed table. Both ranges are
    pinned together on purpose -- the stressed-day result and the
    not-robust-over-all-days counterpart must always be reported as a pair.
    """
    assert export_tables.BOOTSTRAP_P_RANGE_STRESSED == (0.006, 0.044)
    assert export_tables.BOOTSTRAP_P_RANGE_ALL == (0.013, 0.057)


# ==========================================================================
# scripts/task_monitor.py
# ==========================================================================
def test_find_run_pids_fails_closed_when_the_process_probe_breaks(monkeypatch):
    """`except Exception: return []` fails OPEN.

    A PowerShell timeout on a loaded machine -- i.e. exactly when a long
    run is active -- returns [], _resume_incomplete reads that as "nothing
    is running", and launches a DUPLICATE process appending to the same
    CSV. That is the precise corruption the script's own PROCEED_RULES
    warns about, and it already happened once on 2026-08-02. Unknown must
    not mean "not running".
    """
    import subprocess

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="powershell", timeout=30)

    monkeypatch.setattr(subprocess, "run", boom)

    with pytest.raises(RuntimeError, match="(?i)(could not|unable|unknown|determine)"):
        task_monitor._find_run_pids()


def test_resume_does_not_launch_when_process_state_is_unknown(monkeypatch, tmp_path):
    """End-to-end consequence: an unknown process state must block the
    relaunch, never permit it."""
    launched = []
    monkeypatch.setattr(task_monitor, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        task_monitor, "_find_run_pids",
        lambda: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: launched.append(a))

    activity = task_monitor.Activity(
        name="walk-forward: naive", done=10, total=728, started=0.0, last_write=0.0
    )
    task_monitor._resume_incomplete([activity])

    assert launched == [], "spawned a duplicate run while process state was unknown"


# ==========================================================================
# scripts/run_full_baselines.py
# ==========================================================================
def test_resume_removes_partial_origin_rows_instead_of_duplicating_them(tmp_path):
    """completed_origins() distrusts an origin with fewer than 24 rows, but
    the append path never REMOVES those rows.

    A crash mid-origin leaves e.g. 10 rows; the origin goes back into todo
    and 24 more are appended, giving 34 rows for that day, permanently.
    Downstream every consumer then breaks or lies: daily_baseload raises,
    pivot() raises on duplicate entries, and the monitor reports inflated
    progress. Only hand-repair fixes the file.
    """
    out = tmp_path / "naive.csv"
    rows = []
    for hour in range(24):  # a complete origin
        rows.append(dict(origin="2016-01-04", hour=hour, y_true=1.0, y_pred=1.0, model="naive"))
    for hour in range(10):  # torn write
        rows.append(dict(origin="2016-01-05", hour=hour, y_true=2.0, y_pred=2.0, model="naive"))
    pd.DataFrame(rows).to_csv(out, index=False)

    run_full_baselines.repair_partial_origins(out)

    after = pd.read_csv(out)
    counts = after.groupby("origin").size()
    assert set(counts.index) == {"2016-01-04"}
    assert counts.loc["2016-01-04"] == 24


# ==========================================================================
# scripts/week5_checkpoint.py
# ==========================================================================
def test_allow_partial_does_not_crash_when_naive_is_the_missing_model(tmp_path, monkeypatch):
    """--allow-partial exists to preview while some model CSVs are missing,
    but the sanity cross-check reads naive.csv unconditionally afterwards --
    a raw FileNotFoundError traceback and a nonzero exit, after the preview
    already succeeded.
    """
    ours = tmp_path / "baselines"
    ours.mkdir()
    rows = [
        dict(origin="2016-01-04", hour=h, y_true=50.0 + h, y_pred=51.0 + h, model="LSTM")
        for h in range(24)
    ]
    pd.DataFrame(rows).to_csv(ours / "lstm.csv", index=False)

    monkeypatch.setattr(week5, "OURS_DIR", ours)

    # Must not raise FileNotFoundError; a clean SystemExit or a normal
    # return are both acceptable outcomes for a partial preview.
    try:
        week5.main(allow_partial=True)
    except SystemExit:
        pass
    except FileNotFoundError as exc:
        pytest.fail(f"--allow-partial still hard-requires naive.csv: {exc}")
