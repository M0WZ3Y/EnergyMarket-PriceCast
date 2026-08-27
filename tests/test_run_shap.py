"""Script-layer guards for scripts/run_shap.py.

The 2026-08-05 sweep found that scripts were the least-tested surface in the
repo (1 of 12 had any test) and that several defects there were of the
silent-wrong-answer kind. run_shap.py is written after that finding, so its
guards get tests from the start.

The two that matter most:

* `_assert_writable` is the only thing standing between this script and a
  frozen v1.0-results artifact. The PreToolUse freeze hook intercepts Edit/
  Write TOOL calls, NOT scripts — a script is precisely the actor that can
  still overwrite `reports/tables/results_canonical.tex`.
* `assert_explained_days_unseen` is the on-disk counterpart of the leakage
  guard in tests/test_shap.py: that one checks the windowing function, this
  one checks the artifacts actually loaded from models/interpretation/.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from conftest import require_thesis_data
from src.evaluation.walk_forward import load_evaluation_config

# Never a literal: configs/evaluation.yaml is the single source (T3).
STRESS_THRESHOLD = float(
    load_evaluation_config()["regime"]["stress_threshold_eur_mwh"]
)


def _load_run_shap():
    """Import scripts/run_shap.py by path — scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(
        "run_shap_under_test", REPO_ROOT / "scripts" / "run_shap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_shap = _load_run_shap()


# ---------------------------------------------------------------------------
# The freeze guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frozen",
    [
        "reports/tables/results_canonical.tex",
        "reports/tables/dm_tests.csv",
        "reports/tables/ood_stress.tex",
        "reports/figures/01_price_distribution.png",
        "data/processed/baselines/lightgbm.csv",
        "data/processed/validation_preds/lightgbm.csv",
        "data/processed/ood/ood_summary.csv",
        "models/frozen/lightgbm",
    ],
)
def test_refuses_to_write_any_frozen_artifact(frozen):
    """Every one of these backs a number in the frozen thesis results."""
    with pytest.raises(RuntimeError, match="refusing to write"):
        run_shap._assert_writable(REPO_ROOT / frozen)


def test_refuses_a_figure_number_it_does_not_own():
    """Only figures 10-15 belong to this script. Writing 09 would silently
    replace an EDA figure that is already in the thesis."""
    with pytest.raises(RuntimeError, match="refusing to write"):
        run_shap._assert_writable(REPO_ROOT / "reports" / "figures" / "09_daily_baseload.png")


@pytest.mark.parametrize(
    "allowed",
    [
        "models/interpretation/lightgbm",
        "models/interpretation/metadata.json",
        "data/processed/shap/group_by_hour.csv",
        "data/processed/shap/facts.json",
        "reports/figures/10_shap_global_importance_hourly.png",
        "reports/figures/15_shap_waterfall_case_study.png",
        "reports/tables/shap_importance.tex",
        "reports/tables/shap_importance.csv",
    ],
)
def test_permits_its_own_new_artifacts(allowed):
    assert run_shap._assert_writable(REPO_ROOT / allowed)


def test_every_path_the_figure_code_actually_writes_is_declared(monkeypatch, tmp_path):
    """Asserting that WRITABLE_FILES passes _assert_writable would be
    circular — the guard's final clause IS that list. The real drift risk is
    between the declared list and the path literals inside figure_*/
    export_table, so those are what get recorded and checked here.
    """
    written: list[Path] = []

    monkeypatch.setattr(run_shap, "_savefig", lambda fig, path: written.append(Path(path)))
    monkeypatch.setattr(run_shap, "_write_csv", lambda frame, path: written.append(Path(path)))
    monkeypatch.setattr(
        pd.DataFrame, "to_csv", lambda self, *a, **k: written.append(Path(a[0])) if a else None
    )
    monkeypatch.setattr(
        pd.DataFrame, "to_latex", lambda self, *a, **k: written.append(Path(a[0])) if a else None
    )

    facts = {
        "n_test_days": 728, "test_start": "2016-01-04", "test_end": "2017-12-31",
        "n_calm": 651, "n_stressed": 77, "threshold": STRESS_THRESHOLD,
        "train_start": "2013-01-07", "train_end": "2016-01-03", "n_train_days": 1092,
        "beeswarm_hour": 18, "waterfall_day": "2017-01-24",
        "waterfall_regime": "stressed", "waterfall_baseload": 101.9,
        "waterfall_prediction": 56.5, "top_decile_bias": -12.3, "n_top_decile": 73,
        "daily_expected_value": 34.1,
    }
    groups = list(run_shap.FEATURE_GROUPS)
    splits = pd.DataFrame(
        {c: np.arange(len(groups), dtype=float) + 1
         for c in ("hourly_all", "hourly_calm", "hourly_stressed",
                   "daily_all", "daily_calm", "daily_stressed")},
        index=groups,
    )
    by_hour = pd.DataFrame(
        np.ones((len(groups), 24)), index=groups, columns=list(range(24))
    )
    beeswarm = pd.DataFrame(np.zeros((6, 3)), columns=["price_D-1_h18", "exog_2_D0_h17", "dow_0"])
    waterfall = pd.DataFrame(
        {"shap_value": [1.0, -1.0], "feature_value": [10.0, 20.0]},
        index=["price_D-1_h18", "exog_2_D0_h17"],
    )
    monkeypatch.setattr(
        run_shap, "load_cache",
        lambda: (by_hour, splits, beeswarm, beeswarm.copy(), waterfall, facts),
    )

    run_shap.make_figures()

    assert written, "make_figures wrote nothing — the recording stubs missed the call sites"
    declared = {p.resolve() for p in run_shap.WRITABLE_FILES}
    undeclared = [p for p in written if p.resolve() not in declared]
    assert not undeclared, f"figure/table code writes undeclared paths: {undeclared}"


def test_escaping_the_namespace_with_a_traversal_is_refused(tmp_path):
    escape = REPO_ROOT / "data" / "processed" / "shap" / ".." / "baselines" / "lightgbm.csv"
    with pytest.raises(RuntimeError, match="refusing to write"):
        run_shap._assert_writable(escape)


# ---------------------------------------------------------------------------
# The on-disk leakage guard
# ---------------------------------------------------------------------------


def test_refuses_to_explain_days_the_model_was_trained_on():
    with pytest.raises(RuntimeError, match="strictly unseen"):
        run_shap.assert_explained_days_unseen(
            pd.Timestamp("2017-12-31"), pd.Timestamp("2016-01-04")
        )


def test_refuses_when_training_ends_exactly_on_the_first_explained_day():
    """The boundary case: training through the first explained day is a
    one-day leak, and `>=` rather than `>` is what catches it."""
    with pytest.raises(RuntimeError, match="strictly unseen"):
        run_shap.assert_explained_days_unseen(
            pd.Timestamp("2016-01-04"), pd.Timestamp("2016-01-04")
        )


def test_accepts_a_training_window_that_ends_the_day_before():
    run_shap.assert_explained_days_unseen(
        pd.Timestamp("2016-01-03"), pd.Timestamp("2016-01-04")
    )


def test_overlapping_windows_are_refused_as_a_set_not_just_by_endpoint():
    """Comparing only train_end against the first explained day is sufficient
    only while both windows are contiguous. A gapped index could interleave
    them while leaving the endpoints looking fine, so the sets are compared.
    """
    train = pd.DatetimeIndex(["2015-12-30", "2015-12-31", "2016-01-05"])
    explained = pd.date_range("2016-01-04", periods=5, freq="D")

    # The endpoint check alone passes here — the sets still overlap.
    run_shap.assert_explained_days_unseen(pd.Timestamp("2015-12-31"), explained.min())
    with pytest.raises(RuntimeError, match="BOTH"):
        run_shap.assert_windows_disjoint(train, explained)


def test_disjoint_windows_are_accepted():
    train = pd.date_range("2013-01-07", "2016-01-03", freq="D")
    explained = pd.date_range("2016-01-04", "2017-12-31", freq="D")
    run_shap.assert_windows_disjoint(train, explained)


# ---------------------------------------------------------------------------
# Actionable failures instead of bare tracebacks
# ---------------------------------------------------------------------------


def test_missing_interpretation_models_name_the_command_that_creates_them(monkeypatch, tmp_path):
    monkeypatch.setattr(run_shap, "INTERP_DIR", tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match=r"--fit"):
        run_shap.load_interpretation_models()


def test_missing_cache_names_the_command_that_creates_it(monkeypatch, tmp_path):
    monkeypatch.setattr(run_shap, "CACHE_DIR", tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="figures-only"):
        run_shap.load_cache()


# ---------------------------------------------------------------------------
# The real artifacts, when they are present
# ---------------------------------------------------------------------------


def _facts() -> dict:
    """Missing SHAP cache FAILS by default, per the tests/conftest.py policy.

    A bare `pytest.skip` here would reintroduce exactly the clean-checkout
    hole commit b246d25 closed: delete data/processed/shap/ and the guards
    below would report green while checking nothing. The cache is committed
    (.gitignore negation), so absence is a real problem, not a normal state.
    """
    facts_path = run_shap.CACHE_DIR / "facts.json"
    require_thesis_data(facts_path, "SHAP cache (run scripts/run_shap.py)")
    return json.loads(facts_path.read_text())


def test_the_committed_run_explained_only_unseen_days():
    facts = _facts()
    assert pd.Timestamp(facts["train_end"]) < pd.Timestamp(facts["test_start"])


def test_the_committed_run_used_the_ensemble_chapters_regime_split():
    """651/77 is the same split the DM regime tests report. If SHAP ever
    disagrees, sections 3-8 and 4-6 are describing different day sets."""
    facts = _facts()
    assert facts["n_calm"] + facts["n_stressed"] == facts["n_test_days"]
    assert facts["n_stressed"] == 77
    assert facts["threshold"] == pytest.approx(STRESS_THRESHOLD)


def test_the_exported_table_matches_the_cached_values():
    """The .tex is the thesis artifact; the .csv is what a reader inspects.
    They must not drift, which is how the ood_stress caption error happened."""
    csv_path = REPO_ROOT / "reports" / "tables" / "shap_importance.csv"
    cache_path = run_shap.CACHE_DIR / "group_importance_splits.csv"
    if not (csv_path.exists() and cache_path.exists()):
        pytest.skip("SHAP artifacts not built in this checkout")

    exported = pd.read_csv(csv_path, index_col=0)
    cached = pd.read_csv(cache_path, index_col=0)

    assert list(exported.columns) == [
        "Hourly calm",
        "Hourly stressed",
        "Daily calm",
        "Daily stressed",
    ]
    for exported_col, cached_col in zip(
        exported.columns, ["hourly_calm", "hourly_stressed", "daily_calm", "daily_stressed"]
    ):
        pd.testing.assert_series_equal(
            exported[exported_col].round(4),
            cached.loc[exported.index, cached_col].round(4),
            check_names=False,
        )
