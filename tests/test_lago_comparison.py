"""Tests for scripts/run_lago_comparison.py — the benchmark comparison.

This table backs the central claim of chapter 4, and every number in it
comes from somewhere else: our side from results_canonical.csv, their side
either transcribed from the paper or computed from their shipped forecasts.
Each of those three routes has its own failure mode, and each is pinned
here:

  * transcription drift on our side (a hand-typed number going stale when
    the canonical table is regenerated)
  * silent editing of the published constants, which would let the
    comparison flatter us with nobody noticing
  * misalignment between their forecast index and our test index, which
    would corrupt every p-value while raising nothing
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "run_lago_comparison", REPO_ROOT / "scripts" / "run_lago_comparison.py"
)
lago = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lago)

CANONICAL = pd.read_csv(REPO_ROOT / "reports" / "tables" / "results_canonical.csv")
HOURLY = CANONICAL[CANONICAL["target"] == "hourly"].set_index("model")

# The values given in the paper (Applied Energy 293:116983, Tables 2 and 3,
# EPEX-DE rows). Duplicated here ON PURPOSE: if someone edits the script's
# constants, this test fails rather than the comparison quietly changing.
EXPECTED_PAPER = {
    "DNN 1": (0.407, 3.716, 77.145, 14.970, 6.796),
    "DNN 2": (0.422, 3.850, 137.449, 15.356, 7.304),
    "DNN 3": (0.406, 3.706, 100.214, 15.508, 6.271),
    "DNN 4": (0.394, 3.592, 90.578, 14.680, 6.080),
    "LEAR 56": (0.506, 4.619, 129.763, 17.600, 8.122),
    "LEAR 84": (0.499, 4.555, 133.580, 17.491, 7.923),
    "LEAR 1092": (0.450, 4.108, 128.295, 16.984, 6.996),
    "LEAR 1456": (0.451, 4.118, 124.191, 17.054, 6.987),
    "DNN Ensemble": (0.374, 3.413, 94.434, 14.078, 5.927),
    "LEAR Ensemble": (0.433, 3.955, 122.412, 15.747, 7.079),
}


# --------------------------------------------------------------------------
# published constants
# --------------------------------------------------------------------------
def test_published_constants_are_exactly_the_paper_values():
    assert set(lago.PAPER) == set(EXPECTED_PAPER)
    for model, (rmae_, mae_, mape_, smape_, rmse_) in EXPECTED_PAPER.items():
        got = lago.PAPER[model]
        assert got["rMAE"] == rmae_, model
        assert got["MAE"] == mae_, model
        assert got["MAPE"] == mape_, model
        assert got["sMAPE"] == smape_, model
        assert got["RMSE"] == rmse_, model


# --------------------------------------------------------------------------
# our side: no transcription drift
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def comparison():
    return lago.build_comparison(lago.load_published())


def test_our_rows_match_results_canonical_exactly(comparison):
    ours = comparison[comparison["source"] == "this thesis"].set_index("model")
    assert set(ours.index) == set(lago.OUR_FILES)
    for model in ours.index:
        for metric in ("rMAE", "MAE", "sMAPE", "RMSE"):
            assert ours.loc[model, metric] == pytest.approx(
                float(HOURLY.loc[model, metric]), rel=1e-12
            ), f"{model}/{metric} drifted from results_canonical.csv"


def test_our_rows_never_carry_a_mape(comparison):
    """MAPE is excluded by decision (negative prices). A number appearing in
    that column for our models would contradict section 3-5."""
    ours = comparison[comparison["source"] == "this thesis"]
    assert ours["MAPE"].isna().all()


def test_published_rows_carry_the_mape_caveat(comparison):
    paper = comparison[comparison["source"] == "Lago2021 (paper Tables 2/3)"]
    assert len(paper) == len(EXPECTED_PAPER)
    assert paper["MAPE"].notna().all()
    assert paper["note"].str.contains("MAPE").all()


# --------------------------------------------------------------------------
# their side: the DNN control, and the LEAR discrepancy
# --------------------------------------------------------------------------
def test_shipped_dnn_forecasts_reproduce_the_paper_table(comparison):
    """The control. If the DNN rows did NOT reproduce, the alignment or the
    metric code would be wrong and nothing else in this file would mean
    anything. They do reproduce, which is what makes the LEAR mismatch a
    finding about the paper rather than a bug in this repo."""
    shipped = comparison[
        comparison["source"] == "Lago2021 (shipped forecasts, our metric code)"
    ].set_index("model")
    for model in ["DNN 1", "DNN 2", "DNN 3", "DNN 4", "DNN Ensemble"]:
        assert shipped.loc[model, "MAE"] == pytest.approx(lago.PAPER[model]["MAE"], abs=0.005)
        assert shipped.loc[model, "rMAE"] == pytest.approx(lago.PAPER[model]["rMAE"], abs=0.005)


def test_shipped_lear_forecasts_do_not_reproduce_the_paper_table(comparison):
    """Pins the discrepancy so it cannot quietly disappear.

    Every LEAR variant scores BETTER from the shipped forecasts than the
    paper prints. That matters directly: against the printed LEAR 1092
    (4.108) our LEAR-LASSO looks comfortably ahead; against the shipped one
    (3.930) it is barely ahead. If a future toolbox release makes these
    agree, this test fails and the caveat should be removed deliberately.
    """
    shipped = comparison[
        comparison["source"] == "Lago2021 (shipped forecasts, our metric code)"
    ].set_index("model")
    for model in ["LEAR 56", "LEAR 84", "LEAR 1092", "LEAR 1456", "LEAR Ensemble"]:
        printed = lago.PAPER[model]["MAE"]
        assert shipped.loc[model, "MAE"] < printed - 0.005, (
            f"{model} now matches the paper table; the caveat may be stale"
        )
        assert "does NOT match" in shipped.loc[model, "note"]


def test_disagreeing_paper_rows_are_flagged_in_their_own_note(comparison):
    """The caveat must sit on the row, not only in prose elsewhere."""
    paper = comparison[comparison["source"] == "Lago2021 (paper Tables 2/3)"].set_index("model")
    for model in ["LEAR 56", "LEAR 84", "LEAR 1092", "LEAR 1456", "LEAR Ensemble"]:
        assert "DISAGREE" in paper.loc[model, "note"], model
    for model in ["DNN 1", "DNN 2", "DNN 3", "DNN 4", "DNN Ensemble"]:
        assert "DISAGREE" not in paper.loc[model, "note"], model


def test_ensemble_rows_record_the_structural_difference(comparison):
    """Their ensembles average runs of ONE family; ours average different
    families. A reader comparing the two must be told."""
    ours = comparison[comparison["source"] == "this thesis"].set_index("model")
    for model in ["Ensemble (static)", "Ensemble (regime-aware)"]:
        assert "FAMILIES" in ours.loc[model, "note"]


# --------------------------------------------------------------------------
# alignment: the check that has to pass before any p-value is believable
# --------------------------------------------------------------------------
def test_their_forecast_index_equals_our_test_index():
    published = lago.load_published()
    ours = lago.our_hourly("LEAR-LASSO")
    assert len(published.index) == len(ours.index) == 17472
    assert published.index.equals(ours.index), "index mismatch — every p-value would be wrong"


def test_realized_prices_are_identical_across_both_pipelines():
    published = lago.load_published()
    frame = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "baselines" / "lear_lasso.csv", parse_dates=["origin"]
    )
    ts = pd.DatetimeIndex(frame["origin"] + pd.to_timedelta(frame["hour"], unit="h"))
    y_true = pd.Series(frame["y_true"].values, index=ts).sort_index()
    diff = (y_true - published["Real price"].reindex(y_true.index)).abs().max()
    assert float(diff) == pytest.approx(0.0, abs=1e-6)


def test_matrix_reshape_is_728_by_24_and_row_ordered_by_date():
    published = lago.load_published()
    m = lago.as_matrix(published["Real price"])
    assert m.shape == (728, 24)
    # first row must be the first test day's 24 hours, in hour order
    first_day = published["Real price"].iloc[:24].to_numpy()
    assert np.allclose(m[0], first_day)


def test_alignment_verifier_accepts_the_real_data():
    lago.verify_alignment(lago.load_published())  # must not raise


# --------------------------------------------------------------------------
# DM output shape
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dm():
    return lago.build_dm(lago.load_published())


def test_dm_covers_every_pair_in_both_directions(dm):
    assert len(dm) == len(lago.OUR_DM) * len(lago.THEIR_DM)
    assert {"DM p (ours better)", "DM p (theirs better)"} <= set(dm.columns)


def test_dm_one_sided_pairs_are_complementary(dm):
    """The two directions test the same statistic with opposite signs, so
    their p-values must sum to 1. A pair that does not is a sign error."""
    total = dm["DM p (ours better)"] + dm["DM p (theirs better)"]
    assert np.allclose(total.to_numpy(dtype=float), 1.0, atol=1e-9)


def test_verdict_agrees_with_the_p_values(dm):
    for _, row in dm.iterrows():
        if row["DM p (ours better)"] < 0.05:
            assert row["verdict"] == "ours better (p<0.05)"
        elif row["DM p (theirs better)"] < 0.05:
            assert row["verdict"] == "theirs better (p<0.05)"
        else:
            assert row["verdict"] == "no significant difference"


def test_a_lower_mae_does_not_by_itself_produce_a_verdict(dm):
    """The whole point of this table: our ensembles have a lower MAE than
    their LEAR Ensemble, and that difference is NOT significant. If this
    ever flips to a significance claim, it must be because the data changed,
    not because the test was loosened."""
    row = dm[(dm["ours"] == "Ensemble (regime-aware)") & (dm["theirs"] == "LEAR Ensemble")].iloc[0]
    assert row["MAE ours"] < row["MAE theirs"]
    assert row["verdict"] == "no significant difference"
