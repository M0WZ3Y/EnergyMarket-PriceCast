"""Tests for scripts/export_seed_ensemble.py — the supplementary table.

This table exists to stop a real result from living only as prose in a
commit message. That makes its failure modes specific, and each is pinned
here:

  * a headline number drifting away from what logs/decisions.md 2026-08-07
    and the thesis text claim, silently
  * the claim-discipline pair breaking apart — the regime-aware p-value
    crossing 0.05 while the static one does not is the whole finding, and
    reporting either half alone would turn a hedged result into an
    unqualified one
  * the table quietly becoming a headline result: it must stay outside
    v1.0-results, and the frozen numbers it sits beside must not move
  * the averaged output file being folded back in as a seed member (the
    defect found on 2026-08-07), which would over-weight it
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
    "export_seed_ensemble", REPO_ROOT / "scripts" / "export_seed_ensemble.py"
)
seedtab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seedtab)

TABLE = REPO_ROOT / "reports" / "tables" / "seed_ensemble.csv"

# Pinned from logs/decisions.md 2026-08-07 and the run that produced the
# table. Duplicated here ON PURPOSE: if the script or its inputs change,
# this fails loudly rather than the thesis quietly citing a stale number.
EXPECTED_MAE = {
    "LSTM seed 42 (frozen)": 3.8734,
    "LSTM 4-seed ensemble": 3.6460,
    "Ensemble (static), frozen LSTM": 3.5742,
    "Ensemble (static), seed-ensembled LSTM": 3.5260,
    "Ensemble (regime-aware), frozen LSTM": 3.5569,
    "Ensemble (regime-aware), seed-ensembled LSTM": 3.4994,
    "Lago et al. DNN Ensemble (reference)": 3.4135,
}
EXPECTED_RMAE = {
    "Ensemble (regime-aware), frozen LSTM": 0.3897,
    "Ensemble (regime-aware), seed-ensembled LSTM": 0.3834,
    "Lago et al. DNN Ensemble (reference)": 0.3740,
}
EXPECTED_P_VS_DNN_ENSEMBLE = {
    "Ensemble (static), frozen LSTM": 0.0082,
    "Ensemble (static), seed-ensembled LSTM": 0.0460,
    "Ensemble (regime-aware), frozen LSTM": 0.0127,
    "Ensemble (regime-aware), seed-ensembled LSTM": 0.0803,
}

# The oracle upper bound quoted in the caption. Not recomputed here — it
# comes from the test-fitted (i.e. cheating) weighting reported on
# 2026-08-07 — but pinned so the caption cannot drift away from it.
ORACLE_WITH_SEED_MEMBER = 3.5019
THEIR_DNN_ENSEMBLE = 3.4135


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    if not TABLE.exists():
        pytest.skip("seed_ensemble.csv not exported yet")
    return pd.read_csv(TABLE).set_index("model")


def test_every_pinned_mae_still_holds(table):
    for model, expected in EXPECTED_MAE.items():
        assert model in table.index, f"row vanished from the table: {model}"
        assert table.loc[model, "MAE"] == pytest.approx(expected, abs=5e-4), model


def test_every_pinned_rmae_still_holds(table):
    for model, expected in EXPECTED_RMAE.items():
        assert table.loc[model, "rMAE"] == pytest.approx(expected, abs=5e-4), model


def test_pinned_dm_pvalues_against_their_dnn_ensemble(table):
    col = "DM p vs DNN Ensemble"
    for model, expected in EXPECTED_P_VS_DNN_ENSEMBLE.items():
        assert table.loc[model, col] == pytest.approx(expected, abs=5e-4), model


def test_the_claim_discipline_pair_still_straddles_005(table):
    """The finding IS the straddle; if both land the same side, the caption lies.

    CLAUDE.md's claim discipline requires the regime-aware and static
    verdicts to be reported together. That rule only has force while they
    actually disagree about 0.05.
    """
    col = "DM p vs DNN Ensemble"
    regime = table.loc["Ensemble (regime-aware), seed-ensembled LSTM", col]
    static = table.loc["Ensemble (static), seed-ensembled LSTM", col]
    assert regime > 0.05, "regime-aware no longer above 0.05 — rewrite the caption"
    assert static < 0.05, "static no longer below 0.05 — rewrite the caption"


def test_the_gap_to_their_dnn_ensemble_is_not_closed(table):
    """Every one of our rows stays behind their DNN Ensemble.

    The caption states the gap was not closed. If this ever fails, the
    caption is wrong and the claim has to be rewritten, not quietly kept.
    """
    ours = table.drop(index="Lago et al. DNN Ensemble (reference)")
    assert ours["MAE"].min() > THEIR_DNN_ENSEMBLE
    assert ORACLE_WITH_SEED_MEMBER > THEIR_DNN_ENSEMBLE


def test_seed_ensemble_beats_every_member_it_averages(table):
    members = [f"LSTM seed {s}" for s in (43, 44, 45)] + ["LSTM seed 42 (frozen)"]
    best_member = table.loc[members, "MAE"].min()
    assert table.loc["LSTM 4-seed ensemble", "MAE"] < best_member


def test_member_rows_carry_no_dm_pvalues(table):
    """Only the ensemble rows are DM-tested; members must show '--', not a number.

    A p-value appearing on a member row would imply a comparison that was
    never run.
    """
    dm_cols = [c for c in table.columns if c.startswith("DM p vs ")]
    members = table.loc[[i for i in table.index if i.startswith("LSTM ")], dm_cols]
    assert members.isna().all().all()


def test_the_averaged_output_is_not_folded_back_in_as_a_member(tmp_path):
    """Regression guard for the 2026-08-07 glob defect.

    A bare lstm_s*.csv also matches lstm_seed_ensemble.csv — the average the
    module itself writes into the same directory — silently over-weighting
    it. It cancelled exactly when found, so no number moved; a stale file
    from a different seed count would not have been so kind.
    """
    for name in ("lstm_s42.csv", "lstm_s43.csv", "lstm_seed_ensemble.csv"):
        pd.DataFrame(
            {
                "origin": pd.to_datetime(["2016-01-04"] * 24),
                "hour": range(24),
                "y_true": np.zeros(24),
                "y_pred": np.ones(24),
                "model": "LSTM",
            }
        ).to_csv(tmp_path / name, index=False)

    _, n_members = seedtab.seed_ensemble_frame(tmp_path)
    assert n_members == 2, "the averaged output file was counted as a seed member"


def test_the_table_stays_outside_the_freeze(table):
    """The supplementary result must not have become the headline.

    results_canonical.csv is the frozen table chapter 4 quotes. Its LSTM and
    ensemble rows must still hold the seed-42 numbers, not the seed-ensembled
    ones — if they ever match, the freeze was broken.
    """
    canonical = pd.read_csv(REPO_ROOT / "reports" / "tables" / "results_canonical.csv")
    hourly = canonical[canonical["target"] == "hourly"].set_index("model")
    assert hourly.loc["LSTM", "MAE"] == pytest.approx(3.8734, abs=5e-4)
    assert hourly.loc["Ensemble (regime-aware)", "MAE"] == pytest.approx(3.5569, abs=5e-4)


def test_the_caption_reports_both_halves_of_the_straddle():
    """The caption must never quote one side of the 0.05 straddle alone."""
    caption = seedtab.CAPTION
    assert "FAILURE TO REJECT" in caption
    assert "no longer significantly worse" in caption
    assert "while the static ensemble" in caption
    assert "SUPPLEMENTARY" in caption
    assert "frozen seed-42 LSTM" in caption
    assert str(ORACLE_WITH_SEED_MEMBER) in caption
