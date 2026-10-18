"""Rung information sets: D = realized inputs (diagnostic), R5 = national load + residual load."""
import numpy as np
import pandas as pd

import scripts.run_live_ladder as RL
from src.evaluation import live_ladder as LL
from src.features.pipeline import build_features


def _frames():
    idx = pd.date_range("2026-01-01", periods=24 * 20, freq="h", name="timestamp")
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"price": 80 + rng.standard_normal(len(idx)), "exog_1": 2e4, "exog_2": 1e4}, index=idx)
    extra = pd.DataFrame({"load_fc_national": 5.5e4, "load_act_amprion": 2.1e4, "load_act_national": 5.6e4,
                          "ws_act": 1.2e4}, index=idx)
    return base, extra


def test_rung_D_uses_realized_inputs(monkeypatch):
    base, extra = _frames()
    monkeypatch.setattr(RL, "load_entsoe", lambda: base.copy())
    monkeypatch.setattr(RL, "load_extra", lambda: extra.copy())
    df, cfg = RL.rung_inputs("D")
    assert (df["exog_1"] == 2.1e4).all() and (df["exog_2"] == 1.2e4).all() and cfg is None


def test_rung_R5_uses_national_load_and_residual_load(monkeypatch):
    base, extra = _frames()
    monkeypatch.setattr(RL, "load_entsoe", lambda: base.copy())
    monkeypatch.setattr(RL, "load_extra", lambda: extra.copy())
    df, cfg = RL.rung_inputs("R5")
    assert (df["exog_1"] == 5.5e4).all()
    X, _ = build_features(df, cfg=cfg)
    assert X["resload_D0_h00"].iloc[-1] == 5.5e4 - 1e4
    assert list(X.columns[-7:]) == [f"dow_{i}" for i in range(7)]  # LEAR ordering contract
    assert not [c for c in LL.exante(X).columns if c.startswith(("exog_2_D0_", "resload_D0_"))]


def test_D_is_never_a_selection_candidate():
    # the lists cmd_score actually passes to val_table (leakage review T1)
    assert "D" not in RL.SHADOW_CANDIDATES and "D" not in RL.EXANTE_CANDIDATES
    src = open(RL.__file__, encoding="utf-8").read()
    assert "val_table(list(SHADOW_CANDIDATES))" in src and "val_table(list(EXANTE_CANDIDATES))" in src


def test_exante_is_unchanged_by_target_day_wind_solar_with_every_block_on():
    # leakage review T2: perturbation, not names. Every physical block on; only the
    # TARGET day's wind/solar changes; the ex-ante features must not move at all.
    import copy

    import numpy as np
    import pandas as pd

    from src.features.pipeline import build_features, load_feature_config

    cfg = copy.deepcopy(load_feature_config())
    cfg["physical_blocks"] = {k: True for k in (cfg.get("physical_blocks") or {})}
    idx = pd.date_range("2022-01-01", "2024-03-31 23:00", freq="h")  # merit/scarcity blocks need ~1 y of history
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"price": rng.normal(80, 20, len(idx)), "exog_1": rng.normal(15000, 2000, len(idx)),
                       "exog_2": rng.normal(20000, 5000, len(idx))}, index=idx)
    target = pd.Timestamp("2024-03-20")
    X0, _ = build_features(df, cfg=cfg)
    hit = df.copy()
    hit.loc[hit.index.normalize() == target, "exog_2"] = -999.0
    X1, _ = build_features(hit, cfg=cfg)
    assert len(X0) > 100 and target in X0.index  # the matrix is real, not empty
    a, b = LL.exante(X0).loc[target], LL.exante(X1).loc[target]
    pd.testing.assert_series_equal(a, b)
    assert (X0.loc[target] != X1.loc[target]).any()  # the full set DOES see it: the test can fail
