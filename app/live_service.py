"""Serving the v2 validation-selected model for tomorrow — app/live_service.py

`ServedModel` exposes the same predict(X) -> DataFrame contract as the model
wrappers, so app/forecast_service.forecast_for_day drives it unchanged,
including that function's placeholder-price guard (the target day's own
price is never read -- tested at -500 and +5000).

What is served is decided by data/processed/live_v2/selection.json (lowest
VALIDATION MAE over rungs R1b..R4; the test window never chose anything).
scripts/refresh_live_model.py fits it on the latest window into models/live/
and writes models/live/served.json; this module only loads and predicts.

History for features comes from the pinned snapshot plus a gitignored live
cache of recent ENTSO-E days (data/live_cache/). Nothing here feeds a
reported number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_MODEL_DIR = REPO_ROOT / "models" / "live"
SERVED_JSON = LIVE_MODEL_DIR / "served.json"  # legacy single-variant location (shadow)
VARIANTS = ("exante", "shadow")


def variant_dir(variant: str, root: Path = LIVE_MODEL_DIR) -> Path:
    """models/live/<variant>/; a legacy models/live/served.json counts as shadow."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    d = Path(root) / variant
    if variant == "shadow" and not (d / "served.json").exists() and (Path(root) / "served.json").exists():
        return Path(root)
    return d
LIVE_CACHE = REPO_ROOT / "data" / "live_cache" / "entsoe_recent.csv"
SNAPSHOT = REPO_ROOT / "data" / "raw" / "live_v2" / "entsoe_de_2019_2026_sdac.csv"  # SDAC price (2026-10-17 c)
STALE_AFTER_DAYS = 7
Y_COLUMNS = [f"y_h{h:02d}" for h in range(24)]
PREV_DAY_PRICE = [f"price_D-1_h{h:02d}" for h in range(24)]


def member_dir(name: str, window: int | None = None) -> str:
    base = name.lower().replace("-", "_")
    return base if window is None else f"{base}_w{window}"


@dataclass
class ServedModel:
    rung: str
    model: str
    members: dict[str, list]  # name -> fitted wrappers (LEAR: one per window)
    kind: str  # "single" | "static" | "regime"
    weights: dict = field(default_factory=dict)
    threshold: float | None = None
    bias_window: int | None = None
    fit_end: str = ""
    fitted_at: str = ""
    variant: str = "shadow"  # "exante": no D+1 wind/solar input (decision 2026-10-15 e)

    def _member_pred(self, name: str, X: pd.DataFrame) -> np.ndarray:
        preds = [m.predict(X).to_numpy(dtype=float) for m in self.members[name]]
        return np.mean(preds, axis=0)  # LEAR window arms: arithmetic mean

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(X) != 1:
            raise ValueError(f"ServedModel.predict expects one target-day row, got {len(X)}")
        if self.variant == "exante":
            from src.evaluation.live_ladder import exante

            X = exante(X)
        if self.kind == "single":
            out = self._member_pred(self.model, X)
        else:
            w = self.weights
            if self.kind == "regime":
                # previous-day information only, as regime_labels() does
                stressed = float(X[PREV_DAY_PRICE].to_numpy().max()) > float(self.threshold)
                w = self.weights["stressed" if stressed else "calm"]
            total = sum(w.values())
            out = sum((w[m] / total) * self._member_pred(m, X) for m in w)
        return pd.DataFrame(np.asarray(out).reshape(1, 24), index=X.index, columns=Y_COLUMNS)


def load_served(path: Path | None = None, variant: str = "shadow") -> ServedModel:
    path = variant_dir(variant) if path is None else Path(path)
    meta_path = Path(path) / "served.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No served model at {meta_path}. Create it with:\n"
            "    ./.venv/Scripts/python.exe scripts/refresh_live_model.py"
        )
    meta = json.loads(meta_path.read_text())
    from src.models import LEARLassoModel, LightGBMModel, LSTMModel, NaiveModel, SARIMAXModel, load_models_config

    cfgs = load_models_config()
    classes = {"naive": (NaiveModel, "naive"), "SARIMAX": (SARIMAXModel, "sarimax"),
               "LEAR-LASSO": (LEARLassoModel, "lear_lasso"), "LightGBM": (LightGBMModel, "lightgbm"),
               "LSTM": (LSTMModel, "lstm")}
    members = {}
    for name, dirs in meta["member_dirs"].items():
        cls, key = classes[name]
        members[name] = [cls(cfgs[key]).load(Path(path) / d) for d in dirs]
    return ServedModel(
        rung=meta["rung"], model=meta["model"], members=members, kind=meta["kind"],
        weights=meta.get("weights", {}), threshold=meta.get("threshold"),
        bias_window=meta.get("bias_window"), fit_end=meta["fit_end"], fitted_at=meta["fitted_at"],
        variant=meta.get("variant", "shadow"),
    )


def staleness_days(served: ServedModel, today: pd.Timestamp | None = None) -> int:
    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today).normalize()
    return int((today - pd.Timestamp(served.fit_end)).days)


def history_frame(recent: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pinned snapshot + live cache (+ `recent`), later sources win on overlap."""
    parts = [pd.read_csv(SNAPSHOT, index_col=0, parse_dates=True)]
    if LIVE_CACHE.exists():
        parts.append(pd.read_csv(LIVE_CACHE, index_col=0, parse_dates=True))
    if recent is not None:
        parts.append(recent)
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[["price", "exog_1", "exog_2"]]


def fetch_recent(days_back: int = 14, through: pd.Timestamp | None = None) -> pd.DataFrame:
    """Live ENTSO-E window ending on `through` (default tomorrow); merged into the cache."""
    from src.data.entsoe import EntsoeLoader

    through = (pd.Timestamp.now().normalize() + pd.Timedelta(days=1)) if through is None else pd.Timestamp(through)
    start = through - pd.Timedelta(days=days_back)
    recent = EntsoeLoader().fetch_exog(start.strftime("%Y-%m-%d"), through.strftime("%Y-%m-%d"))
    LIVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cached = pd.read_csv(LIVE_CACHE, index_col=0, parse_dates=True) if LIVE_CACHE.exists() else None
    merged = recent if cached is None else pd.concat([cached, recent])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    # a NaN from a not-yet-published series must not overwrite a value cached earlier
    if cached is not None:
        merged = merged.combine_first(cached)
    merged.to_csv(LIVE_CACHE)
    return recent


def forecast_day(df: pd.DataFrame, target_day: pd.Timestamp, served: ServedModel):
    """forecast_for_day with the served model, plus R2's causal bias correction if served."""
    from app.forecast_service import ForecastResult, forecast_for_day

    if served.variant == "exante":
        # The ex-ante model never reads D+1 wind/solar (ServedModel.predict drops
        # the column block), but build_features drops a row with any NaN, so an
        # unpublished A69 is filled with a placeholder. tests/test_live_serving.py
        # checks that the placeholder value cannot change the forecast.
        df = df.copy()
        day_rows = df.index.normalize() == pd.Timestamp(target_day).normalize()
        df.loc[day_rows, "exog_2"] = df.loc[day_rows, "exog_2"].fillna(0.0)
    if served.bias_window:
        # leakage review 2026-10-16, F5: re-forecasting past days with a model
        # trained THROUGH them gives in-sample errors, not the out-of-sample
        # correction R2 evaluated. The path was unreachable (refresh never writes
        # bias_window; plan() refuses R1b/R2) and is now refused outright.
        raise ValueError("bias-corrected serving is not supported: its errors would be in-sample")
    return forecast_for_day(df, target_day, served)


LADDER_TABLE = REPO_ROOT / "reports" / "tables" / "live_ladder.csv"


def v2_context(served: ServedModel, path: Path = LADDER_TABLE) -> dict | None:
    """Test-window rMAE of the served (rung, model) and of naive, READ from the v2 table.

    Same rule as forecast_service.ood_context: a statistic shown in the UI is
    read from the committed result, never typed. None when the table is absent.
    """
    if not Path(path).exists():
        return None
    t = pd.read_csv(path)
    row = t[(t["rung"] == served.rung) & (t["model"] == served.model)]
    naive = t[(t["rung"] == "R1b") & (t["model"] == "naive")]
    if row.empty:
        return None
    return {"rmae": float(row["rMAE"].iloc[0]), "mae": float(row["MAE"].iloc[0]),
            "n_days": int(row["n_days"].iloc[0]),
            "naive_mae": float(naive["MAE"].iloc[0]) if len(naive) else float("nan")}


def entsoe_available() -> bool:
    import os

    return bool(os.environ.get("ENTSOE_API_TOKEN")) and any(
        (variant_dir(v) / "served.json").exists() for v in VARIANTS)
