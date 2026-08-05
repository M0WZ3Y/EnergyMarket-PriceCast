"""PriceCast forecast logic — no Streamlit, no I/O beyond loading a model.

Kept deliberately free of Streamlit imports so every rule below is unit
testable without a browser or a Streamlit runtime; app/pricecast.py is a thin
UI shell over these functions.

The one subtlety worth reading before changing anything here
--------------------------------------------------------------
`build_features()` drops any day whose own 24 prices contain a NaN, because a
partial day must never become a silently-imputed feature row. That is correct
for training and evaluation — and it means the pipeline cannot, as written,
build features for the day you actually want to forecast, whose prices do not
exist yet.

`forecast_for_day` therefore substitutes a placeholder into the target day's
`price` column so the row survives. This is legitimate for exactly one reason:
no feature column ever reads the target day's own price (see the module
docstring of src/features/pipeline.py — the target day contributes only its
`exog_*_D0` day-ahead forecasts, which are known before the origin). It is not
taken on trust: tests/test_app.py forecasts the same day with placeholders of
-500 and +5000 and requires identical output, so if any future feature ever
did read the target day's price, that test fails rather than the app quietly
forecasting from a constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.features.pipeline import build_features, load_feature_config
from src.models import load_models_config
from src.models.lgbm import LightGBMModel

SCHEMA = ["price", "exog_1", "exog_2"]
HOURS = [f"h{h:02d}" for h in range(24)]

# Longest price lag (7 days) plus the target day itself. Requesting less
# produces an empty feature frame and an error far from its cause.
MIN_HISTORY_DAYS = max(load_feature_config()["price_lag_days"]) + 1

# Extra days requested from the API beyond the strict minimum. The live feed
# routinely returns a ragged tail (fetch_exog inner-joins four endpoints), so
# asking for exactly the minimum frequently yields one day too few.
HISTORY_MARGIN_DAYS = 3

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "frozen" / "lightgbm"


class InsufficientHistory(ValueError):
    """Raised when the supplied data cannot produce features for the target day."""


@dataclass(frozen=True)
class ForecastResult:
    target_day: pd.Timestamp
    forecast: pd.Series  # indexed h00..h23, EUR/MWh
    actual: pd.Series | None  # None when the day is not yet published


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> LightGBMModel:
    """Load the frozen hourly LightGBM.

    models/ is gitignored, so on a fresh clone this file legitimately does not
    exist. The error says how to create it rather than surfacing a bare
    FileNotFoundError from deep inside pickle.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. The frozen models are not committed "
            "(models/ is gitignored). Recreate them with:\n"
            "    ./.venv/Scripts/python.exe scripts/run_ood_stress.py --fit"
        )
    return LightGBMModel(load_models_config()["lightgbm"]).load(path)


def history_window(
    target_day: pd.Timestamp, margin_days: int = HISTORY_MARGIN_DAYS
) -> tuple[str, str]:
    """The (start, end) date range to request so `target_day` is forecastable.

    Ends ON the target day, not the day before: the target day's own
    `exog_*_D0` columns are required features. They are day-ahead load and
    renewables forecasts, published before the forecast origin, so requesting
    them is not a leak.
    """
    target_day = pd.Timestamp(target_day).normalize()
    start = target_day - pd.Timedelta(days=MIN_HISTORY_DAYS + margin_days)
    return start.strftime("%Y-%m-%d"), target_day.strftime("%Y-%m-%d")


def _day_mask(index: pd.DatetimeIndex, day: pd.Timestamp) -> np.ndarray:
    return np.asarray(index.normalize() == day)


def forecast_for_day(
    df: pd.DataFrame,
    target_day: pd.Timestamp,
    model: LightGBMModel,
    price_placeholder: float = 0.0,
) -> ForecastResult:
    """Forecast the 24 hourly prices of `target_day` from `df`.

    `df` is the shared hourly schema (price, exog_1, exog_2). The target day's
    prices may be absent — that is the live case — but its exogenous columns
    must be present.
    """
    target_day = pd.Timestamp(target_day).normalize()
    if df.index.tz is not None and target_day.tz is None:
        target_day = target_day.tz_localize(df.index.tz)
    elif df.index.tz is None and target_day.tz is not None:
        target_day = target_day.tz_localize(None)

    day_rows = _day_mask(df.index, target_day)
    if day_rows.sum() != 24:
        raise InsufficientHistory(
            f"{target_day.date()} has {int(day_rows.sum())} of 24 hours in the "
            "supplied data. A partial day cannot be forecast: its day-ahead "
            "load and renewables columns are required features."
        )

    published = df.loc[day_rows, "price"]
    actual_is_known = bool(published.notna().all())

    prepared = df.copy()
    if not actual_is_known:
        prepared.loc[day_rows, "price"] = price_placeholder

    X, Y = build_features(prepared)
    if target_day not in X.index:
        available = int(df.index.normalize().nunique())
        raise InsufficientHistory(
            f"Not enough history to forecast {target_day.date()}: "
            f"{MIN_HISTORY_DAYS} complete days ending on the target day are "
            f"required, and the supplied data covers {available} day(s) "
            f"({df.index.min().date()} to {df.index.max().date()}). A gap or a "
            "short day anywhere in that window is enough to drop the target."
        )

    predicted = model.predict(X.loc[[target_day]]).iloc[0]
    forecast = pd.Series(
        predicted.to_numpy(dtype=float), index=HOURS, name="forecast"
    )

    actual = None
    if actual_is_known:
        actual = pd.Series(
            Y.loc[target_day].to_numpy(dtype=float), index=HOURS, name="actual"
        )

    return ForecastResult(target_day=target_day, forecast=forecast, actual=actual)


def validate_uploaded_frame(handle) -> pd.DataFrame:
    """Parse and validate a user-supplied CSV into the shared schema.

    Every rejection here has a matching failure deeper in the pipeline; the
    point is to fail at the door with a message naming what is wrong, rather
    than surfacing a pivot error from build_features to someone who uploaded a
    spreadsheet.
    """
    try:
        df = pd.read_csv(handle, index_col=0, parse_dates=True)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the user
        raise ValueError(f"Could not read the CSV: {exc}") from exc

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "The first column must be a timestamp (e.g. 'timestamp'), parseable "
            f"as a date. Got index dtype {df.index.dtype}."
        )

    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. The expected "
            f"schema is: timestamp, {', '.join(SCHEMA)} — the same schema "
            "BenchmarkLoader and EnergyChartsLoader produce."
        )

    df = df[SCHEMA].sort_index()

    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].unique()
        raise ValueError(
            f"{len(dupes)} duplicate timestamp(s), first {dupes[0]}. Pivoting "
            "would silently discard one real price per duplicate."
        )

    for col in SCHEMA:
        if not pd.api.types.is_numeric_dtype(df[col]):
            coerced = pd.to_numeric(df[col], errors="coerce")
            bad = df[col][coerced.isna() & df[col].notna()]
            raise ValueError(
                f"Column '{col}' must be numeric; {len(bad)} value(s) are not "
                f"(first: {bad.iloc[0]!r} at {bad.index[0]})."
            )

    if len(df) > 1:
        spacing = df.index.to_series().diff().dropna()
        if not (spacing == pd.Timedelta(hours=1)).all():
            common = spacing.mode()
            raise ValueError(
                "The index must be hourly with no gaps; the most common spacing "
                f"is {common.iloc[0] if len(common) else 'irregular'}. Resample "
                "to hourly before uploading."
            )

    return df
