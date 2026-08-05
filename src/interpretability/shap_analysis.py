"""SHAP attribution for the hourly and daily LightGBM arms — thesis 4-6.

Why this module fits a model of its own
---------------------------------------
`models/frozen/` holds models fit on the trailing 1092 days ending
2017-12-31 (see models/frozen/metadata.json). That window *contains* the
whole 2016-01-04..2017-12-31 test period, so SHAP values computed over test
days against those models would be in-sample — indefensible in a chapter
whose entire point is explaining out-of-sample behaviour, and a direct
conflict with the project's leakage rule.

So section 4-6 explains a separate, interpretation-only fit whose training
window ends strictly before the test boundary and is the same length
(`calibration_window_days`, 1092) the walk-forward used at its first origin.
Every explained day is then genuinely unseen. `interpretation_train_days`
is where that guarantee lives, and tests/test_shap.py asserts it.

Why LightGBM specifically
-------------------------
Both explained arms are gradient-boosted trees, so `shap.TreeExplainer`
gives *exact* TreeSHAP — no sampling, no background dataset, no seed
sensitivity. A thesis figure that changed between runs would be
indefensible; this one cannot. The exactness is itself testable, via the
additivity identity sum(shap) + expected_value == prediction.

Nothing here writes to disk. Artifact writing lives in scripts/run_shap.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.evaluation.ensemble import regime_labels
from src.features.pipeline import daily_target
from src.models.daily import DailyLightGBMModel
from src.models.lgbm import LightGBMModel

# Display order for the collapsed feature families. `feature_group` derives a
# group from a column name generically, so a change to configs/features.yaml
# still produces correct groups; this tuple only fixes the ORDER in which the
# known families are presented (any family not listed is appended, never
# dropped -- see group_importance).
FEATURE_GROUPS = (
    "price_D-1",
    "price_D-2",
    "price_D-3",
    "price_D-7",
    "exog_1_D-1",
    "exog_1_D-7",
    "exog_1_D0",
    "exog_2_D-1",
    "exog_2_D-7",
    "exog_2_D0",
    "dow",
)

# price_D-1_h00 / exog_2_D0_h13 -> the part before the hour suffix.
_LAG_COLUMN = re.compile(r"^((?:price|exog_\d+)_D(?:0|-\d+))_h(?:[01]\d|2[0-3])$")
_DOW_COLUMN = re.compile(r"^dow_[0-6]$")


@dataclass(frozen=True)
class ShapResult:
    """SHAP attributions for one estimator over one block of days.

    `columns` is carried explicitly and is not decoration: every model
    wrapper fits on `X.values`, so the boosters hold no feature names at
    all. Without this the figures would be labelled by position and would
    go silently wrong the first time column order changed.
    """

    values: np.ndarray  # (n_days, n_features)
    expected_value: float
    columns: list[str]
    index: pd.Index

    def mean_abs(self) -> pd.Series:
        """Mean |SHAP| per feature — the standard global-importance reduction."""
        return pd.Series(np.abs(self.values).mean(axis=0), index=self.columns)


def split_boundary(test_df: pd.DataFrame) -> pd.Timestamp:
    """First test target_day, taken from the loader's own split.

    Never hardcode this: `configs/data.yaml`'s `years_test` decides it, and a
    hardcoded 2016-01-04 would keep "passing" while silently explaining the
    wrong days if that config moved.
    """
    if test_df.empty:
        raise ValueError("split_boundary: test frame is empty")
    return test_df.index.min().normalize()


def interpretation_train_days(
    index: pd.Index, boundary: pd.Timestamp, calibration_window_days: int
) -> pd.DatetimeIndex:
    """The trailing training window for the interpretation fit.

    Strictly `< boundary`: the boundary day is the FIRST test day, so an
    inclusive comparison here would train on a day the model is later asked
    to explain -- a one-day leak, which the project rule forbids just as much
    as a one-year one.

    Refuses to silently shorten the window. A short window still fits, still
    predicts, and still produces a plausible-looking SHAP figure, while
    describing a model that is not the one the walk-forward used; there is no
    visible symptom, so it has to be an error.
    """
    index = pd.DatetimeIndex(index)
    eligible = index[index < boundary]
    if len(eligible) < calibration_window_days:
        raise ValueError(
            f"interpretation_train_days: only {len(eligible)} day(s) available "
            f"before {boundary.date()}, need {calibration_window_days} -- "
            "refusing to fit the interpretation model on a shorter window than "
            "the walk-forward used"
        )
    out = eligible.sort_values()[-calibration_window_days:]

    # Row count alone does not pin the window: if build_features() dropped a
    # day inside it, the trailing N rows would span more calendar days and
    # reach further back than the walk-forward's window did. That direction is
    # conservative (never forward, so never a leak), but it would silently
    # describe a different model than the one the results chapter reports.
    span = (out.max() - out.min()).days + 1
    if span != calibration_window_days:
        raise ValueError(
            f"interpretation_train_days: {calibration_window_days} rows span "
            f"{span} calendar days -- the feature index has gaps inside the "
            "training window, so this is not the contiguous window the "
            "walk-forward used"
        )
    return out


def fit_interpretation_models(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    train_days: pd.Index,
    models_cfg: dict | None = None,
) -> tuple[LightGBMModel, DailyLightGBMModel]:
    """Fit the hourly (24 per-hour) and direct-daily LightGBM arms.

    Reuses the production wrappers unmodified, so the explained models are
    the same code path that produced the frozen results -- only the training
    window differs. The daily arm is fitted on `daily_target(Y)`, the same
    reduction the RQ4 comparison uses, so the hourly-vs-daily SHAP comparison
    contrasts two models trained on commensurable targets.
    """
    if models_cfg is None:
        from src.models import load_models_config

        models_cfg = load_models_config()

    train_days = pd.DatetimeIndex(train_days)
    missing = train_days.difference(X.index)
    if len(missing):
        raise ValueError(
            f"fit_interpretation_models: {len(missing)} training day(s) are not "
            f"in the feature index (first: {missing[0].date()})"
        )

    X_train, Y_train = X.loc[train_days], Y.loc[train_days]

    hourly = LightGBMModel(models_cfg["lightgbm"])
    hourly.fit(X_train, Y_train)

    daily = DailyLightGBMModel(models_cfg["daily_lightgbm"])
    daily.fit(X_train, daily_target(Y_train))

    return hourly, daily


def _assert_columns_match(model, X: pd.DataFrame) -> None:
    expected = getattr(model, "_feature_columns", None)
    if expected is not None and list(X.columns) != list(expected):
        raise ValueError(
            "shap_analysis: X column order does not match the columns the model "
            "was fitted on -- reordered columns still produce attributions, and "
            "they are attributed to the wrong features"
        )


def _explain(estimator, X: pd.DataFrame) -> ShapResult:
    import shap

    # No background dataset => tree_path_dependent perturbation: exact,
    # deterministic, and additive against the model's own output.
    explainer = shap.TreeExplainer(estimator)
    values = np.asarray(explainer.shap_values(X.values))
    expected = float(np.ravel(explainer.expected_value)[0])
    return ShapResult(
        values=values,
        expected_value=expected,
        columns=list(X.columns),
        index=X.index,
    )


def shap_values_hourly(model: LightGBMModel, X: pd.DataFrame) -> dict[int, ShapResult]:
    """One ShapResult per hour-of-day (0..23) of the hourly LightGBM arm."""
    _assert_columns_match(model, X)
    if not model._models:
        raise ValueError("shap_values_hourly: model is not fitted")
    return {h: _explain(est, X) for h, est in sorted(model._models.items())}


def shap_values_daily(model: DailyLightGBMModel, X: pd.DataFrame) -> ShapResult:
    """ShapResult for the direct daily-baseload arm."""
    _assert_columns_match(model, X)
    if model._model is None:
        raise ValueError("shap_values_daily: model is not fitted")
    return _explain(model._model, X)


def feature_group(column: str) -> str:
    """Collapse a raw feature column into its family.

    247 columns is unreadable in a bar chart; 11 families is the unit a
    reader can reason about. Unknown columns RAISE rather than falling into
    an 'other' bucket -- a silent bucket would understate whichever family
    the column really belongs to, in a published figure.
    """
    if _DOW_COLUMN.match(column):
        return "dow"
    match = _LAG_COLUMN.match(column)
    if match:
        return match.group(1)
    raise ValueError(
        f"feature_group: unrecognised feature column {column!r}. Add it to a "
        "family explicitly rather than letting it be bucketed silently."
    )


def group_importance(values: np.ndarray, columns: Sequence[str]) -> pd.Series:
    """Mean |SHAP| per feature, summed within each family.

    Grouping redistributes attribution and must never create or destroy it,
    so the returned total equals the ungrouped total by construction.
    """
    values = np.asarray(values)
    columns = list(columns)
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError(
            f"group_importance: values has {values.shape} but {len(columns)} "
            "column name(s) were given -- shape and columns must agree"
        )

    per_feature = pd.Series(np.abs(values).mean(axis=0), index=columns)
    grouped = per_feature.groupby([feature_group(c) for c in columns]).sum()

    # Known families first, in display order; anything else appended rather
    # than dropped, so a features.yaml change cannot silently lose a family.
    known = [g for g in FEATURE_GROUPS if g in grouped.index]
    extra = sorted(g for g in grouped.index if g not in FEATURE_GROUPS)
    return grouped.reindex(known + extra)


def regime_split(Y: pd.DataFrame, threshold: float, default: str = "calm") -> pd.Series:
    """Label every target_day 'calm' or 'stressed'.

    Delegates to `evaluation.ensemble.regime_labels` -- deliberately, not
    incidentally. That function is what chapter 3-8's regime-aware ensemble
    uses, and it labels day D from the PREVIOUS day's realized prices only.
    Reimplementing the rule here would risk 4-6's "stressed days" quietly
    meaning something different from 3-8's, which would make the two chapters
    non-comparable without anything ever failing.
    """
    expected = [f"y_h{h:02d}" for h in range(24)]
    if list(Y.columns) != expected:
        raise ValueError(
            "regime_split: Y must have build_features()'s 24 target columns "
            f"in order (got {len(Y.columns)})"
        )

    long = Y.rename(columns=lambda c: int(c[3:])).stack().rename("y_true").reset_index()
    long.columns = ["origin", "hour", "y_true"]

    labels = regime_labels(long, threshold, default=default)
    out = pd.Series(labels, name="regime").reindex(Y.index)
    out.index.name = Y.index.name
    return out
