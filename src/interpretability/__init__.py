"""Model interpretability (thesis 4-6).

Post-freeze by construction: nothing here refits, reruns or rewrites any
v1.0-results artifact. It explains models, it does not produce forecasts.
"""

from src.interpretability.shap_analysis import (
    FEATURE_GROUPS,
    ShapResult,
    feature_group,
    fit_interpretation_models,
    group_importance,
    interpretation_train_days,
    regime_split,
    shap_values_daily,
    shap_values_hourly,
    split_boundary,
)

__all__ = [
    "FEATURE_GROUPS",
    "ShapResult",
    "feature_group",
    "fit_interpretation_models",
    "group_importance",
    "interpretation_train_days",
    "regime_split",
    "shap_values_daily",
    "shap_values_hourly",
    "split_boundary",
]
