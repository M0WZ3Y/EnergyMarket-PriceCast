"""EnergyMarket-PriceCast — Streamlit UI (thesis 5-3).

    streamlit run app/pricecast.py

A deliberately thin shell. Every rule lives in app/forecast_service.py, which
imports no Streamlit and is unit-tested in tests/test_app.py; this file only
arranges widgets and renders results. If you find yourself writing logic here,
it belongs there instead.

Two things this UI is obliged to do and does not treat as optional:

1. **Attribution.** Energy-Charts data is CC BY 4.0. The attribution string is
   rendered on every view, sourced from configs/data.yaml.
2. **Tell the truth about accuracy.** The served model was frozen on
   2017-12-31, and the v1.1-ood result showed every trained model scoring
   rMAE > 1.0 on live 2026 data — worse than a naive forecast. Presenting
   these forecasts without saying so would misrepresent the research this tool
   demonstrates, so the warning is rendered before any chart, with figures read
   from the frozen OOD summary rather than typed in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.forecast_service import (
    InsufficientHistory,
    attribution,
    fetch_live_window,
    forecast_for_day,
    forecastable_days,
    history_window,
    load_cached_demo,
    load_model,
    ood_context,
    validate_uploaded_frame,
)

st.set_page_config(page_title="EnergyMarket-PriceCast", page_icon="⚡", layout="wide")

LIVE = "Live Energy-Charts API"
DEMO = "Cached demo (offline)"
UPLOAD = "Upload CSV"


@st.cache_resource(show_spinner="Loading the frozen model…")
def _model():
    return load_model()


@st.cache_data(show_spinner="Loading the cached demo window…")
def _demo() -> pd.DataFrame:
    return load_cached_demo()


@st.cache_data(ttl=3600, show_spinner="Fetching from Energy-Charts…")
def _live(start: str, end: str) -> pd.DataFrame:
    return fetch_live_window(start, end)


@st.cache_data
def _ood() -> dict | None:
    return ood_context()


def _accuracy_warning() -> None:
    ctx = _ood()
    if ctx:
        st.warning(
            f"**These forecasts are not accurate on current prices, and that is "
            f"a finding, not a bug.** The model is frozen on 2017-12-31 market "
            f"conditions. Evaluated on live 2026 data it scores rMAE "
            f"{ctx['model_rmae']:.2f} — above 1.0, meaning **worse than a naive "
            f"forecast** ({ctx['naive_rmae']:.2f}). German prices are roughly "
            f"2.8x their training-era level. This regime shift is the subject "
            f"of the thesis limitations chapter; the tool exists to demonstrate "
            f"the pipeline, not to trade on it.",
            icon="⚠️",
        )
    else:
        st.warning(
            "The model is frozen on 2017-12-31 market conditions and is known "
            "to perform worse than a naive forecast on present-day prices.",
            icon="⚠️",
        )


def _chart(result, target_day: pd.Timestamp) -> go.Figure:
    hours = list(range(24))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=result.forecast.to_numpy(),
            name="Forecast",
            mode="lines+markers",
            line=dict(width=3, color="#2b6cb0"),
        )
    )
    if result.actual is not None:
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=result.actual.to_numpy(),
                name="Actual",
                mode="lines+markers",
                line=dict(width=3, color="#c05621", dash="dash"),
            )
        )
    fig.update_layout(
        title=f"Day-ahead hourly prices — {target_day.date()}",
        xaxis_title="Hour of day",
        yaxis_title="Price (EUR/MWh)",
        xaxis=dict(tickmode="linear", dtick=2),
        hovermode="x unified",
        height=460,
        margin=dict(t=60, b=40),
    )
    return fig


def _load_data(source: str):
    """Return (frame, error_message). Exactly one is None."""
    if source == DEMO:
        return _demo(), None

    if source == UPLOAD:
        uploaded = st.sidebar.file_uploader(
            "CSV with columns: timestamp, price, exog_1, exog_2", type="csv"
        )
        if uploaded is None:
            return None, "Upload a CSV to continue."
        try:
            return validate_uploaded_frame(uploaded), None
        except ValueError as exc:
            return None, f"That CSV cannot be used: {exc}"

    # Live
    default_target = pd.Timestamp.now('UTC').normalize().tz_localize(None)
    requested = st.sidebar.date_input(
        "Fetch a window ending", value=default_target.date(),
        help="Energy-Charts publishes day-ahead prices around 13:00 CET.",
    )
    start, end = history_window(pd.Timestamp(requested))
    st.sidebar.caption(f"Requesting {start} to {end}")
    try:
        return _live(start, end), None
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim, see below
        return None, (
            f"The Energy-Charts API did not answer: {exc}\n\n"
            "The loader already retries connection errors, so this is the API, "
            "not your network — it drops TLS intermittently. Switch to "
            f"**{DEMO}** to carry on with the committed 2026 window."
        )


def main() -> None:
    st.title("⚡ EnergyMarket-PriceCast")
    st.caption(
        "Day-ahead hourly electricity price forecasting for the German market "
        "(DE-LU) — the applied deliverable of the MSc thesis."
    )

    source = st.sidebar.radio("Data source", [DEMO, LIVE, UPLOAD], index=0)
    st.sidebar.divider()

    _accuracy_warning()

    frame, error = _load_data(source)
    if error:
        st.info(error)
        st.caption(attribution())
        return

    days = forecastable_days(frame)
    if len(days) == 0:
        st.error(
            "No day in this data can be forecast. Each target day needs itself "
            "and the previous 7 days complete — 24 hours each, with day-ahead "
            "load and renewables present. "
            f"The data covers {frame.index.min().date()} to "
            f"{frame.index.max().date()}."
        )
        st.caption(attribution())
        return

    target = st.sidebar.selectbox(
        "Target day",
        options=list(days),
        index=len(days) - 1,
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d (%a)"),
    )
    st.sidebar.caption(f"{len(days)} forecastable day(s) available")

    try:
        with st.spinner("Forecasting…"):
            result = forecast_for_day(frame, target, _model())
    except InsufficientHistory as exc:
        st.error(str(exc))
        st.caption(attribution())
        return
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.caption(attribution())
        return

    st.plotly_chart(_chart(result, pd.Timestamp(target)), width="stretch")

    left, mid, right = st.columns(3)
    left.metric("Forecast baseload (EUR/MWh)", f"{result.forecast.mean():.2f}")
    left.caption("Mean of the 24 forecast hours")

    if result.actual is None:
        mid.metric("Actual", "not published")
        mid.caption("This day's prices are not out yet — a genuine forecast.")
    else:
        errors = (result.forecast - result.actual).abs()
        mid.metric("Actual baseload (EUR/MWh)", f"{result.actual.mean():.2f}")
        right.metric("MAE (EUR/MWh)", f"{errors.mean():.2f}")
        right.caption(f"Worst hour: {errors.idxmax()} ({errors.max():.2f})")

    with st.expander("The 24 forecast values"):
        table = result.forecast.to_frame("forecast (EUR/MWh)")
        if result.actual is not None:
            table["actual (EUR/MWh)"] = result.actual
            table["error"] = result.forecast - result.actual
        st.dataframe(table, width="stretch")

    st.divider()
    st.caption(attribution())
    st.caption(
        "Model: LightGBM, 24 independent per-hour regressors on the shared "
        "leakage-audited feature set, frozen at the end of the benchmark era "
        "(2017-12-31). Benchmark: Lago et al. (2021) via epftoolbox."
    )


if __name__ == "__main__":
    main()
