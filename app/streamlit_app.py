"""
Pearls AQI Predictor — Streamlit dashboard.
Run: streamlit run app/streamlit_app.py
"""
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.config import CITIES, DEFAULT_CITY, AQI_LEVELS
from src.utils.alerts import (
    classify_aqi, aqi_color, aqi_emoji,
    health_recommendations, check_forecast_alerts,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 6px 0;
}
.aqi-big {
    font-size: 4rem;
    font-weight: 700;
    line-height: 1;
}
.level-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #111;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Cached data loaders ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_current(city: str) -> dict:
    try:
        from src.feature_pipeline.data_fetcher import DataFetcher
        fetcher = DataFetcher()
        return fetcher.fetch_current(city)
    except Exception as e:
        st.warning(f"Live fetch failed ({e}). Showing demo data.")
        return {
            "city": city, "timestamp": datetime.utcnow(),
            "aqi": 72.0, "pm25": 18.4, "pm10": 34.2,
            "o3": 45.1, "no2": 22.3, "so2": 5.1, "co": 0.4,
            "temperature": 19.0, "humidity": 65.0, "wind_speed": 3.2,
        }


@st.cache_data(ttl=3600)
def load_forecast(city: str, model_name: str = None) -> pd.DataFrame:
    try:
        from src.inference_pipeline.predict import predict_next_72h
        return predict_next_72h(city=city, model_name=model_name or None)
    except Exception as e:
        st.info(f"No trained model yet ({e}). Showing demo forecast.")
        return _demo_forecast(city)


@st.cache_data(ttl=1800)
def load_history(city: str, days: int = 7) -> pd.DataFrame:
    try:
        from src.feature_pipeline.feature_store import load_features
        end   = datetime.utcnow()
        start = end - timedelta(days=days)
        return load_features(city, start=start, end=end)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=7200)
def load_shap(city: str) -> pd.DataFrame:
    try:
        from src.training_pipeline.model_registry import load_model, list_models
        from src.training_pipeline.models import random_forest_model
        from src.feature_pipeline.feature_store import load_features
        from src.feature_pipeline.feature_engineer import get_feature_columns
        from src.utils.shap_explainer import get_shap_summary

        models = list_models(city)
        rf_meta = next((m for m in models if m["model_name"] == "random_forest"), None)
        if not rf_meta:
            return pd.DataFrame()

        model, meta = load_model(city, "random_forest")
        df = load_features(city)
        if df.empty:
            return pd.DataFrame()
        feature_cols = meta["feature_columns"]
        df = df.dropna(subset=feature_cols)
        X  = df[feature_cols].values
        return get_shap_summary(model, X, feature_cols, model_type="tree")
    except Exception as e:
        logger.warning(f"SHAP load failed: {e}")
        return pd.DataFrame()


def _demo_forecast(city: str) -> pd.DataFrame:
    base = datetime.utcnow()
    aqi_vals = 70 + 30 * np.sin(np.linspace(0, 4 * np.pi, 72)) + np.random.normal(0, 5, 72)
    aqi_vals = np.clip(aqi_vals, 20, 200)
    return pd.DataFrame({
        "timestamp":     [base + timedelta(hours=i+1) for i in range(72)],
        "predicted_aqi": aqi_vals,
        "lower_bound":   aqi_vals * 0.9,
        "upper_bound":   aqi_vals * 1.1,
        "model_used":    "demo",
    })


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌬️ AQI Predictor")
    st.caption("Pearls — 3-day air quality forecast")
    st.divider()

    city = st.selectbox(
        "Select city", list(CITIES.keys()),
        index=list(CITIES.keys()).index(DEFAULT_CITY),
        format_func=str.title,
    )

    st.subheader("Model")
    model_choice = st.radio(
        "Forecasting model",
        ["Auto (best RMSE)", "Random Forest", "Ridge Regression", "LSTM"],
        index=0,
    )
    model_map = {
        "Auto (best RMSE)": None,
        "Random Forest":    "random_forest",
        "Ridge Regression": "ridge",
        "LSTM":             "lstm",
    }
    selected_model = model_map[model_choice]

    st.divider()
    history_days = st.slider("History window (days)", 1, 30, 7)
    show_shap = st.checkbox("Show SHAP feature importance", value=True)

    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")


# ── Load data ────────────────────────────────────────────────────────────────

current   = load_current(city)
forecast  = load_forecast(city, selected_model)
history   = load_history(city, history_days)


# ── Header ───────────────────────────────────────────────────────────────────

st.title(f"Air Quality Index — {city.title()}")

current_aqi = current.get("aqi", 0) or 0
level       = classify_aqi(current_aqi)
color       = aqi_color(current_aqi)
emoji       = aqi_emoji(current_aqi)

col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="color:#888;font-size:0.8rem;">Current AQI {emoji}</div>
            <div class="aqi-big" style="color:{color};">{current_aqi:.0f}</div>
            <span class="level-badge" style="background:{color};">{level}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.metric("PM₂.₅", f"{current.get('pm25', 0):.1f} μg/m³")
with col3:
    st.metric("PM₁₀",  f"{current.get('pm10', 0):.1f} μg/m³")
with col4:
    st.metric("O₃",    f"{current.get('o3', 0):.1f} ppb")
with col5:
    st.metric("NO₂",   f"{current.get('no2', 0):.1f} ppb")


# ── Alerts ───────────────────────────────────────────────────────────────────

alerts = check_forecast_alerts(forecast, city)
if alerts:
    worst = max(alerts, key=lambda a: float(a["aqi"]))
    st.error(worst["message"])
elif current_aqi > 100:
    st.warning(f"Moderate air quality expected. AQI: {current_aqi:.0f}")
else:
    st.success("Air quality is currently good.")


# ── 72-hour Forecast chart ────────────────────────────────────────────────────

st.subheader("72-hour AQI Forecast")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=forecast["timestamp"],
    y=forecast["upper_bound"],
    line=dict(width=0),
    showlegend=False,
    name="Upper bound",
    mode="lines",
))
fig.add_trace(go.Scatter(
    x=forecast["timestamp"],
    y=forecast["lower_bound"],
    fill="tonexty",
    fillcolor="rgba(99,110,250,0.15)",
    line=dict(width=0),
    showlegend=True,
    name="Uncertainty band",
    mode="lines",
))
fig.add_trace(go.Scatter(
    x=forecast["timestamp"],
    y=forecast["predicted_aqi"],
    mode="lines+markers",
    line=dict(color="#636efa", width=2),
    marker=dict(size=4),
    name="Predicted AQI",
))

# Threshold bands
for label, (lo, hi) in AQI_LEVELS.items():
    if lo > 0:
        fig.add_hline(
            y=lo, line_dash="dot", line_color="rgba(255,255,255,0.15)",
            annotation_text=label, annotation_position="left",
        )

fig.update_layout(
    xaxis_title="Date / Time (UTC)",
    yaxis_title="AQI",
    template="plotly_dark",
    height=400,
    margin=dict(t=20, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# Day-summary cards
st.subheader("Daily Summary")
day_cols = st.columns(3)
for i, (col, label) in enumerate(zip(day_cols, ["Day 1", "Day 2", "Day 3"])):
    day_data = forecast.iloc[i*24:(i+1)*24]
    if day_data.empty:
        continue
    avg_aqi  = day_data["predicted_aqi"].mean()
    max_aqi  = day_data["predicted_aqi"].max()
    clr      = aqi_color(avg_aqi)
    lv       = classify_aqi(avg_aqi)
    with col:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left:4px solid {clr};">
                <div style="color:#888;font-size:0.75rem;">{label} — avg</div>
                <div style="font-size:1.8rem;font-weight:700;color:{clr};">{avg_aqi:.0f}</div>
                <div style="font-size:0.8rem;color:#aaa;">{lv}</div>
                <div style="font-size:0.75rem;color:#666;">Peak: {max_aqi:.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Historical trend ─────────────────────────────────────────────────────────

if not history.empty and "aqi" in history.columns:
    st.subheader(f"Historical AQI — Last {history_days} Days")
    hist_fig = px.line(
        history, x="timestamp", y="aqi",
        template="plotly_dark", height=300,
        labels={"aqi": "AQI", "timestamp": "Date"},
    )
    hist_fig.update_traces(line_color="#ef553b")
    hist_fig.update_layout(margin=dict(t=10, b=30))
    st.plotly_chart(hist_fig, use_container_width=True)
else:
    st.info("Run the feature pipeline to populate historical data.")


# ── Pollutant breakdown ───────────────────────────────────────────────────────

st.subheader("Current Pollutant Breakdown")
pollutants = {
    "PM₂.₅": current.get("pm25", 0),
    "PM₁₀":  current.get("pm10", 0),
    "O₃":    current.get("o3", 0),
    "NO₂":   current.get("no2", 0),
    "SO₂":   current.get("so2", 0),
    "CO":    current.get("co", 0),
}
poll_df = pd.DataFrame(
    {"Pollutant": list(pollutants.keys()), "Value": list(pollutants.values())}
)
poll_fig = px.bar(
    poll_df, x="Pollutant", y="Value", color="Pollutant",
    template="plotly_dark", height=300,
    labels={"Value": "Concentration"},
)
poll_fig.update_layout(showlegend=False, margin=dict(t=10, b=30))
st.plotly_chart(poll_fig, use_container_width=True)


# ── SHAP feature importance ───────────────────────────────────────────────────

if show_shap:
    st.subheader("Feature Importance (SHAP)")
    shap_df = load_shap(city)
    if not shap_df.empty:
        top10 = shap_df.head(10)
        shap_fig = px.bar(
            top10, x="mean_abs_shap", y="feature",
            orientation="h", template="plotly_dark", height=350,
            labels={"mean_abs_shap": "Mean |SHAP|", "feature": "Feature"},
            color="mean_abs_shap",
            color_continuous_scale="Viridis",
        )
        shap_fig.update_layout(margin=dict(t=10), showlegend=False)
        shap_fig.update_yaxes(autorange="reversed")
        st.plotly_chart(shap_fig, use_container_width=True)
    else:
        st.info("Train models first to see SHAP feature importance.")


# ── Health recommendations ────────────────────────────────────────────────────

st.subheader("Health Recommendations")
recs = health_recommendations(current_aqi)
for r in recs:
    st.markdown(f"• {r}")


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
model_used = forecast["model_used"].iloc[0] if not forecast.empty else "—"
st.caption(
    f"Model: **{model_used}** | "
    f"Data: AQICN + OpenWeather | "
    f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
)
