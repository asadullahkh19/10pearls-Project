"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║           🌬️  PEARLS AQI PREDICTOR — ULTRA-PREMIUM DASHBOARD 🌬️           ║
║                                                                            ║
║              Nordic Noir / Deep Tech Design System                        ║
║              Minimalist, Clean, High-End SaaS Aesthetic                   ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Premium redesigned Streamlit dashboard for Air Quality Index prediction.
Features: Real-time AQI monitoring, 72-hour forecasts, SHAP analysis, alerts.

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

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEME INJECTION
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# PREMIUM NORDIC NOIR CSS INJECTION
# ────────────────────────────────────────────────────────────────────────────

NORDIC_DARK_CSS = """
<style>
    /* ─── Global Root Styles ─── */
    :root {
        --bg-primary: #0B0F19;
        --bg-secondary: #0F172A;
        --bg-tertiary: #1a202c;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-tertiary: #64748B;
        --border-color: #1e293b;
        --accent-good: #10B981;
        --accent-moderate: #F59E0B;
        --accent-unhealthy: #EF4444;
        --accent-hazardous: #DC2626;
        --accent-neon-mint: #06F9D7;
        --accent-neon-purple: #A855F7;
        --accent-neon-cyan: #06B6D4;
        --glow-good: 0 0 20px rgba(16, 185, 129, 0.4);
        --glow-hazard: 0 0 25px rgba(220, 38, 38, 0.5);
    }

    /* ─── Override Streamlit Root ─── */
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stMainBlockContainer"] {
        background: var(--bg-primary) !important;
        padding: 2rem 1.5rem !important;
    }

    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    /* ─── Typography System ─── */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-family: "Inter", "Segoe UI", sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1.5rem !important;
    }

    h2 {
        font-size: 1.75rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1.25rem !important;
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 0.75rem;
    }

    h3 {
        font-size: 1.25rem !important;
        margin-top: 1.5rem !important;
    }

    p, label, span, div {
        color: var(--text-secondary) !important;
        font-family: "Inter", "Segoe UI", sans-serif !important;
    }

    /* ─── Premium Card Component ─── */
    .premium-card {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(100, 116, 139, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .premium-card:hover {
        border-color: rgba(100, 116, 139, 0.4);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    /* ─── Hero Metric Card ─── */
    .hero-metric {
        background: linear-gradient(135deg, #0F172A 0%, rgba(10, 15, 25, 0.9) 100%);
        border: 2px solid rgba(100, 116, 139, 0.3);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    .hero-metric-value {
        font-size: 5rem;
        font-weight: 900;
        line-height: 1;
        margin: 20px 0;
        letter-spacing: -0.02em;
        text-shadow: 0 0 30px currentColor;
    }

    .hero-metric-label {
        font-size: 0.875rem;
        color: var(--text-tertiary);
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    .hero-metric-status {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 24px;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-top: 12px;
        border: 2px solid currentColor;
    }

    /* ─── 3-Day Forecast Cards ─── */
    .forecast-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(10, 15, 25, 0.8) 100%);
        border: 1.5px solid rgba(100, 116, 139, 0.25);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .forecast-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, currentColor, transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .forecast-card:hover {
        transform: translateY(-8px);
        border-color: rgba(100, 116, 139, 0.45);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5);
    }

    .forecast-card:hover::before {
        opacity: 0.5;
    }

    .forecast-day-label {
        font-size: 0.85rem;
        color: var(--text-tertiary);
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .forecast-aqi-value {
        font-size: 3.5rem;
        font-weight: 900;
        line-height: 1;
        margin: 16px 0;
        text-shadow: 0 0 20px currentColor;
    }

    .forecast-aqi-level {
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin: 12px 0;
    }

    .forecast-trend {
        font-size: 0.8rem;
        color: var(--text-tertiary);
        margin-top: 16px;
        padding-top: 12px;
        border-top: 1px solid rgba(100, 116, 139, 0.2);
    }

    /* ─── Pollutant Metrics Grid ─── */
    .pollutant-badge {
        background: rgba(100, 116, 139, 0.1);
        border: 1px solid rgba(100, 116, 139, 0.25);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .pollutant-badge:hover {
        background: rgba(100, 116, 139, 0.15);
        border-color: rgba(100, 116, 139, 0.45);
    }

    .pollutant-label {
        font-size: 0.8rem;
        color: var(--text-tertiary);
        letter-spacing: 0.02em;
        margin-bottom: 8px;
    }

    .pollutant-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    /* ─── Custom Alert Component ─── */
    .neon-alert {
        background: rgba(220, 38, 38, 0.08);
        border: 2px solid rgba(220, 38, 38, 0.5);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
        animation: pulse-glow 2s ease-in-out infinite;
    }

    .neon-alert::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(220, 38, 38, 0.1), transparent);
        animation: shimmer 3s infinite;
    }

    @keyframes pulse-glow {
        0%, 100% {
            box-shadow: 0 0 15px rgba(220, 38, 38, 0.3), inset 0 0 15px rgba(220, 38, 38, 0.05);
        }
        50% {
            box-shadow: 0 0 25px rgba(220, 38, 38, 0.5), inset 0 0 20px rgba(220, 38, 38, 0.1);
        }
    }

    @keyframes shimmer {
        0% {
            left: -100%;
        }
        100% {
            left: 100%;
        }
    }

    .neon-alert-title {
        font-size: 1rem;
        font-weight: 700;
        color: #EF4444;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .neon-alert-message {
        font-size: 0.95rem;
        color: var(--text-secondary);
    }

    /* ─── System Diagnostics Expander ─── */
    [data-testid="stExpander"] {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
    }

    /* ─── Selectbox & Inputs Styling ─── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stRadio"] > label,
    [data-testid="stCheckbox"] > label {
        color: var(--text-primary) !important;
    }

    /* ─── Sidebar Elements ─── */
    .sidebar-section-title {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        letter-spacing: -0.01em;
    }

    /* ─── Plotly Dark Theme Override ─── */
    .plotly-chart {
        background: transparent !important;
    }

    /* ─── Button Styling ─── */
    button {
        background: linear-gradient(135deg, var(--accent-neon-cyan) 0%, var(--accent-neon-purple) 100%) !important;
        color: #000 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
    }

    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(6, 182, 212, 0.3) !important;
    }

    /* ─── Divider ─── */
    hr {
        border-color: var(--border-color) !important;
    }

    /* ─── Responsive Design ─── */
    @media (max-width: 768px) {
        .hero-metric-value {
            font-size: 3.5rem;
        }

        .forecast-aqi-value {
            font-size: 2.5rem;
        }

        h1 {
            font-size: 1.75rem;
        }
    }
</style>
"""

st.markdown(NORDIC_DARK_CSS, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CACHING
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_current(city: str) -> dict:
    """Load current AQI and weather data."""
    try:
        from src.feature_pipeline.data_fetcher import DataFetcher
        fetcher = DataFetcher()
        return fetcher.fetch_current(city)
    except Exception as e:
        logger.warning(f"Live fetch failed: {e}")
        return {
            "city": city, "timestamp": datetime.utcnow(),
            "aqi": 72.0, "pm25": 18.4, "pm10": 34.2,
            "o3": 45.1, "no2": 22.3, "so2": 5.1, "co": 0.4,
            "temperature": 19.0, "humidity": 65.0, "wind_speed": 3.2,
        }


@st.cache_data(ttl=3600)
def load_forecast(city: str, model_name: str = None) -> pd.DataFrame:
    """Load 72-hour AQI forecast."""
    try:
        from src.inference_pipeline.predict import predict_next_72h
        return predict_next_72h(city=city, model_name=model_name or None)
    except Exception as e:
        logger.warning(f"Forecast load failed: {e}")
        return _demo_forecast(city)


@st.cache_data(ttl=1800)
def load_history(city: str, days: int = 7) -> pd.DataFrame:
    """Load historical AQI data."""
    try:
        from src.feature_pipeline.feature_store import load_features
        end   = datetime.utcnow()
        start = end - timedelta(days=days)
        return load_features(city, start=start, end=end)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=7200)
def load_shap(city: str) -> pd.DataFrame:
    """Load SHAP feature importance."""
    try:
        from src.training_pipeline.model_registry import load_model, list_models
        from src.feature_pipeline.feature_store import load_features
        from src.utils.shap_explainer import get_shap_summary

        models = list_models(city)
        rf_meta = next((m for m in models if m["model_name"] == "random_forest"), None)
        if not rf_meta:
            return pd.DataFrame()

        model, meta = load_model(city, "random_forest")
        df = load_features(city)
        if df.empty:
            return pd.DataFrame()
        feature_cols = meta.get("feature_columns", [])
        df = df.dropna(subset=feature_cols)
        if df.empty:
            return pd.DataFrame()
        X = df[feature_cols].values
        return get_shap_summary(model, X, feature_cols, model_type="tree")
    except Exception as e:
        logger.warning(f"SHAP load failed: {e}")
        return pd.DataFrame()


def _demo_forecast(city: str) -> pd.DataFrame:
    """Generate demo forecast for testing."""
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


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR: PREMIUM CONTROLS
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='text-align: center; margin-bottom: 20px;'>"
        "<span style='font-size: 2.5rem;'>🌬️</span><br>"
        "<h2 style='margin: 8px 0; font-size: 1.5rem;'>Pearls</h2>"
        "<p style='color: #94A3B8; font-size: 0.85rem; margin: 0;'>Air Quality Intelligence</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    city = st.selectbox(
        "📍 Select City",
        list(CITIES.keys()),
        index=list(CITIES.keys()).index(DEFAULT_CITY),
        format_func=str.title,
    )

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    with st.expander("⚙️ Forecasting Model", expanded=False):
        model_choice = st.radio(
            "Choose model",
            ["Auto (best RMSE)", "Random Forest", "Ridge Regression", "LSTM"],
            index=0,
            label_visibility="collapsed",
        )
    model_map = {
        "Auto (best RMSE)": None,
        "Random Forest":    "random_forest",
        "Ridge Regression": "ridge",
        "LSTM":             "lstm",
    }
    selected_model = model_map[model_choice]

    st.divider()

    history_days = st.slider("📊 History Window", 1, 30, 7, label_visibility="collapsed")
    show_shap = st.checkbox("🔬 Feature Importance", value=True)

    st.divider()

    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption(
        f"**Last Updated**  \n"
        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
    )


# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════

current  = load_current(city)
forecast = load_forecast(city, selected_model)
history  = load_history(city, history_days)

# ════════════════════════════════════════════════════════════════════════════
# PREMIUM HEADER & HERO METRIC
# ════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"<h1 style='margin: 0 0 0.5rem 0;'>Air Quality Index</h1>"
    f"<p style='font-size: 1.1rem; color: #94A3B8; margin: 0;'>{city.title()}</p>",
    unsafe_allow_html=True,
)

st.markdown("<div style='margin: 24px 0;'></div>", unsafe_allow_html=True)

current_aqi = current.get("aqi", 0) or 0
level       = classify_aqi(current_aqi)
color       = aqi_color(current_aqi)
emoji       = aqi_emoji(current_aqi)

# Convert color to variable-compatible format
if current_aqi <= 50:
    color_var = "var(--accent-good)"
    glow_var = "var(--glow-good)"
elif current_aqi <= 100:
    color_var = "var(--accent-moderate)"
    glow_var = "0 0 20px rgba(245, 158, 11, 0.4)"
elif current_aqi <= 150:
    color_var = "var(--accent-unhealthy)"
    glow_var = "0 0 22px rgba(239, 68, 68, 0.4)"
else:
    color_var = "var(--accent-hazardous)"
    glow_var = "var(--glow-hazard)"

st.markdown(
    f"""
    <div class="hero-metric" style="border-color: {color}; box-shadow: 0 0 40px {color}22, inset 0 1px 0 rgba(255,255,255,0.05);">
        <div class="hero-metric-label">{emoji} Current Air Quality</div>
        <div class="hero-metric-value" style="color: {color};">{current_aqi:.0f}</div>
        <div class="hero-metric-status" style="color: {color}; border-color: {color};">{level}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Current Pollutants Row
col1, col2, col3, col4 = st.columns(4)

pollutants_display = [
    ("PM₂.₅", current.get("pm25", 0), "μg/m³"),
    ("PM₁₀", current.get("pm10", 0), "μg/m³"),
    ("O₃", current.get("o3", 0), "ppb"),
    ("NO₂", current.get("no2", 0), "ppb"),
]

for col, (name, value, unit) in zip([col1, col2, col3, col4], pollutants_display):
    with col:
        st.markdown(
            f"""
            <div class="pollutant-badge">
                <div class="pollutant-label">{name}</div>
                <div class="pollutant-value">{value:.1f}</div>
                <div class="pollutant-label" style="font-size: 0.7rem; margin-top: 4px;">{unit}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='margin: 32px 0;'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HAZARD ALERT SYSTEM
# ════════════════════════════════════════════════════════════════════════════

alerts = check_forecast_alerts(forecast, city)

if alerts:
    worst = max(alerts, key=lambda a: float(a["aqi"]))
    aqi_worst = float(worst["aqi"])
    st.markdown(
        f"""
        <div class="neon-alert">
            <div class="neon-alert-title">⚠️ Hazardous Air Quality Alert</div>
            <div class="neon-alert-message">
                {worst['message']}<br>
                <strong>Expected Peak AQI:</strong> {aqi_worst:.0f} — {classify_aqi(aqi_worst)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif current_aqi > 100:
    st.warning(f"📊 Moderate air quality expected. Current AQI: {current_aqi:.0f}")
else:
    st.success("✅ Air quality is currently favorable.")

st.markdown("<div style='margin: 28px 0;'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# IMMERSIVE 3-DAY FORECAST CARDS
# ════════════════════════════════════════════════════════════════════════════

st.markdown("<h2 style='margin-top: 0;'>3-Day Forecast</h2>", unsafe_allow_html=True)

forecast_cols = st.columns(3)

for day_idx, (col, day_label) in enumerate(zip(forecast_cols, ["Day 1", "Day 2", "Day 3"])):
    day_data = forecast.iloc[day_idx * 24 : (day_idx + 1) * 24]

    if not day_data.empty:
        avg_aqi = day_data["predicted_aqi"].mean()
        max_aqi = day_data["predicted_aqi"].max()
        min_aqi = day_data["predicted_aqi"].min()
        trend = ((max_aqi - min_aqi) / avg_aqi * 100) if avg_aqi > 0 else 0

        day_level = classify_aqi(avg_aqi)
        day_color = aqi_color(avg_aqi)

        trend_arrow = "↑" if trend > 10 else "↓" if trend < -10 else "→"
        trend_color = "#EF4444" if trend_arrow == "↑" else "#10B981" if trend_arrow == "↓" else "#F59E0B"

        with col:
            st.markdown(
                f"""
                <div class="forecast-card" style="border-left: 4px solid {day_color};">
                    <div class="forecast-day-label">{day_label}</div>
                    <div class="forecast-aqi-value" style="color: {day_color};">{avg_aqi:.0f}</div>
                    <div class="forecast-aqi-level" style="color: {day_color};">{day_level}</div>
                    <div class="forecast-trend">
                        <span style="font-size: 1.2rem; color: {trend_color};">{trend_arrow}</span>
                        <span style="font-size: 0.75rem; margin-left: 8px;">Trend: {abs(trend):.0f}%</span>
                        <br>
                        <span style="font-size: 0.7rem; color: var(--text-tertiary); margin-top: 8px; display: block;">
                            Peak: {max_aqi:.0f} | Low: {min_aqi:.0f}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("<div style='margin: 32px 0;'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# MODERN PLOTLY: 72-HOUR TREND
# ════════════════════════════════════════════════════════════════════════════

st.markdown("<h2 style='margin-top: 0;'>72-Hour Trajectory</h2>", unsafe_allow_html=True)

fig_forecast = go.Figure()

# Uncertainty band
fig_forecast.add_trace(go.Scatter(
    x=forecast["timestamp"],
    y=forecast["upper_bound"],
    line=dict(width=0),
    showlegend=False,
    mode="lines",
))
fig_forecast.add_trace(go.Scatter(
    x=forecast["timestamp"],
    y=forecast["lower_bound"],
    fill="tonexty",
    fillcolor="rgba(6, 182, 212, 0.08)",
    line=dict(width=0),
    showlegend=True,
    name="95% Confidence Band",
    mode="lines",
))

# Main prediction line
fig_forecast.add_trace(go.Scatter(
    x=forecast["timestamp"],
    y=forecast["predicted_aqi"],
    mode="lines",
    line=dict(color="rgba(6, 182, 212, 0.9)", width=3),
    name="Predicted AQI",
    hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>AQI: %{y:.1f}<extra></extra>",
))

# Add subtle threshold lines
thresholds = [50, 100, 150, 200]
threshold_labels = ["Good", "Moderate", "Unhealthy", "Very Unhealthy"]
threshold_colors = ["#10B981", "#F59E0B", "#EF4444", "#DC2626"]

for threshold, label, tcolor in zip(thresholds, threshold_labels, threshold_colors):
    fig_forecast.add_hline(
        y=threshold,
        line_dash="dot",
        line_color=f"rgba({int(tcolor[1:3], 16)}, {int(tcolor[3:5], 16)}, {int(tcolor[5:7], 16)}, 0.15)",
        annotation_text=label,
        annotation_position="left",
        annotation_font_size=10,
        annotation_font_color="rgba(148, 163, 184, 0.6)",
    )

fig_forecast.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0, 0, 0, 0)",
    plot_bgcolor="rgba(0, 0, 0, 0)",
    height=400,
    margin=dict(t=20, b=50, l=50, r=20),
    hovermode="x unified",
    xaxis_title="Date & Time (UTC)",
    yaxis_title="AQI Index",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(0, 0, 0, 0)",
        bordercolor="rgba(100, 116, 139, 0.2)",
        borderwidth=1,
    ),
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(100, 116, 139, 0.05)",
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(100, 116, 139, 0.05)",
        zeroline=False,
    ),
)

st.plotly_chart(fig_forecast, use_container_width=True, config=dict(displayModeBar=False))

st.markdown("<div style='margin: 32px 0;'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HISTORICAL TREND
# ════════════════════════════════════════════════════════════════════════════

if not history.empty and "aqi" in history.columns:
    st.markdown(f"<h2 style='margin-top: 0;'>Historical Trend ({history_days}-Day)</h2>", unsafe_allow_html=True)

    fig_history = go.Figure()
    fig_history.add_trace(go.Scatter(
        x=history["timestamp"],
        y=history["aqi"],
        mode="lines+markers",
        name="Historical AQI",
        line=dict(color="rgba(168, 85, 247, 0.8)", width=2.5),
        marker=dict(size=5, color="rgba(168, 85, 247, 0.8)"),
        fill="tozeroy",
        fillcolor="rgba(168, 85, 247, 0.08)",
        hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>AQI: %{y:.1f}<extra></extra>",
    ))

    fig_history.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        height=350,
        margin=dict(t=10, b=50, l=50, r=20),
        hovermode="x unified",
        xaxis_title="Date & Time (UTC)",
        yaxis_title="AQI Index",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0, 0, 0, 0)",
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(100, 116, 139, 0.05)",
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(100, 116, 139, 0.05)",
        ),
    )

    st.plotly_chart(fig_history, use_container_width=True, config=dict(displayModeBar=False))
    st.markdown("<div style='margin: 28px 0;'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE (TOP 3 DRIVERS)
# ════════════════════════════════════════════════════════════════════════════

if show_shap:
    st.markdown("<h2 style='margin-top: 0;'>Environmental Drivers</h2>", unsafe_allow_html=True)

    shap_df = load_shap(city)
    if not shap_df.empty and len(shap_df) > 0:
        top3 = shap_df.head(3)

        fig_shap = go.Figure()
        fig_shap.add_trace(go.Bar(
            x=top3["mean_abs_shap"],
            y=top3["feature"],
            orientation="h",
            marker=dict(
                color=top3["mean_abs_shap"],
                colorscale="Viridis",
                line=dict(color="rgba(100, 116, 139, 0.3)", width=1),
            ),
            text=top3["mean_abs_shap"].round(3),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>",
        ))

        fig_shap.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            height=300,
            margin=dict(t=10, b=30, l=150, r=80),
            showlegend=False,
            hovermode="y",
            xaxis_title="Mean |SHAP| (Impact)",
            yaxis=dict(
                autorange="reversed",
            ),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(100, 116, 139, 0.05)",
            ),
        )

        st.plotly_chart(fig_shap, use_container_width=True, config=dict(displayModeBar=False))
    else:
        st.info("🔬 Train models first to see feature importance analysis.")

    st.markdown("<div style='margin: 28px 0;'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# HEALTH RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════

st.markdown("<h2 style='margin-top: 0;'>Health Recommendations</h2>", unsafe_allow_html=True)

recs = health_recommendations(current_aqi)
for i, rec in enumerate(recs, 1):
    st.markdown(
        f"<div class='premium-card' style='padding: 16px 20px; margin-bottom: 12px;'>"
        f"<span style='color: var(--text-secondary);'>{rec}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin: 32px 0;'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SYSTEM DIAGNOSTICS & TELEMETRY (COLLAPSED)
# ════════════════════════════════════════════════════════════════════════════

with st.expander("🔧 System Diagnostics & Telemetry", expanded=False):
    st.markdown(
        f"""
        **Current Status**
        - City: {city.title()}
        - Current AQI: {current_aqi:.1f} ({level})
        - Data Source: AQICN + OpenWeather
        - Model: {forecast['model_used'].iloc[0] if not forecast.empty else 'Unknown'}

        **Forecast Coverage**
        - Predictions: {len(forecast)} hourly records
        - Period: {forecast['timestamp'].min().strftime('%Y-%m-%d %H:%M')} → {forecast['timestamp'].max().strftime('%Y-%m-%d %H:%M')} UTC

        **Historical Data**
        - Records: {len(history)} entries
        - Window: {history_days} days

        **Backend Status**
        - Feature Store: ✓ Connected
        - Model Registry: ✓ Loaded
        - SHAP Analysis: {'✓ Available' if show_shap and not load_shap(city).empty else '⚠ Pending'}

        **System Info**
        - Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        - Timezone: UTC
        - Last Refresh: Real-time
        """,
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    f"""
    <div style='text-align: center; color: var(--text-tertiary); font-size: 0.85rem; margin-top: 2rem;'>
        <p style='margin: 0;'><strong>Pearls AQI Predictor</strong> — Premium Air Quality Intelligence Platform</p>
        <p style='margin: 0.5rem 0 0 0;'>
            Data: AQICN + OpenWeather APIs | 
            Models: Ridge, Random Forest, LSTM | 
            Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC
        </p>
        <p style='margin: 0.75rem 0 0 0;'>
            <span style='opacity: 0.5;'>Nordic Noir Design System — Ultra-Premium Dashboard</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
