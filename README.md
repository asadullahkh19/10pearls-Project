# Pearls AQI Predictor

End-to-end machine learning pipeline for Air Quality Index forecasting. Predicts AQI for the next **3 days (72 hours)** using a fully automated feature pipeline, multiple ML models, and an interactive web dashboard.

---

## Architecture

```
AQICN API ──┐
             ├──▶ Feature Pipeline ──▶ Feature Store (SQLite / Hopsworks)
OpenWeather ─┘                                    │
                                                   ▼
                                        Training Pipeline
                                   (Ridge / Random Forest / LSTM)
                                                   │
                                                   ▼
                                        Model Registry (local / Hopsworks)
                                                   │
                                                   ▼
                                        Inference Pipeline
                                      (72-hour AQI forecast)
                                                   │
                              ┌────────────────────┘
                              ▼                    ▼
                      Streamlit Dashboard     Flask REST API
```

---

## Setup

### 1. Clone and install

```bash
cd "C:\Users\Admin\Desktop\asad_project\AQI_Predictor"
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
copy .env.example .env
# Edit .env and add your API keys:
#   AQICN_API_TOKEN  — free at https://aqicn.org/data-platform/token/
#   OPENWEATHER_API_KEY — free at https://openweathermap.org/api
```

### 3. Backfill historical data (run once)

```bash
python scripts/backfill.py --city london --days 90
```

### 4. Train models

```bash
python scripts/run_training_pipeline.py --city london --days 90
```

### 5. Launch dashboard

```bash
# In one terminal — Flask API
python app/flask_api.py

# In another terminal — Streamlit
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
AQI_Predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml      # Runs hourly (GitHub Actions)
│   └── training_pipeline.yml     # Runs daily  (GitHub Actions)
├── app/
│   ├── streamlit_app.py          # Interactive dashboard
│   └── flask_api.py              # REST API backend
├── data/
│   ├── raw/                      # Raw API responses
│   └── processed/                # SQLite feature store + CSV backups
├── models/registry/              # Saved model files + metadata JSON
├── notebooks/EDA.ipynb           # Exploratory Data Analysis
├── scripts/
│   ├── backfill.py               # Historical data backfill
│   ├── run_feature_pipeline.py   # Hourly pipeline runner
│   └── run_training_pipeline.py  # Daily training runner
├── src/
│   ├── config.py                 # Central configuration
│   ├── feature_pipeline/
│   │   ├── data_fetcher.py       # AQICN + OpenWeather API clients
│   │   ├── feature_engineer.py   # Feature computation
│   │   └── feature_store.py      # Local SQLite + Hopsworks storage
│   ├── training_pipeline/
│   │   ├── models/
│   │   │   ├── random_forest_model.py
│   │   │   ├── ridge_model.py
│   │   │   └── lstm_model.py
│   │   ├── train.py              # Training orchestration
│   │   ├── evaluate.py           # RMSE / MAE / R² metrics
│   │   └── model_registry.py     # Model save/load
│   ├── inference_pipeline/
│   │   └── predict.py            # 72-hour forecast
│   └── utils/
│       ├── alerts.py             # AQI alert system
│       └── shap_explainer.py     # SHAP feature importance
└── tests/
    ├── test_feature_pipeline.py
    └── test_inference.py
```

---

## API Endpoints (Flask)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Service health check |
| GET | `/api/cities` | List supported cities |
| GET | `/api/current?city=london` | Current AQI reading |
| GET | `/api/predict?city=london&model=auto` | 72-hour forecast |
| GET | `/api/history?city=london&days=7` | Historical features |
| GET | `/api/alerts?city=london` | Active AQI alerts |
| GET | `/api/models?city=london` | List trained models |

---

## Models

| Model | Type | Notes |
|-------|------|-------|
| Ridge Regression | Linear baseline | Fast, interpretable |
| Random Forest | Ensemble | Best balance of speed and accuracy |
| LSTM | Deep learning | Best for long sequences (needs ≥ 500 rows) |

The best model (lowest CV RMSE) is selected automatically for forecasting.

---

## CI/CD (GitHub Actions)

Add these secrets to your GitHub repository (`Settings → Secrets`):

- `AQICN_API_TOKEN`
- `OPENWEATHER_API_KEY`
- `HOPSWORKS_API_KEY` *(optional)*
- `HOPSWORKS_PROJECT` *(optional)*

Pipelines run automatically:
- **Feature pipeline**: Every hour at `:00`
- **Training pipeline**: Daily at `02:30 UTC`

---

## AQI Scale (US EPA)

| AQI | Level | Action |
|-----|-------|--------|
| 0–50 | Good | None |
| 51–100 | Moderate | Sensitive groups be cautious |
| 101–150 | Unhealthy for Sensitive Groups | Limit outdoor exertion |
| 151–200 | Unhealthy | Everyone limit outdoor activity |
| 201–300 | Very Unhealthy | Health alert |
| 301–500 | Hazardous | Emergency conditions — stay indoors |

---

## Run Tests

```bash
pytest tests/ -v --tb=short
```

---

## Supported Cities

london, new york, beijing, delhi, paris, karachi, lahore

To add a new city, edit `CITIES` in [`src/config.py`](src/config.py).

---

## Technologies

Python · Scikit-learn · TensorFlow/Keras · Streamlit · Flask · Plotly · SHAP · SQLite · GitHub Actions · AQICN API · OpenWeather API · Hopsworks (optional)
