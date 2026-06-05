# Complete Setup Guide — Pearls AQI Predictor

> Written for someone setting this up for the first time.
> Estimated time: 20–30 minutes.

---

## What Is This Project?

This project predicts the **Air Quality Index (AQI)** for any city for the **next 3 days (72 hours)**. It:

- Fetches live air quality and weather data from free APIs
- Engineers dozens of features (time patterns, lag features, rolling averages)
- Trains 3 machine learning models (Ridge Regression, Random Forest, LSTM)
- Displays everything on an interactive web dashboard
- Runs automatically every hour (feature collection) and every day (model retraining) via GitHub Actions

---

## Before You Start — What You Need

### 1. Python 3.10 or 3.11
Check if you have it:
```
python --version
```
If not, download from: https://www.python.org/downloads/  
**During install, tick the box "Add Python to PATH".**

### 2. Git (to clone from GitHub)
Check if you have it:
```
git --version
```
If not, download from: https://git-scm.com/downloads

### 3. Two Free API Keys
You need these before running anything. Both are free, both take under 2 minutes to get.

#### API Key 1 — AQICN (for real AQI data)
1. Go to: https://aqicn.org/data-platform/token/
2. Enter your email → click "Send Token"
3. Check your email → copy the token (looks like: `a1b2c3d4e5f6...`)

#### API Key 2 — OpenWeather (for weather + historical data)
1. Go to: https://openweathermap.org/api
2. Click "Sign Up" → create a free account
3. Go to: https://home.openweathermap.org/api_keys
4. Copy the key shown (looks like: `ab12cd34ef56gh78...`)
5. **Important:** New keys take up to 10 minutes to activate. Get this key first, then do the other steps.

---

## Step-by-Step Setup

### Step 1 — Get the project files

**Option A — From GitHub (if shared as a repo link):**
```bash
git clone https://github.com/YOUR_FRIENDS_USERNAME/AQI_Predictor.git
cd AQI_Predictor
```

**Option B — From a ZIP file:**
1. Extract the ZIP
2. Open a terminal / command prompt inside the extracted folder
3. Make sure you are inside the `AQI_Predictor` folder (you should see `requirements.txt` when you run `dir` or `ls`)

---

### Step 2 — Create a virtual environment

A virtual environment keeps this project's dependencies separate from everything else on your computer.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` at the start of your terminal line. Keep this active for all the steps below.

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs everything: scikit-learn, TensorFlow, Streamlit, Flask, SHAP, Plotly, etc.

It will take 3–5 minutes. You'll see a lot of text scrolling — that's normal.

---

### Step 4 — Set up your API keys

Copy the example config file:

**Windows:**
```bash
copy .env.example .env
```

**Mac / Linux:**
```bash
cp .env.example .env
```

Now open the `.env` file (use Notepad, VS Code, or any text editor) and fill in your keys:

```
AQICN_API_TOKEN=paste_your_aqicn_token_here
OPENWEATHER_API_KEY=paste_your_openweather_key_here
DEFAULT_CITY=london
```

Save and close the file.

> **Tip:** You can change `DEFAULT_CITY` to any of these:  
> `london`, `new york`, `beijing`, `delhi`, `paris`, `karachi`, `lahore`

---

### Step 5 — Run the historical backfill (run this ONCE)

This fetches 90 days of historical air quality data so the models have enough data to train on.

```bash
python scripts/backfill.py --city london --days 90
```

Replace `london` with your chosen city.

Expected output:
```
2024-xx-xx [INFO] Starting backfill for 'london' — 90 days.
2024-xx-xx [INFO] Retrieved 2160 raw rows from ...
2024-xx-xx [INFO] Engineered 2160 feature rows. Storing...
2024-xx-xx [INFO] Backfill complete.
```

> **If you see an error about the API key:** Wait a few more minutes for the OpenWeather key to activate, then try again.

---

### Step 6 — Train the models

```bash
python scripts/run_training_pipeline.py --city london --days 90
```

This trains three models:
- **Ridge Regression** (fast, ~5 seconds)
- **Random Forest** (moderate, ~30 seconds)  
- **LSTM neural network** (slow, ~2–5 minutes depending on your computer)

Expected output at the end:
```
========================================
 Random Forest (CV avg)
========================================
  RMSE : 12.34
  MAE  : 8.92
  R²   : 0.8741
  MAPE : 11.23%
```

The model with the lowest RMSE will be used automatically for forecasting.

> **To skip LSTM training** (if it's too slow):
> ```bash
> python scripts/run_training_pipeline.py --city london --no-lstm
> ```

---

### Step 7 — Launch the web dashboard

You need **two terminal windows open at the same time** (both with the venv activated).

**Terminal 1 — Start the Flask API:**
```bash
python app/flask_api.py
```
You'll see: `Running on http://0.0.0.0:5000`

**Terminal 2 — Start the Streamlit dashboard:**
```bash
streamlit run app/streamlit_app.py
```
Your browser will open automatically at **http://localhost:8501**

---

## What You'll See on the Dashboard

```
┌─────────────────────────────────────────────────────┐
│  🌬️ AQI Predictor — London                          │
│                                                     │
│  Current AQI: 72  [Moderate 🟡]                     │
│  PM2.5: 18.4   PM10: 34.2   O3: 45.1   NO2: 22.3   │
│                                                     │
│  ⚠️  Moderate air quality expected today.           │
│                                                     │
│  [72-hour forecast chart with uncertainty band]     │
│                                                     │
│  Day 1: avg 72   Day 2: avg 85   Day 3: avg 91      │
│                                                     │
│  [Historical trend — last 7 days]                   │
│  [Pollutant bar chart]                              │
│  [SHAP feature importance chart]                    │
│  [Health recommendations]                           │
└─────────────────────────────────────────────────────┘
```

Use the **left sidebar** to:
- Switch between cities
- Choose which model to use for forecasting
- Change the historical window
- Toggle SHAP feature importance

---

## Project Structure Explained

```
AQI_Predictor/
│
├── .env                        ← Your API keys (you create this from .env.example)
├── requirements.txt            ← All Python packages needed
│
├── src/                        ← All the core logic
│   ├── config.py               ← Settings: cities, thresholds, paths
│   │
│   ├── feature_pipeline/
│   │   ├── data_fetcher.py     ← Talks to AQICN and OpenWeather APIs
│   │   ├── feature_engineer.py ← Computes 50+ features from raw data
│   │   └── feature_store.py    ← Saves/loads data from SQLite database
│   │
│   ├── training_pipeline/
│   │   ├── models/
│   │   │   ├── ridge_model.py          ← Linear regression baseline
│   │   │   ├── random_forest_model.py  ← Ensemble model
│   │   │   └── lstm_model.py           ← Deep learning (TensorFlow)
│   │   ├── train.py            ← Orchestrates training of all models
│   │   ├── evaluate.py         ← Computes RMSE, MAE, R²
│   │   └── model_registry.py   ← Saves trained models to disk
│   │
│   ├── inference_pipeline/
│   │   └── predict.py          ← Loads best model, generates 72-hour forecast
│   │
│   └── utils/
│       ├── alerts.py           ← AQI level classification and alert messages
│       └── shap_explainer.py   ← Explains which features drive predictions
│
├── app/
│   ├── streamlit_app.py        ← The web dashboard you see in the browser
│   └── flask_api.py            ← REST API (JSON endpoints)
│
├── scripts/
│   ├── backfill.py             ← One-time historical data fetch
│   ├── run_feature_pipeline.py ← Run hourly to collect new data
│   └── run_training_pipeline.py ← Run daily to retrain models
│
├── data/
│   └── processed/
│       ├── feature_store.db    ← SQLite database (auto-created)
│       └── london_features.csv ← CSV backup (auto-created)
│
├── models/
│   └── registry/               ← Saved model files (auto-created after training)
│       └── london/
│           ├── random_forest/  ← .pkl files + metadata JSON
│           ├── ridge/
│           └── lstm/
│
├── notebooks/
│   └── EDA.ipynb               ← Exploratory Data Analysis notebook
│
├── tests/                      ← Automated tests
│   ├── test_feature_pipeline.py
│   └── test_inference.py
│
└── .github/workflows/          ← GitHub Actions CI/CD
    ├── feature_pipeline.yml    ← Runs every hour automatically (on GitHub)
    └── training_pipeline.yml   ← Runs every day automatically (on GitHub)
```

---

## How the Data Flows

```
1. EVERY HOUR  →  data_fetcher.py fetches live AQI + weather data
                          ↓
2.              feature_engineer.py computes 50+ features:
                  - Time: hour, day, month, season, cyclical encodings
                  - Lag: AQI 1h ago, 2h ago, 6h ago, 12h ago, 24h ago...
                  - Rolling: mean/std/min/max over 3h, 6h, 12h, 24h windows
                  - Change rate: how fast AQI is rising/falling
                  - Interactions: wind dispersion, heat index
                          ↓
3.              feature_store.py saves to SQLite + CSV

4. EVERY DAY   →  train.py loads last 90 days of features
                          ↓
5.              Trains Ridge, Random Forest, LSTM models
                Uses TimeSeriesSplit (no future data leaks into training)
                          ↓
6.              model_registry.py saves best model to disk

7. ON DEMAND   →  predict.py loads best model
                          ↓
8.              Generates 72-hour AQI forecast (autoregressive)
                          ↓
9.              streamlit_app.py + flask_api.py display results
```

---

## API Endpoints (for developers)

If you want to use the data programmatically instead of the dashboard:

```bash
# Check the service is running
curl http://localhost:5000/api/health

# Get current AQI for London
curl http://localhost:5000/api/current?city=london

# Get 72-hour forecast
curl http://localhost:5000/api/predict?city=london

# Get forecast using a specific model
curl http://localhost:5000/api/predict?city=london&model=random_forest

# Get last 7 days of history
curl http://localhost:5000/api/history?city=london&days=7

# Get active AQI alerts
curl http://localhost:5000/api/alerts?city=london

# List all trained models
curl http://localhost:5000/api/models?city=london
```

---

## Running the EDA Notebook

The notebook walks through data exploration, visualizations, and SHAP analysis.

```bash
# Make sure you're in the project folder with venv active
jupyter notebook notebooks/EDA.ipynb
```

Your browser will open with the notebook. Run each cell top to bottom with **Shift + Enter**.

> **Requirement:** Run the backfill and training steps first so the notebook has data to work with.

---

## Run Tests

To verify everything is working correctly:

```bash
pytest tests/ -v
```

Expected output:
```
28 passed in ~26s
```

---

## Setting Up Automatic Pipelines (GitHub Actions)

If you push this project to GitHub, the pipelines run automatically:

- **Feature pipeline** → runs every hour, collects new AQI data
- **Training pipeline** → runs every day at 2:30 AM UTC, retrains models

**To enable this:**

1. Push the project to GitHub (see "How to Share" section below)
2. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
3. Add these secrets:

| Secret Name | Value |
|---|---|
| `AQICN_API_TOKEN` | Your AQICN token |
| `OPENWEATHER_API_KEY` | Your OpenWeather key |
| `HOPSWORKS_API_KEY` | Leave blank (optional) |
| `HOPSWORKS_PROJECT` | Leave blank (optional) |

4. Go to **Actions** tab → enable workflows if prompted

---

## Troubleshooting

**"No module named X"**
→ Make sure your venv is activated (`venv\Scripts\activate` on Windows) and you ran `pip install -r requirements.txt`

**"AQICN error" or AQI shows as 0**
→ Your AQICN token may be `demo` (limited). Get a free real token at https://aqicn.org/data-platform/token/

**"OpenWeather API 401 Unauthorized"**
→ New OpenWeather keys take 10–30 minutes to activate. Wait and try again.

**"No trained models found for 'london'"**
→ Run `python scripts/run_training_pipeline.py --city london` first

**"Not enough feature data"**
→ Run `python scripts/backfill.py --city london --days 90` first

**"Port 8501 already in use"**
→ Another Streamlit app is running. Stop it or run: `streamlit run app/streamlit_app.py --server.port 8502`

**"Port 5000 already in use"**
→ Run: `python app/flask_api.py --port 5001`

**LSTM training is very slow**
→ Add `--no-lstm` to the training command. Random Forest is just as good in most cases.

---

## Adding a New City

Edit `src/config.py` and add to the `CITIES` dictionary:

```python
CITIES = {
    ...
    "istanbul": {"lat": 41.0082, "lon": 28.9784, "aqicn_name": "istanbul"},
}
```

Then backfill and train for that city:
```bash
python scripts/backfill.py --city istanbul --days 90
python scripts/run_training_pipeline.py --city istanbul
```

---

## AQI Scale Reference

| AQI Range | Level | Who is affected |
|---|---|---|
| 0–50 | 🟢 Good | No one |
| 51–100 | 🟡 Moderate | Very sensitive individuals |
| 101–150 | 🟠 Unhealthy for Sensitive Groups | Children, elderly, asthma patients |
| 151–200 | 🔴 Unhealthy | Everyone |
| 201–300 | 🟣 Very Unhealthy | Everyone — serious effects |
| 301–500 | ⚫ Hazardous | Emergency conditions |

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| Scikit-learn | Ridge Regression, Random Forest, preprocessing |
| TensorFlow / Keras | LSTM neural network |
| SHAP | Feature importance explanations |
| Streamlit | Web dashboard |
| Flask | REST API backend |
| Plotly | Interactive charts |
| SQLite | Local feature store database |
| GitHub Actions | Automated CI/CD pipelines |
| AQICN API | Real-time AQI data (free) |
| OpenWeather API | Weather + historical data (free) |
| Hopsworks | Cloud feature store (optional) |

---

*Guide written for Pearls AQI Predictor — end-to-end ML pipeline for air quality forecasting.*
