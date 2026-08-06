# ⚡ Hourly Energy Consumption Forecasting
### PJM Interconnection — 15 Years of Demand Data | XGBoost · Random Forest · Linear Regression

---

## Overview

End-to-end machine learning project forecasting hourly electricity demand (in megawatts) across the PJM Interconnection grid — a regional operator covering Pennsylvania, New Jersey, Maryland, Ohio, and surrounding states.

The goal is to predict consumption **24 hours ahead** — the realistic horizon utilities use when scheduling generation. Short-horizon lags (1h, 2h) that essentially copy the current value are deliberately excluded.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) |
| File | `PJME_hourly.csv` |
| Columns | `Datetime`, `PJME_MW` |
| Range | ~2002 – 2018 |
| Size | ~130,000 rows |

---

## Project Structure

```
energy-forecasting/
├── energy_forecasting.ipynb   # Main notebook (all steps end-to-end)
├── PJME_hourly.csv            # Raw dataset (download from Kaggle)
└── README.md
```

---

## Methodology

### 1 · Data Cleaning
- Parsed and sorted the datetime index
- Removed anomalous low values (< 19,000 MW) — likely sensor errors or grid anomalies
- Verified no duplicate timestamps or missing values

### 2 · Exploratory Data Analysis
Key patterns discovered:
- **Hour of day** — consumption troughs at 4–5 AM, peaks at 5–7 PM
- **Day of week** — weekdays run ~8% higher than weekends
- **Month** — classic dual-peak: winter heating (Jan/Feb) and summer cooling (Jul/Aug)
- **Year** — slow downward trend since ~2008, likely from efficiency improvements

### 3 · Feature Engineering

| Feature | Description |
|---|---|
| `hour`, `dayofweek`, `month`, `year`, `quarter`, `dayofyear` | Calendar features |
| `is_weekend` | Binary weekend flag |
| `hour_sin`, `hour_cos` | Cyclical hour encoding (hour 23 ↔ hour 0 are adjacent) |
| `hour_x_month` | Interaction term — season-of-day proxy |
| `lag_24h_delta` | Change in consumption from 48h → 24h ago (trend signal) |
| `lag_48h`, `lag_72h`, `lag_168h` | Lagged consumption at 2 days, 3 days, 1 week ago |
| `rolling_mean_24h` | 24h rolling average, shifted 24h (no leakage) |
| `rolling_mean_7d` | 7-day rolling average, shifted 24h (weekly baseline) |

> **Why `lag_24h_delta` instead of raw `lag_24h`?**
> The raw same-hour-yesterday value is so predictive that XGBoost assigned it ~0.80 importance, ignoring almost everything else. Replacing it with a *delta* forces the model to use the full feature set to set the level, producing a genuinely balanced and more robust model.

### 4 · Train / Test Split

Temporal split — no shuffling, no leakage:
- **Training:** 2002 → end of 2016
- **Test:** 2017 → 2018

### 5 · Models

Three models trained and compared, all using the same feature set:

| Model | Notes |
|---|---|
| Linear Regression | Baseline — no hyperparameters |
| Random Forest | 300 trees, `max_depth=None`, `max_features=0.7` |
| **XGBoost** *(best)* | 500 trees, `colsample_bytree=0.7`, `subsample=0.8` |

---

## Results

<img width="622" height="251" alt="image" src="https://github.com/user-attachments/assets/404dc518-5c26-4a79-9dc9-5079bf91c0ae" />


> *Run the notebook to populate exact numbers — they depend on your environment.*

XGBoost wins due to its sequential boosting (each tree corrects prior errors) and the `colsample_bytree` regularisation preventing over-reliance on any single feature.

---

## Key Learnings

**What worked well**
- Replacing `lag_24h` with `lag_24h_delta` — the single biggest fix for a balanced importance chart
- Cyclical hour encoding (`sin`/`cos`) — better than raw integers for the model
- `colsample_bytree=0.7` on XGBoost — distributes learning across the full feature set
- Temporal train/test split — essential for honest evaluation on time series

**What to watch out for**
- Sub-24h lags inflate R² to ~0.99 but are cheating — the model just copies the current value
- Random Forest with `max_depth=10` underperforms Linear Regression; use `max_depth=None` with `min_samples_leaf` for regularisation instead
- The model struggles at extreme peaks (heatwaves, cold snaps) — rare events are underrepresented in training

**What could improve this further**
- Weather data (temperature is the single biggest missing signal)
- Holiday flags (Christmas, July 4th look very different from regular days)
- Walk-forward (rolling window) cross-validation
- Time-series native models: Prophet, SARIMA, LSTM, or Temporal Fusion Transformer

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

Download `PJME_hourly.csv` from [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) and update the path in the first data-loading cell, then run all cells top to bottom.

---

## Acknowledgements

Dataset by [Rob Mulla](https://www.kaggle.com/robikscube) on Kaggle.
