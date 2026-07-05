# 🚆 Train Ridership Forecasting

Predicting daily train ridership and allocating the required number of trains per station using machine learning — with a comparative analysis of **ElasticNet regression** and **LSTM** models across a dataset spanning 2019–2022, including the COVID-19 pandemic period.

---

## 📌 Project Overview

Rail operators need to know how many trains to deploy at each station and time period to avoid both overcrowding and resource waste. This project builds an end-to-end pipeline that:

1. Explores ridership patterns and the structural impact of COVID-19
2. Engineers time-series features (lag, rolling mean, cyclical encoding)
3. Trains and tunes two distinct model families
4. Compares their performance at both global and per-station levels
5. Outputs concrete train allocation recommendations

---

## 📂 Repository Structure

```
Train-Ridership-Forecasting/
│
├── data/
│   ├── Ridership.csv                            # Raw dataset (2019–2022)
│   ├── elasticnet_allocation_summary.csv        # ElasticNet train allocation output
│   └── lstm_allocation_summary.csv             # LSTM train allocation output
│
├── models/
│   ├── ElasticNet_model/
│   │   ├── elasticnet_model.pkl                 # Trained ElasticNet model
│   │   └── preprocessor.pkl                    # Fitted ColumnTransformer
│   └── LSTM_model/
│       ├── lstm_model.h5                        # Trained LSTM model
│       ├── scaler_X.pkl                         # Feature scaler (MinMaxScaler)
│       └── scaler_y.pkl                         # Target scaler (MinMaxScaler)
│
├── notebooks/
│   ├── EDA.ipynb                                # Exploratory Data Analysis
│   ├── ElasticNet_Ridership_Forecasting.ipynb   # ElasticNet pipeline
│   ├── LSTM.ipynb                               # LSTM pipeline
│   └── Model_Comparison.ipynb                  # Side-by-side model comparison
│
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

| Column | Type | Description |
|--------|------|-------------|
| `Year` | int | Calendar year (2019–2022) |
| `Month` | str | Month name |
| `Day` | int | Day of month |
| `Week Number` | int | ISO week number |
| `Corridor` | str | Train route corridor (7 unique) |
| `Workday` | str | `y` = workday, `n` = weekend/holiday |
| `Station` | str | Station identifier (45 unique) |
| `Period` | str | Time-of-day segment (5 unique: AM Peak, Midday, PM Peak, Evening, Weekend/Holiday) |
| `Ridership` | int | **Target** — number of passengers |
| `N_trains` | int | Number of trains deployed |
| `Covid19` | int | Binary flag: 1 = pandemic period |

**Key statistics:**
- 64,369 observations · No missing values · No duplicate rows
- Ridership ranges from 0 to 26,798 (strongly right-skewed)
- COVID-19 reduces average ridership by ~73% (from ~1,676 to ~448)

---

## 🔧 Feature Engineering

All features are computed on the **training set only** and applied to the test set to prevent data leakage.

| Feature | Description |
|---------|-------------|
| `month_sin` / `month_cos` | Cyclical month encoding (preserves Dec→Jan continuity) |
| `day_of_week` / `is_weekend` | Day-level calendar features |
| `lag_1/2/3_period` | Ridership 1–3 steps ago (per Station × Period group) |
| `rolling_mean_3/7_period` | Rolling averages shifted by 1 to avoid leakage |
| `Station_enc` | Target encoding of Station by mean ridership (train-only) |

---

## 🤖 Models

### ElasticNet Regression

Combines L1 (Lasso) and L2 (Ridge) regularization — well-suited for datasets with correlated lag and rolling features.

- Hyperparameter tuning via `GridSearchCV` with `TimeSeriesSplit(n_splits=5)`
- Search space: `alpha` ∈ [0.001 … 100], `l1_ratio` ∈ [0.1, 0.5, 0.9]
- Preprocessing: `OneHotEncoder` for categoricals + `StandardScaler` for numerics
- Low-frequency stations (< 30 records) are grouped into an `Other` category

### LSTM (Long Short-Term Memory)

Deep learning model for sequential pattern learning.

- Sequences built **per station** (window = 7 timesteps) to avoid cross-station contamination
- Architecture: 2× stacked LSTM → Dropout(0.2) → BatchNorm → Dense(16) → Dense(1)
- Training: `EarlyStopping(patience=10)` + `ReduceLROnPlateau(patience=5)`
- Scaling: `MinMaxScaler` for both features and target
- Low-frequency stations (< 30 records) are **dropped entirely** — too few records to build reliable sequences
- Original row indices are tracked inside `create_sequences_per_station` to prevent length-mismatch errors when aligning predictions back to the DataFrame

---

## 📈 Model Comparison

Four comparison dimensions are evaluated in `Model_Comparison.ipynb`:

| # | Method | What it measures |
|---|--------|-----------------|
| 1 | **Global Metrics** | RMSE, MAE, R² on the shared aligned test set |
| 2 | **Visual Comparison** | Time series overlay, scatter plots, residual distributions |
| 3 | **Per-Station RMSE** | Winner at each individual station + delta chart |
| 4 | **Train Allocation** | Over/under-allocation rates, heatmaps per Station × Period |

### Results

| Criterion | ElasticNet | LSTM | Winner |
|-----------|:----------:|:----:|:------:|
| Global RMSE | **433.82** | 563.71 | ✅ ElasticNet |
| Global MAE | **275.28** | 374.11 | ✅ ElasticNet |
| Global R² | **0.4847** | 0.1299 | ✅ ElasticNet |
| Per-Station RMSE wins | **21 / 23** | 2 / 23 | ✅ ElasticNet |
| Train Allocation MAE | **0.431** | 0.572 | ✅ ElasticNet |

**ElasticNet wins across all 5 criteria.**

### Why ElasticNet outperforms LSTM on this dataset

- **Rich feature engineering** — lag and rolling features hand the model pre-digested temporal patterns, removing LSTM's main advantage
- **Moderate data volume per station** — LSTM needs large sequence counts to generalize; splitting by station limits this significantly
- **Near-linear relationships** — ridership correlates strongly and linearly with `Covid19`, `Period`, and `Station`, which ElasticNet captures efficiently
- **ElasticNet's L1+L2 regularization** controls overfitting effectively at this scale, while LSTM tends to overfit with limited per-station sequences

---

## 🚉 Train Allocation Logic

```python
required_trains = ceil(predicted_ridership / capacity)
required_trains = max(required_trains, min_trains)
```

Default parameters: `capacity = 600 passengers/train`, `min_trains = 1`

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run notebooks in order

```
1. EDA.ipynb
2. ElasticNet_Ridership_Forecasting.ipynb   ← saves models/ElasticNet_model/
3. LSTM.ipynb                               ← saves models/LSTM_model/
4. Model_Comparison.ipynb                  ← loads both saved models
```

> ⚠️ `Model_Comparison.ipynb` depends on saved artifacts from steps 2 and 3. Run them first.

### 3. Loading the LSTM model

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import joblib

lstm_model = load_model(
    '../models/LSTM_model/lstm_model.h5',
    custom_objects={'mse': MeanSquaredError()}
)
scaler_X = joblib.load('../models/LSTM_model/scaler_X.pkl')
scaler_y = joblib.load('../models/LSTM_model/scaler_y.pkl')
```

---

## 🛠️ Tech Stack

| Tool | Version | Role |
|------|---------|------|
| Python | 3.10+ | Core language |
| pandas / numpy | — | Data manipulation |
| scikit-learn | — | ElasticNet, preprocessing, evaluation |
| TensorFlow / Keras | — | LSTM model |
| matplotlib / seaborn | — | Visualization |
| joblib | — | Model serialization |

---

## 📋 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Time-based split at `2022-01-01` | Respects temporal ordering — no future data leaks into training |
| Outlier clipping via IQR | Uses train statistics only; applied identically to test set |
| Per-station sequences for LSTM | Prevents cross-station contamination in sliding windows |
| Index tracking in `create_sequences_per_station` | Avoids `ValueError` from length mismatch when aligning predictions |
| `custom_objects={'mse': MeanSquaredError()}` | Resolves Keras 3 deserialization error when loading `.h5` models |
| Station dropped (not grouped) in LSTM | Too few records produce unreliable sequences; dropping is cleaner |

---

## 👤 Author

**Ali Sarafraz**  
GitHub: [@Ali-sarafraz](https://github.com/Ali-sarafraz)
