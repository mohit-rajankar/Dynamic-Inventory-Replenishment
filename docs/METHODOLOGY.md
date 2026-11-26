# Methodology: Dynamic Inventory Replenishment System

This document outlines the systematic, step-by-step engineering methodology implemented in the project.

---

## 1. Complete Workflow Diagram

```
[Synthetic Dataset Generation]
             │
             ▼
   [Data Preprocessing]
   (Validation, Sorting, Chronological 80/20 Splitting)
             │
             ▼
  [Feature Engineering]
  (Lag 1, Lag 7, 7-Day Rolling Averages — Leakage Free)
             │
             ▼
   [Model Training & Eval]
   (Linear Regression Baseline vs. Random Forest Regressor)
             │
             ▼
  [Winning Model Selection]
  (Serialized as models/best_model.joblib)
             │
             ▼
 [Demand Forecasting Module]
 (Single-step and recursive multi-day future predictions)
             │
             ▼
[Inventory Optimization Engine]
(Lead Time Demand, Safety Stock, ROP, Order-Up-To Level)
             │
             ▼
[Comparative Supply Chain Simulation]
(Traditional Static Policy vs. ML-Driven Dynamic Policy)
             │
             ▼
[Streamlit Interactive Application]
(Live Dashboard, Inventory Controls, Replenishment Alerts)
```

---

## 2. Step-by-Step Methodology Details

### Step 1: Synthetic Dataset Generation (`src/generate_data.py`)
- Standard real-world retail datasets frequently suffer from missing supplier cost fields, unrecorded lead times, or proprietary restrictions.
- To provide a realistic, complete supply chain environment, a domain-driven synthetic generator was constructed using NumPy and Pandas.
- Generates 800 daily records (4 products $\times$ 200 consecutive calendar days) from January 1, 2025 to July 19, 2025.
- Controlled factors include product-specific base demand, day-of-week consumption shifts (weekend lift), monthly seasonal drift, price elasticity, and promotional discounts with corresponding demand spikes.

### Step 2: Data Preprocessing & Validation (`src/preprocessing.py`)
- Verifies that all 13 required schema columns exist with correct numerical and categorical types.
- Confirms zero missing values, zero duplicate timestamps per SKU, and non-negative daily demand.
- Chronologically orders records by product identifier and calendar date.
- Applies chronological train-test splitting: the first 80% of days (Days 1–160, 640 total observations) serve as training data, while the final 20% of days (Days 161–200, 160 total observations) are preserved as the unseen test set.

### Step 3: Feature Engineering Without Data Leakage (`src/feature_engineering.py`)
- Standard random train-test splitting in time series causes severe **look-ahead bias** and data leakage.
- To maintain strict temporal integrity, autoregressive features are calculated per product group:
  - `lag_1`: Demand from day $t-1$ (shifted by 1).
  - `lag_7`: Demand from day $t-7$ (shifted by 7).
  - `rolling_avg_7`: Rolling 7-day mean of past demand (`demand.shift(1).rolling(7).mean()`).
- Because the target observation is shifted before calculating rolling windows, features strictly reflect historical information available at prediction time.

### Step 4: Model Training & Evaluation (`src/train_model.py`)
- A Scikit-Learn `ColumnTransformer` handles categorical one-hot encoding for product identifiers (`product_id`) and standard scaling for numerical features.
- Two regression pipelines are built:
  1. **Linear Regression**: Linear baseline estimating direct additive effects.
  2. **Random Forest Regressor**: Primary non-linear ensemble (120 estimators) capable of modeling complex interactions between price discounts, promotions, and day-of-week.
- Models are trained on the chronological training set and evaluated on the chronological test set using MAE, RMSE, and $R^2$.
- The winning model is saved as `models/best_model.joblib`.

### Step 5: Demand Forecasting Inference (`src/forecasting.py`)
- Loads the serialized model artifact.
- Implements single-step batch inference across feature sets.
- Implements recursive multi-step forecasting: for future horizons (e.g., next 7 days), each day's prediction feeds into subsequent lag and rolling features.

### Step 6: Inventory Optimization Engine (`src/inventory_optimizer.py`)
- Transforms numerical demand predictions into practical supply chain decisions:
  - **Lead Time Demand ($LTD$)**: Anticipated consumption during supplier fulfillment delay.
  - **Safety Stock ($SS$)**: Buffer protecting against demand variance during lead time at a 95% cycle service level ($Z = 1.645$).
  - **Reorder Point ($ROP$)**: Threshold triggering a new order ($ROP = LTD + SS$).
  - **Order-Up-To Level ($S$)**: Ceiling target inventory ($S = \text{Demand} \times (L + R) + SS$).
  - **Replenishment Recommendation**: Recommends order quantity $\max(0, S - IP)$ when inventory position falls to or below ROP.

### Step 7: Supply Chain Simulation (`src/simulation.py`)
- Compares two distinct operational policies over the identical 40-day test set per product:
  - **Traditional Policy**: Static ROP and static Order-Up-To levels derived from historical training averages.
  - **ML-Driven Policy**: Dynamic ROP and dynamic Order-Up-To levels recalculated daily using ML demand predictions.
- Tracks daily order arrivals, customer demand fulfillment, shortages, holding costs, and purchasing costs.

### Step 8: Interactive User Interface (`app.py`)
- Built with Streamlit to enable decision support and live interactive demonstrations across 6 dedicated views.
