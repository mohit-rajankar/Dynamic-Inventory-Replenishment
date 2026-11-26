# Dynamic Inventory Replenishment System Using Machine Learning

An end-to-end, machine-learning-driven inventory optimization and demand forecasting system designed to solve the twin retail challenges of **overstocking** and **stockouts**.

---

## 1. Problem Statement

Retail businesses and supply chains face two costly inventory failures:
1. **Overstocking**: Purchasing more stock than customers demand results in inflated warehouse holding costs, capital lockup, and product depreciation.
2. **Understocking (Stockouts)**: Carrying insufficient stock leads to unfulfilled customer demand, lost sales revenue, and degraded customer trust.

Traditional inventory management relies on static reorder thresholds and fixed historical sales averages that fail to anticipate promotional surges, price elasticity, and weekend traffic patterns.

---

## 2. Proposed Solution

This system combines **supervised machine learning regression** with **operations research inventory formulations**:
- **Demand Forecasting**: Uses an ensemble **Random Forest Regressor** to predict daily customer demand based on price elasticity, active promotions, day-of-week patterns, and leakage-free autoregressive lag features (`lag_1`, `lag_7`, and 7-day rolling averages).
- **Dynamic Inventory Optimization**: Converts predicted future demand into dynamic **Safety Stock (SS)**, **Lead Time Demand (LTD)**, **Reorder Points (ROP)**, and **Order-Up-To Levels (S)**.
- **Interactive Decision-Support Application**: A complete **Streamlit** dashboard giving warehouse managers real-time visibility into inventory health, predictive alerts, interactive scenario modeling, and replenishment recommendations.

---

## 3. Technology Stack

- **Core Language**: Python 3.10+
- **Data Engineering**: NumPy, Pandas
- **Machine Learning**: Scikit-Learn, Joblib
- **Data Visualization**: Matplotlib, Seaborn
- **User Interface**: Streamlit
- **Testing & Verification**: PyTest

---

## 4. Algorithms Used

1. **Linear Regression (Baseline)**: Evaluates basic linear relationships between prices, calendar factors, and demand.
2. **Random Forest Regressor (Primary)**: Captures complex non-linear promotional spikes, holiday patterns, and price elasticity with 120 decision trees.
3. **Inventory Management Formulations**:
   - **Lead Time Demand**: $\text{LTD} = \hat{d} \times L$
   - **Safety Stock**: $\text{SS} = \lceil Z \times \sigma_d \times \sqrt{L} \rceil$ ($Z = 1.645$ for 95% Cycle Service Level)
   - **Reorder Point**: $\text{ROP} = \text{LTD} + \text{SS}$
   - **Order-Up-To Level**: $S = \lceil \hat{d} \times (L + R) + \text{SS} \rceil$
   - **Recommended Order Quantity**: $Q^* = \max(0, S - \text{IP})$

---

## 5. Dataset Description

- **Generation**: Fully reproducible synthetic dataset generated via `src/generate_data.py` (`RANDOM_SEED = 42`).
- **Scale**: Exactly 800 observations (4 distinct products $\times$ 200 consecutive calendar days from Jan 1, 2025 to Jul 19, 2025).
- **Products**:
  - `P001`: Basmati Rice (Grocery)
  - `P002`: Cooking Oil (Grocery)
  - `P003`: Herbal Shampoo (Personal Care)
  - `P004`: Digestive Biscuits (Food)
- **Features**: `date`, `product_id`, `product_name`, `category`, `price`, `promotion`, `day_of_week`, `month`, `demand`, `lead_time`, `unit_cost`, `holding_cost`, `shortage_cost`.
- **Quality**: Zero missing values, zero duplicates, non-negative integer demand.

---

## 6. Project Structure

```
dynamic-inventory-replenishment/
│
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Project dependencies
├── README.md                   # Core project documentation
├── .gitignore                  # Git ignore rules
│
├── data/
│   └── inventory_data.csv      # Generated 800-record dataset
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py        # Synthetic dataset generator
│   ├── preprocessing.py        # Data loading, validation, and splitting
│   ├── eda.py                  # Exploratory Data Analysis & plot generation
│   ├── feature_engineering.py  # Leakage-free lag & rolling window features
│   ├── train_model.py          # Model training, comparison, & serialization
│   ├── forecasting.py          # Inference & multi-day projection engine
│   ├── inventory_optimizer.py  # Safety stock, ROP, & recommendation rules
│   └── simulation.py           # Traditional vs. ML-based supply chain simulation
│
├── models/
│   └── best_model.joblib       # Serialized winning Random Forest pipeline
│
├── outputs/
│   ├── eda/                    # 6 saved EDA visualization charts
│   ├── model_comparison.csv    # Test evaluation metrics (MAE, RMSE, R²)
│   ├── model_comparison.png    # Metric comparison bar charts
│   ├── simulation_results.csv  # 40-day test simulation comparison
│   └── simulation_comparison.png # Simulation cost & stockout charts
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py            # Unit tests for dataset integrity
│   ├── test_forecasting.py     # Unit tests for ML inference & forecasting
│   └── test_inventory.py       # Unit tests for replenishment calculations
│
└── docs/
    ├── PROJECT_OVERVIEW.md     # High-level architecture & objectives
    ├── METHODOLOGY.md          # Complete pipeline workflow guide
    ├── ALGORITHMS.md           # Mathematical formulas & algorithms
    ├── DATASET_DOCUMENTATION.md# Dataset design, schema, & assumptions
    ├── HOW_TO_RUN.md           # Installation, execution, & troubleshooting
    └── PRESENTATION_GUIDE.md   # Presentation script & 10 viva Q&As
```

---

## 7. Installation Instructions

```bash
# 1. Clone or open repository folder
cd d:\College\Dynamic-Inventory-Replenishment

# 2. (Optional) Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 8. Execution Pipeline

```bash
# Step 1: Generate dataset
python src/generate_data.py

# Step 2: Run Exploratory Data Analysis
python src/eda.py

# Step 3: Train and evaluate ML models
python src/train_model.py

# Step 4: Run supply chain simulation
python src/simulation.py

# Step 5: Run automated test suite
pytest

# Step 6: Launch Streamlit Web Application
streamlit run app.py
```

---

## 9. Key Results & Expected Outputs

### A. Model Performance (Chronological Test Set)

| Model Name | Test MAE | Test RMSE | Test $R^2$ Score | Selection |
|---|---|---|---|---|
| **Linear Regression (Baseline)** | 8.221 | 11.311 | 0.8452 | Baseline |
| **Random Forest Regressor (Primary)** | **4.619** | **6.894** | **0.9425** | **Selected Winner** |

*Random Forest achieved a ~39% reduction in test RMSE over the linear baseline.*

### B. Simulation Findings (Traditional vs. ML-Based Policy)

- **Stockout Days**: Traditional policy suffered **2 stockout days**; ML-driven policy achieved **0 stockout days** (100% service fulfillment).
- **Shortage Cost**: Traditional policy lost **$33.00**; ML-driven policy incurred **$0.00**.
- **Holding Cost**: ML policy maintains dynamic buffer stock ahead of anticipated promotional spikes, resulting in slightly higher holding costs in exchange for zero lost sales.

---

## 10. Authors & Acknowledgments

Developed as a major college project demonstrating the intersection of Applied Machine Learning and Operations Research in Supply Chain Management.
