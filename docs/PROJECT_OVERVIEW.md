# Project Overview: Dynamic Inventory Replenishment System Using Machine Learning

## 1. Executive Summary

In supply chain and inventory management, companies consistently face two conflicting challenges:
1. **Overstocking**: Holding excessive inventory increases warehouse holding costs, working capital blockage, and risks product spoilage or obsolescence.
2. **Understocking (Stockouts)**: Carrying too little inventory leads to unfulfilled customer demand, lost revenue, and poor customer retention.

The **Dynamic Inventory Replenishment System Using Machine Learning** addresses this fundamental trade-off by combining:
- **Machine Learning Demand Forecasting**: Predicting future consumer demand using historical sales trends, promotional calendars, day-of-week patterns, price changes, and autoregressive lag features.
- **Operations Research Inventory Optimization**: Converting dynamic demand forecasts into actionable replenishment decisions—specifically calculating dynamic **Safety Stock (SS)**, **Reorder Points (ROP)**, and **Order-Up-To Levels (S)**.

---

## 2. Problem Statement & Motivation

Traditional inventory policies commonly rely on:
- Static historical sales averages (e.g., rolling 30-day mean).
- Fixed reorder thresholds (e.g., reordering whenever stock drops below 100 units).
- Periodic manual visual inspections.

These conventional methods fail during:
- **Promotional campaigns**: Where sales surge by 30% to 60%, exhausting inventory before a replenishment arrives.
- **Weekend demand peaks**: Higher customer traffic causing recurring weekend stockouts.
- **Supplier lead time variability**: Delays between placing a purchase order and receiving physical stock at the warehouse.

By introducing predictive machine learning, the inventory management system shifts from **reactive replenishment** (ordering after stock has already crashed) to **proactive dynamic replenishment** (ordering just in time to cover anticipated spikes).

---

## 3. Project Objectives

1. **Synthesize a Realistic Retail Dataset**: Build a reproducible 800-record sales and supply chain dataset covering 4 realistic product categories across 200 consecutive calendar days.
2. **Conduct Exploratory Data Analysis (EDA)**: Identify seasonal trends, price elasticity, day-of-week patterns, and promotional lift.
3. **Train & Evaluate Regression Models**:
   - Baseline: **Linear Regression**.
   - Primary: **Random Forest Regressor**.
   - Evaluation on an unseen chronological test set (80/20 train-test split) using MAE, RMSE, and $R^2$.
4. **Develop an Inventory Optimization Engine**: Implement formulas for Lead Time Demand, Safety Stock, Reorder Point, and Order-Up-To replenishment quantities.
5. **Conduct a Comparative Supply Chain Simulation**: Simulate day-by-day operations comparing a static Traditional policy against the ML-Driven Dynamic policy across identical demand scenarios.
6. **Deliver an Interactive Streamlit Interface**: Provide a dashboard for store managers to inspect inventory health, run forecasts, modify stock parameters, and view replenishment alerts.

---

## 4. System Architecture

The project consists of three core layers:

```
┌────────────────────────────────────────────────────────┐
│                   Data & Feature Layer                 │
│  - Synthetic Generation (800 rows, 4 products)         │
│  - Lag-1, Lag-7, 7-Day Rolling Averages (No Leakage)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                  Machine Learning Layer                │
│  - Chronological 80/20 Train-Test Split                │
│  - Linear Regression vs. Random Forest Regressor       │
│  - Model Serialization (.joblib)                       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│              Supply Chain Optimization Layer           │
│  - Lead Time Demand & Safety Stock (95% Service Level) │
│  - Dynamic Reorder Point (ROP) & Order-Up-To Level     │
│  - 40-Day Test Simulation & Cost Comparison            │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                 Streamlit User Interface               │
│  - Executive Dashboard & Stockout Alerts               │
│  - Interactive Forecasting & Scenario Exploration      │
│  - Live Reorder Recommendations & Explanations         │
└────────────────────────────────────────────────────────┘
```

---

## 5. Technology Stack

- **Language**: Python 3.10+
- **Data Manipulation**: NumPy, Pandas
- **Machine Learning**: Scikit-Learn, Joblib
- **Visualization**: Matplotlib, Seaborn
- **Application Frontend**: Streamlit
- **Testing**: PyTest
