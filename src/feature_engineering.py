"""
Feature Engineering module for Dynamic Inventory Replenishment System.
Constructs leakage-free historical lag, rolling average, temporal, and categorical features.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from typing import Tuple, List
from src.preprocessing import load_data


FEATURE_COLS = [
    "product_id",
    "price",
    "promotion",
    "day_of_week",
    "month",
    "lag_1",
    "lag_7",
    "rolling_avg_7",
]

TARGET_COL = "demand"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate lag and rolling features for each product chronologically without look-ahead bias.

    - lag_1: Demand from previous day (shift 1).
    - lag_7: Demand from 7 days ago (shift 7).
    - rolling_avg_7: 7-day rolling average of historical demand (shifted by 1 day first).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["product_id", "date"]).reset_index(drop=True)

    engineered_dfs = []

    for _, group in df.groupby("product_id", sort=False):
        group = group.copy().sort_values(by="date").reset_index(drop=True)

        # Shift target by 1 day so rolling window strictly uses past days
        group["lag_1"] = group["demand"].shift(1)
        group["lag_7"] = group["demand"].shift(7)
        group["rolling_avg_7"] = group["demand"].shift(1).rolling(window=7, min_periods=1).mean()

        # Cleanly impute early boundary values (first 7 days) without future data:
        # Use initial day observation or expanding mean
        initial_demand = group["demand"].iloc[0]
        group["lag_1"] = group["lag_1"].fillna(initial_demand)
        group["lag_7"] = group["lag_7"].fillna(group["rolling_avg_7"]).fillna(initial_demand)
        group["rolling_avg_7"] = group["rolling_avg_7"].fillna(initial_demand)

        engineered_dfs.append(group)

    result_df = pd.concat(engineered_dfs, ignore_index=True)
    return result_df


def prepare_train_test_data(
    df: pd.DataFrame,
    test_ratio: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chronologically splits data per product into train and test sets,
    returning (X_train, X_test, y_train, y_test).
    """
    df_feat = create_features(df)

    train_rows = []
    test_rows = []

    for _, group in df_feat.groupby("product_id", sort=False):
        group_sorted = group.sort_values(by="date").reset_index(drop=True)
        split_idx = int(len(group_sorted) * (1.0 - test_ratio))
        train_rows.append(group_sorted.iloc[:split_idx])
        test_rows.append(group_sorted.iloc[split_idx:])

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    raw_df = load_data()
    feat_df = create_features(raw_df)
    print("Features engineered successfully.")
    print(f"Total shape: {feat_df.shape}")
    print(f"Feature columns: {FEATURE_COLS}")
    print(f"Any nulls: {feat_df[FEATURE_COLS + [TARGET_COL]].isnull().sum().to_dict()}")

    X_train, X_test, y_train, y_test = prepare_train_test_data(raw_df)
    print(f"Train split: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test split : X_test={X_test.shape}, y_test={y_test.shape}")
