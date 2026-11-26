"""
Preprocessing and data validation utilities for the Dynamic Inventory Replenishment System.
"""

import os
import pandas as pd
from typing import Tuple, List

REQUIRED_COLUMNS = [
    "date", "product_id", "product_name", "category",
    "price", "promotion", "day_of_week", "month", "demand",
    "lead_time", "unit_cost", "holding_cost", "shortage_cost"
]


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load inventory dataset from CSV file with proper parsing and chronological sorting.
    """
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "inventory_data.csv"
        )

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Inventory dataset not found at {filepath}. Run src/generate_data.py first.")

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["product_id", "date"]).reset_index(drop=True)
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Verify all required columns are present and check for missing values.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    if df[REQUIRED_COLUMNS].isnull().sum().sum() > 0:
        raise ValueError("Dataset contains unexpected null values.")

    return True


def get_product_list(df: pd.DataFrame) -> List[dict]:
    """
    Extract unique product metadata for display and configuration.
    """
    meta_cols = ["product_id", "product_name", "category", "lead_time", "unit_cost", "holding_cost", "shortage_cost"]
    products = df[meta_cols].drop_duplicates().to_dict(orient="records")
    return products


def chronological_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform chronological train-test split for each product independently to avoid data leakage.
    For each product, the first (1 - test_ratio) of records are placed into train,
    and the remaining records are placed into test.
    """
    train_dfs = []
    test_dfs = []

    for _, group in df.groupby("product_id", sort=False):
        group_sorted = group.sort_values(by="date").reset_index(drop=True)
        split_idx = int(len(group_sorted) * (1.0 - test_ratio))
        train_dfs.append(group_sorted.iloc[:split_idx])
        test_dfs.append(group_sorted.iloc[split_idx:])

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    return train_df, test_df


if __name__ == "__main__":
    df = load_data()
    validate_schema(df)
    train, test = chronological_train_test_split(df, 0.20)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Chronological split: Train = {len(train)} rows, Test = {len(test)} rows")
    print(f"Train date range: {train['date'].min().strftime('%Y-%m-%d')} to {train['date'].max().strftime('%Y-%m-%d')}")
    print(f"Test date range : {test['date'].min().strftime('%Y-%m-%d')} to {test['date'].max().strftime('%Y-%m-%d')}")
