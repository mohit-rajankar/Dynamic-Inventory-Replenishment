"""
Automated tests for dataset generation, schema validation, and integrity.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
from src.preprocessing import load_data, validate_schema
from src.generate_data import generate_inventory_data, validate_dataset


def test_generate_inventory_data():
    df = generate_inventory_data(n_days=10, seed=123)
    assert len(df) == 40
    assert df["product_id"].nunique() == 4
    assert (df["demand"] >= 0).all()


def test_saved_dataset_integrity():
    df = load_data()
    assert len(df) == 800
    assert df["product_id"].nunique() == 4
    assert (df.groupby("product_id").size() == 200).all()

    # Schema validation
    assert validate_schema(df) is True

    # No nulls
    assert df.isnull().sum().sum() == 0

    # No duplicate observations per product and date
    assert df.duplicated(subset=["product_id", "date"]).sum() == 0

    # Non-negative demands
    assert (df["demand"] >= 0).all()

    # Check required columns
    required = [
        "date", "product_id", "product_name", "category",
        "price", "promotion", "day_of_week", "month", "demand",
        "lead_time", "unit_cost", "holding_cost", "shortage_cost"
    ]
    for col in required:
        assert col in df.columns
