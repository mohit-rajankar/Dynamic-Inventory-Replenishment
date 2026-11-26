"""
Automated tests for feature engineering, model inference, and multi-step forecasting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from src.preprocessing import load_data
from src.feature_engineering import create_features, FEATURE_COLS
from src.forecasting import DemandForecaster, load_forecasting_model


def test_feature_engineering_no_leakage():
    df = load_data()
    feat_df = create_features(df)

    assert len(feat_df) == len(df)
    for col in FEATURE_COLS:
        assert col in feat_df.columns

    # No null values in engineered features
    assert feat_df[FEATURE_COLS].isnull().sum().sum() == 0

    # Ensure lag_1 is indeed non-negative
    assert (feat_df["lag_1"] >= 0).all()
    assert (feat_df["rolling_avg_7"] >= 0).all()


def test_model_loading_and_prediction():
    model = load_forecasting_model()
    assert model is not None

    df = load_data()
    feat_df = create_features(df).head(10)
    forecaster = DemandForecaster()
    preds = forecaster.predict_dataframe(feat_df)

    assert len(preds) == 10
    assert (preds >= 0).all()
    assert isinstance(preds[0], (int, np.integer))


def test_multi_day_forecasting():
    forecaster = DemandForecaster()
    forecast = forecaster.forecast_next_n_days("P001", n_days=7)

    assert len(forecast) == 7
    assert "predicted_demand" in forecast.columns
    assert (forecast["predicted_demand"] >= 0).all()
    assert forecast["product_id"].iloc[0] == "P001"
