"""
Demand Forecasting Inference Module for Dynamic Inventory Replenishment System.
Loads the trained best model and generates future demand predictions for products.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import joblib

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import load_data
from src.feature_engineering import create_features, FEATURE_COLS


def load_forecasting_model(model_path: Optional[str] = None):
    """
    Load the serialized model pipeline from disk.
    """
    if model_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "best_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run src/train_model.py first.")

    return joblib.load(model_path)


class DemandForecaster:
    """
    Inference class to generate predictions and multi-day forecasts.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = load_forecasting_model(model_path)

    def predict_dataframe(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Generate demand predictions for a dataframe containing the required feature columns.
        """
        X = df_features[FEATURE_COLS]
        preds = self.model.predict(X)
        # Ensure predictions are non-negative integers
        return np.maximum(0, np.round(preds)).astype(int)

    def forecast_next_n_days(
        self,
        product_id: str,
        n_days: int = 7,
        future_promotions: Optional[List[int]] = None,
        custom_price: Optional[float] = None,
        historical_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate recursive multi-step daily forecast for a given product for the next n_days.
        Uses historical data to seed the initial lag and rolling features.
        """
        if historical_df is None:
            historical_df = load_data()

        prod_history = historical_df[historical_df["product_id"] == product_id].copy()
        if prod_history.empty:
            raise ValueError(f"Product ID '{product_id}' not found in historical data.")

        prod_history["date"] = pd.to_datetime(prod_history["date"])
        prod_history = prod_history.sort_values(by="date").reset_index(drop=True)

        last_date = prod_history["date"].iloc[-1]
        prod_name = prod_history["product_name"].iloc[-1]
        category = prod_history["category"].iloc[-1]
        base_price = prod_history["price"].iloc[-1] if custom_price is None else custom_price
        lead_time = prod_history["lead_time"].iloc[-1]
        unit_cost = prod_history["unit_cost"].iloc[-1]
        holding_cost = prod_history["holding_cost"].iloc[-1]
        shortage_cost = prod_history["shortage_cost"].iloc[-1]

        # Keep a growing list of observed + forecasted demand for lag/rolling features
        recent_demands = list(prod_history["demand"].values)

        future_records = []

        for step in range(1, n_days + 1):
            target_date = last_date + pd.Timedelta(days=step)
            dow = target_date.day_of_week
            month = target_date.month

            is_promo = 0
            if future_promotions is not None and len(future_promotions) >= step:
                is_promo = int(future_promotions[step - 1])

            # Apply promo discount to price if applicable
            price = round(base_price * 0.88, 2) if is_promo else round(base_price, 2)

            # Lag features from recent history
            lag_1 = recent_demands[-1]
            lag_7 = recent_demands[-7] if len(recent_demands) >= 7 else recent_demands[0]
            rolling_avg_7 = float(np.mean(recent_demands[-7:]))

            feat_row = pd.DataFrame([{
                "product_id": product_id,
                "price": price,
                "promotion": is_promo,
                "day_of_week": dow,
                "month": month,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "rolling_avg_7": rolling_avg_7,
            }])

            pred_val = int(self.predict_dataframe(feat_row)[0])
            recent_demands.append(pred_val)

            future_records.append({
                "date": target_date.strftime("%Y-%m-%d"),
                "product_id": product_id,
                "product_name": prod_name,
                "category": category,
                "price": price,
                "promotion": is_promo,
                "day_of_week": dow,
                "month": month,
                "predicted_demand": pred_val,
                "lead_time": lead_time,
                "unit_cost": unit_cost,
                "holding_cost": holding_cost,
                "shortage_cost": shortage_cost,
            })

        return pd.DataFrame(future_records)


if __name__ == "__main__":
    forecaster = DemandForecaster()
    forecast_df = forecaster.forecast_next_n_days(product_id="P001", n_days=7)
    print("Next 7 Days Forecast for P001 (Basmati Rice):")
    print(forecast_df[["date", "product_name", "price", "promotion", "predicted_demand"]])
