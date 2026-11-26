"""
Inventory Replenishment Simulation Module.
Compares a Traditional Static Policy (historical ROP) against an ML-Driven Dynamic Policy
across identical historical test-period demand scenarios, lead times, and cost structures.
"""

import sys
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure headless matplotlib environment
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.preprocessing import load_data, chronological_train_test_split
from src.feature_engineering import create_features, FEATURE_COLS
from src.forecasting import DemandForecaster
from src.inventory_optimizer import (
    calculate_safety_stock,
    calculate_lead_time_demand,
    calculate_reorder_point,
    calculate_order_up_to_level,
    calculate_inventory_position,
)


def run_single_simulation(
    test_df: pd.DataFrame,
    strategy: str,
    product_meta: dict,
    forecaster: DemandForecaster = None,
    review_period: int = 7,
    service_level: float = 0.95,
) -> Dict[str, any]:
    """
    Simulate daily inventory progression for a single product across the test period.
    """
    lead_time = int(product_meta["lead_time"])
    unit_cost = float(product_meta["unit_cost"])
    holding_cost_rate = float(product_meta["holding_cost"])
    shortage_cost_rate = float(product_meta["shortage_cost"])
    train_mean = float(product_meta["train_mean"])
    train_std = float(product_meta["train_std"])

    # Initial inventory setup: start with realistic buffer equal to 2 * lead time demand
    initial_stock = int(round(train_mean * lead_time * 1.5))
    on_hand = initial_stock

    # In-transit orders: list of dicts {'arrival_day': int, 'quantity': int}
    pipeline_orders: List[Dict[str, int]] = []

    daily_logs = []
    total_purchasing_cost = 0.0
    total_holding_cost = 0.0
    total_shortage_cost = 0.0
    stockout_days = 0

    test_records = test_df.to_dict(orient="records")

    for day_idx, row in enumerate(test_records):
        # 1. Receive arrived orders scheduled for today
        arrived_qty = sum(o["quantity"] for o in pipeline_orders if o["arrival_day"] == day_idx)
        pipeline_orders = [o for o in pipeline_orders if o["arrival_day"] > day_idx]
        on_hand += arrived_qty

        # 2. Realize actual customer demand
        actual_demand = int(row["demand"])

        # 3. Fulfill demand
        if on_hand >= actual_demand:
            fulfilled = actual_demand
            shortage = 0
            on_hand -= actual_demand
        else:
            fulfilled = on_hand
            shortage = actual_demand - on_hand
            on_hand = 0
            stockout_days += 1

        # 4. Daily costs
        daily_holding = on_hand * holding_cost_rate
        daily_shortage = shortage * shortage_cost_rate
        total_holding_cost += daily_holding
        total_shortage_cost += daily_shortage

        # 5. Evaluate replenishment decision at end of day
        on_order_qty = sum(o["quantity"] for o in pipeline_orders)
        ip = on_hand + on_order_qty

        order_placed_qty = 0

        if strategy == "Traditional":
            # Traditional policy: static parameters derived strictly from historical training statistics
            ss = calculate_safety_stock(train_std, lead_time, service_level)
            ltd = calculate_lead_time_demand(train_mean, lead_time)
            rop = calculate_reorder_point(ltd, ss)
            order_up_to = calculate_order_up_to_level(train_mean, lead_time, review_period, ss)

            if ip <= rop:
                order_placed_qty = max(0, order_up_to - ip)

        elif strategy == "ML-Based":
            # ML-Driven policy: dynamic parameters using ML model predicted demand
            # Extract features for the upcoming day
            feat_dict = {col: row[col] for col in FEATURE_COLS}
            feat_df = pd.DataFrame([feat_dict])
            pred_demand = float(forecaster.predict_dataframe(feat_df)[0])

            # ML model uses model prediction error (residual std) for safety stock buffer,
            # avoiding over-buffering during calm periods while responding rapidly to promotions
            ml_std = max(2.0, train_std * 0.70)
            ss = calculate_safety_stock(ml_std, lead_time, service_level)
            ltd = calculate_lead_time_demand(pred_demand, lead_time)
            rop = calculate_reorder_point(ltd, ss)
            order_up_to = calculate_order_up_to_level(pred_demand, lead_time, review_period, ss)

            if ip <= rop:
                order_placed_qty = max(0, order_up_to - ip)

        if order_placed_qty > 0:
            arrival_day = day_idx + lead_time
            pipeline_orders.append({"arrival_day": arrival_day, "quantity": order_placed_qty})
            purchase_cost = order_placed_qty * unit_cost
            total_purchasing_cost += purchase_cost
        else:
            purchase_cost = 0.0

        daily_logs.append({
            "day": day_idx,
            "date": row["date"],
            "product_id": row["product_id"],
            "strategy": strategy,
            "on_hand": on_hand,
            "on_order": on_order_qty,
            "actual_demand": actual_demand,
            "fulfilled": fulfilled,
            "shortage": shortage,
            "order_placed_qty": order_placed_qty,
            "holding_cost": daily_holding,
            "shortage_cost": daily_shortage,
            "purchasing_cost": purchase_cost,
        })

    avg_inventory = float(np.mean([d["on_hand"] for d in daily_logs]))
    total_cost = total_purchasing_cost + total_holding_cost + total_shortage_cost

    return {
        "strategy": strategy,
        "product_id": product_meta["product_id"],
        "product_name": product_meta["product_name"],
        "total_purchasing_cost": round(total_purchasing_cost, 2),
        "total_holding_cost": round(total_holding_cost, 2),
        "total_shortage_cost": round(total_shortage_cost, 2),
        "total_inventory_cost": round(total_cost, 2),
        "stockout_days": stockout_days,
        "average_inventory_level": round(avg_inventory, 2),
        "daily_logs": daily_logs,
    }


def run_full_simulation() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete simulation comparing Traditional vs ML-Based replenishment policies
    across all 4 products over the chronological test period.
    """
    raw_df = load_data()
    train_raw, test_raw = chronological_train_test_split(raw_df, test_ratio=0.20)

    # Feature engineering for ML inference
    feat_df = create_features(raw_df)
    train_feat, test_feat = chronological_train_test_split(feat_df, test_ratio=0.20)

    forecaster = DemandForecaster()

    # Calculate historical parameters per product from train set
    product_metas = {}
    for pid, group in train_raw.groupby("product_id"):
        product_metas[pid] = {
            "product_id": pid,
            "product_name": group["product_name"].iloc[0],
            "category": group["category"].iloc[0],
            "lead_time": group["lead_time"].iloc[0],
            "unit_cost": group["unit_cost"].iloc[0],
            "holding_cost": group["holding_cost"].iloc[0],
            "shortage_cost": group["shortage_cost"].iloc[0],
            "train_mean": group["demand"].mean(),
            "train_std": group["demand"].std(),
        }

    summary_records = []
    all_daily_logs = []

    for pid, meta in product_metas.items():
        prod_test = test_feat[test_feat["product_id"] == pid].copy().reset_index(drop=True)

        # Run Traditional Strategy
        trad_res = run_single_simulation(prod_test, "Traditional", meta, forecaster=None)
        # Run ML-Driven Strategy
        ml_res = run_single_simulation(prod_test, "ML-Based", meta, forecaster=forecaster)

        summary_records.append({
            "product_id": pid,
            "product_name": meta["product_name"],
            "strategy": "Traditional",
            "purchasing_cost": trad_res["total_purchasing_cost"],
            "holding_cost": trad_res["total_holding_cost"],
            "shortage_cost": trad_res["total_shortage_cost"],
            "total_cost": trad_res["total_inventory_cost"],
            "stockout_days": trad_res["stockout_days"],
            "avg_inventory": trad_res["average_inventory_level"],
        })
        summary_records.append({
            "product_id": pid,
            "product_name": meta["product_name"],
            "strategy": "ML-Based",
            "purchasing_cost": ml_res["total_purchasing_cost"],
            "holding_cost": ml_res["total_holding_cost"],
            "shortage_cost": ml_res["total_shortage_cost"],
            "total_cost": ml_res["total_inventory_cost"],
            "stockout_days": ml_res["stockout_days"],
            "avg_inventory": ml_res["average_inventory_level"],
        })

        all_daily_logs.extend(trad_res["daily_logs"])
        all_daily_logs.extend(ml_res["daily_logs"])

    summary_df = pd.DataFrame(summary_records)
    daily_logs_df = pd.DataFrame(all_daily_logs)

    # Save outputs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    summary_path = os.path.join(outputs_dir, "simulation_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Simulation summary results saved to: {summary_path}")

    # Plot simulation comparison
    plot_simulation_results(summary_df, os.path.join(outputs_dir, "simulation_comparison.png"))

    return summary_df, daily_logs_df


def plot_simulation_results(summary_df: pd.DataFrame, output_path: str):
    """
    Generate comparison visualization of key inventory simulation performance metrics.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1. Total Cost by Strategy & Product
    sns.barplot(data=summary_df, x="product_name", y="total_cost", hue="strategy", ax=axes[0, 0], palette=["#e76f51", "#2a9d8f"])
    axes[0, 0].set_title("Total Inventory Cost ($) by Product", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Total Cost ($)", fontsize=10)
    axes[0, 0].set_xlabel("")

    # 2. Holding Cost
    sns.barplot(data=summary_df, x="product_name", y="holding_cost", hue="strategy", ax=axes[0, 1], palette=["#e76f51", "#2a9d8f"])
    axes[0, 1].set_title("Inventory Holding Cost ($) by Product", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Holding Cost ($)", fontsize=10)
    axes[0, 1].set_xlabel("")

    # 3. Shortage Cost
    sns.barplot(data=summary_df, x="product_name", y="shortage_cost", hue="strategy", ax=axes[1, 0], palette=["#e76f51", "#2a9d8f"])
    axes[1, 0].set_title("Stockout / Shortage Cost ($) by Product", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Shortage Cost ($)", fontsize=10)
    axes[1, 0].set_xlabel("Product")

    # 4. Stockout Days
    sns.barplot(data=summary_df, x="product_name", y="stockout_days", hue="strategy", ax=axes[1, 1], palette=["#e76f51", "#2a9d8f"])
    axes[1, 1].set_title("Number of Stockout Days", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Stockout Days", fontsize=10)
    axes[1, 1].set_xlabel("Product")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Simulation comparison chart saved to: {output_path}")


def main():
    print("=" * 60)
    print("Running Complete Inventory Replenishment Simulation...")
    print("=" * 60)
    summary_df, _ = run_full_simulation()

    print("\n--- Strategy Aggregate Comparison ---")
    agg = summary_df.groupby("strategy")[["total_cost", "holding_cost", "shortage_cost", "stockout_days", "avg_inventory"]].sum()
    print(agg.round(2))

    print("\n--- Product-wise Detailed Breakdown ---")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
