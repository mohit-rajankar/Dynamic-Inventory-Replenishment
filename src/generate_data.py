"""
Data generation module for Dynamic Inventory Replenishment System.
Generates a realistic synthetic dataset of 800 records (4 products x 200 daily observations)
with seasonal, day-of-week, promotional, and price elasticity effects.
"""

import os
import numpy as np
import pandas as pd

# Set fixed random seed for reproducibility
RANDOM_SEED = 42

PRODUCTS = [
    {
        "product_id": "P001",
        "product_name": "Basmati Rice",
        "category": "Grocery",
        "base_demand": 55.0,
        "base_price": 25.0,
        "unit_cost": 18.0,
        "lead_time": 3,
        "holding_cost": 0.20,
        "shortage_cost": 8.00,
        "weekend_boost": 0.25,
        "promo_boost": 0.45,
        "price_elasticity": -0.8,
        "noise_std": 4.5,
    },
    {
        "product_id": "P002",
        "product_name": "Cooking Oil",
        "category": "Grocery",
        "base_demand": 42.0,
        "base_price": 15.0,
        "unit_cost": 10.0,
        "lead_time": 4,
        "holding_cost": 0.15,
        "shortage_cost": 6.00,
        "weekend_boost": 0.20,
        "promo_boost": 0.40,
        "price_elasticity": -0.9,
        "noise_std": 3.8,
    },
    {
        "product_id": "P003",
        "product_name": "Herbal Shampoo",
        "category": "Personal Care",
        "base_demand": 28.0,
        "base_price": 8.0,
        "unit_cost": 4.5,
        "lead_time": 5,
        "holding_cost": 0.08,
        "shortage_cost": 4.00,
        "weekend_boost": 0.15,
        "promo_boost": 0.35,
        "price_elasticity": -0.6,
        "noise_std": 3.0,
    },
    {
        "product_id": "P004",
        "product_name": "Digestive Biscuits",
        "category": "Food",
        "base_demand": 72.0,
        "base_price": 3.0,
        "unit_cost": 1.8,
        "lead_time": 2,
        "holding_cost": 0.04,
        "shortage_cost": 1.50,
        "weekend_boost": 0.30,
        "promo_boost": 0.50,
        "price_elasticity": -1.1,
        "noise_std": 5.5,
    },
]


def generate_inventory_data(n_days: int = 200, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic daily inventory sales and replenishment data for 4 products.

    Parameters:
        n_days (int): Number of consecutive observation days per product (default 200).
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Clean dataset with 4 * n_days records and required schema.
    """
    rng = np.random.default_rng(seed)
    start_date = pd.Timestamp("2025-01-01")
    dates = [start_date + pd.Timedelta(days=i) for i in range(n_days)]

    records = []

    for prod in PRODUCTS:
        # Determine promotional days: ~18% probability of promotional campaigns
        # Promotions run in small clusters (1-3 days) for realism
        promo_prob = 0.18
        promo_flags = (rng.uniform(0, 1, n_days) < promo_prob).astype(int)

        for i, dt in enumerate(dates):
            dow = dt.day_of_week  # 0=Monday, 6=Sunday
            month = dt.month
            is_weekend = 1 if dow in (5, 6) else 0
            is_promo = int(promo_flags[i])

            # Price dynamics: slight natural fluctuation plus discount during promotion
            base_p = prod["base_price"]
            if is_promo:
                # 10% to 15% discount during promotions
                discount = rng.uniform(0.10, 0.15)
                selling_price = round(base_p * (1.0 - discount), 2)
            else:
                # Small standard market price variance (+/- 2%)
                noise_price = rng.uniform(-0.02, 0.02)
                selling_price = round(base_p * (1.0 + noise_price), 2)

            # Demand components:
            # 1. Base demand
            d = prod["base_demand"]

            # 2. Weekend multiplier
            if is_weekend:
                d *= (1.0 + prod["weekend_boost"])

            # 3. Monthly/Seasonal wave (gentle sinusoidal variation representing consumer cycles)
            # Cycle through months with peak in spring/early summer
            seasonal_factor = 1.0 + 0.08 * np.sin(2 * np.pi * (dt.day_of_year) / 365.25)
            d *= seasonal_factor

            # 4. Price elasticity effect: relative price change against base price
            price_ratio = selling_price / base_p
            elasticity_multiplier = (price_ratio) ** prod["price_elasticity"]
            d *= elasticity_multiplier

            # 5. Promotional uplift
            if is_promo:
                d *= (1.0 + prod["promo_boost"])

            # 6. Random fluctuation / consumer noise
            noise = rng.normal(0, prod["noise_std"])
            raw_demand = d + noise

            # Ensure non-negative integer demand
            final_demand = int(max(0, round(raw_demand)))

            records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "product_id": prod["product_id"],
                "product_name": prod["product_name"],
                "category": prod["category"],
                "price": selling_price,
                "promotion": is_promo,
                "day_of_week": dow,
                "month": month,
                "demand": final_demand,
                "lead_time": prod["lead_time"],
                "unit_cost": prod["unit_cost"],
                "holding_cost": prod["holding_cost"],
                "shortage_cost": prod["shortage_cost"],
            })

    df = pd.DataFrame(records)
    # Sort chronologically by product and date
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["product_id", "date"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate the generated dataset against project requirements.
    """
    required_cols = [
        "date", "product_id", "product_name", "category",
        "price", "promotion", "day_of_week", "month", "demand",
        "lead_time", "unit_cost", "holding_cost", "shortage_cost"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    null_counts = df.isnull().sum().to_dict()
    duplicates = int(df.duplicated(subset=["date", "product_id"]).sum())
    negative_demands = int((df["demand"] < 0).sum())

    summary = {
        "total_records": len(df),
        "products_count": df["product_id"].nunique(),
        "records_per_product": df.groupby("product_id").size().to_dict(),
        "missing_columns": missing_cols,
        "total_null_values": sum(null_counts.values()),
        "duplicate_records": duplicates,
        "negative_demand_count": negative_demands,
        "date_range": (df["date"].min(), df["date"].max()),
        "is_valid": len(missing_cols) == 0 and sum(null_counts.values()) == 0 and duplicates == 0 and negative_demands == 0
    }
    return summary


def main():
    print("=" * 60)
    print("Generating Synthetic Inventory Dataset...")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "inventory_data.csv")

    df = generate_inventory_data(n_days=200, seed=RANDOM_SEED)
    validation = validate_dataset(df)

    print(f"Total Records Generated : {validation['total_records']}")
    print(f"Unique Products         : {validation['products_count']}")
    print(f"Records Per Product     : {validation['records_per_product']}")
    print(f"Date Range              : {validation['date_range'][0]} to {validation['date_range'][1]}")
    print(f"Duplicate Count         : {validation['duplicate_records']}")
    print(f"Total Null Values       : {validation['total_null_values']}")
    print(f"Negative Demand Count   : {validation['negative_demand_count']}")
    print(f"Validation Passed       : {validation['is_valid']}")

    if not validation["is_valid"]:
        raise ValueError("Dataset validation failed! Please check constraints.")

    df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Dataset successfully saved to: {output_path}")

    # Display descriptive statistics
    print("\n--- Summary Statistics (Numeric Columns) ---")
    numeric_cols = ["price", "demand", "lead_time", "unit_cost", "holding_cost", "shortage_cost"]
    print(df[numeric_cols].describe().round(2))


if __name__ == "__main__":
    main()
