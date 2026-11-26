"""
Exploratory Data Analysis (EDA) module for Dynamic Inventory Replenishment System.
Generates focused, high-clarity visualizations and statistical summaries saved to outputs/eda/.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure headless matplotlib environment
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.preprocessing import load_data


def generate_eda(output_dir: str = None) -> dict:
    """
    Perform complete exploratory data analysis and save visual charts.
    """
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "outputs", "eda")

    os.makedirs(output_dir, exist_ok=True)
    df = load_data()

    # Apply clean presentation style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 150

    charts_generated = []

    # 1. Overall Demand Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["demand"], kde=True, color="#2b5c8f", bins=30)
    plt.title("Distribution of Daily Product Demand", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Daily Demand (Units)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.axvline(df["demand"].mean(), color="red", linestyle="--", label=f"Mean: {df['demand'].mean():.1f}")
    plt.axvline(df["demand"].median(), color="green", linestyle=":", label=f"Median: {df['demand'].median():.1f}")
    plt.legend(frameon=True)
    plt.tight_layout()
    chart1 = os.path.join(output_dir, "01_demand_distribution.png")
    plt.savefig(chart1)
    plt.close()
    charts_generated.append(chart1)

    # 2. Product-wise Demand Comparison
    plt.figure(figsize=(9, 5))
    order = list(df.groupby("product_name")["demand"].median().sort_values(ascending=False).index)
    sns.boxplot(data=df, x="product_name", y="demand", order=order, hue="product_name", palette="Set2", legend=False)
    plt.title("Product-wise Demand Comparison", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Product", fontsize=11)
    plt.ylabel("Daily Demand (Units)", fontsize=11)
    plt.tight_layout()
    chart2 = os.path.join(output_dir, "02_product_demand_comparison.png")
    plt.savefig(chart2)
    plt.close()
    charts_generated.append(chart2)

    # 3. Daily and Monthly Demand Patterns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_df = df.groupby("day_of_week")["demand"].mean().reset_index()
    sns.barplot(data=dow_df, x="day_of_week", y="demand", ax=axes[0], color="#3a86c8")
    axes[0].set_xticks(range(len(day_labels)))
    axes[0].set_xticklabels(day_labels)
    axes[0].set_title("Average Demand by Day of Week", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Day of Week", fontsize=11)
    axes[0].set_ylabel("Average Demand (Units)", fontsize=11)

    month_df = df.groupby("month")["demand"].mean().reset_index()
    sns.lineplot(data=month_df, x="month", y="demand", ax=axes[1], marker="o", color="#e76f51", linewidth=2.5)
    axes[1].set_title("Average Demand Trend by Month", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Month Number", fontsize=11)
    axes[1].set_ylabel("Average Demand (Units)", fontsize=11)
    plt.tight_layout()
    chart3 = os.path.join(output_dir, "03_day_and_month_patterns.png")
    plt.savefig(chart3)
    plt.close()
    charts_generated.append(chart3)

    # 4. Impact of Promotional Activities
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="product_name", y="demand", hue="promotion", palette=["#708090", "#e63946"])
    plt.title("Impact of Promotions on Daily Product Demand", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Product", fontsize=11)
    plt.ylabel("Average Demand (Units)", fontsize=11)
    plt.legend(title="Promotion Active", labels=["No (0)", "Yes (1)"], frameon=True)
    plt.tight_layout()
    chart4 = os.path.join(output_dir, "04_promotion_impact.png")
    plt.savefig(chart4)
    plt.close()
    charts_generated.append(chart4)

    # 5. Price vs. Demand Relationship
    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=df, x="price", y="demand", hue="product_name", palette="tab10", alpha=0.75, s=60)
    plt.title("Price vs. Daily Demand (by Product)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Selling Price ($)", fontsize=11)
    plt.ylabel("Daily Demand (Units)", fontsize=11)
    plt.legend(title="Product", frameon=True)
    plt.tight_layout()
    chart5 = os.path.join(output_dir, "05_price_vs_demand.png")
    plt.savefig(chart5)
    plt.close()
    charts_generated.append(chart5)

    # 6. Product-wise Demand Trends Over Time
    plt.figure(figsize=(13, 6))
    for prod_name, group in df.groupby("product_name"):
        plt.plot(group["date"], group["demand"], label=prod_name, alpha=0.8, linewidth=1.5)
    plt.title("200-Day Daily Demand Trajectory Across Products", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Observation Date", fontsize=11)
    plt.ylabel("Daily Demand (Units)", fontsize=11)
    plt.legend(title="Product", frameon=True)
    plt.xticks(rotation=30)
    plt.tight_layout()
    chart6 = os.path.join(output_dir, "06_demand_time_series.png")
    plt.savefig(chart6)
    plt.close()
    charts_generated.append(chart6)

    # Compute key statistical findings
    findings = {
        "overall_mean_demand": round(float(df["demand"].mean()), 2),
        "overall_std_demand": round(float(df["demand"].std()), 2),
        "product_means": df.groupby("product_name")["demand"].mean().round(2).to_dict(),
        "promo_lift_pct": round(
            float((df[df["promotion"] == 1]["demand"].mean() - df[df["promotion"] == 0]["demand"].mean()) /
                  df[df["promotion"] == 0]["demand"].mean() * 100), 2
        ),
        "weekend_lift_pct": round(
            float((df[df["day_of_week"].isin([5, 6])]["demand"].mean() - df[~df["day_of_week"].isin([5, 6])]["demand"].mean()) /
                  df[~df["day_of_week"].isin([5, 6])]["demand"].mean() * 100), 2
        ),
        "charts_generated": charts_generated
    }

    return findings


def main():
    print("=" * 60)
    print("Running Exploratory Data Analysis (EDA)...")
    print("=" * 60)
    findings = generate_eda()
    print(f"Overall Mean Demand       : {findings['overall_mean_demand']} units")
    print(f"Overall Demand Std Dev    : {findings['overall_std_demand']} units")
    print(f"Promotion Demand Lift     : +{findings['promo_lift_pct']}%")
    print(f"Weekend Demand Lift       : +{findings['weekend_lift_pct']}%")
    print("\nProduct Average Demand:")
    for prod, val in findings["product_means"].items():
        print(f"  - {prod:20s}: {val:.1f} units/day")
    print(f"\nSaved {len(findings['charts_generated'])} charts to outputs/eda/")


if __name__ == "__main__":
    main()
