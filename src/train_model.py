"""
Model training and evaluation module for Dynamic Inventory Replenishment System.
Trains Linear Regression (Baseline) and Random Forest Regressor, compares them on chronological test set,
and serializes the winning model using Joblib.
"""

import sys
import os
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
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.preprocessing import load_data
from src.feature_engineering import prepare_train_test_data, FEATURE_COLS


def build_preprocessor() -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for one-hot encoding product IDs
    and scaling numerical features.
    """
    categorical_cols = ["product_id"]
    numerical_cols = [c for c in FEATURE_COLS if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ]
    )
    return preprocessor


def evaluate_model(name: str, model: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    """
    Train model pipeline and compute MAE, RMSE, and R2 on both train and test splits.
    """
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Predictions cannot be negative in retail demand
    train_preds = np.clip(train_preds, a_min=0, a_max=None)
    test_preds = np.clip(test_preds, a_min=0, a_max=None)

    metrics = {
        "model_name": name,
        "train_mae": round(float(mean_absolute_error(y_train, train_preds)), 3),
        "train_rmse": round(float(root_mean_squared_error(y_train, train_preds)), 3),
        "train_r2": round(float(r2_score(y_train, train_preds)), 4),
        "test_mae": round(float(mean_absolute_error(y_test, test_preds)), 3),
        "test_rmse": round(float(root_mean_squared_error(y_test, test_preds)), 3),
        "test_r2": round(float(r2_score(y_test, test_preds)), 4),
    }
    return metrics, test_preds


def plot_model_comparison(results_df: pd.DataFrame, output_path: str):
    """
    Plot test-set evaluation metrics comparison between models.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    metrics_to_plot = [
        ("test_mae", "Mean Absolute Error (MAE)", "Lower is Better", "#2a9d8f"),
        ("test_rmse", "Root Mean Squared Error (RMSE)", "Lower is Better", "#e76f51"),
        ("test_r2", "R² Score (Goodness of Fit)", "Higher is Better", "#457b9d"),
    ]

    for idx, (col, title, note, color) in enumerate(metrics_to_plot):
        bars = axes[idx].bar(results_df["model_name"], results_df[col], color=color, width=0.45, edgecolor="black")
        axes[idx].set_title(f"{title}\n({note})", fontsize=11, fontweight="bold")
        axes[idx].set_ylabel("Score", fontsize=10)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.7)

        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            axes[idx].annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_and_compare() -> Tuple[pd.DataFrame, Pipeline, str]:
    """
    Execute full training and comparison pipeline.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    raw_df = load_data()
    X_train, X_test, y_train, y_test = prepare_train_test_data(raw_df, test_ratio=0.20)

    # 1. Linear Regression Baseline Pipeline
    lr_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", LinearRegression()),
    ])

    # 2. Random Forest Regressor Pipeline
    rf_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
    ])

    print("Training Linear Regression (Baseline)...")
    lr_metrics, lr_preds = evaluate_model("Linear Regression", lr_pipeline, X_train, y_train, X_test, y_test)

    print("Training Random Forest Regressor (Primary)...")
    rf_metrics, rf_preds = evaluate_model("Random Forest Regressor", rf_pipeline, X_train, y_train, X_test, y_test)

    results = [lr_metrics, rf_metrics]
    results_df = pd.DataFrame(results)

    # Save comparison CSV
    csv_path = os.path.join(outputs_dir, "model_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nModel comparison table saved to: {csv_path}")

    # Plot comparison chart
    chart_path = os.path.join(outputs_dir, "model_comparison.png")
    plot_model_comparison(results_df, chart_path)
    print(f"Model comparison chart saved to: {chart_path}")

    # Select best model based on Test RMSE and Test R2
    # The lower the test RMSE and higher test R2, the better.
    if rf_metrics["test_rmse"] <= lr_metrics["test_rmse"]:
        best_model = rf_pipeline
        best_name = "Random Forest Regressor"
    else:
        best_model = lr_pipeline
        best_name = "Linear Regression"

    model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\n[WINNER] Selected: {best_name} (Test RMSE: {min(lr_metrics['test_rmse'], rf_metrics['test_rmse'])})")
    print(f"Saved winning model artifact to: {model_path}")

    return results_df, best_model, best_name


def main():
    print("=" * 60)
    print("Executing Demand Forecasting Model Training & Evaluation")
    print("=" * 60)
    results_df, _, best_name = train_and_compare()
    print("\n--- Model Evaluation Summary ---")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
