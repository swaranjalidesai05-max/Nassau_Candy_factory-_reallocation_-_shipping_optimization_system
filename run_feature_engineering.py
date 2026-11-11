"""
Step 4A: Date Anomaly Resolution & Feature Engineering for Nassau Candy Distributor.

Resolves the two systematic date issues discovered in Steps 2-3:
  1. Raw dates are formatted DD-MM-YYYY (%d-%m-%Y); parsing without day-first
     format previously produced 6,064 false NaTs (59.49% of Order Dates).
  2. Order Date years were shifted to 2024-2025 vs the true Order ID year (2021-2024),
     and Ship Date was shifted forward by exactly +2,000 days.

Exports a fully validated dataset for optimization and ML modeling without
modifying the raw CSV.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.geo_and_network import (
    BASELINE_PRODUCT_FACTORY,
    FACTORIES,
    STATE_CENTROIDS,
    estimate_freight_cost,
    haversine_miles,
)

BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "Nassau Candy Distributor.csv"
OUT_DATA = BASE_DIR / "data" / "processed" / "nassau_cleaned_validated.csv"
OUT_DIR = BASE_DIR / "outputs" / "feature_engineering"
PLOT_DIR = OUT_DIR / "plots"


def resolve_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse DD-MM-YYYY dates and reverse the systematic +2,000-day / year shifts."""
    df = df.copy()
    df["raw_order_date_str"] = df["Order Date"].astype(str)
    df["raw_ship_date_str"] = df["Ship Date"].astype(str)

    parsed_order = pd.to_datetime(df["Order Date"], format="%d-%m-%Y", errors="raise")
    parsed_ship = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y", errors="raise")

    df["raw_order_date"] = parsed_order
    df["raw_ship_date"] = parsed_ship
    df["raw_lead_time_days"] = (parsed_ship - parsed_order).dt.days

    # Extract true order year encoded in Order ID (e.g., US-2021-...)
    order_id_year = df["Order ID"].str.slice(3, 7).astype(int)
    df["order_id_year"] = order_id_year

    # Restore Order Date to its true Order ID year (no Feb 29 dates exist in dataset)
    df["clean_order_date"] = [
        dt.replace(year=int(yr)) for dt, yr in zip(parsed_order, order_id_year)
    ]
    df["clean_order_date"] = pd.to_datetime(df["clean_order_date"])

    # Reverse the +2,000-day forward shift on Ship Date
    df["clean_ship_date"] = parsed_ship - pd.Timedelta(days=2000)
    df["clean_lead_time_days"] = (df["clean_ship_date"] - df["clean_order_date"]).dt.days
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, unit-economics, geographic, and baseline routing features."""
    df = df.copy()

    # Calendar features from clean_order_date
    df["order_year"] = df["clean_order_date"].dt.year
    df["order_month"] = df["clean_order_date"].dt.month
    df["order_quarter"] = df["clean_order_date"].dt.quarter
    df["order_dayofweek"] = df["clean_order_date"].dt.dayofweek
    df["is_weekend_order"] = df["order_dayofweek"].isin([5, 6]).astype(int)

    # Unit economics
    df["unit_price"] = df["Sales"] / df["Units"]
    df["unit_cost"] = df["Cost"] / df["Units"]
    df["unit_profit"] = df["Gross Profit"] / df["Units"]
    df["profit_margin_pct"] = (df["Gross Profit"] / df["Sales"]) * 100.0

    # Geographic centroids
    missing_states = set(df["State/Province"].unique()) - set(STATE_CENTROIDS.keys())
    if missing_states:
        raise ValueError(f"Missing centroid definitions for states: {missing_states}")

    df["dest_lat"] = df["State/Province"].map(lambda s: STATE_CENTROIDS[s][0])
    df["dest_lon"] = df["State/Province"].map(lambda s: STATE_CENTROIDS[s][1])

    # Baseline factory routing
    missing_products = set(df["Product Name"].unique()) - set(BASELINE_PRODUCT_FACTORY.keys())
    if missing_products:
        raise ValueError(f"Missing baseline factory mapping for products: {missing_products}")

    df["baseline_factory"] = df["Product Name"].map(BASELINE_PRODUCT_FACTORY)
    df["baseline_factory_lat"] = df["baseline_factory"].map(lambda f: FACTORIES[f]["lat"])
    df["baseline_factory_lon"] = df["baseline_factory"].map(lambda f: FACTORIES[f]["lon"])
    df["baseline_factory_region"] = df["baseline_factory"].map(lambda f: FACTORIES[f]["region"])
    df["is_cross_region_baseline"] = (df["Region"] != df["baseline_factory_region"]).astype(int)

    df["baseline_distance_miles"] = haversine_miles(
        df["dest_lat"].to_numpy(),
        df["dest_lon"].to_numpy(),
        df["baseline_factory_lat"].to_numpy(),
        df["baseline_factory_lon"].to_numpy(),
    )
    df["modeled_baseline_freight"] = estimate_freight_cost(
        df["Units"].to_numpy(),
        df["baseline_distance_miles"].to_numpy(),
        df["Ship Mode"],
    )
    df["net_profit_after_freight"] = df["Gross Profit"] - df["modeled_baseline_freight"]
    df["freight_pct_of_gross_profit"] = np.where(
        df["Gross Profit"] > 0,
        (df["modeled_baseline_freight"] / df["Gross Profit"]) * 100.0,
        100.0,
    )
    return df


def save_outputs(df: pd.DataFrame) -> None:
    """Persist validated dataset, audit recovery tables, and plots."""
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUT_DATA, index=False)

    # Recovery summary table
    recovery_summary = pd.DataFrame(
        [
            {
                "Metric": "Total Rows",
                "Before_Stage2_Misparse": len(df),
                "After_Stage4_Validation": len(df),
            },
            {
                "Metric": "Invalid / Missing Order Dates",
                "Before_Stage2_Misparse": 6064,
                "After_Stage4_Validation": int(df["clean_order_date"].isna().sum()),
            },
            {
                "Metric": "Invalid / Missing Ship Dates",
                "Before_Stage2_Misparse": 0,
                "After_Stage4_Validation": int(df["clean_ship_date"].isna().sum()),
            },
            {
                "Metric": "Median Lead Time (Days)",
                "Before_Stage2_Misparse": float(df["raw_lead_time_days"].median()),
                "After_Stage4_Validation": float(df["clean_lead_time_days"].median()),
            },
            {
                "Metric": "Min Lead Time (Days)",
                "Before_Stage2_Misparse": float(df["raw_lead_time_days"].min()),
                "After_Stage4_Validation": float(df["clean_lead_time_days"].min()),
            },
            {
                "Metric": "Max Lead Time (Days)",
                "Before_Stage2_Misparse": float(df["raw_lead_time_days"].max()),
                "After_Stage4_Validation": float(df["clean_lead_time_days"].max()),
            },
        ]
    )
    recovery_summary.to_csv(OUT_DIR / "date_recovery_summary.csv", index=False)

    lt_by_mode = (
        df.groupby("Ship Mode", as_index=False)["clean_lead_time_days"]
        .agg(["count", "min", "mean", "median", "std", "max"])
        .reset_index()
    )
    lt_by_mode.to_csv(OUT_DIR / "clean_lead_time_by_ship_mode.csv", index=False)

    sns.set_theme(style="whitegrid")

    # Plot 1: Raw vs Clean Lead Time Distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(df["raw_lead_time_days"], bins=40, ax=axes[0], color="#c0392b")
    axes[0].set_title("Raw Lead Time (Corrupted +2,000-Day / Year Shift)")
    axes[0].set_xlabel("Days")

    sns.countplot(x="clean_lead_time_days", data=df, ax=axes[1], color="#27ae60")
    axes[1].set_title("Validated Lead Time After Date Shift Recovery")
    axes[1].set_xlabel("Days")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "raw_vs_clean_lead_time.png", dpi=150)
    plt.close()

    # Plot 2: Clean Lead Time by Ship Mode
    plt.figure(figsize=(9, 5))
    order = ["Same Day", "First Class", "Second Class", "Standard Class"]
    sns.boxplot(data=df, x="Ship Mode", y="clean_lead_time_days", order=order, palette="viridis", hue="Ship Mode", legend=False)
    plt.title("Validated Lead Time (Days) by Ship Mode")
    plt.ylabel("Clean Lead Time (Days)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "clean_lead_time_by_ship_mode.png", dpi=150)
    plt.close()

    # Plot 3: Baseline Shipping Distance by Region
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="Region", y="baseline_distance_miles", palette="Set2", hue="Region", legend=False)
    plt.title("Baseline Shipping Distance (Miles) by Destination Region")
    plt.ylabel("Distance from Baseline Factory (Miles)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "baseline_distance_by_region.png", dpi=150)
    plt.close()


def main() -> None:
    print("--- STEP 4A: DATE RESOLUTION & FEATURE ENGINEERING ---")
    df_raw = pd.read_csv(RAW_CSV)
    df = resolve_dates(df_raw)
    df = engineer_features(df)
    save_outputs(df)
    print(f"Validated dataset saved: {OUT_DATA} | Shape: {df.shape}")
    print(
        f"Clean lead time range: {df['clean_lead_time_days'].min()} to "
        f"{df['clean_lead_time_days'].max()} days (median {df['clean_lead_time_days'].median():.1f})"
    )
    print(
        f"Cross-region baseline orders: {df['is_cross_region_baseline'].mean()*100:.1f}% | "
        f"Avg baseline haul: {df['baseline_distance_miles'].mean():.1f} miles"
    )
    print("STEP 4A COMPLETED")


if __name__ == "__main__":
    main()
