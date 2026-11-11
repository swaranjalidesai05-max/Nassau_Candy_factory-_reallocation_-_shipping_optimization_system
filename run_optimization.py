"""
Step 5: Factory Reallocation Linear Programming Optimization & Predictive ML Models
for Nassau Candy Distributor.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.optimizer import solve_factory_reallocation

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "nassau_cleaned_validated.csv"
OUT_DIR = BASE_DIR / "outputs" / "optimization"
PLOT_DIR = OUT_DIR / "plots"
MODEL_DIR = BASE_DIR / "models"


def build_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    """Create shared preprocessing pipeline for categorical and numeric features."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )


def train_lead_time_models(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    """Train and benchmark ML regressors predicting clean_lead_time_days."""
    cat_cols = ["Ship Mode", "Region", "Division", "Product Name"]
    num_cols = [
        "Units",
        "Sales",
        "order_month",
        "order_dayofweek",
        "is_weekend_order",
        "dest_lat",
        "dest_lon",
        "baseline_distance_miles",
    ]
    feature_cols = cat_cols + num_cols
    X = df[feature_cols]
    y = df["clean_lead_time_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["Ship Mode"]
    )

    candidates = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, learning_rate=0.05, random_state=42
        ),
    }

    metrics = []
    fitted: dict[str, Pipeline] = {}
    for name, est in candidates.items():
        pipe = Pipeline(
            [("prep", build_preprocessor(cat_cols, num_cols)), ("model", est)]
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        metrics.append(
            {
                "Task": "Lead Time Regression",
                "Model": name,
                "Primary_Metric_Name": "RMSE (Days)",
                "Primary_Metric_Value": rmse,
                "MAE_Days": mae,
                "R2": r2,
            }
        )
        fitted[name] = pipe

    metrics_df = pd.DataFrame(metrics).sort_values("Primary_Metric_Value")
    best_name = metrics_df.iloc[0]["Model"]
    best_pipe = fitted[best_name]

    perm = permutation_importance(
        best_pipe, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )
    imp_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance_Mean": perm.importances_mean,
            "Importance_Std": perm.importances_std,
        }
    ).sort_values("Importance_Mean", ascending=False)
    return best_pipe, metrics_df, imp_df


def train_freight_risk_models(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    """Train classifiers flagging orders where baseline freight >= 20% of Gross Profit."""
    cat_cols = ["Ship Mode", "Region", "Division", "Product Name"]
    num_cols = ["Units", "Sales", "order_month", "dest_lat", "dest_lon"]
    feature_cols = cat_cols + num_cols

    X = df[feature_cols]
    y = (df["freight_pct_of_gross_profit"] >= 20.0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting Classifier": HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05, random_state=42
        ),
    }

    metrics = []
    fitted: dict[str, Pipeline] = {}
    for name, est in candidates.items():
        pipe = Pipeline(
            [("prep", build_preprocessor(cat_cols, num_cols)), ("model", est)]
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, probs))
        f1 = float(f1_score(y_test, preds))
        acc = float(accuracy_score(y_test, preds))
        metrics.append(
            {
                "Task": "Freight Erosion Risk (>=20% Profit)",
                "Model": name,
                "Primary_Metric_Name": "ROC-AUC",
                "Primary_Metric_Value": roc_auc,
                "F1_Score": f1,
                "Accuracy": acc,
            }
        )
        fitted[name] = pipe

    metrics_df = pd.DataFrame(metrics).sort_values("Primary_Metric_Value", ascending=False)
    best_name = metrics_df.iloc[0]["Model"]
    return fitted[best_name], metrics_df


def save_optimization_plots(
    opt_res: dict, lt_imp_df: pd.DataFrame
) -> None:
    """Generate clean summary plots for optimization and ML findings."""
    sns.set_theme(style="whitegrid")

    # 1. Baseline vs Optimized Freight by Region
    reg_df = opt_res["region_summary"].melt(
        id_vars="Region",
        value_vars=["Baseline_Freight", "Optimized_Freight"],
        var_name="Scenario",
        value_name="Freight_USD",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=reg_df, x="Region", y="Freight_USD", hue="Scenario", palette=["#c0392b", "#27ae60"])
    plt.title("Modeled Freight Cost by Region: Baseline vs Optimized Reallocation")
    plt.ylabel("Total Freight Cost (USD)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "freight_by_region_comparison.png", dpi=150)
    plt.close()

    # 2. Factory Capacity Utilization Comparison
    fac_df = opt_res["factory_summary"].melt(
        id_vars="Factory",
        value_vars=["Baseline_Utilization_Pct", "Optimized_Utilization_Pct"],
        var_name="Scenario",
        value_name="Utilization_Pct",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=fac_df, x="Factory", y="Utilization_Pct", hue="Scenario", palette=["#7f8c8d", "#2980b9"])
    plt.axhline(100.0, color="#c0392b", linestyle="--", linewidth=1.2, label="100% Cap")
    plt.title("Factory Capacity Utilization (%): Baseline vs Optimized")
    plt.ylabel("Utilization (%)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "factory_utilization_comparison.png", dpi=150)
    plt.close()

    # 3. Top 10 States by Freight Savings
    top_states = (
        opt_res["state_recommendations"]
        .groupby("State/Province", as_index=False)["Freight_Savings_USD"]
        .sum()
        .sort_values("Freight_Savings_USD", ascending=False)
        .head(10)
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_states, x="Freight_Savings_USD", y="State/Province", color="#16a085")
    plt.title("Top 10 States/Provinces by Modeled Freight Cost Savings")
    plt.xlabel("Freight Savings (USD)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "top_10_states_savings.png", dpi=150)
    plt.close()

    # 4. Lead Time Model Feature Importance
    plt.figure(figsize=(9, 5))
    sns.barplot(data=lt_imp_df.head(10), x="Importance_Mean", y="Feature", color="#34495e")
    plt.title("Lead-Time Prediction Model: Permutation Feature Importance")
    plt.xlabel("Mean Permutation Importance (R² Drop)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "lead_time_feature_importance.png", dpi=150)
    plt.close()


def main() -> None:
    print("--- STEP 5: FACTORY REALLOCATION OPTIMIZATION & ML MODELING ---")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # 1. Solve Capacitated LP Optimization
    opt_res = solve_factory_reallocation(df)
    summary = opt_res["summary"]

    pd.DataFrame([summary]).to_csv(OUT_DIR / "baseline_vs_optimized_summary.csv", index=False)
    opt_res["factory_summary"].to_csv(OUT_DIR / "factory_utilization_comparison.csv", index=False)
    opt_res["region_summary"].to_csv(OUT_DIR / "region_optimization_summary.csv", index=False)
    opt_res["product_summary"].to_csv(OUT_DIR / "product_optimization_summary.csv", index=False)
    opt_res["state_recommendations"].to_csv(OUT_DIR / "state_factory_reallocation_plan.csv", index=False)
    opt_res["lane_allocations"].to_csv(OUT_DIR / "lane_allocations.csv", index=False)

    print(
        f"LP Optimization Status: {summary['lp_message']}\n"
        f"Avg Haul Reduced: {summary['baseline_avg_distance_miles']:.1f} -> "
        f"{summary['optimized_avg_distance_miles']:.1f} miles "
        f"(-{summary['unit_miles_reduction_pct']:.1f}%)\n"
        f"Modeled Freight Reduced: ${summary['baseline_freight_usd']:,.2f} -> "
        f"${summary['optimized_freight_usd']:,.2f} "
        f"(Savings: ${summary['freight_savings_usd']:,.2f} / {summary['freight_savings_pct']:.1f}%)\n"
        f"Reallocated Volume: {summary['reallocated_units']:,.0f} units "
        f"({summary['reallocated_units_pct']:.1f}%)"
    )

    # 2. Train ML Models
    best_lt_model, lt_metrics, lt_imp_df = train_lead_time_models(df)
    best_risk_model, risk_metrics = train_freight_risk_models(df)

    ml_metrics = pd.concat([lt_metrics, risk_metrics], ignore_index=True)
    ml_metrics.to_csv(OUT_DIR / "ml_model_metrics.csv", index=False)
    lt_imp_df.to_csv(OUT_DIR / "lead_time_feature_importance.csv", index=False)

    joblib.dump(best_lt_model, MODEL_DIR / "lead_time_model.joblib")
    joblib.dump(best_risk_model, MODEL_DIR / "freight_risk_model.joblib")
    metadata = {
        "best_lead_time_model": lt_metrics.iloc[0]["Model"],
        "lead_time_rmse": float(lt_metrics.iloc[0]["Primary_Metric_Value"]),
        "lead_time_r2": float(lt_metrics.iloc[0]["R2"]),
        "best_risk_model": risk_metrics.iloc[0]["Model"],
        "risk_roc_auc": float(risk_metrics.iloc[0]["Primary_Metric_Value"]),
        "risk_f1": float(risk_metrics.iloc[0]["F1_Score"]),
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\nML Model Benchmark:")
    print(ml_metrics.to_string(index=False))

    # 3. Generate Visualizations
    save_optimization_plots(opt_res, lt_imp_df)
    print("\nSTEP 5 OPTIMIZATION & ML COMPLETED")


if __name__ == "__main__":
    main()
