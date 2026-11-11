"""End-to-end verification checks for the Nassau Candy Optimization project."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optimizer import solve_factory_reallocation


def main() -> None:
    # 1. Validated dataset checks
    validated_path = ROOT / "data" / "processed" / "nassau_cleaned_validated.csv"
    assert validated_path.exists(), f"Missing {validated_path}"
    df = pd.read_csv(validated_path)
    assert len(df) == 10194, f"Expected 10194 rows, got {len(df)}"
    assert df["clean_order_date"].isna().sum() == 0, "clean_order_date has NaNs"
    assert df["clean_ship_date"].isna().sum() == 0, "clean_ship_date has NaNs"
    assert df["clean_lead_time_days"].between(0, 15).all(), "clean_lead_time_days out of [0, 15]"

    # 2. Optimization invariants
    opt = solve_factory_reallocation(df)
    summary = opt["summary"]
    assert summary["lp_status"] == 0, f"LP did not terminate optimally: {summary['lp_message']}"
    assert np.isclose(
        opt["lane_allocations"]["Allocated_Units"].sum(), df["Units"].sum(), atol=1e-4
    ), "Total allocated units mismatch total demand"
    assert (
        opt["factory_summary"]["Optimized_Units"]
        <= opt["factory_summary"]["Capacity_Units"] + 1e-4
    ).all(), "Factory capacity exceeded"
    assert summary["optimized_freight_usd"] < summary["baseline_freight_usd"], "No freight savings"

    # 3. Saved ML model checks
    lt_model = joblib.load(ROOT / "models" / "lead_time_model.joblib")
    risk_model = joblib.load(ROOT / "models" / "freight_risk_model.joblib")
    meta = json.loads((ROOT / "models" / "model_metadata.json").read_text())
    assert meta["lead_time_r2"] > 0.65, "Lead time R2 unexpectedly low"
    assert meta["risk_roc_auc"] > 0.95, "Risk ROC-AUC unexpectedly low"

    sample_lt = df[
        [
            "Ship Mode",
            "Region",
            "Division",
            "Product Name",
            "Units",
            "Sales",
            "order_month",
            "order_dayofweek",
            "is_weekend_order",
            "dest_lat",
            "dest_lon",
            "baseline_distance_miles",
        ]
    ].head(5)
    lt_preds = lt_model.predict(sample_lt)
    assert len(lt_preds) == 5 and np.all(np.isfinite(lt_preds)) and np.all(lt_preds >= 0)

    sample_risk = df[
        [
            "Ship Mode",
            "Region",
            "Division",
            "Product Name",
            "Units",
            "Sales",
            "order_month",
            "dest_lat",
            "dest_lon",
        ]
    ].head(5)
    risk_probs = risk_model.predict_proba(sample_risk)[:, 1]
    assert len(risk_probs) == 5 and np.all((risk_probs >= 0.0) & (risk_probs <= 1.0))

    # 4. Required output artifacts check
    required_files = [
        ROOT / "app.py",
        ROOT / "requirements.txt",
        ROOT / "notebooks" / "04_date_resolution_and_feature_engineering.ipynb",
        ROOT / "notebooks" / "05_factory_optimization_and_ml.ipynb",
        ROOT / "outputs" / "optimization" / "baseline_vs_optimized_summary.csv",
        ROOT / "outputs" / "optimization" / "state_factory_reallocation_plan.csv",
        ROOT / "outputs" / "optimization" / "ml_model_metrics.csv",
    ]
    for path in required_files:
        assert path.exists(), f"Missing artifact: {path}"

    print("ALL VERIFICATION CHECKS PASSED")


if __name__ == "__main__":
    main()
