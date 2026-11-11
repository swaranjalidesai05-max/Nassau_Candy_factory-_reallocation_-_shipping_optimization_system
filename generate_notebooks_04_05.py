"""Generate Jupyter notebooks 04 and 05 for Steps 4-5."""

from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent / "notebooks"


def make_notebook(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for cell_type, source in cells:
        lines = [line + "\n" for line in source.splitlines()]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": lines,
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)

    nb04 = make_notebook(
        [
            (
                "markdown",
                "# Step 4: Date Anomaly Resolution & Feature Engineering\n\n"
                "This notebook resolves the `DD-MM-YYYY` parsing mismatch and the systematic "
                "`+2,000-day` Ship Date / `Order ID` year shifts identified in Steps 2-3, "
                "and engineers geographic & unit-economics features for optimization and ML.",
            ),
            (
                "code",
                "import sys\nfrom pathlib import Path\n\n"
                "ROOT = Path('..').resolve()\n"
                "if str(ROOT) not in sys.path:\n"
                "    sys.path.insert(0, str(ROOT))\n\n"
                "import pandas as pd\n"
                "from run_feature_engineering import resolve_dates, engineer_features\n\n"
                "df_raw = pd.read_csv(ROOT / 'data' / 'Nassau Candy Distributor.csv')\n"
                "df = engineer_features(resolve_dates(df_raw))\n"
                "print('Validated shape:', df.shape)\n"
                "df[['Order ID', 'clean_order_date', 'clean_ship_date', 'clean_lead_time_days', 'baseline_distance_miles']].head()",
            ),
            (
                "code",
                "print('Missing clean_order_date:', df['clean_order_date'].isna().sum())\n"
                "print(df.groupby('Ship Mode')['clean_lead_time_days'].agg(['count', 'min', 'mean', 'median', 'max']))",
            ),
        ]
    )
    (NB_DIR / "04_date_resolution_and_feature_engineering.ipynb").write_text(
        json.dumps(nb04, indent=2)
    )

    nb05 = make_notebook(
        [
            (
                "markdown",
                "# Step 5: Factory Reallocation Optimization & Machine Learning\n\n"
                "Runs the HiGHS Linear Programming network optimization (`src/optimizer.py`) "
                "and compares predictive ML models for lead-time forecasting and freight-erosion risk.",
            ),
            (
                "code",
                "import sys\nfrom pathlib import Path\n\n"
                "ROOT = Path('..').resolve()\n"
                "if str(ROOT) not in sys.path:\n"
                "    sys.path.insert(0, str(ROOT))\n\n"
                "import pandas as pd\n"
                "from src.optimizer import solve_factory_reallocation\n\n"
                "df = pd.read_csv(ROOT / 'data' / 'processed' / 'nassau_cleaned_validated.csv')\n"
                "opt_res = solve_factory_reallocation(df)\n"
                "pd.DataFrame([opt_res['summary']]).T",
            ),
            (
                "code",
                "opt_res['factory_summary']",
            ),
            (
                "code",
                "pd.read_csv(ROOT / 'outputs' / 'optimization' / 'ml_model_metrics.csv')",
            ),
        ]
    )
    (NB_DIR / "05_factory_optimization_and_ml.ipynb").write_text(
        json.dumps(nb05, indent=2)
    )
    print("Notebooks 04 and 05 generated successfully.")


if __name__ == "__main__":
    main()
