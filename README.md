# Factory Reallocation & Shipping Optimization Recommendation System — Nassau Candy Distributor

An end-to-end supply chain analytics, Linear Programming network optimization, Machine Learning lead-time/freight-risk modeling, and interactive Streamlit decision-support system for **Nassau Candy Distributor**.

---

## Project Architecture

```text
Nassau_Candy_Optimization/
├── data/
│   ├── Nassau Candy Distributor.csv                  # Original raw dataset (10,194 rows x 18 cols)
│   └── processed/
│       ├── nassau_cleaned_stage2.csv                 # Stage 2 audit snapshot
│       └── nassau_cleaned_validated.csv              # Stage 4 validated & engineered dataset (10,194 x 47)
├── src/
│   ├── geo_and_network.py                            # 59 state/province centroids, Haversine distance, 5-factory network
│   └── optimizer.py                                  # Capacitated HiGHS Linear Programming solver
├── models/
│   ├── lead_time_model.joblib                        # Best Lead-Time Regressor (Random Forest)
│   ├── freight_risk_model.joblib                     # Best Freight-Erosion Risk Classifier (HistGradientBoosting)
│   └── model_metadata.json
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_date_resolution_and_feature_engineering.ipynb
│   └── 05_factory_optimization_and_ml.ipynb
├── outputs/
│   ├── data_audit/                                   # Step 1 audit tables & plots
│   ├── data_cleaning/                                # Step 2 lead-time investigation tables & plots
│   ├── eda/                                          # Step 3 business KPIs, summaries & 14 plots
│   ├── feature_engineering/                          # Step 4A date recovery summary & plots
│   └── optimization/                                 # Step 5 LP reallocation tables, ML benchmarks & plots
├── tests/
│   └── verify_project.py                             # End-to-end automated verification suite
├── run_audit.py                                      # Step 1 pipeline
├── run_cleaning.py                                   # Step 2 pipeline
├── run_eda.py                                        # Step 3 pipeline
├── run_feature_engineering.py                        # Step 4A date resolution & feature engineering
├── run_optimization.py                               # Step 5 LP optimization & ML training
├── app.py                                            # Step 6 interactive Streamlit dashboard
└── requirements.txt
```

---

## Key Analytical Findings & Results

### 1. Business Overview (Observed Historical Transactions)
- **Total Sales:** \$141,783.63 across **8,549 unique orders**, **5,044 customers**, and **38,654 units**
- **Total Gross Profit:** \$93,442.80 (**65.9% overall gross margin**)
- **Regional Demand Leaders:** Pacific (\$46,301.53 sales, 32.7%) and Atlantic (\$41,197.24 sales, 29.1%) account for 61.7% of total revenue.
- **Product Concentration:** The Chocolate division drives 96.4% of unit volume (37,275 units), led by `Wonka Bar - Milk Chocolate` (8,267 units) and `Wonka Bar - Triple Dazzle Caramel` (\$28,485.00 sales).

### 2. Root-Cause Date Anomaly Resolution (`run_feature_engineering.py`)
Steps 2–3 flagged severe lead-time anomalies (`613` to `1,965` days) and `6,064` missing `Order Date` values. Root-cause investigation established two exact issues:
1. Raw dates use `DD-MM-YYYY` (`%d-%m-%Y`); parsing without day-first format coerced `59.49%` of `Order Date` entries to `NaT`.
2. `Ship Date` was shifted forward by **exactly +2,000 days**, while `Order Date` years were shifted to `2024–2025` instead of the true order year encoded in `Order ID` (`2021–2024`).

Reversing both shifts recovers **100% of dates (0 missing)** and restores clean `0–11` day lead times (`median = 4.0 days`) aligned with Ship Mode SLAs (`Same Day`: 0–1d, `First Class`: 1–4d, `Second Class`: 1–5d, `Standard Class`: 3–11d).

### 3. Capacitated Factory Reallocation Optimization (`run_optimization.py`)
> *Note: Because the raw CSV does not contain factory coordinates or freight tariffs, the 5-factory network (`Wicked Choccy's`, `Lot's O' Nuts`, `Sugar Shack`, `The Other Factory`, `Secret Factory`), plant capacity shares, and freight rates are explicitly parameterized network assumptions.*

- **Baseline Imbalance:** Legacy product-specialized routing overloads `Lot's O' Nuts` (191.9% capacity) and `Wicked Choccy's` (138.8% capacity) while underutilizing `Sugar Shack` (1.2%), `The Other Factory` (7.6%), and `Secret Factory` (5.2%), causing 79.8% cross-region shipments.
- **HiGHS LP Optimal Reallocation:**
  - **Average Shipping Distance:** `1,029.5` -> `392.0` miles (**-61.9%** unit-miles)
  - **Modeled Freight Cost:** `\$14,243.05` -> `\$6,853.09` (**\$7,389.96 / 51.9% savings**)
  - **Modeled Transit Time:** `-0.60 days` average reduction across 38,654 units
  - **Reallocated Volume:** `30,972` units (`80.1%`), balancing all 5 plants within capacity (`25.7%–100.0%`).

### 4. Machine Learning Benchmarks (`run_optimization.py`)
- **Lead-Time Regressor (`Random Forest Regressor`):** Test `RMSE = 0.960 days`, `MAE = 0.775 days`, `R² = 0.695`
- **Freight-Erosion Risk Classifier (`HistGradientBoosting Classifier`):** Test `ROC-AUC = 0.9999`, `F1 = 0.9959`, `Accuracy = 99.80%`

---

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the full reproducible pipeline:
   ```bash
   python run_audit.py
   python run_cleaning.py
   python run_eda.py
   python run_feature_engineering.py
   python run_optimization.py
   ```
3. Verify outputs, optimization invariants, and ML models:
   ```bash
   python tests/verify_project.py
   ```
4. Launch the interactive Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
