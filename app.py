"""
Nassau Candy Distributor — Factory Reallocation & Shipping Optimization Dashboard.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.geo_and_network import (
    BASELINE_PRODUCT_FACTORY,
    DEFAULT_BASE_HANDLING_PER_UNIT,
    DEFAULT_RATE_PER_UNIT_MILE,
    FACTORIES,
    STATE_CENTROIDS,
    estimate_freight_cost,
    get_factory_dataframe,
    haversine_miles,
)
from src.optimizer import solve_factory_reallocation

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "nassau_cleaned_validated.csv"
MODEL_DIR = BASE_DIR / "models"
OPT_OUT_DIR = BASE_DIR / "outputs" / "optimization"
FE_OUT_DIR = BASE_DIR / "outputs" / "feature_engineering"

st.set_page_config(
    page_title="Nassau Candy | Factory & Shipping Optimization",
    page_icon="🍬",
    layout="wide",
)

# Force blue filter tags and clean dark-mode spacing
st.markdown(
    """
    <style>
    span[data-baseweb="tag"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
    }
    span[data-baseweb="tag"] svg {
        fill: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


FACTORY_COLORS: dict[str, str] = {
    "Wicked Choccy's": "#38BDF8",
    "Lot's O' Nuts": "#34D399",
    "Sugar Shack": "#FBBF24",
    "The Other Factory": "#A78BFA",
    "Secret Factory": "#F472B6",
}

FACTORY_LABEL_POSITIONS: dict[str, str] = {
    "Wicked Choccy's": "bottom right",
    "Lot's O' Nuts": "bottom center",
    "Sugar Shack": "top left",
    "The Other Factory": "top right",
    "Secret Factory": "top center",
}


def build_network_map(opt_res: dict, fac_df: pd.DataFrame) -> go.Figure:
    """Create a high-contrast dark-themed North America factory allocation map."""
    top_flows = (
        opt_res["lane_allocations"]
        .groupby(["State/Province", "Optimized_Factory"], as_index=False)["Allocated_Units"]
        .sum()
        .sort_values("Allocated_Units", ascending=False)
        .head(45)
    )

    fig_map = go.Figure()

    # 1. Color-coded shipping lanes per assigned factory
    for _, row in top_flows.iterrows():
        fname = row["Optimized_Factory"]
        fmeta = FACTORIES[fname]
        dlat, dlon = STATE_CENTROIDS[row["State/Province"]]
        color = FACTORY_COLORS.get(fname, "#38BDF8")
        fig_map.add_trace(
            go.Scattergeo(
                lon=[fmeta["lon"], dlon],
                lat=[fmeta["lat"], dlat],
                mode="lines",
                line=dict(
                    width=max(1.0, min(4.0, row["Allocated_Units"] / 400.0)),
                    color=color,
                ),
                opacity=0.45,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # 2. Destination state hubs sized by allocated volume
    dest_lats = [STATE_CENTROIDS[s][0] for s in top_flows["State/Province"]]
    dest_lons = [STATE_CENTROIDS[s][1] for s in top_flows["State/Province"]]
    dest_colors = [FACTORY_COLORS.get(f, "#38BDF8") for f in top_flows["Optimized_Factory"]]
    dest_hover = [
        f"<b>{s}</b><br>Factory: {f}<br>Allocated Units: {int(u):,}"
        for s, f, u in zip(
            top_flows["State/Province"],
            top_flows["Optimized_Factory"],
            top_flows["Allocated_Units"],
        )
    ]
    fig_map.add_trace(
        go.Scattergeo(
            lon=dest_lons,
            lat=dest_lats,
            mode="markers",
            marker=dict(
                size=np.clip(top_flows["Allocated_Units"] / 140.0, 6, 18),
                color=dest_colors,
                opacity=0.85,
                line=dict(width=1, color="#0F172A"),
            ),
            text=dest_hover,
            hoverinfo="text",
            name="Destination States",
        )
    )

    # 3. Factory hub markers with high-contrast, non-overlapping labels
    for _, frow in fac_df.iterrows():
        fname = frow["Factory"]
        fcolor = FACTORY_COLORS.get(fname, "#38BDF8")
        tpos = FACTORY_LABEL_POSITIONS.get(fname, "top center")
        fig_map.add_trace(
            go.Scattergeo(
                lon=[frow["lon"]],
                lat=[frow["lat"]],
                text=[fname],
                customdata=[[f"{frow['city']}, {frow['state']}", frow["region"]]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Location: %{customdata[0]}<br>"
                    "Region: %{customdata[1]}<extra></extra>"
                ),
                mode="markers+text",
                textposition=tpos,
                textfont=dict(size=12, color="#F8FAFC", family="Arial, sans-serif"),
                marker=dict(
                    size=15,
                    color=fcolor,
                    symbol="diamond",
                    line=dict(width=2, color="#FFFFFF"),
                ),
                name=fname,
            )
        )

    fig_map.update_layout(
        title=dict(
            text="Optimized Factory-to-State Allocation Network (Top 45 Flows)",
            font=dict(size=16, color="#F8FAFC"),
        ),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        geo=dict(
            scope="north america",
            projection_type="mercator",
            lonaxis=dict(range=[-130, -58]),
            lataxis=dict(range=[23, 55]),
            bgcolor="#0B1120",
            showocean=True,
            oceancolor="#0F172A",
            showlakes=True,
            lakecolor="#0F172A",
            showland=True,
            landcolor="#1E293B",
            showsubunits=True,
            subunitcolor="#334155",
            showcountries=True,
            countrycolor="#475569",
            coastlinecolor="#475569",
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#E2E8F0"),
            bgcolor="rgba(15, 23, 42, 0.7)",
        ),
        height=480,
        margin=dict(l=0, r=0, t=42, b=30),
    )
    return fig_map


@st.cache_data
def load_validated_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["clean_order_date"] = pd.to_datetime(df["clean_order_date"])
    df["clean_ship_date"] = pd.to_datetime(df["clean_ship_date"])
    return df


@st.cache_data
def load_static_reports() -> dict[str, pd.DataFrame]:
    return {
        "date_recovery": pd.read_csv(FE_OUT_DIR / "date_recovery_summary.csv"),
        "ml_metrics": pd.read_csv(OPT_OUT_DIR / "ml_model_metrics.csv"),
        "lt_importance": pd.read_csv(OPT_OUT_DIR / "lead_time_feature_importance.csv"),
    }


@st.cache_resource
def load_models() -> tuple[object, object, dict]:
    lt_model = joblib.load(MODEL_DIR / "lead_time_model.joblib")
    risk_model = joblib.load(MODEL_DIR / "freight_risk_model.joblib")
    meta = json.loads((MODEL_DIR / "model_metadata.json").read_text())
    return lt_model, risk_model, meta


def main() -> None:
    st.title("Nassau Candy Distributor: Factory Reallocation & Shipping Optimization")
    st.caption(
        "Validated transactional analytics, HiGHS Linear Programming network optimization, "
        "and ML lead-time & freight-risk decision support."
    )

    df = load_validated_data()
    reports = load_static_reports()
    lt_model, risk_model, model_meta = load_models()

    # Sidebar global filters
    st.sidebar.header("Exploratory Filters")
    years = sorted(df["order_year"].unique().tolist())
    sel_years = st.sidebar.multiselect("Order Year", years, default=years)
    regions = sorted(df["Region"].unique().tolist())
    sel_regions = st.sidebar.multiselect("Destination Region", regions, default=regions)
    divisions = sorted(df["Division"].unique().tolist())
    sel_divisions = st.sidebar.multiselect("Product Division", divisions, default=divisions)
    ship_modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
    sel_modes = st.sidebar.multiselect("Ship Mode", ship_modes, default=ship_modes)

    filtered_df = df[
        df["order_year"].isin(sel_years)
        & df["Region"].isin(sel_regions)
        & df["Division"].isin(sel_divisions)
        & df["Ship Mode"].isin(sel_modes)
    ].copy()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Executive Summary & Data Recovery",
            "Exploratory Insights",
            "Factory Reallocation Optimizer",
            "ML Order & Routing Recommender",
        ]
    )

    # ------------------------------------------------------------------
    # TAB 1: EXECUTIVE SUMMARY & DATA QUALITY RECOVERY
    # ------------------------------------------------------------------
    with tab1:
        st.subheader("Validated Business KPIs (All Records)")
        c1, c2, c3, c4, c5 = st.columns(5)
        total_sales = df["Sales"].sum()
        total_profit = df["Gross Profit"].sum()
        margin_pct = (total_profit / total_sales) * 100.0
        c1.metric("Total Sales", f"${total_sales:,.2f}")
        c2.metric("Gross Profit", f"${total_profit:,.2f}")
        c3.metric("Gross Margin", f"{margin_pct:.1f}%")
        c4.metric("Unique Orders", f"{df['Order ID'].nunique():,}")
        c5.metric("Total Units Sold", f"{int(df['Units'].sum()):,}")

        st.divider()
        col_a, col_b = st.columns([1.15, 1.0])
        with col_a:
            st.subheader("Date Anomaly Resolution (Steps 2 -> 4)")
            st.info(
                "Root-cause audit identified two systematic issues in the raw CSV: "
                "(1) `DD-MM-YYYY` dates parsed without `dayfirst` created 6,064 false NaTs, and "
                "(2) `Ship Date` was shifted forward by **+2,000 days** while `Order Date` years "
                "were shifted relative to `Order ID` (`2021-2024`). Recovering both restores clean "
                "0-11 day lead times aligned with Ship Mode SLAs."
            )
            st.dataframe(reports["date_recovery"], width="stretch", hide_index=True)

        with col_b:
            st.subheader("Clean Lead Time by Ship Mode")
            fig_lt = px.box(
                df,
                x="Ship Mode",
                y="clean_lead_time_days",
                category_orders={"Ship Mode": ship_modes},
                color="Ship Mode",
                title="Validated Transit Lead Time Distribution",
            )
            fig_lt.update_layout(showlegend=False, xaxis_title="", yaxis_title="Days")
            st.plotly_chart(fig_lt, width="stretch")

        st.subheader("Parameterized 5-Factory Baseline Network")
        st.caption(
            "Note: The raw transactional CSV does not contain factory locations or freight tariffs; "
            "factory nodes and capacities below are explicit network modeling assumptions."
        )
        st.dataframe(get_factory_dataframe(), width="stretch", hide_index=True)

    # ------------------------------------------------------------------
    # TAB 2: EXPLORATORY INSIGHTS
    # ------------------------------------------------------------------
    with tab2:
        if filtered_df.empty:
            st.warning("No records match the current sidebar filters.")
        else:
            st.subheader("Filtered Slice Overview")
            f1, f2, f3, f4 = st.columns(4)
            f_sales = filtered_df["Sales"].sum()
            f_profit = filtered_df["Gross Profit"].sum()
            f1.metric("Filtered Sales", f"${f_sales:,.2f}")
            f2.metric("Filtered Profit", f"${f_profit:,.2f}")
            f3.metric("Filtered Units", f"{int(filtered_df['Units'].sum()):,}")
            f4.metric("Avg Baseline Haul", f"{filtered_df['baseline_distance_miles'].mean():.0f} mi")

            r_col1, r_col2 = st.columns(2)
            with r_col1:
                reg_agg = (
                    filtered_df.groupby("Region", as_index=False)
                    .agg(Sales=("Sales", "sum"), Gross_Profit=("Gross Profit", "sum"), Units=("Units", "sum"))
                    .sort_values("Sales", ascending=False)
                )
                fig_reg = px.bar(
                    reg_agg,
                    x="Region",
                    y=["Sales", "Gross_Profit"],
                    barmode="group",
                    title="Sales & Gross Profit by Region",
                )
                st.plotly_chart(fig_reg, width="stretch")

            with r_col2:
                monthly = (
                    filtered_df.assign(YearMonth=filtered_df["clean_order_date"].dt.to_period("M").astype(str))
                    .groupby("YearMonth", as_index=False)
                    .agg(Sales=("Sales", "sum"), Gross_Profit=("Gross Profit", "sum"))
                    .sort_values("YearMonth")
                )
                fig_m = px.line(
                    monthly,
                    x="YearMonth",
                    y=["Sales", "Gross_Profit"],
                    title="Validated Monthly Revenue & Profit Trend (2021-2024)",
                )
                st.plotly_chart(fig_m, width="stretch")

            prod_agg = (
                filtered_df.groupby(["Division", "Product Name"], as_index=False)
                .agg(
                    Units=("Units", "sum"),
                    Sales=("Sales", "sum"),
                    Gross_Profit=("Gross Profit", "sum"),
                    Baseline_Freight=("modeled_baseline_freight", "sum"),
                )
                .sort_values("Gross_Profit", ascending=False)
            )
            prod_agg["Gross_Margin_Pct"] = (prod_agg["Gross_Profit"] / prod_agg["Sales"]) * 100.0
            fig_prod = px.scatter(
                prod_agg,
                x="Units",
                y="Gross_Margin_Pct",
                size="Sales",
                color="Division",
                hover_name="Product Name",
                title="Product Portfolio: Volume vs Gross Margin (%) (Bubble Size = Sales)",
            )
            st.plotly_chart(fig_prod, width="stretch")

    # ------------------------------------------------------------------
    # TAB 3: FACTORY REALLOCATION OPTIMIZER (LIVE LP SOLVER)
    # ------------------------------------------------------------------
    with tab3:
        st.subheader("Live Capacitated Factory Reallocation Optimizer (HiGHS LP)")
        p1, p2, p3, p4 = st.columns(4)
        rate_mile = p1.number_input(
            "Freight Rate ($ / unit-mile)",
            min_value=0.00005,
            max_value=0.00200,
            value=DEFAULT_RATE_PER_UNIT_MILE,
            step=0.00005,
            format="%.5f",
        )
        base_hand = p2.number_input(
            "Handling Cost ($ / unit)",
            min_value=0.0,
            max_value=0.50,
            value=DEFAULT_BASE_HANDLING_PER_UNIT,
            step=0.01,
            format="%.2f",
        )
        cap_mult = p3.slider(
            "Factory Capacity Slack Multiplier",
            min_value=0.80,
            max_value=1.50,
            value=1.00,
            step=0.05,
            help="Scales each plant's baseline max capacity share.",
        )
        restrict_div = p4.checkbox(
            "Restrict to Primary Division Only",
            value=False,
            help="Compare cross-division co-production vs division-locked plants.",
        )

        custom_caps = {f: meta["max_capacity_share"] * cap_mult for f, meta in FACTORIES.items()}
        try:
            opt_res = solve_factory_reallocation(
                df,
                capacity_shares=custom_caps,
                rate_per_unit_mile=rate_mile,
                base_handling=base_hand,
                restrict_to_primary_division=restrict_div,
            )
        except ValueError as exc:
            st.error(f"Infeasible capacity configuration: {exc}. Increase the capacity slack multiplier.")
            return

        s = opt_res["summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Modeled Freight Cost",
            f"${s['optimized_freight_usd']:,.2f}",
            delta=f"-${s['freight_savings_usd']:,.2f} (-{s['freight_savings_pct']:.1f}%)",
            delta_color="inverse",
        )
        m2.metric(
            "Avg Haul Distance",
            f"{s['optimized_avg_distance_miles']:.1f} mi",
            delta=f"-{s['baseline_avg_distance_miles'] - s['optimized_avg_distance_miles']:.1f} mi (-{s['unit_miles_reduction_pct']:.1f}%)",
            delta_color="inverse",
        )
        m3.metric(
            "Avg Transit Lead Time",
            f"{s['optimized_avg_lead_time_days']:.2f} days",
            delta=f"-{s['lead_time_reduction_days']:.2f} days",
            delta_color="inverse",
        )
        m4.metric(
            "Reallocated Volume",
            f"{s['reallocated_units']:,.0f} units",
            delta=f"{s['reallocated_units_pct']:.1f}% of total",
        )

        map_col, util_col = st.columns([1.2, 1.0])
        with map_col:
            fac_df = get_factory_dataframe()
            fig_map = build_network_map(opt_res, fac_df)
            st.plotly_chart(fig_map, width="stretch")

        with util_col:
            fac_util = opt_res["factory_summary"]
            fig_u = px.bar(
                fac_util,
                x="Factory",
                y=["Baseline_Utilization_Pct", "Optimized_Utilization_Pct"],
                barmode="group",
                color_discrete_map={
                    "Baseline_Utilization_Pct": "#64748B",
                    "Optimized_Utilization_Pct": "#2563EB",
                },
                title="Factory Capacity Utilization (%): Baseline vs Optimized",
            )
            fig_u.add_hline(y=100, line_dash="dash", line_color="#EF4444", annotation_text="100% Cap")
            fig_u.update_layout(height=480, xaxis_title="", yaxis_title="Utilization (%)")
            st.plotly_chart(fig_u, width="stretch")

        st.subheader("Top State × Product Reallocation Recommendations")
        rec_df = opt_res["state_recommendations"]
        st.dataframe(rec_df.head(50), width="stretch", hide_index=True)
        st.download_button(
            "Download Full State Reallocation Plan (CSV)",
            data=rec_df.to_csv(index=False).encode("utf-8"),
            file_name="nassau_state_factory_reallocation_plan.csv",
            mime="text/csv",
        )

    # ------------------------------------------------------------------
    # TAB 4: ML LEAD-TIME & ORDER ROUTING RECOMMENDER
    # ------------------------------------------------------------------
    with tab4:
        st.subheader("Interactive Order Routing & Lead-Time Recommender")
        st.caption(
            f"Lead-Time Predictor: **{model_meta['best_lead_time_model']}** "
            f"(RMSE: {model_meta['lead_time_rmse']:.3f} days, R²: {model_meta['lead_time_r2']:.3f}) | "
            f"Freight-Risk Classifier: **{model_meta['best_risk_model']}** "
            f"(ROC-AUC: {model_meta['risk_roc_auc']:.4f})"
        )

        # Lookup product unit price/cost/division from validated data
        prod_ref = (
            df.groupby("Product Name", as_index=False)
            .agg(Division=("Division", "first"), Unit_Price=("unit_price", "mean"), Unit_Profit=("unit_profit", "mean"))
        )
        state_reg_map = df.groupby("State/Province")["Region"].first().to_dict()

        c_in1, c_in2, c_in3, c_in4 = st.columns(4)
        sel_state = c_in1.selectbox("Destination State/Province", sorted(STATE_CENTROIDS.keys()), index=sorted(STATE_CENTROIDS.keys()).index("California"))
        sel_prod = c_in2.selectbox("Product Name", sorted(prod_ref["Product Name"].tolist()))
        sel_units = c_in3.number_input("Order Units", min_value=1, max_value=50, value=5, step=1)
        sel_mode = c_in4.selectbox("Ship Mode", ship_modes)

        c_in5, c_in6 = st.columns(2)
        sel_month = c_in5.slider("Order Month", 1, 12, 11)
        sel_dow = c_in6.slider("Order Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

        p_info = prod_ref[prod_ref["Product Name"] == sel_prod].iloc[0]
        dest_region = state_reg_map[sel_state]
        dlat, dlon = STATE_CENTROIDS[sel_state]
        est_sales = float(p_info["Unit_Price"] * sel_units)
        est_gross_profit = float(p_info["Unit_Profit"] * sel_units)
        base_fac = BASELINE_PRODUCT_FACTORY[sel_prod]
        base_dist = float(haversine_miles(dlat, dlon, FACTORIES[base_fac]["lat"], FACTORIES[base_fac]["lon"]))

        # Evaluate all 5 factories for this order
        cand_rows = []
        for fname, fmeta in FACTORIES.items():
            dist = float(haversine_miles(dlat, dlon, fmeta["lat"], fmeta["lon"]))
            freight = float(estimate_freight_cost(sel_units, dist, sel_mode))
            net_profit = est_gross_profit - freight

            X_pred = pd.DataFrame(
                [
                    {
                        "Ship Mode": sel_mode,
                        "Region": dest_region,
                        "Division": p_info["Division"],
                        "Product Name": sel_prod,
                        "Units": sel_units,
                        "Sales": est_sales,
                        "order_month": sel_month,
                        "order_dayofweek": sel_dow,
                        "is_weekend_order": int(sel_dow in [5, 6]),
                        "dest_lat": dlat,
                        "dest_lon": dlon,
                        "baseline_distance_miles": dist,
                    }
                ]
            )
            pred_lt = float(lt_model.predict(X_pred)[0])
            X_risk = X_pred[["Ship Mode", "Region", "Division", "Product Name", "Units", "Sales", "order_month", "dest_lat", "dest_lon"]]
            risk_prob = float(risk_model.predict_proba(X_risk)[0, 1])

            cand_rows.append(
                {
                    "Factory": fname,
                    "Factory_Location": f"{fmeta['city']}, {fmeta['state']}",
                    "Distance_Miles": round(dist, 1),
                    "Modeled_Freight_USD": round(freight, 2),
                    "Net_Profit_After_Freight_USD": round(net_profit, 2),
                    "ML_Predicted_Lead_Time_Days": round(pred_lt, 2),
                    "Is_Baseline_Factory": fname == base_fac,
                    "Baseline_Freight_Risk_Prob": round(risk_prob, 3),
                }
            )

        cand_df = pd.DataFrame(cand_rows).sort_values("Modeled_Freight_USD")
        best_row = cand_df.iloc[0]
        base_row = cand_df[cand_df["Is_Baseline_Factory"]].iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Recommended Factory", best_row["Factory"], delta=f"vs {base_fac}")
        k2.metric(
            "Freight Cost vs Baseline",
            f"${best_row['Modeled_Freight_USD']:.2f}",
            delta=f"-${base_row['Modeled_Freight_USD'] - best_row['Modeled_Freight_USD']:.2f} saved",
            delta_color="inverse",
        )
        k3.metric("Predicted Lead Time", f"{best_row['ML_Predicted_Lead_Time_Days']:.2f} days")
        k4.metric(
            "Baseline Freight-Erosion Risk",
            f"{best_row['Baseline_Freight_Risk_Prob']*100:.1f}%",
            help="Probability under legacy baseline routing that freight >= 20% of Gross Profit.",
        )

        st.dataframe(
            cand_df.drop(columns=["Baseline_Freight_Risk_Prob"]),
            width="stretch",
            hide_index=True,
        )

        with st.expander("View ML Model Benchmark & Feature Importance"):
            st.dataframe(reports["ml_metrics"], width="stretch", hide_index=True)
            st.dataframe(reports["lt_importance"], width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
