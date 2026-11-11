"""
Linear Programming (HiGHS) Factory Reallocation & Shipping Optimization Engine
for Nassau Candy Distributor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from src.geo_and_network import (
    BASELINE_PRODUCT_FACTORY,
    DEFAULT_BASE_HANDLING_PER_UNIT,
    DEFAULT_RATE_PER_UNIT_MILE,
    FACTORIES,
    SHIP_MODE_MULTIPLIERS,
    STATE_CENTROIDS,
    estimate_freight_cost,
    haversine_miles,
)


def solve_factory_reallocation(
    df: pd.DataFrame,
    capacity_shares: dict[str, float] | None = None,
    rate_per_unit_mile: float = DEFAULT_RATE_PER_UNIT_MILE,
    base_handling: float = DEFAULT_BASE_HANDLING_PER_UNIT,
    restrict_to_primary_division: bool = False,
) -> dict[str, Any]:
    """
    Solve a capacitated minimum-freight transportation LP assigning demand lanes
    (State/Province, Region, Division, Product Name, Ship Mode) to the 5 factories.

    Returns a dictionary containing:
      - summary: baseline vs optimized KPIs
      - factory_summary: baseline vs optimized volume, capacity, and utilization
      - lane_allocations: optimal factory flows per lane
      - state_recommendations: dominant recommended factory per State & Product
      - product_summary: baseline vs optimized distance & freight by Product
      - region_summary: baseline vs optimized distance & freight by Region
    """
    factory_names = list(FACTORIES.keys())
    n_factories = len(factory_names)

    # Aggregate demand to unique shipping lanes for fast exact LP solution
    lanes = (
        df.groupby(
            ["State/Province", "Region", "Division", "Product Name", "Ship Mode"],
            as_index=False,
        )
        .agg(
            Units=("Units", "sum"),
            Sales=("Sales", "sum"),
            Cost=("Cost", "sum"),
            Gross_Profit=("Gross Profit", "sum"),
            Orders=("Order ID", "nunique"),
            Avg_Clean_Lead_Time=("clean_lead_time_days", "mean"),
        )
    )
    lanes["dest_lat"] = lanes["State/Province"].map(lambda s: STATE_CENTROIDS[s][0])
    lanes["dest_lon"] = lanes["State/Province"].map(lambda s: STATE_CENTROIDS[s][1])
    lanes["Baseline_Factory"] = lanes["Product Name"].map(BASELINE_PRODUCT_FACTORY)

    # Baseline metrics
    lanes["base_lat"] = lanes["Baseline_Factory"].map(lambda f: FACTORIES[f]["lat"])
    lanes["base_lon"] = lanes["Baseline_Factory"].map(lambda f: FACTORIES[f]["lon"])
    lanes["Baseline_Distance_Miles"] = haversine_miles(
        lanes["dest_lat"].to_numpy(),
        lanes["dest_lon"].to_numpy(),
        lanes["base_lat"].to_numpy(),
        lanes["base_lon"].to_numpy(),
    )
    lanes["Baseline_Unit_Miles"] = lanes["Units"] * lanes["Baseline_Distance_Miles"]
    lanes["Baseline_Freight"] = estimate_freight_cost(
        lanes["Units"].to_numpy(),
        lanes["Baseline_Distance_Miles"].to_numpy(),
        lanes["Ship Mode"],
        rate_per_unit_mile=rate_per_unit_mile,
        base_handling=base_handling,
    )

    n_lanes = len(lanes)
    total_units = float(lanes["Units"].sum())

    # Build factory capacities
    cap_shares = capacity_shares or {
        f: FACTORIES[f]["max_capacity_share"] for f in factory_names
    }
    capacities = np.array([cap_shares[f] * total_units for f in factory_names], dtype=float)
    if capacities.sum() < total_units - 1e-6:
        raise ValueError(
            f"Total factory capacity ({capacities.sum():.1f}) is less than total demand ({total_units:.1f})."
        )

    # Compute distance & unit freight cost matrix: shape (n_lanes, n_factories)
    dist_mat = np.zeros((n_lanes, n_factories), dtype=float)
    cost_mat = np.zeros((n_lanes, n_factories), dtype=float)
    ship_mult = lanes["Ship Mode"].map(SHIP_MODE_MULTIPLIERS).fillna(1.0).to_numpy()

    for j, fname in enumerate(factory_names):
        fmeta = FACTORIES[fname]
        d = haversine_miles(
            lanes["dest_lat"].to_numpy(),
            lanes["dest_lon"].to_numpy(),
            fmeta["lat"],
            fmeta["lon"],
        )
        dist_mat[:, j] = d
        unit_cost = (base_handling + rate_per_unit_mile * d) * ship_mult
        if restrict_to_primary_division:
            # Penalize mismatched division heavily if restricted
            mismatch = (lanes["Division"] != fmeta["primary_division"]).to_numpy()
            unit_cost = np.where(mismatch, unit_cost + 1e4, unit_cost)
        cost_mat[:, j] = unit_cost

    # Decision vector x of length n_lanes * n_factories, ordered by (lane i, factory j)
    c = cost_mat.ravel()

    # Equality constraints: sum_j x_{i,j} = demand_i
    eq_rows = np.repeat(np.arange(n_lanes), n_factories)
    eq_cols = np.arange(n_lanes * n_factories)
    eq_data = np.ones(n_lanes * n_factories, dtype=float)
    A_eq = coo_matrix((eq_data, (eq_rows, eq_cols)), shape=(n_lanes, n_lanes * n_factories)).tocsc()
    b_eq = lanes["Units"].to_numpy(dtype=float)

    # Inequality constraints: sum_i x_{i,j} <= capacity_j
    ub_rows = np.tile(np.arange(n_factories), n_lanes)
    ub_cols = np.arange(n_lanes * n_factories)
    ub_data = np.ones(n_lanes * n_factories, dtype=float)
    A_ub = coo_matrix((ub_data, (ub_rows, ub_cols)), shape=(n_factories, n_lanes * n_factories)).tocsc()
    b_ub = capacities

    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    alloc_mat = res.x.reshape((n_lanes, n_factories))
    # Zero out tiny numerical dust
    alloc_mat[alloc_mat < 1e-7] = 0.0

    # Build expanded non-zero flow table
    flow_records = []
    for i in range(n_lanes):
        row_lane = lanes.iloc[i]
        lane_units = float(row_lane["Units"])
        for j, fname in enumerate(factory_names):
            u = float(alloc_mat[i, j])
            if u <= 0.0:
                continue
            share = u / lane_units
            d_opt = float(dist_mat[i, j])
            f_opt = float(u * cost_mat[i, j])
            # Modeled transit time adjustment (~1 day per 1,000 miles reduced, bounded >= 0)
            dist_saved = float(row_lane["Baseline_Distance_Miles"]) - d_opt
            base_lt = float(row_lane["Avg_Clean_Lead_Time"])
            modeled_lt = max(0.0, base_lt - (dist_saved / 1000.0)) if row_lane["Ship Mode"] != "Same Day" else base_lt
            flow_records.append(
                {
                    "State/Province": row_lane["State/Province"],
                    "Region": row_lane["Region"],
                    "Division": row_lane["Division"],
                    "Product Name": row_lane["Product Name"],
                    "Ship Mode": row_lane["Ship Mode"],
                    "Baseline_Factory": row_lane["Baseline_Factory"],
                    "Optimized_Factory": fname,
                    "Allocated_Units": u,
                    "Allocated_Sales": float(row_lane["Sales"]) * share,
                    "Allocated_Gross_Profit": float(row_lane["Gross_Profit"]) * share,
                    "Baseline_Distance_Miles": float(row_lane["Baseline_Distance_Miles"]),
                    "Optimized_Distance_Miles": d_opt,
                    "Baseline_Unit_Miles": u * float(row_lane["Baseline_Distance_Miles"]),
                    "Optimized_Unit_Miles": u * d_opt,
                    "Baseline_Freight": float(row_lane["Baseline_Freight"]) * share,
                    "Optimized_Freight": f_opt,
                    "Baseline_Lead_Time_Days": base_lt,
                    "Modeled_Optimized_Lead_Time_Days": modeled_lt,
                    "Reallocated": fname != row_lane["Baseline_Factory"],
                }
            )

    flows_df = pd.DataFrame(flow_records)

    # Aggregate KPIs
    base_unit_miles = float(flows_df["Baseline_Unit_Miles"].sum())
    opt_unit_miles = float(flows_df["Optimized_Unit_Miles"].sum())
    base_freight = float(flows_df["Baseline_Freight"].sum())
    opt_freight = float(flows_df["Optimized_Freight"].sum())
    base_avg_dist = base_unit_miles / total_units
    opt_avg_dist = opt_unit_miles / total_units
    base_avg_lt = float((flows_df["Baseline_Lead_Time_Days"] * flows_df["Allocated_Units"]).sum() / total_units)
    opt_avg_lt = float((flows_df["Modeled_Optimized_Lead_Time_Days"] * flows_df["Allocated_Units"]).sum() / total_units)
    realloc_units = float(flows_df.loc[flows_df["Reallocated"], "Allocated_Units"].sum())

    summary = {
        "total_units": total_units,
        "baseline_unit_miles": base_unit_miles,
        "optimized_unit_miles": opt_unit_miles,
        "unit_miles_saved": base_unit_miles - opt_unit_miles,
        "unit_miles_reduction_pct": ((base_unit_miles - opt_unit_miles) / base_unit_miles) * 100.0,
        "baseline_avg_distance_miles": base_avg_dist,
        "optimized_avg_distance_miles": opt_avg_dist,
        "baseline_freight_usd": base_freight,
        "optimized_freight_usd": opt_freight,
        "freight_savings_usd": base_freight - opt_freight,
        "freight_savings_pct": ((base_freight - opt_freight) / base_freight) * 100.0,
        "baseline_avg_lead_time_days": base_avg_lt,
        "optimized_avg_lead_time_days": opt_avg_lt,
        "lead_time_reduction_days": base_avg_lt - opt_avg_lt,
        "reallocated_units": realloc_units,
        "reallocated_units_pct": (realloc_units / total_units) * 100.0,
        "lp_status": int(res.status),
        "lp_message": res.message,
    }

    # Factory capacity utilization comparison
    base_fac = (
        flows_df.groupby("Baseline_Factory", as_index=False)["Allocated_Units"]
        .sum()
        .rename(columns={"Baseline_Factory": "Factory", "Allocated_Units": "Baseline_Units"})
    )
    opt_fac = (
        flows_df.groupby("Optimized_Factory", as_index=False)["Allocated_Units"]
        .sum()
        .rename(columns={"Optimized_Factory": "Factory", "Allocated_Units": "Optimized_Units"})
    )
    fac_df = pd.DataFrame({"Factory": factory_names})
    fac_df["City_State"] = fac_df["Factory"].map(lambda f: f"{FACTORIES[f]['city']}, {FACTORIES[f]['state']}")
    fac_df["Region"] = fac_df["Factory"].map(lambda f: FACTORIES[f]["region"])
    fac_df["Capacity_Units"] = [cap_shares[f] * total_units for f in factory_names]
    fac_df = fac_df.merge(base_fac, on="Factory", how="left").merge(opt_fac, on="Factory", how="left").fillna(0.0)
    fac_df["Baseline_Utilization_Pct"] = (fac_df["Baseline_Units"] / fac_df["Capacity_Units"]) * 100.0
    fac_df["Optimized_Utilization_Pct"] = (fac_df["Optimized_Units"] / fac_df["Capacity_Units"]) * 100.0
    fac_df["Baseline_Share_Pct"] = (fac_df["Baseline_Units"] / total_units) * 100.0
    fac_df["Optimized_Share_Pct"] = (fac_df["Optimized_Units"] / total_units) * 100.0

    # Dominant state-product reallocation recommendations
    state_prod = (
        flows_df.groupby(
            ["State/Province", "Region", "Division", "Product Name", "Baseline_Factory", "Optimized_Factory"],
            as_index=False,
        )
        .agg(
            Units=("Allocated_Units", "sum"),
            Baseline_Freight=("Baseline_Freight", "sum"),
            Optimized_Freight=("Optimized_Freight", "sum"),
            Baseline_Unit_Miles=("Baseline_Unit_Miles", "sum"),
            Optimized_Unit_Miles=("Optimized_Unit_Miles", "sum"),
        )
    )
    state_prod["Freight_Savings_USD"] = state_prod["Baseline_Freight"] - state_prod["Optimized_Freight"]
    state_prod["Baseline_Avg_Miles"] = state_prod["Baseline_Unit_Miles"] / state_prod["Units"]
    state_prod["Optimized_Avg_Miles"] = state_prod["Optimized_Unit_Miles"] / state_prod["Units"]
    state_prod["Action"] = np.where(
        state_prod["Baseline_Factory"] != state_prod["Optimized_Factory"],
        "Reallocate",
        "Keep Current",
    )

    # Region & Product summaries
    region_summary = (
        flows_df.groupby("Region", as_index=False)
        .agg(
            Units=("Allocated_Units", "sum"),
            Baseline_Freight=("Baseline_Freight", "sum"),
            Optimized_Freight=("Optimized_Freight", "sum"),
            Baseline_Unit_Miles=("Baseline_Unit_Miles", "sum"),
            Optimized_Unit_Miles=("Optimized_Unit_Miles", "sum"),
        )
    )
    region_summary["Freight_Savings_USD"] = region_summary["Baseline_Freight"] - region_summary["Optimized_Freight"]
    region_summary["Freight_Savings_Pct"] = (region_summary["Freight_Savings_USD"] / region_summary["Baseline_Freight"]) * 100.0
    region_summary["Baseline_Avg_Miles"] = region_summary["Baseline_Unit_Miles"] / region_summary["Units"]
    region_summary["Optimized_Avg_Miles"] = region_summary["Optimized_Unit_Miles"] / region_summary["Units"]

    product_summary = (
        flows_df.groupby(["Division", "Product Name"], as_index=False)
        .agg(
            Units=("Allocated_Units", "sum"),
            Gross_Profit=("Allocated_Gross_Profit", "sum"),
            Baseline_Freight=("Baseline_Freight", "sum"),
            Optimized_Freight=("Optimized_Freight", "sum"),
            Baseline_Unit_Miles=("Baseline_Unit_Miles", "sum"),
            Optimized_Unit_Miles=("Optimized_Unit_Miles", "sum"),
        )
    )
    product_summary["Freight_Savings_USD"] = product_summary["Baseline_Freight"] - product_summary["Optimized_Freight"]
    product_summary["Freight_Savings_Pct"] = (product_summary["Freight_Savings_USD"] / product_summary["Baseline_Freight"]) * 100.0
    product_summary["Baseline_Avg_Miles"] = product_summary["Baseline_Unit_Miles"] / product_summary["Units"]
    product_summary["Optimized_Avg_Miles"] = product_summary["Optimized_Unit_Miles"] / product_summary["Units"]

    return {
        "summary": summary,
        "factory_summary": fac_df,
        "lane_allocations": flows_df,
        "state_recommendations": state_prod.sort_values("Freight_Savings_USD", ascending=False),
        "region_summary": region_summary.sort_values("Freight_Savings_USD", ascending=False),
        "product_summary": product_summary.sort_values("Freight_Savings_USD", ascending=False),
    }
