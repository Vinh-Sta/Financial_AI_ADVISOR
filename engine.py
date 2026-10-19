"""Deterministic aggregation engine (Pandas).

Pipeline (from the design doc):
  CSV + Goal  ->  baseline monthly burn & net savings rate
              ->  category anomaly detection (MoM % spikes)
              ->  deterministic target estimate (months = remaining / avg savings)
              ->  JSON context for the LLM advisor
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

CATEGORIES = [
    "Groceries & Food",
    "Dining Out",
    "Transportation",
    "Utilities & Rent",
    "Shopping & Tech",
]

COLUMN_ALIASES = {
    "date": ("date", "day", "ngay"),
    "item": ("item", "name", "description", "desc", "merchant", "ten"),
    "category": ("category", "cat", "danhmuc"),
    "price": ("price", "amount", "cost", "spend", "value", "sotien"),
    "income": ("income", "thunhap"),
}


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    mapping: dict[str, str] = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                mapping[a] = canon
    df = df.rename(columns=mapping)

    if "date" not in df.columns:
        raise ValueError("CSV is missing the 'date' column")
    if "price" not in df.columns:
        raise ValueError("CSV is missing the 'price' column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    if "income" in df.columns:
        df["income"] = pd.to_numeric(df["income"], errors="coerce").fillna(0.0)
    else:
        df["income"] = 0.0
    if "item" not in df.columns:
        df["item"] = ""
    df["item"] = df["item"].fillna("").astype(str).str.strip()
    if "category" not in df.columns:
        df["category"] = ""
    df["category"] = df["category"].fillna("").astype(str).str.strip()
    df.loc[(df["income"] > 0) & (df["category"] == ""), "category"] = "Income"
    return df


def compute_report(
    df: pd.DataFrame,
    goal_amount: float,
    goal_months: int,
    monthly_income_override: float | None = None,
    anomaly_threshold: float = 1.3,
) -> dict[str, Any]:
    df = normalize(df)

    exp = df[df["income"] == 0].copy()
    inc = df[df["income"] > 0].copy()

    exp["ym"] = exp["date"].dt.to_period("M")
    inc["ym"] = inc["date"].dt.to_period("M")

    monthly_exp = exp.groupby("ym")["price"].sum()
    monthly_inc_raw = inc.groupby("ym")["price"].sum()

    all_yms = sorted(set(monthly_exp.index) | set(monthly_inc_raw.index))
    if not all_yms:
        raise ValueError("No valid transaction data found.")
    idx = pd.period_range(all_yms[0], all_yms[-1], freq="M")

    monthly_exp = monthly_exp.reindex(idx, fill_value=0.0)
    monthly_inc = monthly_inc_raw.reindex(idx, fill_value=0.0)

    if monthly_income_override and monthly_income_override > 0:
        monthly_inc = pd.Series(float(monthly_income_override), index=idx)
    elif (monthly_inc == 0).all():
        monthly_inc = pd.Series(0.0, index=idx)

    monthly_savings = monthly_inc - monthly_exp
    cum_savings = monthly_savings.cumsum()

    avg_income = float(monthly_inc.mean())
    avg_expense = float(monthly_exp.mean())
    avg_savings = float(monthly_savings.mean())
    net_savings_rate = (avg_savings / avg_income) if avg_income > 0 else 0.0

    total_saved = float(cum_savings.iloc[-1])
    remaining = goal_amount - total_saved

    if avg_savings > 0:
        projected_months = remaining / avg_savings
    else:
        projected_months = math.inf

    monthly_deficit = -avg_savings if avg_savings < 0 else 0.0

    # ---- Category anomaly detection (month-over-month % spikes) --------------
    cat_monthly = (
        exp.pivot_table(index="ym", columns="category", values="price", aggfunc="sum")
        .reindex(idx, fill_value=0.0)
    )
    anomalies: list[dict[str, Any]] = []
    for cat in cat_monthly.columns:
        series = cat_monthly[cat]
        if len(series) < 2:
            continue
        prev_avg = float(series.iloc[:-1].mean())
        latest = float(series.iloc[-1])
        if prev_avg > 0 and latest > prev_avg * anomaly_threshold:
            anomalies.append({
                "category": str(cat),
                "latest_month": str(idx[-1]),
                "latest": latest,
                "previous_avg": prev_avg,
                "change_pct": (latest / prev_avg - 1) * 100,
                "extra_spend": latest - prev_avg,
            })
    anomalies.sort(key=lambda a: a["change_pct"], reverse=True)

    # ---- Deterministic target estimate ----------------------------------------
    start_date = idx[0].to_timestamp()
    last_date = idx[-1].to_timestamp()
    elapsed_months = len(idx)
    target_at_now = goal_amount * min(elapsed_months / goal_months, 1.0) if goal_months > 0 else goal_amount
    gap_to_target = target_at_now - total_saved

    if math.isfinite(projected_months):
        projected_date = pd.Timestamp.now().normalize() + pd.DateOffset(months=int(round(max(projected_months, 0))))
    else:
        projected_date = None

    return {
        "df": df,
        "expenses": exp,
        "monthly_expense": monthly_exp,
        "monthly_income": monthly_inc,
        "monthly_savings": monthly_savings,
        "cum_savings": cum_savings,
        "cat_monthly": cat_monthly,
        "anomalies": anomalies,
        "kpis": {
            "avg_income": avg_income,
            "avg_expense": avg_expense,
            "avg_savings": avg_savings,
            "net_savings_rate": net_savings_rate,
            "total_saved": total_saved,
            "remaining": remaining,
            "projected_months": projected_months,
            "projected_date": projected_date,
            "monthly_deficit": monthly_deficit,
            "gap_to_target": gap_to_target,
            "elapsed_months": elapsed_months,
        },
        "goal": {"amount": goal_amount, "months": goal_months},
        "start_date": start_date,
        "last_date": last_date,
    }


def forecast_cone(report: dict[str, Any], horizon: int | None = None) -> dict[str, Any]:
    """Project cumulative savings forward with a confidence band (mean +- 1.96*std*sqrt(t))."""
    kpi = report["kpis"]
    cum = report["cum_savings"]
    savings = report["monthly_savings"]
    avg = kpi["avg_savings"]
    std = float(savings.std()) if len(savings) > 1 else abs(avg) * 0.3
    goal = report["goal"]["amount"]

    if horizon is None:
        pm = kpi["projected_months"]
        horizon = max(int(math.ceil(pm)) if math.isfinite(pm) else 6, report["goal"]["months"], 6)
    horizon = max(1, min(horizon, 72))

    last_cum = float(cum.iloc[-1])
    last_date = report["last_date"]
    future_idx = pd.date_range(
        last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    t = np.arange(1, horizon + 1, dtype=float)
    mean = last_cum + avg * t
    band = 1.96 * std * np.sqrt(t)
    upper = np.minimum(mean + band, goal)
    lower = mean - band

    return {
        "dates": future_idx,
        "mean": mean,
        "upper": upper,
        "lower": lower,
    }


def build_json_context(report: dict[str, Any], currency: str) -> str:
    kpi = report["kpis"]
    cat_monthly = report["cat_monthly"]
    top_cats = cat_monthly.iloc[-1].sort_values(ascending=False).head(5)
    top_cats = {str(k): float(v) for k, v in top_cats.items() if v > 0}

    monthly = []
    for ym, exp_v in report["monthly_expense"].items():
        monthly.append({
            "month": str(ym),
            "income": float(report["monthly_income"][ym]),
            "expense": float(exp_v),
            "savings": float(report["monthly_savings"][ym]),
            "cumulative_savings": float(report["cum_savings"][ym]),
        })

    ctx = {
        "currency": currency,
        "goal": report["goal"],
        "kpis": {
            "average_monthly_income": round(kpi["avg_income"]),
            "average_monthly_expense": round(kpi["avg_expense"]),
            "average_monthly_savings": round(kpi["avg_savings"]),
            "net_savings_rate_pct": round(kpi["net_savings_rate"] * 100, 1),
            "total_saved_so_far": round(kpi["total_saved"]),
            "remaining_to_goal": round(kpi["remaining"]),
            "projected_months_at_current_pace": (
                round(kpi["projected_months"], 1) if math.isfinite(kpi["projected_months"]) else None
            ),
            "monthly_deficit_if_negative": round(kpi["monthly_deficit"]),
        },
        "monthly_series": monthly,
        "category_anomalies": [
            {
                "category": a["category"],
                "latest_month_spend": round(a["latest"]),
                "previous_average": round(a["previous_avg"]),
                "increase_pct": round(a["change_pct"], 1),
            }
            for a in report["anomalies"]
        ],
        "top_spending_categories_last_month": top_cats,
    }
    return json.dumps(ctx, ensure_ascii=False, indent=2)