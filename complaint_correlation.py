"""
Correlate cybercrime complaints with the transactions they describe.

A complaint is matched to a transaction when the victim paid the reported
account, the amount is close to the reported loss, and the payment happened
shortly before the complaint was filed.
"""

from __future__ import annotations

import pandas as pd


def correlate_complaints(
    complaints: pd.DataFrame,
    transactions: pd.DataFrame,
    amount_tolerance: float = 0.02,
    window_days: int = 7,
) -> pd.DataFrame:
    tx = transactions.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    cp = complaints.copy()
    cp["complaint_date"] = pd.to_datetime(cp["complaint_date"])

    matches = []
    for _, c in cp.iterrows():
        candidates = tx[
            (tx["sender"] == c["victim_account"])
            & (tx["receiver"] == c["reported_account"])
        ]
        for _, t in candidates.iterrows():
            amount_ok = abs(t["amount"] - c["amount_lost"]) <= amount_tolerance * c["amount_lost"]
            # Compare at day granularity: complaint dates are stored as calendar dates
            time_gap = (c["complaint_date"].normalize() - t["timestamp"].normalize()).days
            time_ok = 0 <= time_gap <= window_days
            if amount_ok and time_ok:
                matches.append({
                    "complaint_id": c["complaint_id"],
                    "tx_id": t["tx_id"],
                    "victim_account": c["victim_account"],
                    "reported_account": c["reported_account"],
                    "amount": t["amount"],
                    "days_before_complaint": time_gap,
                    "category": c["category"],
                })

    return pd.DataFrame(matches)


def complaint_incident_groups(complaints: pd.DataFrame) -> pd.DataFrame:
    """
    Group complaints that point to the same reported account within a short
    period, so investigators can see related incidents across victims.
    """
    cp = complaints.copy()
    cp["complaint_date"] = pd.to_datetime(cp["complaint_date"])
    summary = (
        cp.groupby("reported_account")
        .agg(
            complaint_count=("complaint_id", "count"),
            distinct_victims=("victim_account", "nunique"),
            total_reported_loss=("amount_lost", "sum"),
            first_complaint=("complaint_date", "min"),
            last_complaint=("complaint_date", "max"),
            categories=("category", lambda s: ", ".join(sorted(set(s)))),
        )
        .reset_index()
        .sort_values("distinct_victims", ascending=False)
    )
    summary["total_reported_loss"] = summary["total_reported_loss"].round(2)
    return summary
