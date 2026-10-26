"""
Behavioural feature engineering and explainable risk scoring.

Each account gets a set of interpretable features. A weighted rule score
combines them, and every account carries the top contributing indicators so
investigators can see *why* it was flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Feature -> (weight, human-readable explanation template)
RULES: dict[str, tuple[float, str]] = {
    "fan_in_ratio": (0.20, "Receives funds from many distinct senders"),
    "pass_through_ratio": (0.25, "Forwards almost all received funds (pass-through behaviour)"),
    "rapid_forward_score": (0.20, "Forwards funds within hours of receiving them"),
    "counterparty_diversity": (0.10, "Unusually high number of distinct counterparties"),
    "dormant_activation": (0.10, "Recently opened or previously dormant account suddenly active"),
    "complaint_exposure": (0.15, "Reported by multiple victims in cybercrime complaints"),
}


def build_transaction_frame(transactions: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    return tx


def compute_features(
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    complaints: pd.DataFrame,
) -> pd.DataFrame:
    tx = build_transaction_frame(transactions)
    feats = accounts[["account_id", "account_type", "opened_on"]].copy()
    feats["opened_on"] = pd.to_datetime(feats["opened_on"])
    feats = feats.set_index("account_id")

    inflow = tx.groupby("receiver")["amount"].sum()
    outflow = tx.groupby("sender")["amount"].sum()
    in_count = tx.groupby("receiver")["sender"].nunique()
    out_count = tx.groupby("sender")["receiver"].nunique()

    feats["total_in"] = inflow.reindex(feats.index).fillna(0.0)
    feats["total_out"] = outflow.reindex(feats.index).fillna(0.0)
    feats["distinct_senders"] = in_count.reindex(feats.index).fillna(0).astype(int)
    feats["distinct_receivers"] = out_count.reindex(feats.index).fillna(0).astype(int)

    # Fan-in: distinct senders relative to the population, capped to [0, 1]
    # Ramp: below 6 senders contributes nothing, 20+ senders is full weight
    feats["fan_in_ratio"] = np.clip((feats["distinct_senders"] - 6) / 14.0, 0, 1)

    # Pass-through: share of inflow forwarded out, only meaningful when money came in
    with np.errstate(divide="ignore", invalid="ignore"):
        pt = np.where(feats["total_in"] > 0, feats["total_out"] / feats["total_in"], 0.0)
    # Ramp: only near-complete forwarding (>= 60%) counts, full weight at 95%+
    feats["pass_through_ratio"] = np.clip((pt - 0.6) / 0.35, 0, 1)

    # Rapid forwarding: for each account, average hours between an inbound tx
    # and the next outbound tx. Shorter gaps yield higher scores.
    feats["rapid_forward_score"] = _rapid_forward_scores(tx, feats.index)

    # Counterparty diversity: distinct counterparties relative to transaction count
    counterparties = (
        pd.concat([
            tx[["sender", "receiver"]].rename(columns={"sender": "acc", "receiver": "cp"}),
            tx[["receiver", "sender"]].rename(columns={"receiver": "acc", "sender": "cp"}),
        ])
        .groupby("acc")["cp"].nunique()
    )
    feats["counterparty_diversity"] = np.clip(
        (counterparties.reindex(feats.index).fillna(0) - 15) / 35.0, 0, 1
    )

    # Dormant activation: account opened long before the activity window but
    # with very few prior transactions, or opened very recently with large flows
    activity_start = tx["timestamp"].min()
    first_tx = pd.concat([
        tx[["sender", "timestamp"]].rename(columns={"sender": "acc"}),
        tx[["receiver", "timestamp"]].rename(columns={"receiver": "acc"}),
    ]).groupby("acc")["timestamp"].min()
    tx_count = pd.concat([tx["sender"], tx["receiver"]]).value_counts()
    age_days = (activity_start - feats["opened_on"]).dt.days.clip(lower=0)
    activity = tx_count.reindex(feats.index).fillna(0)
    recent_open = (age_days < 60).astype(float)
    sparse = (activity <= 3).astype(float)
    feats["dormant_activation"] = np.clip(0.5 * recent_open + 0.5 * sparse, 0, 1)
    feats["first_tx"] = first_tx.reindex(feats.index)

    # Complaint exposure: number of distinct victims reporting the account
    victims = complaints.groupby("reported_account")["victim_account"].nunique()
    feats["complaint_victims"] = victims.reindex(feats.index).fillna(0).astype(int)
    feats["complaint_exposure"] = np.clip(feats["complaint_victims"] / 3.0, 0, 1)

    return feats


def _rapid_forward_scores(tx: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Score accounts that forward received money within a short window."""
    scores = {}
    inbound = tx[["receiver", "timestamp", "amount"]].rename(columns={"receiver": "acc"})
    outbound = tx[["sender", "timestamp", "amount"]].rename(columns={"sender": "acc"})

    for acc, ins in inbound.groupby("acc"):
        outs = outbound[outbound["acc"] == acc]
        if outs.empty or ins.empty:
            scores[acc] = 0.0
            continue
        ins_sorted = ins.sort_values("timestamp")
        outs_sorted = outs.sort_values("timestamp")
        gaps = []
        for _, row in ins_sorted.iterrows():
            nxt = outs_sorted[outs_sorted["timestamp"] > row["timestamp"]]
            if not nxt.empty:
                gap_h = (nxt.iloc[0]["timestamp"] - row["timestamp"]).total_seconds() / 3600
                gaps.append(gap_h)
        if not gaps:
            scores[acc] = 0.0
        else:
            median_gap = float(np.median(gaps))
            # 0 hours -> 1.0, 72+ hours -> 0.0
            scores[acc] = float(np.clip(1 - median_gap / 72.0, 0, 1))

    return pd.Series(scores).reindex(index).fillna(0.0)


def score_accounts(features: pd.DataFrame) -> pd.DataFrame:
    """Combine features into a 0-100 risk score with per-feature explanations."""
    result = features.copy()
    total_weight = sum(w for w, _ in RULES.values())

    contributions = pd.DataFrame(index=result.index)
    for name, (weight, _) in RULES.items():
        contributions[name] = weight * result[name] / total_weight

    result["risk_score"] = (contributions.sum(axis=1) * 100).round(1)

    explanations = []
    for acc in result.index:
        row = contributions.loc[acc].sort_values(ascending=False)
        top = [
            {
                "indicator": name,
                "points": round(float(row[name] * 100), 1),
                "reason": RULES[name][1],
            }
            for name in row.index[:3]
            if row[name] > 0.01
        ]
        explanations.append(top)
    result["explanation"] = explanations

    result["risk_band"] = pd.cut(
        result["risk_score"],
        bins=[-0.1, 35, 55, 100],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    return result.sort_values("risk_score", ascending=False)
