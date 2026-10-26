"""
End-to-end analysis pipeline. The dashboard and CLI both call run_pipeline().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from complaint_correlation import complaint_incident_groups, correlate_complaints
from graph_analysis import build_graph, cluster_membership, detect_clusters, find_intermediaries
from scoring import compute_features, score_accounts

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class AnalysisResult:
    accounts: pd.DataFrame
    transactions: pd.DataFrame
    complaints: pd.DataFrame
    scores: pd.DataFrame
    matches: pd.DataFrame
    incidents: pd.DataFrame
    clusters: list[dict]
    intermediaries: list[str]
    graph_nodes: int = 0
    graph_edges: int = 0
    extras: dict = field(default_factory=dict)


def load_data(data_dir: str | Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    return {
        "accounts": pd.read_csv(d / "accounts.csv"),
        "transactions": pd.read_csv(d / "transactions.csv"),
        "complaints": pd.read_csv(d / "complaints.csv"),
    }


def run_pipeline(data: dict[str, pd.DataFrame]) -> AnalysisResult:
    accounts = data["accounts"]
    transactions = data["transactions"]
    complaints = data["complaints"]

    features = compute_features(accounts, transactions, complaints)
    scores = score_accounts(features)

    graph = build_graph(transactions, accounts)
    clusters = detect_clusters(graph)
    membership = cluster_membership(clusters)

    # Boost accounts that sit inside a suspicious cluster together with other high-risk accounts
    scores = scores.copy()
    high_risk = set(scores[scores["risk_score"] >= 30].index)
    for acc in scores.index:
        idx = membership.get(acc)
        if idx is None:
            continue
        members = set(clusters[idx]["members"])
        if len(members & high_risk) >= 2:
            scores.loc[acc, "risk_score"] = min(100.0, scores.loc[acc, "risk_score"] + 10.0)
    scores["risk_band"] = pd.cut(
        scores["risk_score"], bins=[-0.1, 35, 55, 100], labels=["Low", "Medium", "High"]
    ).astype(str)
    scores["cluster_id"] = [membership.get(acc, -1) for acc in scores.index]
    scores = scores.sort_values("risk_score", ascending=False)

    matches = correlate_complaints(complaints, transactions)
    incidents = complaint_incident_groups(complaints)

    return AnalysisResult(
        accounts=accounts,
        transactions=transactions,
        complaints=complaints,
        scores=scores,
        matches=matches,
        incidents=incidents,
        clusters=clusters,
        intermediaries=find_intermediaries(graph),
        graph_nodes=graph.number_of_nodes(),
        graph_edges=graph.number_of_edges(),
        extras={"graph": graph},
    )


if __name__ == "__main__":
    result = run_pipeline(load_data())
    print(f"Accounts scored: {len(result.scores)}")
    print(f"High-risk accounts: {(result.scores['risk_band'] == 'High').sum()}")
    print(f"Suspicious clusters: {len(result.clusters)}")
    print(f"Complaint-transaction matches: {len(result.matches)}")
    print("\nTop 10 accounts:")
    print(result.scores[["account_type", "risk_score", "risk_band"]].head(10).to_string())
