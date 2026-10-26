"""
Graph analytics for money-trail tracing and suspicious cluster detection.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


def build_graph(transactions: pd.DataFrame, accounts: pd.DataFrame | None = None) -> nx.DiGraph:
    """Directed, weighted graph: edges are aggregated transfers between accounts."""
    g = nx.DiGraph()
    agg = (
        transactions.groupby(["sender", "receiver"])
        .agg(total_amount=("amount", "sum"), tx_count=("amount", "size"))
        .reset_index()
    )
    for _, row in agg.iterrows():
        g.add_edge(
            row["sender"],
            row["receiver"],
            total_amount=float(row["total_amount"]),
            tx_count=int(row["tx_count"]),
        )

    if accounts is not None:
        for _, row in accounts.iterrows():
            if row["account_id"] in g:
                g.nodes[row["account_id"]]["account_type"] = row["account_type"]

    return g


def trace_money_trail(
    graph: nx.DiGraph,
    source: str,
    max_depth: int = 4,
) -> list[list[str]]:
    """
    Find the main fund-flow paths starting at a victim account.

    Returns simple paths up to max_depth, ordered by total value moved along the path
    (weakest-link amount), so the most significant chains come first.
    """
    if source not in graph:
        return []

    paths = []
    for target in graph.nodes:
        if target == source:
            continue
        try:
            for path in nx.all_simple_paths(graph, source, target, cutoff=max_depth):
                flow = min(graph[u][v]["total_amount"] for u, v in zip(path, path[1:]))
                paths.append((flow, path))
        except nx.NetworkXNoPath:
            continue

    paths.sort(key=lambda p: p[0], reverse=True)
    return [path for _, path in paths[:10]]


def find_intermediaries(graph: nx.DiGraph, min_in: int = 2, min_out: int = 2) -> list[str]:
    """Accounts with multiple inbound and outbound counterparties, typical of layering nodes."""
    return [
        n for n in graph.nodes
        if graph.in_degree(n) >= min_in and graph.out_degree(n) >= min_out
    ]


def detect_clusters(graph: nx.DiGraph, min_size: int = 3) -> list[dict]:
    """
    Group accounts into connected clusters on the undirected view of the graph.
    Returns clusters with size and the total value moved inside them.
    """
    undirected = graph.to_undirected()
    clusters = []
    for component in nx.connected_components(undirected):
        if len(component) < min_size:
            continue
        sub = graph.subgraph(component)
        internal_value = sum(d["total_amount"] for _, _, d in sub.edges(data=True))
        clusters.append({
            "members": sorted(component),
            "size": len(component),
            "internal_value": round(internal_value, 2),
            "density": round(nx.density(sub), 3),
        })

    clusters.sort(key=lambda c: c["internal_value"], reverse=True)
    return clusters


def cluster_membership(clusters: list[dict]) -> dict[str, int]:
    """Map each account to its cluster index (or absent if not in a suspicious cluster)."""
    mapping: dict[str, int] = {}
    for idx, cluster in enumerate(clusters):
        for member in cluster["members"]:
            mapping[member] = idx
    return mapping
