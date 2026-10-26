import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from complaint_correlation import correlate_complaints  # noqa: E402
from data_generator import GeneratorConfig, generate  # noqa: E402
from graph_analysis import build_graph, detect_clusters, trace_money_trail  # noqa: E402
from pipeline import run_pipeline  # noqa: E402


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    cfg = GeneratorConfig(n_normal_accounts=120, n_fraud_rings=3, n_normal_transactions=800)
    return generate(cfg, out_dir=out)


def test_generator_produces_all_tables(data):
    assert set(data) == {"accounts", "transactions", "complaints"}
    assert len(data["transactions"]) > 0
    assert len(data["complaints"]) > 0


def test_mule_accounts_rank_above_normal(data):
    result = run_pipeline(data)
    mules = result.scores[result.scores["account_type"] == "mule"]["risk_score"]
    normals = result.scores[result.scores["account_type"] == "normal"]["risk_score"]
    assert mules.mean() > normals.mean()


def test_scores_are_bounded_and_explained(data):
    result = run_pipeline(data)
    assert result.scores["risk_score"].between(0, 100).all()
    flagged = result.scores[result.scores["risk_band"] == "High"]
    assert all(len(exp) > 0 for exp in flagged["explanation"])


def test_money_trail_reaches_cashout(data):
    graph = build_graph(data["transactions"], data["accounts"])
    victim = data["complaints"]["victim_account"].iloc[0]
    paths = trace_money_trail(graph, victim, max_depth=4)
    assert paths, "expected at least one fund path from a complaint victim"
    end_types = {graph.nodes[p[-1]].get("account_type") for p in paths}
    assert "cashout" in end_types or "mule" in end_types


def test_complaints_correlate_with_transactions(data):
    matches = correlate_complaints(data["complaints"], data["transactions"])
    assert len(matches) == len(data["complaints"])


def test_clusters_detected(data):
    graph = build_graph(data["transactions"], data["accounts"])
    clusters = detect_clusters(graph)
    assert clusters
    assert all(c["size"] >= 3 for c in clusters)
