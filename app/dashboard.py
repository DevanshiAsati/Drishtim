"""
Investigation dashboard.

Run from the project root:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from data_generator import generate  # noqa: E402
from graph_analysis import trace_money_trail  # noqa: E402
from pipeline import DATA_DIR, load_data, run_pipeline  # noqa: E402

st.set_page_config(page_title="Mule Account Investigation", layout="wide")


@st.cache_data(show_spinner="Running analysis...")
def get_result(data_dir: str):
    data = load_data(data_dir)
    return run_pipeline(data)


def ensure_data() -> None:
    if not (DATA_DIR / "transactions.csv").exists():
        generate(out_dir=DATA_DIR)


def network_figure(graph: nx.DiGraph, focus: set[str], risk: dict[str, float]) -> go.Figure:
    """Plotly network figure for a subgraph around the focus accounts."""
    nodes = set(focus)
    for n in list(focus):
        nodes.update(graph.predecessors(n))
        nodes.update(graph.successors(n))
    sub = graph.subgraph(nodes)
    pos = nx.spring_layout(sub, seed=7, k=0.8)

    edge_x, edge_y = [], []
    for u, v in sub.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    node_x = [pos[n][0] for n in sub.nodes()]
    node_y = [pos[n][1] for n in sub.nodes()]
    colors = [risk.get(n, 0.0) for n in sub.nodes()]
    labels = [
        f"{n}<br>risk {risk.get(n, 0):.1f}<br>type {sub.nodes[n].get('account_type', '?')}"
        for n in sub.nodes()
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=1, color="#999"), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=list(sub.nodes()), textposition="top center",
        hovertext=labels, hoverinfo="text",
        marker=dict(size=16, color=colors, colorscale="RdYlGn_r",
                    cmin=0, cmax=100, showscale=True,
                    colorbar=dict(title="Risk")),
    ))
    fig.update_layout(showlegend=False, height=520,
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def main() -> None:
    st.title("Intelligent Mule Account Detection")
    st.caption("Behavioural scoring, money-trail tracing and complaint correlation")

    ensure_data()
    result = get_result(str(DATA_DIR))
    scores = result.scores
    graph = result.extras["graph"]
    risk_map = scores["risk_score"].to_dict()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts scored", len(scores))
    c2.metric("High-risk", int((scores["risk_band"] == "High").sum()))
    c3.metric("Suspicious clusters", len(result.clusters))
    c4.metric("Complaint matches", len(result.matches))

    tab_alerts, tab_network, tab_trail, tab_complaints, tab_explain = st.tabs(
        ["Ranked alerts", "Network view", "Money trail", "Complaints", "Explanations"]
    )

    with tab_alerts:
        st.subheader("Prioritised investigative leads")
        band = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium"])
        view = scores[scores["risk_band"].isin(band)][
            ["account_type", "risk_score", "risk_band", "cluster_id",
             "distinct_senders", "pass_through_ratio", "complaint_victims"]
        ]
        st.dataframe(view, use_container_width=True)

    with tab_network:
        st.subheader("Account relationships")
        high = scores[scores["risk_band"] == "High"].index.tolist()
        default_focus = high[:3] if high else scores.index[:3].tolist()
        focus = st.multiselect("Focus accounts", list(scores.index), default=default_focus)
        if focus:
            st.plotly_chart(network_figure(graph, set(focus), risk_map), use_container_width=True)

        st.subheader("Suspicious clusters")
        if result.clusters:
            st.dataframe(pd.DataFrame([
                {"cluster": i, "size": c["size"], "internal_value": c["internal_value"],
                 "density": c["density"], "members": ", ".join(c["members"][:8])}
                for i, c in enumerate(result.clusters)
            ]), use_container_width=True)

    with tab_trail:
        st.subheader("Trace fund flow from a victim account")
        victims = sorted(result.complaints["victim_account"].unique())
        victim = st.selectbox("Victim account", victims)
        paths = trace_money_trail(graph, victim, max_depth=4)
        if not paths:
            st.info("No outbound fund paths found for this account.")
        for path in paths[:5]:
            st.markdown(" → ".join(f"`{p}`" for p in path))
            hops = []
            for u, v in zip(path, path[1:]):
                d = graph[u][v]
                hops.append({"from": u, "to": v, "amount": round(d["total_amount"], 2),
                             "transfers": d["tx_count"]})
            st.dataframe(pd.DataFrame(hops), use_container_width=True)

        st.subheader("Transaction timeline")
        tl = result.transactions[
            (result.transactions["sender"] == victim) | (result.transactions["receiver"] == victim)
        ].copy()
        tl["timestamp"] = pd.to_datetime(tl["timestamp"])
        if not tl.empty:
            fig = go.Figure(go.Scatter(
                x=tl["timestamp"], y=tl["amount"], mode="markers",
                text=tl["sender"] + " → " + tl["receiver"],
                marker=dict(size=10, color=tl["amount"], colorscale="Viridis"),
            ))
            fig.update_layout(height=320, yaxis_title="Amount", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab_complaints:
        st.subheader("Complaint-based incident groups")
        st.dataframe(result.incidents, use_container_width=True)
        st.subheader("Complaint-to-transaction matches")
        if result.matches.empty:
            st.info("No matches found under the current tolerance.")
        else:
            st.dataframe(result.matches, use_container_width=True)

    with tab_explain:
        st.subheader("Why an account was flagged")
        account = st.selectbox("Account", scores.index.tolist())
        row = scores.loc[account]
        st.markdown(f"**Risk score:** {row['risk_score']} ({row['risk_band']})")
        for item in row["explanation"]:
            st.markdown(f"- **{item['indicator']}** (+{item['points']} pts): {item['reason']}")
        if not row["explanation"]:
            st.write("No significant indicators.")

        st.download_button(
            "Download case report (CSV)",
            data=scores.loc[[account]].drop(columns=["explanation"]).to_csv().encode(),
            file_name=f"case_{account}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
