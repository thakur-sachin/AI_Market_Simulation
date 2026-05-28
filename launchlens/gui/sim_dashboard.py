"""
Streamlit dashboard for LaunchLens Lite — decision evolution visualization.

Run:
    streamlit run launchlens/gui/sim_dashboard.py
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from launchlens.phase1.schemas import AgentPersona
from launchlens.phase1.persona_gen import sample_demographic_vectors
from launchlens.phase2.graph import build_graph, validate_small_world
from launchlens.phase2.influencers import inject_influencers
from launchlens.phase2.schemas import SimGraph
from launchlens.phase3.schemas import ProductStimulus
from launchlens.phase3.memory import MemoryStore
from launchlens.phase4.propagation import propagate_decisions
from launchlens.phase4.loop import SimulationLog, TimestepLog
from launchlens.sim_lite import _indore_profile, _mock_decision


# ── Constants ─────────────────────────────────────────────────────────────────

STATE_COLORS = {
    "IGNORE":         "#4a4a4a",
    "AWARE":          "#8e8e8e",
    "RESEARCH":       "#f39c12",
    "CONSIDER":       "#f1c40f",
    "BUY":            "#27ae60",
    "SHARE_POSITIVE": "#16a085",
    "REJECT":         "#c0392b",
    "SHARE_NEGATIVE": "#922b21",
    "COMPLAIN":       "#e67e22",
}

STATE_ORDER = [
    "IGNORE", "AWARE", "RESEARCH", "CONSIDER",
    "BUY", "SHARE_POSITIVE", "REJECT", "SHARE_NEGATIVE", "COMPLAIN",
]

ISEC_ORDER = ["A1","A2","A3","B1","B2","C1","C2","D1","D2","E1","E2","E3"]

ARCHETYPE_ORDER = ["innovator","early_adopter","early_majority","late_majority","laggard"]


# ── Async helper ──────────────────────────────────────────────────────────────
# Reuse a single thread + a single event loop across calls. Streamlit invokes
# this from its own event loop, so we route async work to a sidecar loop.

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: concurrent.futures.ThreadPoolExecutor | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        _loop_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        _loop_thread.submit(_loop.run_forever)
    return _loop


def _run_async(coro):
    """Run coroutine on a persistent sidecar event loop."""
    loop = _get_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


# ── Simulation runner ─────────────────────────────────────────────────────────

@dataclass
class SimResult:
    sim_log: SimulationLog
    personas: list[AgentPersona]
    G: nx.Graph
    node_meta: dict
    product: ProductStimulus
    sw_metrics: dict
    state_df: pd.DataFrame
    signal_counts: list[int]
    cumulative_buyers: list[int]
    graph_pos: dict   # nx layout positions


@st.cache_data(show_spinner=False)
def run_simulation(
    n_agents: int,
    n_timesteps: int,
    seed: int,
    product_name: str,
    price_launch: int,
    price_mrp: int,
) -> SimResult:
    rng = random.Random(seed)

    # Population
    profile = _indore_profile()
    vectors = sample_demographic_vectors(profile, n=n_agents, seed=seed)
    personas = [
        AgentPersona(
            agent_id=f"agent_{i:04d}",
            demographic=v,
            biography=(
                f"{v.sex.title()}, age {v.age}, {v.occupation.replace('_',' ')}, "
                f"ISEC {v.isec_tier}, income ₹{v.monthly_hh_income:,}, "
                f"{v.primary_language}-speaking, {v.tech_adoption.replace('_',' ')}."
            ),
            llm_route="sarvam" if v.primary_language != "english" else "claude",
        )
        for i, v in enumerate(vectors)
    ]
    persona_map = {p.agent_id: p for p in personas}

    # Graph
    G = build_graph(personas, k=6, beta=0.15, seed=seed)
    node_meta = inject_influencers(G, personas, seed=seed)
    graph = SimGraph(
        node_ids=[p.agent_id for p in personas],
        adjacency={n: list(G.neighbors(n)) for n in G.nodes()},
        node_meta=node_meta,
        k=6, beta=0.15, n_agents=n_agents,
    )
    sw_metrics = validate_small_world(G, k=6)
    graph_pos = nx.spring_layout(G, seed=seed, k=1.5 / math.sqrt(n_agents))

    # Product
    product = ProductStimulus(
        product_id="prod_001",
        product_name=product_name,
        category="Health & Nutrition",
        price_mrp=price_mrp,
        price_launch=price_launch,
        key_features=["20g protein", "No added sugar", "Mango / Chocolate flavors"],
        distribution_channels=["Amazon India", "BigBasket", "Modern Trade"],
        marketing_copy="Fuel your grind. India's first truly tasty protein bar.",
        competitor_context="Yoga Bar (₹50-80), RiteBite Max (₹80-120)",
        target_segment="Health-conscious urban millennials, 22-35, SEC A/B",
    )

    # Memory
    store = MemoryStore()
    _run_async(store.init_all({p.agent_id: p.biography for p in personas}))

    # Simulate
    sim_log = SimulationLog(product_id=product.product_id, n_agents=n_agents)
    cumulative_buyers: set[str] = set()
    cumulative_buyer_counts: list[int] = []
    signal_counts: list[int] = []
    state_records: list[dict] = []

    for t in range(n_timesteps):
        all_memories = _run_async(store.get_many(graph.node_ids))
        ts_log = TimestepLog(timestep=t)

        for agent_id in graph.node_ids:
            mem = all_memories.get(agent_id)
            persona = persona_map.get(agent_id)
            if not mem or not persona:
                continue

            peer_signals = mem.pending_peer_signals(product.product_id)
            dec = _mock_decision(persona, mem, product, peer_signals, rng)
            dec.timestep = t
            ts_log.decisions.append(dec)

            mem.current_decision[product.product_id] = dec.decision
            mem.product_opinion[product.product_id] = dec.internal_reasoning
            mem.add_event(f"t{t}: {dec.decision} — {dec.primary_reason[:60]}")
            if dec.decision == "BUY":
                cumulative_buyers.add(agent_id)
                mem.purchase_history.append({"timestep": t, "product_id": product.product_id})
            _run_async(store.update(mem))

            nm = node_meta.get(agent_id)
            state_records.append({
                "agent_id": agent_id,
                "timestep": t,
                "state": dec.decision,
                "isec_tier": persona.demographic.isec_tier,
                "tech_adoption": persona.demographic.tech_adoption,
                "age": persona.demographic.age,
                "income": persona.demographic.monthly_hh_income,
                "archetype": nm.archetype if nm else "standard",
                "degree": G.degree(agent_id),
            })

        n_signals = propagate_decisions(ts_log.decisions, graph, all_memories, t)
        signal_counts.append(n_signals)
        for mem in all_memories.values():
            _run_async(store.update(mem))

        cumulative_buyer_counts.append(len(cumulative_buyers))
        sim_log.timesteps.append(ts_log)

    return SimResult(
        sim_log=sim_log,
        personas=personas,
        G=G,
        node_meta=node_meta,
        product=product,
        sw_metrics=sw_metrics,
        state_df=pd.DataFrame(state_records),
        signal_counts=signal_counts,
        cumulative_buyers=cumulative_buyer_counts,
        graph_pos=graph_pos,
    )


# ── Chart builders ────────────────────────────────────────────────────────────

def chart_stacked_area(state_df: pd.DataFrame, n_agents: int) -> go.Figure:
    counts = (
        state_df.groupby(["timestep", "state"])
        .size()
        .reset_index(name="count")
    )
    fig = go.Figure()
    for state in reversed(STATE_ORDER):
        sub = counts[counts["state"] == state]
        ts_vals = sorted(state_df["timestep"].unique())
        y_vals = []
        for t in ts_vals:
            row = sub[sub["timestep"] == t]
            y_vals.append(int(row["count"].values[0]) if len(row) else 0)
        fig.add_trace(go.Scatter(
            x=ts_vals, y=y_vals,
            name=state,
            mode="lines",
            stackgroup="one",
            fillcolor=STATE_COLORS[state],
            line=dict(color=STATE_COLORS[state], width=0.5),
            hovertemplate=f"<b>{state}</b>: %{{y}} agents<br>Timestep %{{x}}<extra></extra>",
        ))
    fig.update_layout(
        title="Decision State Distribution Over Time",
        xaxis_title="Timestep (≈ weeks)",
        yaxis_title="Agents",
        yaxis=dict(range=[0, n_agents]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        hovermode="x unified",
        height=380,
    )
    return fig


def chart_adoption_and_entropy(
    cumulative_buyers: list[int],
    n_agents: int,
    state_df: pd.DataFrame,
) -> go.Figure:
    timesteps = list(range(len(cumulative_buyers)))
    adoption_pct = [b / n_agents * 100 for b in cumulative_buyers]

    entropy_vals = []
    for t in timesteps:
        counts = state_df[state_df["timestep"] == t]["state"].value_counts()
        total = counts.sum()
        probs = counts / total
        e = -sum(p * math.log2(p) for p in probs if p > 0)
        entropy_vals.append(round(e, 3))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timesteps, y=adoption_pct,
        name="Adoption %",
        mode="lines+markers",
        line=dict(color="#27ae60", width=2),
        marker=dict(size=7),
        yaxis="y1",
        hovertemplate="<b>Adoption</b>: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=timesteps, y=entropy_vals,
        name="Decision Entropy (bits)",
        mode="lines+markers",
        line=dict(color="#3498db", width=2, dash="dot"),
        marker=dict(size=7),
        yaxis="y2",
        hovertemplate="<b>Entropy</b>: %{y:.2f} bits<extra></extra>",
    ))
    fig.update_layout(
        title="Adoption Curve vs. Decision Entropy",
        xaxis_title="Timestep (≈ weeks)",
        yaxis=dict(title="Cumulative Adoption %", color="#27ae60", side="left"),
        yaxis2=dict(title="Entropy (bits)", color="#3498db", overlaying="y", side="right"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        height=340,
    )
    return fig


def chart_network(
    G: nx.Graph,
    pos: dict,
    node_meta: dict,
    state_df: pd.DataFrame,
    personas: list[AgentPersona],
    timestep: int,
) -> go.Figure:
    t_states = state_df[state_df["timestep"] == timestep].set_index("agent_id")["state"].to_dict()
    persona_map = {p.agent_id: p for p in personas}

    # Edge trace
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.4, color="#333333"),
        hoverinfo="none", showlegend=False,
    )

    # Node traces — one per state for legend grouping
    node_traces = []
    nodes_by_state: dict[str, list] = defaultdict(list)
    for node in G.nodes():
        state = t_states.get(node, "IGNORE")
        nodes_by_state[state].append(node)

    for state in STATE_ORDER:
        nodes = nodes_by_state.get(state, [])
        if not nodes:
            continue
        xs, ys, texts, sizes = [], [], [], []
        for n in nodes:
            x, y = pos[n]
            xs.append(x); ys.append(y)
            nm = node_meta.get(n)
            p = persona_map.get(n)
            arch = nm.archetype if nm else "standard"
            isec = p.demographic.isec_tier if p else "?"
            age = p.demographic.age if p else "?"
            texts.append(f"{n}<br>{arch}<br>ISEC {isec} | Age {age}<br><b>{state}</b>")
            # Influencers get larger markers
            sizes.append(14 if nm and nm.archetype != "standard" else 8)
        node_traces.append(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(color=STATE_COLORS[state], size=sizes, line=dict(width=0.5, color="#ffffff")),
            text=texts, hoverinfo="text",
            name=state,
            legendgroup=state,
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title=f"Social Network — Timestep {timestep}  (large nodes = influencers)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def chart_isec_heatmap(state_df: pd.DataFrame) -> go.Figure:
    final_t = state_df["timestep"].max()
    final = state_df[state_df["timestep"] == final_t]
    pivot = (
        final.groupby(["isec_tier", "state"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=[t for t in ISEC_ORDER if t in final["isec_tier"].unique()])
    )
    present_states = [s for s in STATE_ORDER if s in pivot.columns]
    pivot = pivot[present_states]

    fig = px.imshow(
        pivot,
        labels=dict(x="Decision State", y="ISEC Tier", color="Agents"),
        color_continuous_scale="Viridis",
        title="Decision Distribution by ISEC Tier (final timestep)",
        aspect="auto",
    )
    fig.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), height=360,
        coloraxis_colorbar=dict(title="Count"),
    )
    return fig


def chart_archetype_adoption(state_df: pd.DataFrame) -> go.Figure:
    buy_counts = state_df[state_df["state"] == "BUY"].groupby("tech_adoption").size()
    total_counts = state_df.groupby("tech_adoption")["agent_id"].nunique()
    adoption_rates = (buy_counts / total_counts * 100).fillna(0).reset_index()
    adoption_rates.columns = ["tech_adoption", "adoption_pct"]
    adoption_rates["tech_adoption"] = pd.Categorical(
        adoption_rates["tech_adoption"],
        categories=[a for a in ARCHETYPE_ORDER if a in adoption_rates["tech_adoption"].values],
        ordered=True,
    )
    adoption_rates = adoption_rates.sort_values("tech_adoption")

    fig = px.bar(
        adoption_rates, x="tech_adoption", y="adoption_pct",
        title="Adoption Rate by Tech Archetype (any timestep agent reached BUY)",
        labels={"tech_adoption": "Adoption Archetype", "adoption_pct": "Adoption %"},
        color="adoption_pct",
        color_continuous_scale=["#4a4a4a", "#27ae60"],
    )
    fig.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), height=320,
        showlegend=False, coloraxis_showscale=False,
    )
    return fig


def chart_signals(signal_counts: list[int]) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(range(len(signal_counts))),
        y=signal_counts,
        marker_color="#3498db",
        hovertemplate="t%{x}: %{y} signals<extra></extra>",
    ))
    fig.update_layout(
        title="Social Propagation Signals Written Per Timestep",
        xaxis_title="Timestep", yaxis_title="Signals",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), height=280,
    )
    return fig


# ── App ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="LaunchLens Lite",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        "<h1 style='margin-bottom:0'>🔭 LaunchLens Lite</h1>"
        "<p style='color:#888;margin-top:4px'>Synthetic consumer decision simulation — Indore market</p>",
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Simulation Parameters")
        n_agents    = st.slider("Agents", 50, 500, 100, step=50)
        n_timesteps = st.slider("Timesteps (≈weeks)", 4, 24, 10)
        seed        = st.number_input("Random Seed", value=42, step=1)

        st.divider()
        st.subheader("Product")
        product_name  = st.text_input("Name", value="FreshBite Protein Bar")
        price_mrp     = st.number_input("MRP (₹)", value=99, step=5)
        price_launch  = st.number_input("Launch Price (₹)", value=79, step=5)

        st.divider()
        run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    # ── Run ───────────────────────────────────────────────────────────────────
    if "result" not in st.session_state or run_btn:
        with st.spinner("Running simulation…"):
            st.session_state.result = run_simulation(
                n_agents, n_timesteps, int(seed),
                product_name, price_launch, price_mrp,
            )

    res: SimResult = st.session_state.result
    n = res.sim_log.n_agents
    final_buyers = res.cumulative_buyers[-1] if res.cumulative_buyers else 0
    final_rejected = int(
        (res.state_df[res.state_df["timestep"] == res.state_df["timestep"].max()]["state"] == "REJECT").sum()
    )
    final_entropy = 0.0
    t_max = res.state_df["timestep"].max()
    counts = res.state_df[res.state_df["timestep"] == t_max]["state"].value_counts()
    probs = counts / counts.sum()
    final_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    sw = res.sw_metrics

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Adoption Rate", f"{final_buyers/n:.1%}")
    c2.metric("Cumulative Buyers", final_buyers)
    c3.metric("Rejected", final_rejected)
    c4.metric("Decision Entropy", f"{final_entropy:.2f} bits")
    c5.metric("Small-World σ", f"{sw['small_world_sigma']:.2f}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🕸️ Network", "📊 Segments"])

    with tab1:
        st.plotly_chart(
            chart_stacked_area(res.state_df, n),
            use_container_width=True,
        )
        st.plotly_chart(
            chart_adoption_and_entropy(res.cumulative_buyers, n, res.state_df),
            use_container_width=True,
        )

    with tab2:
        t_slider = st.slider(
            "Scrub timestep", 0, n_timesteps - 1, n_timesteps - 1, key="net_t"
        )
        st.plotly_chart(
            chart_network(
                res.G, res.graph_pos, res.node_meta,
                res.state_df, res.personas, t_slider,
            ),
            use_container_width=True,
        )
        with st.expander("Graph Metrics"):
            m = res.sw_metrics
            st.markdown(
                f"| Metric | Value |\n|---|---|\n"
                f"| Clustering Coefficient | {m['clustering_coefficient']:.3f} |\n"
                f"| Avg Path Length | {m['avg_path_length']:.2f} |\n"
                f"| Small-World σ | {m['small_world_sigma']:.2f} |\n"
                f"| Random CC baseline | {m['cc_random_baseline']:.3f} |\n"
            )

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(chart_isec_heatmap(res.state_df), use_container_width=True)
        with col_b:
            st.plotly_chart(chart_archetype_adoption(res.state_df), use_container_width=True)
        st.plotly_chart(chart_signals(res.signal_counts), use_container_width=True)

        with st.expander("Raw Agent State Table"):
            t_filter = st.slider("Timestep", 0, n_timesteps - 1, n_timesteps - 1, key="tbl_t")
            tbl = res.state_df[res.state_df["timestep"] == t_filter][
                ["agent_id", "state", "isec_tier", "tech_adoption", "archetype", "age", "income"]
            ].sort_values("state")
            st.dataframe(tbl, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
