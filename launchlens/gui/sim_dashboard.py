"""
Streamlit dashboard for LaunchLens — decision evolution visualization.

Engines:
  * mock   — stochastic heuristic decision engine (sim_lite._mock_decision).
              Always available; default for fast iteration with no LLM cost.
  * local  — Ollama on http://localhost:11434 (Qwen2.5-3B etc.).
              Default when a model is detected and engine=auto.
  * sarvam / claude — remote, only available when API keys are set.

Run:
    streamlit run launchlens/gui/sim_dashboard.py
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import math
import random
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from launchlens.config import get_settings
from launchlens.llm import (
    LLMRoute,
    LocalModelUnavailable,
    OllamaProvider,
    estimate_cost,
    get_usage_tracker,
    reset_provider_cache,
    select_provider,
)
from launchlens.phase1.persona_gen import sample_demographic_vectors
from launchlens.phase1.schemas import AgentPersona, DistrictProfile
from launchlens.phase1.sources import load_local_district_profile
from launchlens.phase2.graph import build_graph, to_sim_graph, validate_small_world
from launchlens.phase2.influencers import inject_influencers
from launchlens.phase2.schemas import SimGraph
from launchlens.phase3.memory import MemoryStore
from launchlens.phase3.schemas import ProductStimulus
from launchlens.phase4.loop import SimulationLog, TimestepLog
from launchlens.phase4.loop import run_simulation as run_llm_simulation
from launchlens.phase4.propagation import propagate_decisions
from launchlens.sim_lite import _indore_profile, _mock_decision

_cfg = get_settings()


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

def _run_async(coro):
    """Run coroutine in a fresh thread to avoid Streamlit event-loop conflicts."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ── Provider discovery (cached) ──────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=30)
def discover_providers() -> dict:
    info = {
        "ollama_running": False,
        "ollama_models": [],
        "sarvam_available": bool(_cfg.sarvam_api_key),
        "claude_available": bool(_cfg.anthropic_api_key),
    }
    try:
        tags = OllamaProvider.probe()
        info["ollama_running"] = True
        info["ollama_models"] = tags
    except LocalModelUnavailable:
        pass
    return info


@st.cache_data(show_spinner=False)
def list_local_district_files() -> list[str]:
    if not _cfg.district_profiles_dir.exists():
        return []
    return sorted(p.stem for p in _cfg.district_profiles_dir.glob("*.json"))


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
    graph_pos: dict
    engine: str
    parse_failures: int
    usage: dict
    profile: DistrictProfile


def _build_personas(profile: DistrictProfile, n: int, seed: int) -> list[AgentPersona]:
    vectors = sample_demographic_vectors(profile, n=n, seed=seed)
    return [
        AgentPersona(
            agent_id=f"agent_{i:04d}",
            demographic=v,
            biography=(
                f"{v.sex.title()}, age {v.age}, {v.occupation.replace('_', ' ')}, "
                f"ISEC {v.isec_tier}, monthly household income ₹{v.monthly_hh_income:,}, "
                f"{v.primary_language}-speaking, {v.tech_adoption.replace('_', ' ')}. "
                f"Lives in {profile.district_name}, {profile.state_name}."
            ),
            llm_route="sarvam" if v.primary_language.lower() != "english" else "claude",
        )
        for i, v in enumerate(vectors)
    ]


def _build_product(name: str, mrp: int, launch: int) -> ProductStimulus:
    return ProductStimulus(
        product_id="prod_dash",
        product_name=name,
        category="Health & Nutrition",
        price_mrp=mrp,
        price_launch=launch,
        key_features=["20g protein", "No added sugar", "Mango / Chocolate"],
        distribution_channels=["Amazon India", "BigBasket"],
        marketing_copy="Fuel your grind. India's first truly tasty protein bar.",
        competitor_context="Yoga Bar (₹50-80), RiteBite Max (₹80-120)",
        target_segment="Health-conscious urban millennials, 22-35, SEC A/B",
    )


@st.cache_data(show_spinner=False)
def run_mock_simulation(
    n_agents: int,
    n_timesteps: int,
    seed: int,
    product_name: str,
    price_launch: int,
    price_mrp: int,
    district_source: str,
) -> SimResult:
    """Stochastic heuristic engine — no LLM calls."""
    rng = random.Random(seed)
    profile = _load_profile(district_source)
    personas = _build_personas(profile, n_agents, seed)
    persona_map = {p.agent_id: p for p in personas}

    G = build_graph(personas, k=6, beta=0.15, seed=seed)
    node_meta = inject_influencers(G, personas, seed=seed)
    graph = SimGraph(
        node_ids=[p.agent_id for p in personas],
        adjacency={n: list(G.neighbors(n)) for n in G.nodes()},
        node_meta=node_meta, k=6, beta=0.15, n_agents=n_agents,
    )
    sw_metrics = validate_small_world(G, k=6)
    graph_pos = nx.spring_layout(G, seed=seed, k=1.5 / math.sqrt(n_agents))

    product = _build_product(product_name, price_mrp, price_launch)
    store = MemoryStore()
    _run_async(store.init_all({p.agent_id: p.biography for p in personas}))

    sim_log = SimulationLog(product_id=product.product_id, n_agents=n_agents, engine="mock")
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
            mem.peer_signals = []
            _run_async(store.update(mem))
            nm = node_meta.get(agent_id)
            state_records.append({
                "agent_id": agent_id, "timestep": t, "state": dec.decision,
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
        sim_log=sim_log, personas=personas, G=G, node_meta=node_meta, product=product,
        sw_metrics=sw_metrics, state_df=pd.DataFrame(state_records),
        signal_counts=signal_counts, cumulative_buyers=cumulative_buyer_counts,
        graph_pos=graph_pos, engine="mock", parse_failures=0, usage={},
        profile=profile,
    )


def _load_profile(source: str) -> DistrictProfile:
    if source == "synthetic_indore":
        return _indore_profile()
    try:
        return load_local_district_profile(source)
    except FileNotFoundError:
        st.warning(f"Saved profile '{source}' not found — falling back to synthetic Indore.")
        return _indore_profile()


def run_llm_backed_simulation(
    n_agents: int,
    n_timesteps: int,
    seed: int,
    product_name: str,
    price_launch: int,
    price_mrp: int,
    district_source: str,
    engine: str,
) -> SimResult:
    """Phase 4 loop with real (or local) LLM. NOT cached — usage tracker matters."""
    profile = _load_profile(district_source)
    personas = _build_personas(profile, n_agents, seed)

    G = build_graph(personas, k=6, beta=0.15, seed=seed)
    node_meta = inject_influencers(G, personas, seed=seed)
    sim_graph = to_sim_graph(G, personas, node_meta, k=6, beta=0.15)
    sw_metrics = validate_small_world(G, k=6)
    graph_pos = nx.spring_layout(G, seed=seed, k=1.5 / math.sqrt(n_agents))

    product = _build_product(product_name, price_mrp, price_launch)
    store = MemoryStore()
    _run_async(store.init_all({p.agent_id: p.biography for p in personas}))

    routes = {p.agent_id: LLMRoute(p.llm_route) for p in personas}
    get_usage_tracker().reset()

    sim_log = _run_async(run_llm_simulation(
        product=product, graph=sim_graph, memory_store=store,
        agent_llm_routes=routes, n_timesteps=n_timesteps,
        seed=seed, engine_override=engine,
    ))

    # Flatten into state_records using SimulationLog
    state_records: list[dict] = []
    cumulative_buyers: set[str] = set()
    cumulative_buyer_counts: list[int] = []
    signal_counts: list[int] = []
    persona_map = {p.agent_id: p for p in personas}
    # Carry forward current state if an agent had no decision in a given timestep.
    last_state: dict[str, str] = {p.agent_id: "IGNORE" for p in personas}

    for ts in sim_log.timesteps:
        ts_states: dict[str, str] = dict(last_state)
        for d in ts.decisions:
            ts_states[d.agent_id] = d.decision
            if d.decision == "BUY":
                cumulative_buyers.add(d.agent_id)
        last_state = ts_states
        for agent_id, state in ts_states.items():
            persona = persona_map[agent_id]
            nm = node_meta.get(agent_id)
            state_records.append({
                "agent_id": agent_id, "timestep": ts.timestep, "state": state,
                "isec_tier": persona.demographic.isec_tier,
                "tech_adoption": persona.demographic.tech_adoption,
                "age": persona.demographic.age,
                "income": persona.demographic.monthly_hh_income,
                "archetype": nm.archetype if nm else "standard",
                "degree": G.degree(agent_id),
            })
        cumulative_buyer_counts.append(len(cumulative_buyers))
        # signal count proxy: sum of propagating decisions × avg neighbours
        propagating = sum(
            1 for d in ts.decisions
            if d.decision in ("BUY", "SHARE_POSITIVE", "SHARE_NEGATIVE", "COMPLAIN")
        )
        signal_counts.append(propagating)

    return SimResult(
        sim_log=sim_log, personas=personas, G=G, node_meta=node_meta, product=product,
        sw_metrics=sw_metrics, state_df=pd.DataFrame(state_records),
        signal_counts=signal_counts, cumulative_buyers=cumulative_buyer_counts,
        graph_pos=graph_pos, engine=sim_log.engine,
        parse_failures=sim_log.total_parse_failures,
        usage=get_usage_tracker().summary(),
        profile=profile,
    )


# ── Chart builders (unchanged from prior dashboard) ──────────────────────────

def chart_stacked_area(state_df: pd.DataFrame, n_agents: int) -> go.Figure:
    counts = state_df.groupby(["timestep", "state"]).size().reset_index(name="count")
    fig = go.Figure()
    for state in reversed(STATE_ORDER):
        sub = counts[counts["state"] == state]
        ts_vals = sorted(state_df["timestep"].unique())
        y_vals = []
        for t in ts_vals:
            row = sub[sub["timestep"] == t]
            y_vals.append(int(row["count"].values[0]) if len(row) else 0)
        fig.add_trace(go.Scatter(
            x=ts_vals, y=y_vals, name=state, mode="lines",
            stackgroup="one", fillcolor=STATE_COLORS[state],
            line=dict(color=STATE_COLORS[state], width=0.5),
            hovertemplate=f"<b>{state}</b>: %{{y}} agents<br>Timestep %{{x}}<extra></extra>",
        ))
    fig.update_layout(
        title="Decision State Distribution Over Time",
        xaxis_title="Timestep (≈ weeks)", yaxis_title="Agents",
        yaxis=dict(range=[0, n_agents]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), hovermode="x unified", height=380,
    )
    return fig


def chart_adoption_and_entropy(
    cumulative_buyers: list[int], n_agents: int, state_df: pd.DataFrame,
) -> go.Figure:
    timesteps = list(range(len(cumulative_buyers)))
    adoption_pct = [b / n_agents * 100 for b in cumulative_buyers]
    entropy_vals = []
    for t in timesteps:
        counts = state_df[state_df["timestep"] == t]["state"].value_counts()
        total = counts.sum()
        probs = counts / total if total else counts
        e = -sum(p * math.log2(p) for p in probs if p > 0)
        entropy_vals.append(round(e, 3))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timesteps, y=adoption_pct, name="Adoption %",
        mode="lines+markers", line=dict(color="#27ae60", width=2),
        marker=dict(size=7), yaxis="y1",
        hovertemplate="<b>Adoption</b>: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=timesteps, y=entropy_vals, name="Decision Entropy (bits)",
        mode="lines+markers", line=dict(color="#3498db", width=2, dash="dot"),
        marker=dict(size=7), yaxis="y2",
        hovertemplate="<b>Entropy</b>: %{y:.2f} bits<extra></extra>",
    ))
    fig.update_layout(
        title="Adoption Curve vs. Decision Entropy",
        xaxis_title="Timestep (≈ weeks)",
        yaxis=dict(title="Cumulative Adoption %", color="#27ae60", side="left"),
        yaxis2=dict(title="Entropy (bits)", color="#3498db", overlaying="y", side="right"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified", height=340,
    )
    return fig


def chart_network(G, pos, node_meta, state_df, personas, timestep) -> go.Figure:
    t_states = state_df[state_df["timestep"] == timestep].set_index("agent_id")["state"].to_dict()
    persona_map = {p.agent_id: p for p in personas}
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=0.4, color="#333333"),
                            hoverinfo="none", showlegend=False)
    node_traces = []
    nodes_by_state = defaultdict(list)
    for node in G.nodes():
        nodes_by_state[t_states.get(node, "IGNORE")].append(node)
    for state in STATE_ORDER:
        nodes = nodes_by_state.get(state, [])
        if not nodes:
            continue
        xs, ys, texts, sizes = [], [], [], []
        for n in nodes:
            x, y = pos[n]; xs.append(x); ys.append(y)
            nm = node_meta.get(n); p = persona_map.get(n)
            arch = nm.archetype if nm else "standard"
            isec = p.demographic.isec_tier if p else "?"
            age = p.demographic.age if p else "?"
            texts.append(f"{n}<br>{arch}<br>ISEC {isec} | Age {age}<br><b>{state}</b>")
            sizes.append(14 if nm and nm.archetype != "standard" else 8)
        node_traces.append(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(color=STATE_COLORS[state], size=sizes,
                        line=dict(width=0.5, color="#ffffff")),
            text=texts, hoverinfo="text", name=state, legendgroup=state,
        ))
    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title=f"Social Network — Timestep {timestep}  (large nodes = influencers)",
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), height=520,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def chart_isec_heatmap(state_df: pd.DataFrame) -> go.Figure:
    final_t = state_df["timestep"].max()
    final = state_df[state_df["timestep"] == final_t]
    pivot = (final.groupby(["isec_tier", "state"]).size().unstack(fill_value=0)
             .reindex(index=[t for t in ISEC_ORDER if t in final["isec_tier"].unique()]))
    present_states = [s for s in STATE_ORDER if s in pivot.columns]
    pivot = pivot[present_states]
    fig = px.imshow(
        pivot, labels=dict(x="Decision State", y="ISEC Tier", color="Agents"),
        color_continuous_scale="Viridis",
        title="Decision Distribution by ISEC Tier (final timestep)", aspect="auto",
    )
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#fafafa"), height=360,
                      coloraxis_colorbar=dict(title="Count"))
    return fig


def chart_archetype_adoption(state_df: pd.DataFrame) -> go.Figure:
    buy_counts = state_df[state_df["state"] == "BUY"].groupby("tech_adoption").size()
    total_counts = state_df.groupby("tech_adoption")["agent_id"].nunique()
    rates = (buy_counts / total_counts * 100).fillna(0).reset_index()
    rates.columns = ["tech_adoption", "adoption_pct"]
    rates["tech_adoption"] = pd.Categorical(
        rates["tech_adoption"],
        categories=[a for a in ARCHETYPE_ORDER if a in rates["tech_adoption"].values],
        ordered=True,
    )
    rates = rates.sort_values("tech_adoption")
    fig = px.bar(
        rates, x="tech_adoption", y="adoption_pct",
        title="Adoption Rate by Tech Archetype",
        labels={"tech_adoption": "Adoption Archetype", "adoption_pct": "Adoption %"},
        color="adoption_pct",
        color_continuous_scale=["#4a4a4a", "#27ae60"],
    )
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#fafafa"), height=320,
                      showlegend=False, coloraxis_showscale=False)
    return fig


def chart_signals(signal_counts: list[int]) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(range(len(signal_counts))), y=signal_counts,
        marker_color="#3498db",
        hovertemplate="t%{x}: %{y} signals<extra></extra>",
    ))
    fig.update_layout(
        title="Social Propagation Signals Per Timestep",
        xaxis_title="Timestep", yaxis_title="Signals",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), height=280,
    )
    return fig


# ── App ───────────────────────────────────────────────────────────────────────

def _render_provider_banner(info: dict, chosen_engine: str) -> None:
    if chosen_engine == "mock":
        st.info("Engine: **mock** — heuristic decisions, no LLM calls.")
        return
    if chosen_engine == "local":
        if info["ollama_running"]:
            st.success(f"Engine: **local Ollama** — models available: "
                       f"{', '.join(info['ollama_models']) or '(none pulled)'}")
        else:
            st.warning("Engine 'local' selected but Ollama is not reachable at "
                       f"{_cfg.ollama_base_url}. Start it with `ollama serve` or pick "
                       "another engine. Run will fall back to mock.")
        return
    if chosen_engine == "sarvam":
        if info["sarvam_available"]:
            st.success("Engine: **Sarvam API** (key detected).")
        else:
            st.warning("SARVAM_API_KEY not configured — will fall back to mock.")
        return
    if chosen_engine == "claude":
        if info["claude_available"]:
            st.success("Engine: **Anthropic Claude** (key detected).")
        else:
            st.warning("ANTHROPIC_API_KEY not configured — will fall back to mock.")
        return
    # auto
    if info["ollama_running"] and info["ollama_models"]:
        st.success(f"Engine: **auto → local Ollama** ({info['ollama_models'][0]})")
    elif info["sarvam_available"]:
        st.success("Engine: **auto → Sarvam API**")
    elif info["claude_available"]:
        st.success("Engine: **auto → Anthropic Claude**")
    else:
        st.info("Engine: **auto → mock** (no LLM provider available)")


def main() -> None:
    st.set_page_config(
        page_title="LaunchLens", page_icon="🔭",
        layout="wide", initial_sidebar_state="expanded",
    )
    st.markdown(
        "<h1 style='margin-bottom:0'>🔭 LaunchLens</h1>"
        "<p style='color:#888;margin-top:4px'>Synthetic consumer decision simulation</p>",
        unsafe_allow_html=True,
    )

    info = discover_providers()

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Engine")
        engine = st.selectbox(
            "LLM engine",
            options=["auto", "mock", "local", "sarvam", "claude"],
            index=0,
            help="auto = local if Ollama present, else remote if key present, else mock.",
        )
        if engine == "local" and info["ollama_models"]:
            st.selectbox("Ollama model", options=info["ollama_models"], key="model_pick")
        if not info["ollama_running"] and engine in ("local", "auto"):
            st.caption("⚠ Ollama not detected at localhost:11434")
        if st.button("↻ Re-probe providers", use_container_width=True):
            reset_provider_cache()
            discover_providers.clear()
            st.rerun()

        st.divider()
        st.header("District")
        local_districts = list_local_district_files()
        district_options = ["synthetic_indore"] + local_districts
        district_source = st.selectbox(
            "Source", options=district_options, index=0,
            help=("synthetic_indore = hardcoded baseline. Items below are real "
                  "profiles saved by `python -m launchlens.cli fetch-data`."),
        )

        st.divider()
        st.header("Simulation")
        n_agents    = st.slider("Agents", 20, 300, 60, step=20)
        n_timesteps = st.slider("Timesteps (≈weeks)", 4, 16, 8)
        seed        = st.number_input("Random Seed", value=42, step=1)

        st.divider()
        st.subheader("Product")
        product_name  = st.text_input("Name", value="FreshBite Protein Bar")
        price_mrp     = st.number_input("MRP (₹)", value=99, step=5)
        price_launch  = st.number_input("Launch Price (₹)", value=79, step=5)

        st.divider()
        run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    # ── Provider status banner ────────────────────────────────────────────
    _render_provider_banner(info, engine)

    # Cost pre-flight for remote providers
    try:
        projected, provider_name = estimate_cost(
            n_agents=n_agents, n_timesteps=n_timesteps,
            engine_override=engine if engine != "auto" else None,
        )
    except Exception:
        projected, provider_name = 0.0, "auto"
    if projected > 0.01:
        st.warning(f"Projected cost on **{provider_name}**: ${projected:.4f}")

    # ── Run ───────────────────────────────────────────────────────────────
    needs_run = run_btn or "result" not in st.session_state or \
                st.session_state.get("last_config") != (engine, district_source, n_agents,
                                                       n_timesteps, int(seed), product_name,
                                                       price_launch, price_mrp)

    if needs_run:
        with st.spinner(f"Running simulation ({engine})..."):
            if engine == "mock":
                result = run_mock_simulation(
                    n_agents, n_timesteps, int(seed),
                    product_name, price_launch, price_mrp, district_source,
                )
            else:
                # Resolve to ensure provider is actually available before we run
                try:
                    select_provider(LLMRoute.SARVAM, engine_override=engine)
                except Exception as exc:
                    st.error(f"Engine '{engine}' unavailable: {exc}")
                    st.info("Falling back to mock engine.")
                    result = run_mock_simulation(
                        n_agents, n_timesteps, int(seed),
                        product_name, price_launch, price_mrp, district_source,
                    )
                else:
                    result = run_llm_backed_simulation(
                        n_agents, n_timesteps, int(seed),
                        product_name, price_launch, price_mrp,
                        district_source, engine,
                    )
        st.session_state.result = result
        st.session_state.last_config = (engine, district_source, n_agents, n_timesteps,
                                        int(seed), product_name, price_launch, price_mrp)

    res: SimResult = st.session_state.result
    n = res.sim_log.n_agents
    final_buyers = res.cumulative_buyers[-1] if res.cumulative_buyers else 0
    final_rejected = int(
        (res.state_df[res.state_df["timestep"] == res.state_df["timestep"].max()]["state"]
         == "REJECT").sum()
    )
    final_entropy = 0.0
    t_max = res.state_df["timestep"].max()
    counts = res.state_df[res.state_df["timestep"] == t_max]["state"].value_counts()
    if counts.sum() > 0:
        probs = counts / counts.sum()
        final_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    total_cost = sum(v.get("cost_usd", 0.0) for v in res.usage.values()) if res.usage else 0.0
    total_calls = sum(v.get("calls", 0) for v in res.usage.values()) if res.usage else 0

    # ── KPIs ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Adoption Rate", f"{final_buyers/n:.1%}")
    c2.metric("Rejected", final_rejected)
    c3.metric("Decision Entropy", f"{final_entropy:.2f} bits")
    c4.metric("Parse Failures", res.parse_failures)
    c5.metric("LLM Calls", total_calls)
    c6.metric("Run Cost (USD)", f"${total_cost:.4f}")

    # ── Provenance banner for real district ──────────────────────────────
    fallback_fields = [k for k, v in res.profile.provenance.items() if v == "fallback"]
    if res.profile.provenance and fallback_fields:
        st.warning(
            f"District profile **{res.profile.district_id}** has {len(fallback_fields)} "
            f"fields on baseline fallbacks: {', '.join(fallback_fields)}. "
            f"Run `python -m launchlens.cli fetch-data --district {res.profile.district_id}` "
            f"after placing real Census/NFHS/TRAI CSVs in `data/raw/`."
        )

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🕸️ Network", "📊 Segments", "🔬 Engine"])

    with tab1:
        st.plotly_chart(chart_stacked_area(res.state_df, n), use_container_width=True)
        st.plotly_chart(
            chart_adoption_and_entropy(res.cumulative_buyers, n, res.state_df),
            use_container_width=True,
        )

    with tab2:
        t_slider = st.slider("Scrub timestep", 0, n_timesteps - 1, n_timesteps - 1, key="net_t")
        st.plotly_chart(
            chart_network(res.G, res.graph_pos, res.node_meta,
                          res.state_df, res.personas, t_slider),
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

    with tab4:
        st.subheader("Active engine")
        st.code(f"engine: {res.engine}", language=None)
        if res.usage:
            st.subheader("LLM usage")
            usage_df = pd.DataFrame.from_dict(res.usage, orient="index")
            st.dataframe(usage_df, use_container_width=True)
        else:
            st.info("Mock engine — no LLM usage recorded.")
        st.subheader("District profile")
        st.json({
            "district_id": res.profile.district_id,
            "district_name": res.profile.district_name,
            "state_name": res.profile.state_name,
            "provenance": res.profile.provenance,
        })


if __name__ == "__main__":
    main()
