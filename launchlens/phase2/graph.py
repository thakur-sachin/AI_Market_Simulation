"""
Social graph construction — Phase 2a.
Watts-Strogatz small-world topology with homophily-biased ring ordering.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import networkx as nx
import structlog

from launchlens.phase1.schemas import AgentPersona
from launchlens.phase2.schemas import InfluencerArchetype, NodeMeta, SimGraph

log = structlog.get_logger()

# ── ISEC tier numeric rank for similarity scoring ─────────────────────────────
_ISEC_RANK = {
    "A1": 11, "A2": 10, "A3": 9,
    "B1": 8,  "B2": 7,
    "C1": 6,  "C2": 5,
    "D1": 4,  "D2": 3,
    "E1": 2,  "E2": 1, "E3": 0,
}


def _similarity_key(p: AgentPersona) -> tuple:
    """Sort key: same district > same ISEC band > same age bracket > same language."""
    isec_band = _ISEC_RANK.get(p.demographic.isec_tier, 0) // 3   # 0-3 bands
    age_band = p.demographic.age // 10
    return (
        p.demographic.district_id,
        isec_band,
        age_band,
        p.demographic.primary_language,
    )


# ── Watts-Strogatz builder ────────────────────────────────────────────────────

def build_graph(
    personas: list[AgentPersona],
    k: int = 8,
    beta: float = 0.15,
    seed: int | None = None,
) -> nx.Graph:
    """
    Build a Watts-Strogatz small-world graph over the agent population.

    Agents are ordered by _similarity_key before ring construction so
    the k nearest ring-neighbors are demographically similar.
    β-rewiring introduces cross-demographic long-range edges.
    """
    if k % 2 != 0:
        k += 1  # WS requires even k

    ordered = sorted(personas, key=_similarity_key)
    ids = [p.agent_id for p in ordered]
    n = len(ids)

    rng = random.Random(seed)

    # Build initial k-regular ring lattice
    G = nx.Graph()
    G.add_nodes_from(ids)

    for i in range(n):
        for j in range(1, k // 2 + 1):
            neighbor = ids[(i + j) % n]
            G.add_edge(ids[i], neighbor)

    # β-rewiring: for each edge, with probability β replace target with random node
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < beta:
                u = ids[i]
                v = ids[(i + j) % n]
                # pick a random node that isn't u and isn't already connected
                candidates = [ids[x] for x in range(n) if ids[x] != u and not G.has_edge(u, ids[x])]
                if candidates:
                    w = rng.choice(candidates)
                    G.remove_edge(u, v)
                    G.add_edge(u, w)

    log.info("graph_built", nodes=G.number_of_nodes(), edges=G.number_of_edges(), k=k, beta=beta)
    return G


def add_cross_district_edges(
    G: nx.Graph,
    personas: list[AgentPersona],
    cross_edge_fraction: float = 0.03,
    seed: int | None = None,
) -> nx.Graph:
    """Add inter-district long-range edges (2-5% of total edges)."""
    rng = random.Random(seed)
    by_district: dict[str, list[str]] = {}
    for p in personas:
        by_district.setdefault(p.demographic.district_id, []).append(p.agent_id)

    districts = list(by_district.keys())
    if len(districts) < 2:
        return G

    target_cross = int(G.number_of_edges() * cross_edge_fraction)
    added = 0
    attempts = 0
    while added < target_cross and attempts < target_cross * 10:
        d1, d2 = rng.sample(districts, 2)
        u = rng.choice(by_district[d1])
        v = rng.choice(by_district[d2])
        if not G.has_edge(u, v):
            G.add_edge(u, v, cross_district=True)
            added += 1
        attempts += 1

    log.info("cross_district_edges_added", added=added)
    return G


# ── Graph → SimGraph serialization ────────────────────────────────────────────

def to_sim_graph(
    G: nx.Graph,
    personas: list[AgentPersona],
    node_meta: dict[str, NodeMeta],
    k: int,
    beta: float,
) -> SimGraph:
    persona_map = {p.agent_id: p for p in personas}
    adj: dict[str, list[str]] = {n: list(G.neighbors(n)) for n in G.nodes()}
    return SimGraph(
        node_ids=list(G.nodes()),
        adjacency=adj,
        node_meta={aid: node_meta.get(aid, NodeMeta(agent_id=aid)) for aid in G.nodes()},
        k=k,
        beta=beta,
        n_agents=G.number_of_nodes(),
    )


def save_graph(sim_graph: SimGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sim_graph.model_dump_json(indent=2))


def load_graph(path: Path) -> SimGraph:
    return SimGraph.model_validate_json(path.read_text())


# ── Small-world metric validation ─────────────────────────────────────────────

def validate_small_world(G: nx.Graph, k: int) -> dict[str, float]:
    """
    Check graph meets small-world criteria.
    Gate criterion (CLAUDE.md Week 4-5): clustering coefficient and avg path length
    match small-world benchmarks vs. equivalent random graph.
    """
    # Use largest connected component for path length
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    cc = nx.average_clustering(lcc)

    # Sample path length on large graphs (full computation is O(N²))
    if lcc.number_of_nodes() > 500:
        sample = random.sample(list(lcc.nodes()), 200)
        lengths = []
        for s in sample:
            lens = nx.single_source_shortest_path_length(lcc, s)
            lengths.extend(lens.values())
        avg_path = sum(lengths) / len(lengths)
    else:
        avg_path = nx.average_shortest_path_length(lcc)

    # Equivalent random graph baselines
    n, m = G.number_of_nodes(), G.number_of_edges()
    p_rand = (2 * m) / (n * (n - 1))
    cc_rand = p_rand                       # Erdős–Rényi clustering coefficient
    path_rand = math.log(n) / math.log(k) # approx random graph avg path

    # Small-world: CC much higher than random, path length similar to random
    sigma = (cc / cc_rand) / (avg_path / path_rand) if path_rand > 0 else 0.0

    metrics = {
        "clustering_coefficient": round(cc, 4),
        "avg_path_length": round(avg_path, 4),
        "cc_random_baseline": round(cc_rand, 4),
        "path_random_baseline": round(path_rand, 4),
        "small_world_sigma": round(sigma, 4),  # >1 = small world
    }
    log.info("small_world_metrics", **metrics)
    return metrics
