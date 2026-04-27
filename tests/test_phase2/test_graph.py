"""Phase 2 graph tests — no LLM calls."""
import pytest
import networkx as nx

from launchlens.phase1.schemas import (
    AgentPersona, DemographicVector, AgeDistribution, DistrictProfile
)
from launchlens.phase2.graph import build_graph, validate_small_world, to_sim_graph
from launchlens.phase2.influencers import inject_influencers, _ARCHETYPE_CONFIG
from launchlens.phase2.schemas import NodeMeta


def _make_persona(i: int, district: str = "MP001", isec: str = "B1", lang: str = "hindi") -> AgentPersona:
    return AgentPersona(
        agent_id=f"agent_{i:04d}",
        demographic=DemographicVector(
            age=25 + (i % 30),
            sex="male" if i % 2 == 0 else "female",
            urban=True,
            isec_tier=isec,  # type: ignore[arg-type]
            primary_language=lang,
            occupation="services_formal",
            monthly_hh_income=30000,
            tech_adoption="early_majority",
            smartphone_owner=True,
            upi_user=True,
            district_id=district,
            district_name="Indore",
            state_name="Madhya Pradesh",
        ),
        biography="Test persona.",
        llm_route="claude",
    )


@pytest.fixture
def personas_100():
    return [_make_persona(i) for i in range(100)]


@pytest.fixture
def personas_mixed():
    # Mix of districts, ISEC, languages
    p = []
    for i in range(60):
        p.append(_make_persona(i, "MP001", "B1", "hindi"))
    for i in range(60, 100):
        p.append(_make_persona(i, "MP002", "D1", "marathi"))
    return p


def test_graph_node_count(personas_100):
    G = build_graph(personas_100, k=6, beta=0.1, seed=0)
    assert G.number_of_nodes() == 100


def test_graph_is_connected(personas_100):
    G = build_graph(personas_100, k=6, beta=0.2, seed=42)
    assert nx.is_connected(G)


def test_all_agents_present(personas_100):
    G = build_graph(personas_100, k=6, beta=0.1, seed=0)
    for p in personas_100:
        assert p.agent_id in G.nodes


def test_average_degree_near_k(personas_100):
    k = 8
    G = build_graph(personas_100, k=k, beta=0.15, seed=0)
    avg_deg = sum(d for _, d in G.degree()) / G.number_of_nodes()
    # Rewiring preserves degree sequence exactly in WS
    assert abs(avg_deg - k) < 1.0


def test_small_world_sigma(personas_100):
    G = build_graph(personas_100, k=6, beta=0.15, seed=7)
    metrics = validate_small_world(G, k=6)
    # Small-world sigma > 1 indicates high clustering relative to random
    assert metrics["small_world_sigma"] > 1.0
    assert metrics["clustering_coefficient"] > 0.0


def test_homophily_ordering_clusters_similar():
    """Same-district agents should share more edges than cross-district pairs."""
    personas = [_make_persona(i, "MP001" if i < 50 else "MP002") for i in range(100)]
    G = build_graph(personas, k=6, beta=0.05, seed=0)  # low beta → less rewiring

    mp1 = {p.agent_id for p in personas if p.demographic.district_id == "MP001"}
    mp2 = {p.agent_id for p in personas if p.demographic.district_id == "MP002"}

    intra = sum(1 for u, v in G.edges() if (u in mp1 and v in mp1) or (u in mp2 and v in mp2))
    inter = sum(1 for u, v in G.edges() if (u in mp1 and v in mp2) or (u in mp2 and v in mp1))
    assert intra > inter


def test_influencer_archetypes_assigned(personas_100):
    G = build_graph(personas_100, k=6, beta=0.1, seed=0)
    node_meta = inject_influencers(G, personas_100, seed=0)
    archetypes = {m.archetype for m in node_meta.values()}
    assert "family_elder" in archetypes
    assert "local_shopkeeper" in archetypes
    assert "micro_influencer" in archetypes
    assert "whatsapp_hub" in archetypes


def test_influencer_proportions_approximate(personas_100):
    G = build_graph(personas_100, k=6, beta=0.1, seed=0)
    node_meta = inject_influencers(G, personas_100, seed=0)
    n = len(personas_100)
    for archetype, (proportion, _, _, _) in _ARCHETYPE_CONFIG.items():
        count = sum(1 for m in node_meta.values() if m.archetype == archetype)
        expected = int(n * proportion)
        # Allow ±2 agents from target
        assert abs(count - expected) <= 2, f"{archetype}: got {count}, expected ~{expected}"


def test_micro_influencer_high_degree(personas_100):
    G = build_graph(personas_100, k=6, beta=0.1, seed=0)
    node_meta = inject_influencers(G, personas_100, seed=0)
    influencers = [aid for aid, m in node_meta.items() if m.archetype == "micro_influencer"]
    for aid in influencers:
        deg = G.degree(aid)
        # micro-influencers target 50-200 but capped by population size (100 agents)
        assert deg >= 6


def test_sim_graph_serialization(personas_100):
    G = build_graph(personas_100, k=6, beta=0.1, seed=0)
    node_meta = inject_influencers(G, personas_100, seed=0)
    sim_graph = to_sim_graph(G, personas_100, node_meta, k=6, beta=0.1)

    assert sim_graph.n_agents == 100
    assert len(sim_graph.node_ids) == 100
    # Round-trip through JSON
    restored = sim_graph.model_validate_json(sim_graph.model_dump_json())
    assert restored.n_agents == sim_graph.n_agents
    assert len(restored.adjacency) == len(sim_graph.adjacency)


def test_neighbors_are_symmetric(personas_100):
    G = build_graph(personas_100, k=6, beta=0.1, seed=0)
    node_meta = inject_influencers(G, personas_100, seed=0)
    sim_graph = to_sim_graph(G, personas_100, node_meta, k=6, beta=0.1)

    for agent_id, neighbors in sim_graph.adjacency.items():
        for neighbor in neighbors:
            assert agent_id in sim_graph.adjacency.get(neighbor, []), \
                f"Edge {agent_id}↔{neighbor} not symmetric"
