"""Feed builder tests."""
import random

import pytest

from launchlens.phase2.schemas import NodeMeta, SimGraph
from launchlens.phase3.feed import build_feed
from launchlens.phase3.schemas import AgentMemory, PeerSignal, ProductStimulus


@pytest.fixture
def product():
    return ProductStimulus(
        product_id="p1",
        product_name="Test Bar",
        category="Snacks",
        price_mrp=100, price_launch=80,
        key_features=["protein", "tasty"],
        distribution_channels=["Amazon"],
        marketing_copy="Eat well.",
        competitor_context="None",
        target_segment="urban",
    )


@pytest.fixture
def graph():
    ids = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"]
    return SimGraph(
        node_ids=ids,
        adjacency={"a1": ["a2", "a3", "a4", "a5", "a6", "a7", "a8"]},
        node_meta={i: NodeMeta(agent_id=i) for i in ids},
        k=6, beta=0.15, n_agents=len(ids),
    )


def _mem(aid: str, decision=None, opinion: str = "") -> AgentMemory:
    m = AgentMemory(agent_id=aid, biography="bio")
    if decision:
        m.current_decision["p1"] = decision
    if opinion:
        m.product_opinion["p1"] = opinion
    return m


def test_feed_caps_reviews_at_5(product, graph):
    me = _mem("a1")
    all_mem = {"a1": me}
    for i, aid in enumerate(graph.neighbors("a1")):
        all_mem[aid] = _mem(aid, decision="REJECT", opinion=f"opinion {i}")
    feed = build_feed("a1", me, product, graph, all_mem, timestep=0, rng=random.Random(0))
    assert len(feed.peer_reviews) <= 5


def test_feed_caps_purchases_at_3(product, graph):
    me = _mem("a1")
    all_mem = {"a1": me}
    for i, aid in enumerate(graph.neighbors("a1")):
        all_mem[aid] = _mem(aid, decision="BUY")
    feed = build_feed("a1", me, product, graph, all_mem, timestep=0, rng=random.Random(0))
    assert len(feed.peer_purchases) <= 3


def test_feed_dedupes_by_from_agent(product, graph):
    """If a neighbor's review is already in pending peer_signals AND surfaces via
    current_decision, only one entry should remain."""
    me = _mem("a1")
    me.peer_signals = [PeerSignal(
        from_agent_id="a2", decision="SHARE_POSITIVE", reason="loved it",
        salience=0.9, timestep=0,
    )]
    all_mem = {"a1": me, "a2": _mem("a2", decision="CONSIDER", opinion="thinking")}
    feed = build_feed("a1", me, product, graph, all_mem, timestep=1, rng=random.Random(0))
    sources = [s.from_agent_id for s in feed.peer_reviews]
    assert sources.count("a2") == 1


def test_feed_archetype_hint_from_node_meta(product):
    """Archetype hint should come from graph.node_meta, not fall through to 'standard'."""
    graph = SimGraph(
        node_ids=["a1", "a2"],
        adjacency={"a1": ["a2"]},
        node_meta={
            "a1": NodeMeta(agent_id="a1"),
            "a2": NodeMeta(agent_id="a2", archetype="family_elder",
                           awareness_multiplier=1.0, trust_multiplier=2.0),
        },
        k=2, beta=0.0, n_agents=2,
    )
    me = _mem("a1")
    all_mem = {"a1": me, "a2": _mem("a2", decision="BUY", opinion="excellent")}
    feed = build_feed("a1", me, product, graph, all_mem, timestep=0, rng=random.Random(0))
    assert feed.peer_purchases[0].archetype_hint == "family_elder"
