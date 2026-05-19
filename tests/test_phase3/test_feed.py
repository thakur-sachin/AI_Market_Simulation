"""Marketplace feed builder tests."""
from __future__ import annotations

import random

import pytest

from launchlens.phase2.schemas import NodeMeta, SimGraph
from launchlens.phase3.feed import build_feed, render_feed_text
from launchlens.phase3.schemas import AgentMemory, PeerSignal, ProductStimulus


def _product() -> ProductStimulus:
    return ProductStimulus(
        product_id="p1",
        product_name="Test Product",
        category="Snacks",
        price_mrp=100, price_launch=79,
        key_features=["tasty", "cheap"],
        distribution_channels=["Amazon"],
        marketing_copy="Try it!",
        competitor_context="Competitor X",
        target_segment="Urban millennials",
    )


def _make_graph(node_ids: list[str], edges: list[tuple[str, str]]) -> SimGraph:
    adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return SimGraph(
        node_ids=node_ids, adjacency=adj,
        node_meta={n: NodeMeta(agent_id=n) for n in node_ids},
        k=2, beta=0.0, n_agents=len(node_ids),
    )


def test_feed_contains_product_stimulus():
    g = _make_graph(["a0", "a1"], [("a0", "a1")])
    mems = {n: AgentMemory(agent_id=n, biography="b") for n in g.node_ids}
    rng = random.Random(0)
    feed = build_feed("a0", mems["a0"], _product(), g, mems, timestep=0, rng=rng)
    assert "Test Product" in feed.product_stimulus
    assert "₹79" in feed.product_stimulus


def test_feed_caps_reviews_at_five():
    ids = [f"a{i}" for i in range(10)]
    edges = [("a0", x) for x in ids[1:]]
    g = _make_graph(ids, edges)
    mems = {n: AgentMemory(agent_id=n, biography="b") for n in g.node_ids}
    # Every neighbor has a non-BUY opinion → counts as review
    for nid in ids[1:]:
        mems[nid].current_decision["p1"] = "CONSIDER"
        mems[nid].product_opinion["p1"] = f"opinion of {nid}"
    rng = random.Random(0)
    feed = build_feed("a0", mems["a0"], _product(), g, mems, timestep=0, rng=rng)
    assert len(feed.peer_reviews) <= 5


def test_feed_caps_purchases_at_three():
    ids = [f"a{i}" for i in range(6)]
    edges = [("a0", x) for x in ids[1:]]
    g = _make_graph(ids, edges)
    mems = {n: AgentMemory(agent_id=n, biography="b") for n in g.node_ids}
    for nid in ids[1:]:
        mems[nid].current_decision["p1"] = "BUY"
        mems[nid].product_opinion["p1"] = f"bought by {nid}"
    rng = random.Random(0)
    feed = build_feed("a0", mems["a0"], _product(), g, mems, timestep=0, rng=rng)
    assert len(feed.peer_purchases) <= 3


def test_feed_dedups_pending_signals_by_source():
    """If a neighbour's pending signal + a current-decision signal both arrive,
    only the highest-salience entry survives."""
    g = _make_graph(["a0", "a1"], [("a0", "a1")])
    mems = {n: AgentMemory(agent_id=n, biography="b") for n in g.node_ids}
    mems["a1"].current_decision["p1"] = "CONSIDER"
    mems["a1"].product_opinion["p1"] = "neighbour considers"
    # a0 already has a pending signal from a1 with lower salience
    mems["a0"].peer_signals = [PeerSignal(
        from_agent_id="a1", decision="REJECT", reason="old",
        salience=0.2, timestep=0,
    )]
    rng = random.Random(0)
    feed = build_feed("a0", mems["a0"], _product(), g, mems, timestep=1, rng=rng)
    sources = [s.from_agent_id for s in feed.peer_reviews]
    assert sources.count("a1") == 1


def test_feed_is_deterministic_with_seed():
    g = _make_graph(["a0", "a1", "a2"], [("a0", "a1"), ("a0", "a2")])
    mems = {n: AgentMemory(agent_id=n, biography="b") for n in g.node_ids}
    feed_a = build_feed("a0", mems["a0"], _product(), g, mems, 0, random.Random(7))
    feed_b = build_feed("a0", mems["a0"], _product(), g, mems, 0, random.Random(7))
    assert feed_a.competitor_mention == feed_b.competitor_mention
    assert feed_a.market_noise == feed_b.market_noise


def test_render_feed_text_contains_sections():
    g = _make_graph(["a0", "a1"], [("a0", "a1")])
    mems = {n: AgentMemory(agent_id=n, biography="b") for n in g.node_ids}
    mems["a1"].current_decision["p1"] = "BUY"
    mems["a1"].product_opinion["p1"] = "loved it"
    rng = random.Random(0)
    feed = build_feed("a0", mems["a0"], _product(), g, mems, 0, rng)
    txt = render_feed_text(feed)
    assert "[PRODUCT]" in txt
    assert "[PEER PURCHASES]" in txt
