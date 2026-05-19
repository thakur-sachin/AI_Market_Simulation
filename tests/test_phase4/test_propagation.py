"""Propagation engine tests — salience decay, COMPLAIN boost, dedup, idempotency."""
from __future__ import annotations

import pytest

from launchlens.phase2.schemas import NodeMeta, SimGraph
from launchlens.phase3.schemas import AgentDecision, AgentMemory, PeerSignal
from launchlens.phase4.propagation import (
    _COMPLAIN_BOOST,
    _SALIENCE_DECAY,
    _SALIENCE_FLOOR,
    propagate_decisions,
)


def _star_graph(n: int = 5) -> SimGraph:
    """Hub-and-spoke graph: agent_0 is the hub, others spokes."""
    ids = [f"agent_{i}" for i in range(n)]
    adj = {ids[0]: ids[1:]}
    for spoke in ids[1:]:
        adj[spoke] = [ids[0]]
    return SimGraph(
        node_ids=ids,
        adjacency=adj,
        node_meta={i: NodeMeta(agent_id=i) for i in ids},
        k=2, beta=0.0, n_agents=n,
    )


def _empty_mem(agent_id: str) -> AgentMemory:
    return AgentMemory(agent_id=agent_id, biography="t")


def test_buy_decision_propagates_to_neighbors():
    g = _star_graph(5)
    mems = {aid: _empty_mem(aid) for aid in g.node_ids}
    decision = AgentDecision(
        agent_id="agent_0", product_id="p", timestep=0,
        internal_reasoning="r", decision="BUY", primary_reason="r",
        would_discuss_with="friends", language_of_discussion="Hindi",
    )
    n_signals = propagate_decisions([decision], g, mems, timestep=0)
    assert n_signals == 4  # 4 spokes
    for spoke in g.node_ids[1:]:
        assert any(s.decision == "BUY" for s in mems[spoke].peer_signals)
    # Hub itself receives no signal from its own action.
    assert mems["agent_0"].peer_signals == []


def test_non_propagating_states_emit_nothing():
    g = _star_graph(3)
    mems = {aid: _empty_mem(aid) for aid in g.node_ids}
    for state in ("IGNORE", "AWARE", "RESEARCH", "CONSIDER", "REJECT"):
        decision = AgentDecision(
            agent_id="agent_0", product_id="p", timestep=0,
            internal_reasoning="r", decision=state,  # type: ignore[arg-type]
            primary_reason="r",
            would_discuss_with="no_one", language_of_discussion="N/A",
        )
        # Clear state between iterations
        for m in mems.values():
            m.peer_signals = []
        n = propagate_decisions([decision], g, mems, timestep=0)
        assert n == 0, f"{state} should not propagate"


def test_complain_carries_boost():
    """COMPLAIN signals are emitted at 1.5× then decayed once with all other signals.

    The current implementation runs the decay pass over ALL signals (newly written
    or carried over) at the end of each propagation call, so the observed salience
    on the first read is ``_COMPLAIN_BOOST * _SALIENCE_DECAY``. A normal BUY signal
    is emitted at 1.0 and then decayed to ``_SALIENCE_DECAY`` (0.7).
    """
    g = _star_graph(2)  # 1 hub + 1 spoke
    mems = {aid: _empty_mem(aid) for aid in g.node_ids}
    complaint = AgentDecision(
        agent_id="agent_0", product_id="p", timestep=0,
        internal_reasoning="r", decision="COMPLAIN", primary_reason="bad",
        would_discuss_with="friends", language_of_discussion="Hindi",
    )
    propagate_decisions([complaint], g, mems, timestep=0)
    sig = mems["agent_1"].peer_signals[0]
    expected = round(_COMPLAIN_BOOST * _SALIENCE_DECAY, 3)
    assert sig.salience == pytest.approx(expected, abs=1e-3)

    # Sanity: BUY (no boost) on a fresh graph should land at exactly _SALIENCE_DECAY.
    g2 = _star_graph(2)
    mems2 = {aid: _empty_mem(aid) for aid in g2.node_ids}
    buy = AgentDecision(
        agent_id="agent_0", product_id="p", timestep=0,
        internal_reasoning="r", decision="BUY", primary_reason="r",
        would_discuss_with="friends", language_of_discussion="Hindi",
    )
    propagate_decisions([buy], g2, mems2, timestep=0)
    assert mems2["agent_1"].peer_signals[0].salience == pytest.approx(
        _SALIENCE_DECAY, abs=1e-3
    )


def test_salience_decays_each_timestep():
    """Carried-over signals lose 30% per propagation call."""
    g = _star_graph(2)
    mems = {aid: _empty_mem(aid) for aid in g.node_ids}
    # Seed one signal with salience 1.0
    mems["agent_1"].peer_signals = [PeerSignal(
        from_agent_id="agent_0", decision="BUY", reason="r", salience=1.0,
        timestep=0, archetype_hint="standard",
    )]
    propagate_decisions([], g, mems, timestep=1)
    assert len(mems["agent_1"].peer_signals) == 1
    assert mems["agent_1"].peer_signals[0].salience == pytest.approx(_SALIENCE_DECAY, abs=1e-3)
    # Another decay
    propagate_decisions([], g, mems, timestep=2)
    assert mems["agent_1"].peer_signals[0].salience == pytest.approx(
        _SALIENCE_DECAY ** 2, abs=1e-3
    )


def test_salience_floor_drops_low_signals():
    g = _star_graph(2)
    mems = {aid: _empty_mem(aid) for aid in g.node_ids}
    mems["agent_1"].peer_signals = [PeerSignal(
        from_agent_id="agent_0", decision="BUY", reason="r",
        salience=_SALIENCE_FLOOR / _SALIENCE_DECAY * 0.99,  # just under floor after decay
        timestep=0,
    )]
    propagate_decisions([], g, mems, timestep=1)
    assert mems["agent_1"].peer_signals == []


def test_propagation_is_observable_via_count():
    g = _star_graph(10)
    mems = {aid: _empty_mem(aid) for aid in g.node_ids}
    decisions = [
        AgentDecision(
            agent_id="agent_0", product_id="p", timestep=0,
            internal_reasoning="r", decision="SHARE_POSITIVE",
            primary_reason="r", would_discuss_with="friends",
            language_of_discussion="Hindi",
        )
    ]
    n = propagate_decisions(decisions, g, mems, timestep=0)
    assert n == 9  # 9 spokes


def test_idempotency_no_duplicates_on_replay_with_clear():
    g = _star_graph(3)
    mems = {aid: _empty_mem(aid) for aid in g.node_ids}
    decision = AgentDecision(
        agent_id="agent_0", product_id="p", timestep=0,
        internal_reasoning="r", decision="BUY", primary_reason="r",
        would_discuss_with="friends", language_of_discussion="Hindi",
    )
    propagate_decisions([decision], g, mems, timestep=0)
    # Clear consumed signals (as loop._apply_decision does after each timestep)
    for m in mems.values():
        m.peer_signals = []
    propagate_decisions([decision], g, mems, timestep=1)
    for spoke in g.node_ids[1:]:
        assert len(mems[spoke].peer_signals) == 1
