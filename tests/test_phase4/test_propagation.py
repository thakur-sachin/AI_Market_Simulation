"""Propagation tests — peer signal lifecycle, decay, replacement."""
from launchlens.phase2.schemas import NodeMeta, SimGraph
from launchlens.phase3.schemas import AgentDecision, AgentMemory, PeerSignal
from launchlens.phase4.propagation import propagate_decisions


def _graph(ids: list[str], edges: dict[str, list[str]], metas: dict | None = None) -> SimGraph:
    return SimGraph(
        node_ids=ids,
        adjacency=edges,
        node_meta=metas or {i: NodeMeta(agent_id=i) for i in ids},
        k=4, beta=0.1, n_agents=len(ids),
    )


def _decision(aid: str, state: str, reason: str = "r", t: int = 0) -> AgentDecision:
    return AgentDecision(
        agent_id=aid, product_id="p1", timestep=t,
        internal_reasoning="r", decision=state,            # type: ignore[arg-type]
        primary_reason=reason, would_discuss_with="friends",
        language_of_discussion="english",
    )


def test_buy_decision_fans_out_to_neighbors():
    graph = _graph(["a1", "a2", "a3"], {"a1": ["a2", "a3"], "a2": ["a1"], "a3": ["a1"]})
    mems = {i: AgentMemory(agent_id=i, biography="bio") for i in graph.node_ids}
    count = propagate_decisions([_decision("a1", "BUY")], graph, mems, timestep=0)
    assert count == 2
    assert any(s.from_agent_id == "a1" for s in mems["a2"].peer_signals)
    assert any(s.from_agent_id == "a1" for s in mems["a3"].peer_signals)


def test_non_propagating_state_does_not_emit():
    graph = _graph(["a1", "a2"], {"a1": ["a2"], "a2": ["a1"]})
    mems = {i: AgentMemory(agent_id=i, biography="bio") for i in graph.node_ids}
    count = propagate_decisions([_decision("a1", "CONSIDER")], graph, mems, timestep=0)
    assert count == 0
    assert mems["a2"].peer_signals == []


def test_signals_decay_over_multiple_timesteps():
    """A signal emitted at t=0 should be at 0.7 when seen at t=1, 0.49 at t=2."""
    graph = _graph(["a1", "a2"], {"a1": ["a2"], "a2": ["a1"]})
    mems = {i: AgentMemory(agent_id=i, biography="bio") for i in graph.node_ids}

    propagate_decisions([_decision("a1", "BUY")], graph, mems, timestep=0)
    s_t0 = mems["a2"].peer_signals[0].salience
    assert s_t0 == 1.0   # written fresh at t=0; no decay yet (decay happens BEFORE writes)

    # At t=1 no new propagation from a1; existing signal decays
    propagate_decisions([], graph, mems, timestep=1)
    assert abs(mems["a2"].peer_signals[0].salience - 0.7) < 0.01

    # At t=2 decay again
    propagate_decisions([], graph, mems, timestep=2)
    assert abs(mems["a2"].peer_signals[0].salience - 0.49) < 0.01


def test_signals_below_floor_dropped():
    graph = _graph(["a1", "a2"], {"a1": ["a2"], "a2": ["a1"]})
    mems = {i: AgentMemory(agent_id=i, biography="bio") for i in graph.node_ids}
    mems["a2"].peer_signals.append(PeerSignal(
        from_agent_id="a1", decision="BUY", reason="r", salience=0.06,
    ))
    # Decay 0.06 * 0.70 = 0.042 < 0.05 floor → dropped
    propagate_decisions([], graph, mems, timestep=1)
    assert mems["a2"].peer_signals == []


def test_repeat_buy_from_same_source_replaces_not_stacks():
    """If a1 keeps making BUY decisions, a2 should not accumulate signals."""
    graph = _graph(["a1", "a2"], {"a1": ["a2"], "a2": ["a1"]})
    mems = {i: AgentMemory(agent_id=i, biography="bio") for i in graph.node_ids}

    for t in range(3):
        propagate_decisions([_decision("a1", "BUY", t=t)], graph, mems, timestep=t)
    # Only one signal from a1 in a2's memory
    a1_signals = [s for s in mems["a2"].peer_signals if s.from_agent_id == "a1"]
    assert len(a1_signals) == 1
    # And the timestep should be the most recent
    assert a1_signals[0].timestep == 2


def test_family_elder_buy_has_doubled_trust():
    """Family elders carry 2.0× trust multiplier on BUY."""
    metas = {
        "a1": NodeMeta(agent_id="a1", archetype="family_elder",
                       awareness_multiplier=1.0, trust_multiplier=2.0),
        "a2": NodeMeta(agent_id="a2"),
    }
    graph = _graph(["a1", "a2"], {"a1": ["a2"], "a2": ["a1"]}, metas)
    mems = {i: AgentMemory(agent_id=i, biography="bio") for i in graph.node_ids}
    propagate_decisions([_decision("a1", "BUY")], graph, mems, timestep=0)
    assert mems["a2"].peer_signals[0].salience == 2.0


def test_complain_gets_salience_boost():
    graph = _graph(["a1", "a2"], {"a1": ["a2"], "a2": ["a1"]})
    mems = {i: AgentMemory(agent_id=i, biography="bio") for i in graph.node_ids}
    propagate_decisions([_decision("a1", "COMPLAIN")], graph, mems, timestep=0)
    # base 1.0 * complain_boost 1.5 = 1.5
    assert mems["a2"].peer_signals[0].salience == 1.5
