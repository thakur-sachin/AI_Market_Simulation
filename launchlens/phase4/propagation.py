"""
Network propagation engine — Phase 4.
Social actions (SHARE_*/COMPLAIN/BUY) fan out to direct connections.
Salience decays 30% per hop; signals below 0.05 are dropped.
"""
from __future__ import annotations

import structlog

from launchlens.phase2.schemas import SimGraph
from launchlens.phase2.influencers import get_propagation_multiplier
from launchlens.phase3.schemas import (
    AgentDecision,
    AgentMemory,
    PeerSignal,
    PROPAGATING_STATES,
)

log = structlog.get_logger()

_SALIENCE_DECAY = 0.70        # each hop multiplies salience by this
_SALIENCE_FLOOR = 0.05        # signals below this are dropped
_COMPLAIN_BOOST = 1.5         # complaints carry extra salience weight


def propagate_decisions(
    decisions: list[AgentDecision],
    graph: SimGraph,
    all_memories: dict[str, AgentMemory],
    timestep: int,
) -> int:
    """
    For each propagating decision, write PeerSignals into each direct neighbor's
    memory.peer_signals. Returns total signal count written.
    """
    total = 0
    for dec in decisions:
        if dec.decision not in PROPAGATING_STATES:
            continue

        node_meta = graph.node_meta.get(dec.agent_id)
        action_type = "trust" if dec.decision == "BUY" else "awareness"
        mult = get_propagation_multiplier(node_meta, action_type) if node_meta else 1.0

        base_salience = mult * (_COMPLAIN_BOOST if dec.decision == "COMPLAIN" else 1.0)

        neighbors = graph.neighbors(dec.agent_id)
        for neighbor_id in neighbors:
            neighbor_mem = all_memories.get(neighbor_id)
            if neighbor_mem is None:
                continue
            signal = PeerSignal(
                from_agent_id=dec.agent_id,
                decision=dec.decision,
                reason=dec.primary_reason,
                salience=round(base_salience, 3),
                timestep=timestep,
                archetype_hint=node_meta.archetype if node_meta else "standard",
            )
            neighbor_mem.peer_signals.append(signal)
            total += 1

    # Decay existing signals that were carried over from previous timesteps
    for mem in all_memories.values():
        mem.peer_signals = [
            PeerSignal(
                from_agent_id=s.from_agent_id,
                decision=s.decision,
                reason=s.reason,
                salience=round(s.salience * _SALIENCE_DECAY, 3),
                timestep=s.timestep,
                archetype_hint=s.archetype_hint,
            )
            for s in mem.peer_signals
            if s.salience * _SALIENCE_DECAY >= _SALIENCE_FLOOR
        ]

    log.info("propagation_done", timestep=timestep, signals_written=total)
    return total
