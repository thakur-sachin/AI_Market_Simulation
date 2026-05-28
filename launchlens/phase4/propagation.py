"""
Network propagation engine — Phase 4.
Social actions (SHARE_*/COMPLAIN/BUY) fan out to direct connections.
Salience decays 30% per timestep; signals below 0.05 are dropped.

Order of operations per timestep, after all agent decisions for t are recorded:
  1. Decay carried-over signals from prior timesteps (and drop below floor).
  2. For each propagating decision at t, write a fresh PeerSignal at base
     salience to each direct neighbor; if a signal from the same source
     already exists, replace it (keep the freshest).

Net effect: a signal emitted at t is at base_salience when neighbor reads at t+1
(its first opportunity), 0.7× at t+2, 0.49× at t+3, ... falls below floor
around t+8. This matches the "30% decay per hop" spec.
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

_SALIENCE_DECAY = 0.70        # each timestep multiplies salience by this
_SALIENCE_FLOOR = 0.05        # signals below this are dropped
_COMPLAIN_BOOST = 1.5         # complaints carry extra salience weight


def _decay_existing(memories: dict[str, AgentMemory]) -> None:
    """Decay every signal already in memory and drop those below the floor."""
    for mem in memories.values():
        decayed: list[PeerSignal] = []
        for s in mem.peer_signals:
            new_sal = round(s.salience * _SALIENCE_DECAY, 3)
            if new_sal < _SALIENCE_FLOOR:
                continue
            decayed.append(PeerSignal(
                from_agent_id=s.from_agent_id,
                decision=s.decision,
                reason=s.reason,
                salience=new_sal,
                timestep=s.timestep,
                archetype_hint=s.archetype_hint,
            ))
        mem.peer_signals = decayed


def propagate_decisions(
    decisions: list[AgentDecision],
    graph: SimGraph,
    all_memories: dict[str, AgentMemory],
    timestep: int,
) -> int:
    """
    Fan out propagating decisions to direct neighbors and decay older signals.
    Returns total fresh signal count written this timestep.
    """
    # 1. Decay everything that was already in memory before this timestep's events
    _decay_existing(all_memories)

    # 2. Write fresh signals from this timestep's propagating decisions
    total = 0
    for dec in decisions:
        if dec.decision not in PROPAGATING_STATES:
            continue

        node_meta = graph.node_meta.get(dec.agent_id)
        action_type = "trust" if dec.decision == "BUY" else "awareness"
        mult = get_propagation_multiplier(node_meta, action_type) if node_meta else 1.0
        base_salience = round(mult * (_COMPLAIN_BOOST if dec.decision == "COMPLAIN" else 1.0), 3)

        for neighbor_id in graph.neighbors(dec.agent_id):
            neighbor_mem = all_memories.get(neighbor_id)
            if neighbor_mem is None:
                continue
            # Replace any existing signal from the same source to avoid stacking
            neighbor_mem.peer_signals = [
                s for s in neighbor_mem.peer_signals if s.from_agent_id != dec.agent_id
            ]
            neighbor_mem.peer_signals.append(PeerSignal(
                from_agent_id=dec.agent_id,
                decision=dec.decision,
                reason=dec.primary_reason,
                salience=base_salience,
                timestep=timestep,
                archetype_hint=node_meta.archetype if node_meta else "standard",
            ))
            total += 1

    log.info("propagation_done", timestep=timestep, signals_written=total)
    return total
