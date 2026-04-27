"""
Influencer node injection — Phase 2b.
Converts a fraction of standard agents to high-degree archetypes
per the proportions in the implementation plan.
"""
from __future__ import annotations

import random

import networkx as nx
import structlog

from launchlens.phase1.schemas import AgentPersona
from launchlens.phase2.schemas import InfluencerArchetype, NodeMeta

log = structlog.get_logger()

# archetype → (target_proportion, degree_range, awareness_mult, trust_mult)
_ARCHETYPE_CONFIG: dict[InfluencerArchetype, tuple[float, tuple[int, int], float, float]] = {
    "family_elder":      (0.10, (3,   5),  1.0, 2.0),   # strong trust within family cluster
    "local_shopkeeper":  (0.03, (15, 25),  1.5, 1.0),   # broad local awareness
    "micro_influencer":  (0.008,(50,200),  1.3, 0.8),   # wide reach, lower trust
    "whatsapp_hub":      (0.065,(10,  15), 1.2, 1.0),   # group-based spread
}


def inject_influencers(
    G: nx.Graph,
    personas: list[AgentPersona],
    seed: int | None = None,
) -> dict[str, NodeMeta]:
    """
    Assign influencer archetypes to a subset of agents and rewire
    their edges so they achieve the target degree ranges.
    Returns NodeMeta dict for all nodes.
    """
    rng = random.Random(seed)
    n = len(personas)
    all_ids = [p.agent_id for p in personas]
    node_meta: dict[str, NodeMeta] = {
        p.agent_id: NodeMeta(agent_id=p.agent_id) for p in personas
    }

    assigned: set[str] = set()

    for archetype, (proportion, (deg_lo, deg_hi), aware_mult, trust_mult) in _ARCHETYPE_CONFIG.items():
        count = max(1, int(n * proportion))
        candidates = [aid for aid in all_ids if aid not in assigned]
        chosen = rng.sample(candidates, min(count, len(candidates)))

        for aid in chosen:
            assigned.add(aid)
            node_meta[aid] = NodeMeta(
                agent_id=aid,
                archetype=archetype,
                awareness_multiplier=aware_mult,
                trust_multiplier=trust_mult,
            )

            # Rewire to achieve target degree
            target_deg = rng.randint(deg_lo, deg_hi)
            current_deg = G.degree(aid)

            if current_deg < target_deg:
                # Add edges to random unconnected nodes
                pool = [x for x in all_ids if x != aid and not G.has_edge(aid, x)]
                rng.shuffle(pool)
                for neighbor in pool[:target_deg - current_deg]:
                    G.add_edge(aid, neighbor)
            elif current_deg > deg_hi and archetype == "family_elder":
                # Family elders should have small, tight clusters — prune excess
                excess_neighbors = list(G.neighbors(aid))[deg_hi:]
                for neighbor in excess_neighbors:
                    G.remove_edge(aid, neighbor)

    counts = {k: sum(1 for m in node_meta.values() if m.archetype == k)
              for k in list(_ARCHETYPE_CONFIG.keys()) + ["standard"]}
    log.info("influencers_injected", **counts)
    return node_meta


def get_propagation_multiplier(meta: NodeMeta, action_type: str) -> float:
    """
    Return the effective multiplier for a node's social action.
    action_type: 'awareness' (SHARE_*/COMPLAIN) or 'trust' (BUY signal).
    """
    if action_type == "trust":
        return meta.trust_multiplier
    return meta.awareness_multiplier
