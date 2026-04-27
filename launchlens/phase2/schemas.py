"""Phase 2 data models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

InfluencerArchetype = Literal[
    "family_elder", "local_shopkeeper", "micro_influencer", "whatsapp_hub", "standard"
]


class NodeMeta(BaseModel):
    agent_id: str
    archetype: InfluencerArchetype = "standard"
    # Multipliers applied when this node's social actions propagate
    awareness_multiplier: float = 1.0
    trust_multiplier: float = 1.0


class SimGraph(BaseModel):
    """Serializable graph representation."""
    node_ids: list[str]                        # ordered list; index = node index
    adjacency: dict[str, list[str]]            # agent_id → [neighbor agent_ids]
    node_meta: dict[str, NodeMeta]             # agent_id → metadata
    k: int
    beta: float
    n_agents: int

    def neighbors(self, agent_id: str) -> list[str]:
        return self.adjacency.get(agent_id, [])
