"""Phase 3 data models — simulation environment."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DecisionState = Literal[
    "IGNORE", "AWARE", "RESEARCH", "CONSIDER",
    "BUY", "REJECT", "SHARE_POSITIVE", "SHARE_NEGATIVE", "COMPLAIN",
]

# States that propagate to the network
PROPAGATING_STATES: set[DecisionState] = {
    "SHARE_POSITIVE", "SHARE_NEGATIVE", "COMPLAIN", "BUY",
}

DiscussTarget = Literal["family", "friends", "colleagues", "no_one"]


class ProductStimulus(BaseModel):
    product_id: str
    product_name: str
    category: str
    price_mrp: int
    price_launch: int
    currency: str = "INR"
    key_features: list[str]
    distribution_channels: list[str]
    marketing_copy: str
    competitor_context: str
    target_segment: str

    def _symbol(self) -> str:
        return "₹" if self.currency == "INR" else f"{self.currency} "

    def render_for_agent(self) -> str:
        feats = "\n".join(f"  - {f}" for f in self.key_features)
        sym = self._symbol()
        return (
            f"Product: {self.product_name}\n"
            f"Category: {self.category}\n"
            f"Price: {sym}{self.price_launch} (MRP {sym}{self.price_mrp})\n"
            f"Key features:\n{feats}\n"
            f"Available at: {', '.join(self.distribution_channels)}\n"
            f"Ad copy: \"{self.marketing_copy}\"\n"
            f"Competitors: {self.competitor_context}"
        )


class PeerSignal(BaseModel):
    """A social signal from one agent propagating to another."""
    from_agent_id: str
    decision: DecisionState
    reason: str
    salience: float = 1.0        # degrades 30% per hop
    timestep: int = 0
    archetype_hint: str = "standard"   # influences trust weighting


class AgentMemory(BaseModel):
    """Per-agent memory. Persisted across timesteps."""
    agent_id: str
    biography: str                              # immutable
    episodic_buffer: list[str] = Field(default_factory=list)   # last 10 events
    product_opinion: dict[str, str] = Field(default_factory=dict)       # product_id→text
    purchase_history: list[dict] = Field(default_factory=list)
    current_decision: dict[str, DecisionState] = Field(default_factory=dict)  # product_id→state
    peer_signals: list[PeerSignal] = Field(default_factory=list)         # pending signals

    def add_event(self, event: str) -> None:
        self.episodic_buffer.append(event)
        if len(self.episodic_buffer) > 10:
            self.episodic_buffer.pop(0)

    def latest_opinion(self, product_id: str) -> str:
        return self.product_opinion.get(product_id, "You have not formed an opinion yet.")

    def pending_peer_signals(self, product_id: str | None = None) -> list[PeerSignal]:
        # product_id is reserved for future multi-product support;
        # signals currently belong to a single-product simulation.
        return [s for s in self.peer_signals if s.salience > 0.05]


class AgentDecision(BaseModel):
    agent_id: str
    product_id: str
    timestep: int
    internal_reasoning: str
    decision: DecisionState
    primary_reason: str
    would_discuss_with: DiscussTarget
    language_of_discussion: str


class MarketplaceFeed(BaseModel):
    """The environment state an agent sees at one timestep."""
    product_stimulus: str           # rendered ProductStimulus text
    peer_reviews: list[PeerSignal]  # up to 5
    peer_purchases: list[PeerSignal]  # up to 3
    competitor_mention: str = ""
    market_noise: str = ""
