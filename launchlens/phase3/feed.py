"""Marketplace feed builder — personalizes per agent from network connections."""
from __future__ import annotations

import random

from launchlens.phase2.schemas import SimGraph
from launchlens.phase3.schemas import (
    AgentMemory,
    DecisionState,
    MarketplaceFeed,
    PeerSignal,
    ProductStimulus,
    PROPAGATING_STATES,
)

_MARKET_NOISE = [
    "Health and wellness products are trending on Instagram this week.",
    "Consumers are increasingly price-conscious amid rising food inflation.",
    "New-age D2C brands are gaining trust through social proof and YouTube reviews.",
    "Quick-commerce delivery is changing impulse purchase behaviour in metros.",
    "WhatsApp forwards are driving awareness of health products in Tier-2 cities.",
]

_COMPETITOR_MENTIONS = [
    "A competitor launched a similar product at a lower price point last month.",
    "Established brands are running heavy discount campaigns this season.",
    "A foreign brand recently entered the Indian market with aggressive pricing.",
]


def build_feed(
    agent_id: str,
    memory: AgentMemory,
    product: ProductStimulus,
    graph: SimGraph,
    all_memories: dict[str, AgentMemory],
    timestep: int,
    rng: random.Random,
) -> MarketplaceFeed:
    neighbors = graph.neighbors(agent_id)
    neighbor_memories = [all_memories[n] for n in neighbors if n in all_memories]

    # Peer reviews: neighbors who have formed an opinion
    reviews: list[PeerSignal] = []
    purchases: list[PeerSignal] = []

    for nm in neighbor_memories:
        state = nm.current_decision.get(product.product_id)
        if state is None:
            continue
        opinion_text = nm.latest_opinion(product.product_id)
        nm_node = graph.node_meta.get(nm.agent_id)
        signal = PeerSignal(
            from_agent_id=nm.agent_id,
            decision=state,
            reason=opinion_text[:200],
            salience=1.0,
            timestep=timestep,
            archetype_hint=nm_node.archetype if nm_node else "standard",
        )
        if state == "BUY":
            purchases.append(signal)
        elif state in ("SHARE_POSITIVE", "SHARE_NEGATIVE", "COMPLAIN", "REJECT", "CONSIDER"):
            reviews.append(signal)

    # Also pull in propagated peer signals from memory
    pending = memory.pending_peer_signals(product.product_id)
    reviews.extend(pending)

    # Deduplicate by from_agent_id, keep highest salience
    seen: dict[str, PeerSignal] = {}
    for sig in reviews:
        if sig.from_agent_id not in seen or sig.salience > seen[sig.from_agent_id].salience:
            seen[sig.from_agent_id] = sig
    reviews = sorted(seen.values(), key=lambda s: -s.salience)[:5]
    purchases = purchases[:3]

    return MarketplaceFeed(
        product_stimulus=product.render_for_agent(),
        peer_reviews=reviews,
        peer_purchases=purchases,
        competitor_mention=rng.choice(_COMPETITOR_MENTIONS) if rng.random() < 0.4 else "",
        market_noise=rng.choice(_MARKET_NOISE) if rng.random() < 0.6 else "",
    )


def render_feed_text(feed: MarketplaceFeed) -> str:
    parts = [f"[PRODUCT]\n{feed.product_stimulus}"]

    if feed.peer_reviews:
        reviews_text = "\n".join(
            f"  • {s.from_agent_id[:8]}… says ({s.decision}): {s.reason[:120]}"
            for s in feed.peer_reviews
        )
        parts.append(f"[PEER OPINIONS]\n{reviews_text}")

    if feed.peer_purchases:
        parts.append(
            f"[PEER PURCHASES]\n"
            + "\n".join(f"  • {s.from_agent_id[:8]}… purchased this product." for s in feed.peer_purchases)
        )

    if feed.competitor_mention:
        parts.append(f"[MARKET CONTEXT]\n  {feed.competitor_mention}")

    if feed.market_noise:
        parts.append(f"[CATEGORY TRENDS]\n  {feed.market_noise}")

    return "\n\n".join(parts)
