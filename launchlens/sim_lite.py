"""
Lightweight in-memory simulation for decision evolution validation.

No Redis, no pgvector, no real LLM calls needed.
Uses a stochastic mock LLM that models realistic Indian consumer decision dynamics:
  - Price sensitivity by ISEC tier
  - Social proof influence from peer signals
  - Adoption archetype (innovator→laggard) affects adoption speed
  - Homophily means early adopters cluster → cascade through network

Run:
    python -m launchlens.sim_lite [--agents N] [--timesteps T] [--seed S] [--verbose]

Output: per-timestep decision distribution + adoption curve printed to terminal.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from launchlens.phase1.schemas import AgentPersona, AgeDistribution, DistrictProfile
from launchlens.phase1.persona_gen import sample_demographic_vectors
from launchlens.phase2.graph import build_graph, validate_small_world
from launchlens.phase2.influencers import inject_influencers
from launchlens.phase2.schemas import NodeMeta, SimGraph
from launchlens.phase3.schemas import (
    AgentDecision, AgentMemory, DecisionState,
    PeerSignal, ProductStimulus, PROPAGATING_STATES,
)
from launchlens.phase3.memory import MemoryStore
from launchlens.phase3.feed import build_feed
from launchlens.phase4.loop import SimulationLog, TimestepLog
from launchlens.phase4.propagation import propagate_decisions


# ── Mock district profile (Indore) ───────────────────────────────────────────

def _indore_profile() -> DistrictProfile:
    return DistrictProfile(
        district_id="MP001", district_name="Indore", state_name="Madhya Pradesh",
        population=3_276_697,
        age_distribution=AgeDistribution(
            bucket_0_4=0.09, bucket_5_14=0.18, bucket_15_24=0.18,
            bucket_25_34=0.17, bucket_35_44=0.14, bucket_45_54=0.10,
            bucket_55_64=0.07, bucket_65_plus=0.07,
        ),
        sex_ratio=920, urban_share=0.70, literacy_rate=0.82,
        language_distribution={"hindi": 0.85, "urdu": 0.08, "english": 0.07},
        isec_distribution={
            "A1": 0.03, "A2": 0.05, "A3": 0.07, "B1": 0.09, "B2": 0.10,
            "C1": 0.12, "C2": 0.12, "D1": 0.13, "D2": 0.12,
            "E1": 0.08, "E2": 0.05, "E3": 0.04,
        },
        median_monthly_hh_expenditure=18000,
        smartphone_penetration=0.62,
        internet_penetration=0.50,
        upi_adoption=0.38,
    )


# ── Stochastic mock LLM decision engine ──────────────────────────────────────

# Base BUY probability by ISEC tier (price-sensitivity model)
_ISEC_BASE_BUY: dict[str, float] = {
    "A1": 0.55, "A2": 0.45, "A3": 0.38,
    "B1": 0.28, "B2": 0.20,
    "C1": 0.13, "C2": 0.09,
    "D1": 0.05, "D2": 0.03,
    "E1": 0.02, "E2": 0.01, "E3": 0.005,
}

# Adoption archetype affects how much social proof is needed
_ARCHETYPE_SPEED: dict[str, float] = {
    "innovator": 2.0,
    "early_adopter": 1.5,
    "early_majority": 1.0,
    "late_majority": 0.6,
    "laggard": 0.3,
}

# States in the consumer decision funnel (roughly ordered)
_FUNNEL: list[DecisionState] = [
    "IGNORE", "AWARE", "RESEARCH", "CONSIDER", "BUY",
]


def _mock_decision(
    persona: AgentPersona,
    memory: AgentMemory,
    product: ProductStimulus,
    peer_signals: list[PeerSignal],
    rng: random.Random,
) -> AgentDecision:
    isec = persona.demographic.isec_tier
    archetype = persona.demographic.tech_adoption
    income = persona.demographic.monthly_hh_income
    price = product.price_launch

    # Price affordability: reject if price > 15% of monthly income for lower tiers
    affordability = min(1.0, income / (price * 6.67))

    # Social proof: count positive vs negative peer signals
    pos = sum(1 for s in peer_signals if s.decision in ("SHARE_POSITIVE", "BUY") and s.salience > 0.1)
    neg = sum(1 for s in peer_signals if s.decision in ("SHARE_NEGATIVE", "COMPLAIN", "REJECT"))
    social_boost = (pos * 0.08) - (neg * 0.05)

    # Trust from influencer signals
    influencer_trust = sum(
        s.salience * 0.10
        for s in peer_signals
        if s.archetype_hint in ("family_elder", "local_shopkeeper") and s.decision == "BUY"
    )

    base_p = _ISEC_BASE_BUY.get(isec, 0.05) * affordability
    speed = _ARCHETYPE_SPEED.get(archetype, 1.0)
    p_buy = min(0.92, (base_p + social_boost + influencer_trust) * speed)

    # Where is agent in funnel currently?
    current = memory.current_decision.get(product.product_id, "IGNORE")
    try:
        funnel_idx = _FUNNEL.index(current)  # type: ignore[arg-type]
    except ValueError:
        funnel_idx = 0

    roll = rng.random()

    # Funnel progression logic
    if funnel_idx < 4:  # hasn't bought yet
        advance_p = p_buy * (0.3 + 0.2 * funnel_idx)   # faster as awareness grows
        if roll < advance_p:
            new_state: DecisionState = _FUNNEL[min(funnel_idx + 1, 4)]
        elif roll < advance_p + 0.08 and neg > pos:
            new_state = "REJECT"
        else:
            new_state = current  # type: ignore[assignment]
    elif current == "BUY":
        # Post-buy: share or complain
        if roll < 0.25:
            new_state = "SHARE_POSITIVE"
        elif roll < 0.30:
            new_state = "COMPLAIN"
        else:
            new_state = "BUY"
    else:
        new_state = current  # type: ignore[assignment]

    reasons: dict[DecisionState, str] = {
        "IGNORE": "Not relevant to my current needs.",
        "AWARE": "Noticed this product but haven't looked into it yet.",
        "RESEARCH": "Seems interesting — want to read more reviews before deciding.",
        "CONSIDER": "Seriously evaluating; comparing with alternatives.",
        "BUY": f"Convinced by the value. ₹{price} fits my budget and I trust the reviews.",
        "REJECT": f"At ₹{price} this is too expensive given alternatives available.",
        "SHARE_POSITIVE": "Happy with the purchase — telling my friends about it.",
        "SHARE_NEGATIVE": "Disappointed with the product — warning others.",
        "COMPLAIN": "The product did not meet expectations.",
    }

    lang = persona.demographic.primary_language
    discuss = "no_one" if new_state in ("IGNORE","AWARE") else (
        "family" if rng.random() < 0.4 else "friends"
    )

    reasoning = (
        f"Given my income of ₹{income:,} and the price of ₹{price}, "
        f"affordability is {'comfortable' if affordability > 0.7 else 'a concern'}. "
        f"I've seen {pos} positive and {neg} negative signals from people I know. "
        f"As a {archetype.replace('_',' ')}, I {'act quickly' if speed > 1.2 else 'prefer to wait and watch'}."
    )

    return AgentDecision(
        agent_id=persona.agent_id,
        product_id=product.product_id,
        timestep=0,
        internal_reasoning=reasoning,
        decision=new_state,
        primary_reason=reasons.get(new_state, ""),
        would_discuss_with=discuss,  # type: ignore[arg-type]
        language_of_discussion=lang,
    )


# ── Terminal visualisation ────────────────────────────────────────────────────

_BARS = "▏▎▍▌▋▊▉█"
_STATE_ORDER: list[DecisionState] = [
    "IGNORE", "AWARE", "RESEARCH", "CONSIDER",
    "BUY", "REJECT", "SHARE_POSITIVE", "SHARE_NEGATIVE", "COMPLAIN",
]
_STATE_COLORS = {
    "BUY": "\033[92m", "SHARE_POSITIVE": "\033[96m",
    "REJECT": "\033[91m", "SHARE_NEGATIVE": "\033[91m", "COMPLAIN": "\033[91m",
    "CONSIDER": "\033[93m", "RESEARCH": "\033[93m",
    "AWARE": "\033[37m", "IGNORE": "\033[90m",
}
_RESET = "\033[0m"


def _bar(val: float, width: int = 30) -> str:
    filled = int(val * width)
    return "█" * filled + "░" * (width - filled)


def _print_timestep(t: int, counts: dict[str, int], n: int, cum_buyers: int) -> None:
    adoption = cum_buyers / n if n else 0
    print(f"\n{'─'*60}")
    print(f"  Timestep {t:2d}  │  Agents: {n}  │  Cumulative adoption: {adoption:.1%}")
    print(f"{'─'*60}")
    for state in _STATE_ORDER:
        c = counts.get(state, 0)
        share = c / n if n else 0
        color = _STATE_COLORS.get(state, "")
        bar = _bar(share)
        print(f"  {color}{state:<16}{_RESET} {bar}  {c:4d} ({share:5.1%})")


def _print_adoption_curve(curve: list[float]) -> None:
    print(f"\n{'═'*60}")
    print("  ADOPTION CURVE (cumulative BUY rate per timestep)")
    print(f"{'═'*60}")
    max_val = max(curve) if curve else 1
    for t, val in enumerate(curve):
        bar_len = int((val / max(max_val, 0.001)) * 40)
        bar = "█" * bar_len
        print(f"  t{t:02d} │{bar:<40}│ {val:.1%}")


def _print_consensus_metrics(all_decisions: list[dict[str, Any]], n: int) -> None:
    """Show how decision distribution entropy decreases (consensus forming)."""
    print(f"\n{'═'*60}")
    print("  CONSENSUS EVOLUTION (decision entropy per timestep)")
    print(f"{'═'*60}")
    for entry in all_decisions:
        counts = entry["counts"]
        total = sum(counts.values())
        if total == 0:
            continue
        probs = [c / total for c in counts.values() if c > 0]
        entropy = -sum(p * math.log2(p) for p in probs)
        max_entropy = math.log2(len(_STATE_ORDER))
        norm_entropy = entropy / max_entropy
        bar = _bar(1 - norm_entropy, 30)   # high bar = high consensus
        print(f"  t{entry['t']:02d} │ consensus {bar}│ entropy={entropy:.2f} bits")


# ── Main simulation ───────────────────────────────────────────────────────────

async def run_lite(
    n_agents: int = 100,
    n_timesteps: int = 8,
    seed: int = 42,
    verbose: bool = False,
) -> SimulationLog:
    rng = random.Random(seed)
    print(f"\n{'═'*60}")
    print(f"  LaunchLens Lite Simulation")
    print(f"  Agents: {n_agents}  │  Timesteps: {n_timesteps}  │  Seed: {seed}")
    print(f"{'═'*60}")

    # 1. Build population
    profile = _indore_profile()
    vectors = sample_demographic_vectors(profile, n=n_agents, seed=seed)
    personas = [
        AgentPersona(
            agent_id=f"agent_{i:04d}",
            demographic=v,
            biography=(
                f"{v.sex.title()}, age {v.age}, {v.occupation.replace('_',' ')}, "
                f"ISEC {v.isec_tier}, income ₹{v.monthly_hh_income:,}, "
                f"{v.primary_language}-speaking, {v.tech_adoption.replace('_',' ')}."
            ),
            llm_route="sarvam" if v.primary_language != "english" else "claude",
        )
        for i, v in enumerate(vectors)
    ]
    persona_map = {p.agent_id: p for p in personas}

    # 2. Build graph
    G = build_graph(personas, k=6, beta=0.15, seed=seed)
    node_meta = inject_influencers(G, personas, seed=seed)
    graph = SimGraph(
        node_ids=[p.agent_id for p in personas],
        adjacency={n: list(G.neighbors(n)) for n in G.nodes()},
        node_meta=node_meta,
        k=6, beta=0.15, n_agents=n_agents,
    )

    sw = validate_small_world(G, k=6)
    print(f"\n  Graph: σ={sw['small_world_sigma']:.2f}  "
          f"CC={sw['clustering_coefficient']:.3f}  "
          f"AvgPath={sw['avg_path_length']:.2f}")

    # 3. Product stimulus
    product = ProductStimulus(
        product_id="prod_001",
        product_name="FreshBite Protein Bar",
        category="Health & Nutrition",
        price_mrp=99, price_launch=79,
        key_features=["20g protein", "No added sugar", "Mango / Chocolate / Peanut flavors"],
        distribution_channels=["Amazon India", "BigBasket", "Modern Trade"],
        marketing_copy="Fuel your grind. India's first truly tasty protein bar.",
        competitor_context="Yoga Bar (₹50-80), RiteBite Max (₹80-120)",
        target_segment="Health-conscious urban millennials, 22-35, SEC A/B",
    )
    print(f"\n  Product: {product.product_name} @ ₹{product.price_launch}")

    # 4. Init memory (all in-memory)
    store = MemoryStore()
    await store.init_all({p.agent_id: p.biography for p in personas})

    # 5. Simulate
    sim_log = SimulationLog(product_id=product.product_id, n_agents=n_agents)
    all_decisions_meta: list[dict[str, Any]] = []
    cumulative_buyers: set[str] = set()

    for t in range(n_timesteps):
        all_memories = await store.get_many(graph.node_ids)
        ts_log = TimestepLog(timestep=t)

        for agent_id in graph.node_ids:
            mem = all_memories.get(agent_id)
            persona = persona_map.get(agent_id)
            if not mem or not persona:
                continue

            peer_signals = mem.pending_peer_signals(product.product_id)
            dec = _mock_decision(persona, mem, product, peer_signals, rng)
            dec.timestep = t
            ts_log.decisions.append(dec)

            mem.current_decision[product.product_id] = dec.decision
            mem.product_opinion[product.product_id] = dec.internal_reasoning
            mem.add_event(f"t{t}: {dec.decision} — {dec.primary_reason[:60]}")
            if dec.decision == "BUY":
                cumulative_buyers.add(agent_id)
                mem.purchase_history.append({"timestep": t, "product_id": product.product_id})
            await store.update(mem)

        propagate_decisions(ts_log.decisions, graph, all_memories, t)
        for mem in all_memories.values():
            await store.update(mem)

        counts = ts_log.decision_counts()
        all_decisions_meta.append({"t": t, "counts": counts})
        _print_timestep(t, counts, n_agents, len(cumulative_buyers))
        sim_log.timesteps.append(ts_log)

    # 6. Summary
    _print_adoption_curve(sim_log.adoption_curve())
    _print_consensus_metrics(all_decisions_meta, n_agents)

    final_counts = Counter(
        mem.current_decision.get(product.product_id, "IGNORE")
        for mem in (await store.get_many(graph.node_ids)).values()
    )
    print(f"\n{'═'*60}")
    print(f"  FINAL STATE  │  Adoption: {len(cumulative_buyers)/n_agents:.1%}  "
          f"│  Buyers: {final_counts.get('BUY',0)}  "
          f"│  Rejected: {final_counts.get('REJECT',0)}")
    print(f"{'═'*60}\n")

    return sim_log


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LaunchLens Lite Sim")
    p.add_argument("--agents", type=int, default=100)
    p.add_argument("--timesteps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    asyncio.run(run_lite(args.agents, args.timesteps, args.seed, args.verbose))
