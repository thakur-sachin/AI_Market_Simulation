"""
Phase 5 — Bias detection suite.

Run before every calibration cycle to surface systematic distortions in the
agent population's outputs:
  - affluence_bias:  Are SEC D/E agents over- or under-buying vs published benchmarks?
  - positivity_bias: Is REJECT rate suspiciously low (LLMs tend to over-generate BUY)?
  - homogeneity_bias: Within a single ISEC tier, do agents differ enough? (Gini > 0.3)
  - language_bias:   Placeholder — requires human raters; surfaces samples to review.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from launchlens.phase1.schemas import AgentPersona
from launchlens.phase3.schemas import AgentDecision


@dataclass
class BiasFlag:
    name: str
    value: float
    threshold: str
    flagged: bool
    note: str = ""


# ── Affluence bias ───────────────────────────────────────────────────────────

_LOWER_TIERS = {"D1", "D2", "E1", "E2", "E3"}


def affluence_bias(
    decisions: Sequence[AgentDecision],
    personas: Sequence[AgentPersona],
    expected_low_tier_buy_rate: float = 0.05,
    tolerance: float = 0.20,
) -> BiasFlag:
    """
    Compare D/E tier BUY rate against an expected benchmark.
    Default benchmark: 5% (rural/low-income tend to be slow adopters of premium SKUs).
    Flag if abs deviation > 20%.
    """
    persona_by_id = {p.agent_id: p for p in personas}
    low_tier_ids = {p.agent_id for p in personas if p.demographic.isec_tier in _LOWER_TIERS}
    if not low_tier_ids:
        return BiasFlag("affluence_bias", 0.0, "n/a", False, "no low-tier agents in population")

    # Best decision per agent: BUY if ever reached, else final state
    ever_buy: set[str] = set()
    for d in decisions:
        if d.agent_id in low_tier_ids and d.decision == "BUY":
            ever_buy.add(d.agent_id)

    rate = len(ever_buy) / len(low_tier_ids)
    deviation = abs(rate - expected_low_tier_buy_rate)
    flagged = deviation > tolerance
    return BiasFlag(
        "affluence_bias", rate, f"|sim - {expected_low_tier_buy_rate:.0%}| < {tolerance:.0%}",
        flagged,
        f"low-tier BUY rate {rate:.1%}; expected ~{expected_low_tier_buy_rate:.1%}",
    )


# ── Positivity bias ──────────────────────────────────────────────────────────

def positivity_bias(
    decisions: Sequence[AgentDecision],
    expected_reject_rate: float = 0.15,
    tolerance: float = 0.10,
) -> BiasFlag:
    """
    LLMs tend to over-generate BUY. Flag if REJECT rate is < (expected - tolerance).
    Uses the latest decision per agent.
    """
    latest: dict[str, str] = {}
    for d in sorted(decisions, key=lambda x: x.timestep):
        latest[d.agent_id] = d.decision

    if not latest:
        return BiasFlag("positivity_bias", 0.0, "n/a", False, "no decisions recorded")

    counts = Counter(latest.values())
    n = len(latest)
    reject_rate = counts.get("REJECT", 0) / n
    flagged = reject_rate < (expected_reject_rate - tolerance)
    return BiasFlag(
        "positivity_bias", reject_rate, f">= {expected_reject_rate - tolerance:.0%}",
        flagged,
        f"REJECT rate {reject_rate:.1%} vs benchmark {expected_reject_rate:.1%}",
    )


# ── Homogeneity bias ─────────────────────────────────────────────────────────

def _gini(values: Sequence[float]) -> float:
    """Standard Gini coefficient. 0 = uniform, 1 = totally concentrated."""
    arr = sorted(values)
    n = len(arr)
    if n == 0 or sum(arr) == 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(arr, start=1):
        cum += i * v
    total = sum(arr)
    return (2 * cum) / (n * total) - (n + 1) / n


def homogeneity_bias(
    decisions: Sequence[AgentDecision],
    personas: Sequence[AgentPersona],
    min_gini: float = 0.30,
) -> BiasFlag:
    """
    Within a single ISEC tier, agents should produce diverse decision distributions.
    A Gini < 0.30 on the latest decision histogram per tier signals lock-step behavior.
    Reports the worst (lowest) Gini across tiers with >=10 agents.
    """
    persona_by_id = {p.agent_id: p for p in personas}
    latest: dict[str, str] = {}
    for d in sorted(decisions, key=lambda x: x.timestep):
        latest[d.agent_id] = d.decision

    by_tier: dict[str, list[str]] = defaultdict(list)
    for aid, state in latest.items():
        p = persona_by_id.get(aid)
        if not p:
            continue
        by_tier[p.demographic.isec_tier].append(state)

    worst_tier: str | None = None
    worst_gini: float = 1.0
    for tier, states in by_tier.items():
        if len(states) < 10:
            continue
        hist = Counter(states)
        # Gini over the count distribution; uniform across 9 states would be near 0
        gini = _gini(list(hist.values()))
        if gini < worst_gini:
            worst_gini = gini
            worst_tier = tier

    if worst_tier is None:
        return BiasFlag("homogeneity_bias", 0.0, f">= {min_gini}", False,
                        "not enough agents per tier (need >=10)")

    flagged = worst_gini < min_gini
    return BiasFlag(
        "homogeneity_bias", worst_gini, f">= {min_gini}", flagged,
        f"worst tier: {worst_tier} (gini={worst_gini:.2f})",
    )


# ── Language bias (audit-only) ───────────────────────────────────────────────

def language_audit_sample(
    decisions: Sequence[AgentDecision],
    personas: Sequence[AgentPersona],
    sample_size: int = 50,
) -> list[dict]:
    """
    Surface a sample of non-English internal_reasoning outputs for human review.
    Returns a list of {agent_id, language, reasoning} dicts.
    Human raters score these on a 1-5 cultural authenticity scale.
    """
    persona_by_id = {p.agent_id: p for p in personas}
    candidates = [
        {
            "agent_id": d.agent_id,
            "language": persona_by_id[d.agent_id].demographic.primary_language,
            "isec_tier": persona_by_id[d.agent_id].demographic.isec_tier,
            "reasoning": d.internal_reasoning,
        }
        for d in decisions
        if d.agent_id in persona_by_id
        and persona_by_id[d.agent_id].demographic.primary_language.lower() != "english"
    ]
    return candidates[:sample_size]


# ── Aggregate ────────────────────────────────────────────────────────────────

def run_bias_suite(
    decisions: Sequence[AgentDecision],
    personas: Sequence[AgentPersona],
) -> list[BiasFlag]:
    return [
        affluence_bias(decisions, personas),
        positivity_bias(decisions),
        homogeneity_bias(decisions, personas),
    ]
