"""Pre-calibration bias detection suite.

Each function returns a dict so the calibration report can dump everything to
JSON. None of these *gate* a run by themselves — they surface signals the
operator must address before trusting the validation metrics.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from launchlens.phase1.schemas import AgentPersona
from launchlens.phase3.schemas import AgentDecision

_UPPER_TIERS = ("A1", "A2", "A3", "B1", "B2")
_LOWER_TIERS = ("D1", "D2", "E1", "E2", "E3")


def _buy_rate(decisions: Iterable[AgentDecision], agent_ids: set[str]) -> float:
    matching = [d for d in decisions if d.agent_id in agent_ids]
    if not matching:
        return 0.0
    return sum(1 for d in matching if d.decision == "BUY") / len(matching)


def affluence_bias(
    decisions: Sequence[AgentDecision],
    personas: Sequence[AgentPersona],
    flag_threshold: float = 0.20,
) -> dict:
    """Compare upper-tier vs lower-tier BUY rates. Flag if upper:lower ratio is implausible."""
    upper_ids = {p.agent_id for p in personas if p.demographic.isec_tier in _UPPER_TIERS}
    lower_ids = {p.agent_id for p in personas if p.demographic.isec_tier in _LOWER_TIERS}
    upper_rate = _buy_rate(decisions, upper_ids)
    lower_rate = _buy_rate(decisions, lower_ids)

    # We expect upper > lower for premium products; flag when lower-tier BUY rate
    # exceeds upper-tier rate by more than ``flag_threshold`` (LLM affluence bias
    # would more typically *overstate* lower-tier purchase intent).
    flagged = (lower_rate - upper_rate) > flag_threshold
    return {
        "upper_tier_buy_rate": round(upper_rate, 4),
        "lower_tier_buy_rate": round(lower_rate, 4),
        "delta": round(lower_rate - upper_rate, 4),
        "flagged": flagged,
        "note": "lower-tier BUY rate exceeds upper-tier BUY rate beyond threshold"
                if flagged else "ok",
    }


def positivity_bias(
    decisions: Sequence[AgentDecision],
    category_benchmark_reject_rate: float | None = None,
    flag_threshold: float = 0.15,
) -> dict:
    """Compare simulated REJECT rate against a known category benchmark."""
    if not decisions:
        return {"sim_reject_rate": 0.0, "flagged": False, "note": "no decisions"}
    total = len(decisions)
    sim_reject = sum(1 for d in decisions if d.decision == "REJECT") / total
    out: dict = {"sim_reject_rate": round(sim_reject, 4)}
    if category_benchmark_reject_rate is None:
        out["flagged"] = False
        out["note"] = "no benchmark supplied"
        return out
    delta = category_benchmark_reject_rate - sim_reject
    out["benchmark_reject_rate"] = category_benchmark_reject_rate
    out["delta"] = round(delta, 4)
    out["flagged"] = delta > flag_threshold
    out["note"] = (
        "simulated REJECT rate is substantially lower than the category benchmark "
        "(positivity bias suspected)"
        if out["flagged"]
        else "ok"
    )
    return out


def homogeneity_gini(
    decisions: Sequence[AgentDecision],
    personas: Sequence[AgentPersona],
    flag_threshold: float = 0.30,
) -> dict:
    """Within-cohort decision diversity (Gini). Higher = more diverse.

    Returns the population-weighted mean Gini across ISEC cohorts. Flag if
    the *mean* Gini falls below ``flag_threshold`` (cohorts behave too uniformly).
    """
    by_tier: dict[str, list[str]] = defaultdict(list)
    persona_tier = {p.agent_id: p.demographic.isec_tier for p in personas}
    for d in decisions:
        tier = persona_tier.get(d.agent_id)
        if tier:
            by_tier[tier].append(d.decision)

    cohort_ginis: list[tuple[str, float, int]] = []
    for tier, states in by_tier.items():
        if len(states) < 2:
            continue
        counter = Counter(states)
        # 1 - sum(p_i^2): Gini-like impurity index in [0, 1)
        ps = [c / len(states) for c in counter.values()]
        impurity = 1.0 - sum(p * p for p in ps)
        cohort_ginis.append((tier, impurity, len(states)))

    if not cohort_ginis:
        return {"mean_gini": float("nan"), "per_tier": {}, "flagged": False}

    weight_sum = sum(n for _, _, n in cohort_ginis)
    mean_gini = sum(g * n for _, g, n in cohort_ginis) / weight_sum
    return {
        "mean_gini": round(mean_gini, 4),
        "per_tier": {t: round(g, 4) for t, g, _ in cohort_ginis},
        "flagged": mean_gini < flag_threshold,
        "note": "ISEC cohorts behave too uniformly (homogeneity bias)"
                if mean_gini < flag_threshold
                else "ok",
    }


def language_bias_sample(
    decisions: Sequence[AgentDecision],
    personas: Sequence[AgentPersona],
    k: int = 50,
    out_path: Path | None = None,
) -> dict:
    """Sample Indic-language reasoning outputs for human review.

    Writes a JSONL file (defaults to ``outputs/language_audit_<n>.jsonl``)
    containing the sampled (agent, language, reasoning) tuples. Returns
    a summary dict with the sample size and language counts.
    """
    persona_lang = {p.agent_id: p.demographic.primary_language for p in personas}
    indic = [
        d for d in decisions
        if persona_lang.get(d.agent_id, "english").lower() != "english"
    ]
    sample = indic[:k]
    out_path = out_path or Path("outputs") / f"language_audit_{len(sample)}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for d in sample:
            f.write(json.dumps({
                "agent_id": d.agent_id,
                "language": persona_lang.get(d.agent_id, "unknown"),
                "decision": d.decision,
                "reasoning": d.internal_reasoning,
            }) + "\n")
    lang_counts = Counter(persona_lang.get(d.agent_id, "unknown") for d in sample)
    return {
        "sample_size": len(sample),
        "languages": dict(lang_counts),
        "sample_path": str(out_path),
        "note": "Submit this file for human authenticity review on a 1-5 scale.",
    }


def run_bias_suite(
    decisions: Sequence[AgentDecision],
    personas: Sequence[AgentPersona],
    category_benchmark_reject_rate: float | None = None,
) -> dict:
    return {
        "affluence": affluence_bias(decisions, personas),
        "positivity": positivity_bias(decisions, category_benchmark_reject_rate),
        "homogeneity": homogeneity_gini(decisions, personas),
        # language_bias_sample writes a file — only invoke explicitly via CLI.
    }
