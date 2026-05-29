"""Calibration harness: load a real-product case, score a SimulationLog, emit tuning signals."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from launchlens.config import get_settings
from launchlens.phase5.metrics import (
    adoption_rate_deviation,
    dtw_curve_distance,
    passes_all_gates,
    regional_spearman,
    rejection_reason_alignment,
    top_segment_accuracy,
)

_cfg = get_settings()


@dataclass
class CalibrationCase:
    """Ground-truth record for one historical product launch."""
    product_id: str
    product_name: str
    category: str
    district_id: str
    real_adoption_curve: list[float]       # cumulative fraction reaching BUY per timestep
    real_top3_segments: list[str]          # e.g. ["A1", "A2", "B1"]
    real_top3_rejections: list[str]        # natural-language top objection themes
    real_district_rates: dict[str, float] = field(default_factory=dict)
    category_reject_benchmark: float | None = None
    source_citations: list[str] = field(default_factory=list)
    # Optional ProductStimulus payload. When present, `cmd_run_sim --calibrate <id>`
    # uses this to drive the simulation so the sim and the ground truth are about
    # the same product. Without it, the CLI falls back to its hardcoded demo product.
    product: dict | None = None


@dataclass
class CalibrationReport:
    product_id: str
    engine: str
    metrics: dict[str, float]
    gates: dict[str, bool]
    tuning_signals: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def passed(self) -> bool:
        return all(self.gates.values())


def load_calibration_case(product_id: str, base_dir: Path | None = None) -> CalibrationCase:
    base_dir = base_dir or _cfg.calibration_dir
    path = base_dir / f"{product_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No calibration case at {path}. "
            f"Available: {[p.stem for p in base_dir.glob('*.json')]}"
        )
    data = json.loads(path.read_text())
    return CalibrationCase(**data)


def _sim_top_segments(sim_summary: dict, top_n: int = 3) -> list[str]:
    """Pull top-N ISEC tiers by BUY count out of a sim summary, if available.

    Sim summaries currently only carry the adoption curve; segment-level data
    requires the full SimulationLog. For now we return whatever the summary
    contains under ``top_segments``; otherwise an empty list.
    """
    return list(sim_summary.get("top_segments", []))[:top_n]


def _sim_rejection_reasons(sim_summary: dict) -> list[str]:
    return list(sim_summary.get("rejection_reasons", []))


def _sim_district_rates(sim_summary: dict) -> dict[str, float]:
    return dict(sim_summary.get("district_rates", {}))


def run_calibration(case: CalibrationCase, sim_summary: dict) -> CalibrationReport:
    sim_curve = list(sim_summary.get("adoption_curve") or [])
    real_curve = case.real_adoption_curve

    sim_final = sim_curve[-1] if sim_curve else 0.0
    real_final = real_curve[-1] if real_curve else 0.0

    metrics: dict[str, float] = {
        "adoption_rate_deviation": adoption_rate_deviation(sim_final, real_final),
        "dtw_curve_distance": dtw_curve_distance(sim_curve, real_curve),
        "top_segment_accuracy": top_segment_accuracy(
            _sim_top_segments(sim_summary), case.real_top3_segments,
        ),
        "regional_spearman": regional_spearman(
            _sim_district_rates(sim_summary), case.real_district_rates,
        ),
        "rejection_reason_alignment": rejection_reason_alignment(
            _sim_rejection_reasons(sim_summary), case.real_top3_rejections,
        ),
    }
    gates = passes_all_gates(metrics)
    signals = tune_signal(metrics, sim_curve, real_curve)
    return CalibrationReport(
        product_id=case.product_id,
        engine=sim_summary.get("engine", "unknown"),
        metrics={k: _to_jsonable(v) for k, v in metrics.items()},
        gates=gates,
        tuning_signals=signals,
    )


def _to_jsonable(v: float) -> float:
    if isinstance(v, float):
        if v != v:  # NaN
            return None  # type: ignore[return-value]
        if v in (float("inf"), float("-inf")):
            return None  # type: ignore[return-value]
    return v


def tune_signal(metrics: dict[str, float],
                sim_curve: list[float],
                real_curve: list[float]) -> dict[str, str]:
    """Produce one structured tuning recommendation per failed gate.

    Aligns with the NEXT_STEPS.md tuning table:
      adoption too high  → increase price sensitivity
      adoption too low   → increase social proof multiplier
      wrong segments     → audit sample_demographic_vectors
      curve too fast     → reduce archetype speed
      wrong regional spread → audit district disaggregation
    """
    signals: dict[str, str] = {}
    if not sim_curve or not real_curve:
        return signals

    sim_final = sim_curve[-1]
    real_final = real_curve[-1]
    dev = metrics.get("adoption_rate_deviation", 0.0)

    if dev > 0.08:
        if sim_final > real_final:
            signals["adoption_rate"] = (
                "Simulated adoption EXCEEDS real. Increase price_sensitivity_weight in "
                "persona prompts; reduce _ISEC_BASE_BUY for lower tiers; sharpen the "
                "anti-positivity prior in phase4/prompts.py."
            )
        else:
            signals["adoption_rate"] = (
                "Simulated adoption UNDERSHOOTS real. Increase social_proof_multiplier in "
                "the decision prompt; raise propagation salience floor or extend "
                "_COMPLAIN_BOOST symmetry to BUY signals."
            )

    if metrics.get("dtw_curve_distance", 0.0) > 0.15:
        # Compare curve shape: fast vs slow
        if _curve_is_too_fast(sim_curve, real_curve):
            signals["curve_speed"] = (
                "Adoption curve climbs too quickly. Reduce _ARCHETYPE_SPEED multipliers "
                "for innovator/early_adopter; raise social-signal advancement threshold."
            )
        else:
            signals["curve_speed"] = (
                "Adoption curve climbs too slowly. Reduce timestep duration, increase "
                "max peer signals per agent, or relax the social-signal threshold."
            )

    if metrics.get("top_segment_accuracy", 0) < 2:
        signals["segments"] = (
            "Wrong demographic segments are buying. Audit sample_demographic_vectors "
            "for ISEC distribution accuracy; verify district profile provenance is "
            "real (not 'fallback')."
        )

    rho = metrics.get("regional_spearman", float("nan"))
    if rho == rho and rho < 0.70:  # not NaN
        signals["regional"] = (
            "Regional adoption pattern misaligned. Audit _disaggregate_isec for "
            "state-level adjustments; verify NFHS-5 wealth-quintile mapping; "
            "confirm cross-district edge weighting in phase2/graph."
        )

    if metrics.get("rejection_reason_alignment", 0) < 2:
        signals["rejection_reasons"] = (
            "Rejection reasoning doesn't match real-world objections. Inspect "
            "internal_reasoning samples; consider richer ProductStimulus context "
            "(competitor framing, distribution channels)."
        )
    return signals


def _curve_is_too_fast(sim_curve: list[float], real_curve: list[float]) -> bool:
    """Heuristic: cumulative AUC of sim vs real."""
    n = min(len(sim_curve), len(real_curve))
    sim_auc = sum(sim_curve[:n])
    real_auc = sum(real_curve[:n])
    return sim_auc > real_auc


# ── SimulationLog adapter ────────────────────────────────────────────────────
# Build the ``sim_summary`` dict ``run_calibration`` expects directly from a
# ``SimulationLog`` + the persona list that produced it. Lives here (not in
# loop.py) so phase5 owns the contract.

def _isec_band(isec: str) -> str:
    if isec in ("A1", "A2", "A3"):
        return "A1-A3"
    if isec in ("B1", "B2"):
        return "B1-B2"
    if isec in ("C1", "C2"):
        return "C1-C2"
    if isec in ("D1", "D2"):
        return "D1-D2"
    return "E1-E3"


def build_sim_summary(sim_log, personas) -> dict:
    """Project a SimulationLog + persona list into the dict run_calibration consumes.

    Carries adoption curve, top ISEC tiers among buyers, most-common rejection
    reasons, and per-district adoption rates. ``top_segments`` uses raw ISEC
    tier names (e.g. ``"A2"``, ``"B1"``) to match the format used in
    ``CalibrationCase.real_top3_segments``.
    """
    from collections import Counter, defaultdict

    persona_by_id = {p.agent_id: p for p in personas}
    decisions = sim_log.all_decisions() if hasattr(sim_log, "all_decisions") else []

    # Top segments: raw ISEC tier counts among unique buyers.
    seg_counts: Counter[str] = Counter()
    seen_buyer: set[str] = set()
    for d in decisions:
        if d.decision != "BUY" or d.agent_id in seen_buyer:
            continue
        seen_buyer.add(d.agent_id)
        p = persona_by_id.get(d.agent_id)
        if p is None:
            continue
        seg_counts[p.demographic.isec_tier] += 1
    top_segments = [seg for seg, _ in seg_counts.most_common(3)]

    # Top rejection reasons
    rej_counts = Counter(
        d.primary_reason for d in decisions
        if d.decision in ("REJECT", "COMPLAIN") and d.primary_reason.strip()
    )
    rejection_reasons = [r for r, _ in rej_counts.most_common(10)]

    # District rates
    district_totals: Counter[str] = Counter(p.demographic.district_id for p in personas)
    district_buyers: defaultdict[str, set[str]] = defaultdict(set)
    for d in decisions:
        if d.decision != "BUY":
            continue
        p = persona_by_id.get(d.agent_id)
        if p is None:
            continue
        district_buyers[p.demographic.district_id].add(d.agent_id)
    district_rates = {
        d: len(district_buyers[d]) / district_totals[d]
        for d in district_totals if district_totals[d]
    }

    return {
        "engine": getattr(sim_log, "engine", "unknown"),
        "adoption_curve": sim_log.adoption_curve(),
        "top_segments": top_segments,
        "rejection_reasons": rejection_reasons,
        "district_rates": district_rates,
    }


def calibrate_from_sim_log(sim_log, personas, case: CalibrationCase) -> CalibrationReport:
    """Run the 5-gate calibration directly against a SimulationLog."""
    summary = build_sim_summary(sim_log, personas)
    return run_calibration(case, summary)
