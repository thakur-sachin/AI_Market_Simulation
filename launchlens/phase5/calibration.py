"""
Phase 5 — Calibration runner.

Wires together: load a calibration fixture (real product launch data),
compute simulated metrics from a SimulationLog, and evaluate all 5 validation
gates plus the bias suite.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from launchlens.phase1.schemas import AgentPersona
from launchlens.phase3.schemas import AgentDecision
from launchlens.phase4.loop import SimulationLog
from launchlens.phase5.bias import BiasFlag, run_bias_suite
from launchlens.phase5.metrics import ValidationReport, evaluate_all


class CalibrationFixture(BaseModel):
    """Real-world ground truth for a single product launch."""
    product_id: str
    product_name: str
    real_adoption_curve: list[float]          # cumulative adoption per week
    real_final_adoption_rate: float
    real_top_segments: list[str]              # e.g. ["urban_metro_A", ...]
    real_district_rates: dict[str, float]     # district_id → adoption rate
    real_top_rejections: list[str]            # natural-language rejection reasons

    @classmethod
    def from_json(cls, path: Path) -> "CalibrationFixture":
        return cls.model_validate_json(path.read_text())


# ── Extractors: SimulationLog → metric inputs ────────────────────────────────

def sim_final_rate(sim_log: SimulationLog) -> float:
    curve = sim_log.adoption_curve()
    return curve[-1] if curve else 0.0


def sim_top_segments(
    sim_log: SimulationLog,
    personas: Sequence[AgentPersona],
    k: int = 3,
) -> list[str]:
    """
    Identify which (urban/rural × ISEC band) segments produced the most BUY decisions.
    Segment label format: e.g. "urban_A1-A3", "rural_C1-C2".
    """
    persona_by_id = {p.agent_id: p for p in personas}
    seg_counts: Counter[str] = Counter()
    buyers: set[str] = set()
    for d in sim_log.all_decisions():
        if d.decision != "BUY" or d.agent_id in buyers:
            continue
        buyers.add(d.agent_id)
        p = persona_by_id.get(d.agent_id)
        if not p:
            continue
        seg_counts[_segment_label(p)] += 1
    return [seg for seg, _ in seg_counts.most_common(k)]


def _segment_label(p: AgentPersona) -> str:
    isec = p.demographic.isec_tier
    band = (
        "A1-A3" if isec in ("A1", "A2", "A3") else
        "B1-B2" if isec in ("B1", "B2") else
        "C1-C2" if isec in ("C1", "C2") else
        "D1-D2" if isec in ("D1", "D2") else "E1-E3"
    )
    geo = "urban" if p.demographic.urban else "rural"
    return f"{geo}_{band}"


def sim_district_rates(
    sim_log: SimulationLog,
    personas: Sequence[AgentPersona],
) -> dict[str, float]:
    persona_by_id = {p.agent_id: p for p in personas}
    by_district: dict[str, list[bool]] = defaultdict(list)
    seen_buyer: set[str] = set()
    for d in sim_log.all_decisions():
        p = persona_by_id.get(d.agent_id)
        if not p:
            continue
        if d.agent_id not in seen_buyer:
            if d.decision == "BUY":
                seen_buyer.add(d.agent_id)
                by_district[p.demographic.district_id].append(True)
    # Denominator: total agents per district
    district_totals: Counter[str] = Counter(p.demographic.district_id for p in personas)
    return {
        d: len(by_district[d]) / district_totals[d]
        for d in district_totals if district_totals[d] > 0
    }


def sim_reject_reasons(sim_log: SimulationLog, k: int = 10) -> list[str]:
    """Top-k most-common rejection reasons (primary_reason of REJECT/COMPLAIN decisions)."""
    reasons = [
        d.primary_reason for d in sim_log.all_decisions()
        if d.decision in ("REJECT", "COMPLAIN") and d.primary_reason.strip()
    ]
    counts = Counter(reasons)
    return [r for r, _ in counts.most_common(k)]


# ── Runner ───────────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    validation: ValidationReport
    bias_flags: list[BiasFlag]

    def summary(self) -> str:
        lines = [self.validation.summary(), "", "Bias Suite", "─" * 60]
        for f in self.bias_flags:
            mark = "⚠" if f.flagged else "✓"
            v = f"{f.value:.4f}" if isinstance(f.value, float) else str(f.value)
            lines.append(f"  {mark} {f.name:<22} value={v:<10} gate={f.threshold}   {f.note}")
        return "\n".join(lines)


def calibrate(
    sim_log: SimulationLog,
    personas: Sequence[AgentPersona],
    fixture: CalibrationFixture,
) -> CalibrationResult:
    """Evaluate all 5 gates + 3 bias checks against a fixture."""
    validation = evaluate_all(
        product_id=fixture.product_id,
        sim_rate=sim_final_rate(sim_log),
        real_rate=fixture.real_final_adoption_rate,
        sim_curve=sim_log.adoption_curve(),
        real_curve=fixture.real_adoption_curve,
        sim_top_segments=sim_top_segments(sim_log, personas),
        real_top_segments=fixture.real_top_segments,
        sim_district_rates=sim_district_rates(sim_log, personas),
        real_district_rates=fixture.real_district_rates,
        sim_reject_reasons=sim_reject_reasons(sim_log),
        real_reject_reasons=fixture.real_top_rejections,
    )
    bias_flags = run_bias_suite(sim_log.all_decisions(), personas)
    return CalibrationResult(validation=validation, bias_flags=bias_flags)


# ── Fixture I/O helpers ──────────────────────────────────────────────────────

def save_fixture(fixture: CalibrationFixture, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fixture.model_dump_json(indent=2))


def load_fixture(path: Path) -> CalibrationFixture:
    return CalibrationFixture.from_json(path)
