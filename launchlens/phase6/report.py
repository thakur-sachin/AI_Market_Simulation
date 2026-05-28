"""
Phase 6 — Markdown report generator.

Consumes a SimulationLog + persona list + ProductStimulus and produces a
human-readable markdown report covering the 8 standard deliverables.

PDF/HTML rendering is deferred (kaleido + jinja) until Phase 6.2.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

from launchlens.phase1.schemas import AgentPersona
from launchlens.phase3.schemas import ProductStimulus
from launchlens.phase4.loop import SimulationLog
from launchlens.phase5.calibration import (
    CalibrationReport,
    _isec_band,
    build_sim_summary,
)
from launchlens.phase6.analytics import (
    feature_importance,
    message_resonance,
    objection_map,
    segment_breakdown,
)


def _segment_label(p: AgentPersona) -> str:
    geo = "urban" if p.demographic.urban else "rural"
    return f"{geo}_{_isec_band(p.demographic.isec_tier)}"


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "·" * (width - filled)


def _currency_symbol(product: ProductStimulus) -> str:
    return "₹" if product.currency == "INR" else f"{product.currency} "


def generate_report(
    product: ProductStimulus,
    sim_log: SimulationLog,
    personas: Sequence[AgentPersona],
    calibration: CalibrationReport | None = None,
) -> str:
    """Return a markdown report string."""
    decisions = sim_log.all_decisions()
    curve = sim_log.adoption_curve()
    final_rate = curve[-1] if curve else 0.0

    latest: dict[str, str] = {}
    for d in sorted(decisions, key=lambda x: x.timestep):
        latest[d.agent_id] = d.decision
    final_counts = Counter(latest.values())

    persona_segs = {p.agent_id: _segment_label(p) for p in personas}
    summary = build_sim_summary(sim_log, personas)
    sym = _currency_symbol(product)

    lines: list[str] = []
    lines.append(f"# LaunchLens Simulation Report — {product.product_name}")
    lines.append("")
    lines.append(f"- **Product ID:** `{product.product_id}`")
    lines.append(f"- **Category:** {product.category}")
    lines.append(f"- **Launch price:** {sym}{product.price_launch}  (MRP {sym}{product.price_mrp})")
    lines.append(f"- **Agents simulated:** {sim_log.n_agents}")
    lines.append(f"- **Timesteps:** {len(sim_log.timesteps)}")
    lines.append(f"- **Engine:** {sim_log.engine}")
    lines.append("")

    # ── 1. Market Fit ────────────────────────────────────────────────────────
    lines.append("## 1 · Market Fit")
    lines.append("")
    lines.append("| Decision | Count | Share |")
    lines.append("|---|---:|---:|")
    for state, count in final_counts.most_common():
        share = count / sim_log.n_agents if sim_log.n_agents else 0.0
        lines.append(f"| {state} | {count} | {share:.1%} |")
    lines.append("")

    # ── 2. Adoption curve ────────────────────────────────────────────────────
    lines.append("## 2 · Adoption Curve")
    lines.append("")
    lines.append("```")
    for t, val in enumerate(curve):
        lines.append(f"t{t:02d}  {_bar(val)}  {val:.1%}")
    lines.append(f"\nFinal cumulative adoption: {final_rate:.1%}")
    lines.append("```")
    lines.append("")

    # ── 3. District rates ────────────────────────────────────────────────────
    district_rates = summary["district_rates"]
    if len(district_rates) > 1:
        lines.append("## 3 · District Adoption Rates")
        lines.append("")
        lines.append("| District | Adoption rate |")
        lines.append("|---|---:|")
        for d, r in sorted(district_rates.items(), key=lambda x: -x[1]):
            lines.append(f"| {d} | {r:.1%} |")
        lines.append("")

    # ── 4. Segment Depth ─────────────────────────────────────────────────────
    seg_data = segment_breakdown(decisions, persona_segs)
    if seg_data:
        lines.append("## 4 · Segment Depth")
        lines.append("")
        lines.append("| Segment | Size | BUY rate | REJECT rate |")
        lines.append("|---|---:|---:|---:|")
        for s in seg_data:
            lines.append(f"| {s['segment']} | {s['size']} | {s['buy_rate']:.1%} | {s['reject_rate']:.1%} |")
        lines.append("")
        top3 = summary["top_segments"]
        if top3:
            lines.append(f"**Top segments (by BUY count):** {', '.join(top3)}")
            lines.append("")

    # ── 5. Message Resonance ─────────────────────────────────────────────────
    resonance = message_resonance(decisions, product.marketing_copy)
    if resonance:
        lines.append("## 5 · Message Resonance")
        lines.append("")
        lines.append(f"_Marketing copy: \"{product.marketing_copy}\"_")
        lines.append("")
        lines.append("| Keyword | % of BUY reasoning containing it |")
        lines.append("|---|---:|")
        for kw, share in sorted(resonance.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"| {kw} | {share:.1%} |")
        lines.append("")

    # ── 6. Feature Priority ──────────────────────────────────────────────────
    feat_data = feature_importance(decisions, product.key_features)
    if feat_data:
        lines.append("## 6 · Feature Priority")
        lines.append("")
        lines.append("| Feature | BUY mentions | REJECT mentions | Score |")
        lines.append("|---|---:|---:|---:|")
        for f in feat_data:
            lines.append(
                f"| {f['feature']} | {f['mentions_in_buy']} | "
                f"{f['mentions_in_reject']} | {f['importance_score']:+.2f} |"
            )
        lines.append("")

    # ── 7. Objection Map ─────────────────────────────────────────────────────
    objs = objection_map(decisions)
    if objs:
        lines.append("## 7 · Objection Map")
        lines.append("")
        lines.append("Top recurring themes in REJECT / COMPLAIN reasoning:")
        lines.append("")
        for o in objs:
            lines.append(f"- **{o['keyword']}** (cited by {o['count']} agents)")
            for ex in o["example_reasons"][:2]:
                lines.append(f"  - _\"{ex}\"_")
        lines.append("")

    # ── 8. Validation ────────────────────────────────────────────────────────
    if calibration is not None:
        lines.append("## 8 · Validation vs. Real Launch")
        lines.append("")
        lines.append(f"**Calibration case:** `{calibration.product_id}` "
                     f"(engine: {calibration.engine})")
        lines.append("")
        lines.append("| Metric | Value | Passed |")
        lines.append("|---|---:|:---:|")
        for metric, value in calibration.metrics.items():
            passed = calibration.gates.get(metric, False)
            mark = "✓" if passed else "✗"
            val = f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"| {metric} | {val} | {mark} |")
        lines.append("")
        if calibration.tuning_signals:
            lines.append("**Tuning recommendations:**")
            lines.append("")
            for area, message in calibration.tuning_signals.items():
                lines.append(f"- **{area}**: {message}")
            lines.append("")

    return "\n".join(lines)


def write_report(
    path: Path,
    product: ProductStimulus,
    sim_log: SimulationLog,
    personas: Sequence[AgentPersona],
    calibration: CalibrationReport | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    md = generate_report(product, sim_log, personas, calibration)
    path.write_text(md)
    return path
