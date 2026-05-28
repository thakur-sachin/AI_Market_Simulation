"""
Phase 5 — Validation metrics.

Five gates per CLAUDE.md:
  1. adoption_rate_deviation         — gate <8%
  2. dtw_curve_distance              — gate <0.15 normalized
  3. top_segment_accuracy            — gate >=2 of 3 match
  4. regional_spearman               — gate rho > 0.70
  5. rejection_reason_alignment      — gate >=2 of 3 match (cosine similarity)

All functions accept simple, JSON-friendly inputs so they can be wired against
any source of ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# ── 1. Adoption rate deviation ───────────────────────────────────────────────

def adoption_rate_deviation(sim_rate: float, real_rate: float) -> float:
    """Absolute deviation between simulated and real final adoption rate."""
    return float(abs(sim_rate - real_rate))


# ── 2. DTW curve distance ────────────────────────────────────────────────────

def _normalize(curve: Sequence[float]) -> np.ndarray:
    arr = np.asarray(curve, dtype=float)
    max_val = float(arr.max()) if arr.size and arr.max() > 0 else 1.0
    return arr / max_val


def dtw_curve_distance(sim_curve: Sequence[float], real_curve: Sequence[float]) -> float:
    """
    Normalized DTW distance between two adoption curves.
    Each curve is min-max scaled to [0, 1] before alignment so that absolute
    adoption magnitude does not dominate over shape.
    """
    if not sim_curve or not real_curve:
        return float("inf")

    s = _normalize(sim_curve)
    r = _normalize(real_curve)

    try:
        from dtaidistance import dtw
        d = float(dtw.distance(s, r))
    except ImportError:
        # Fallback: pointwise L2 over the shorter length
        n = min(len(s), len(r))
        d = float(np.linalg.norm(s[:n] - r[:n]))

    # Normalize by max possible distance ≈ sqrt(longer_length) for [0,1] series
    norm = float(np.sqrt(max(len(s), len(r))))
    return d / norm if norm > 0 else d


# ── 3. Top segment accuracy ──────────────────────────────────────────────────

def top_segment_accuracy(sim_top: Sequence[str], real_top: Sequence[str], k: int = 3) -> int:
    """Number of overlapping segment labels in the top-k of each list (case-insensitive)."""
    s = {x.strip().lower() for x in sim_top[:k]}
    r = {x.strip().lower() for x in real_top[:k]}
    return len(s & r)


# ── 4. Regional spearman correlation ─────────────────────────────────────────

def regional_spearman(
    sim_rates: dict[str, float],
    real_rates: dict[str, float],
) -> float:
    """Spearman rank correlation between districts that appear in both dicts."""
    keys = sorted(set(sim_rates) & set(real_rates))
    if len(keys) < 2:
        return 0.0
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr([sim_rates[k] for k in keys], [real_rates[k] for k in keys])
        return float(rho) if rho == rho else 0.0   # NaN guard
    except ImportError:
        # Fallback: rank-based Pearson
        sim = _rank([sim_rates[k] for k in keys])
        real = _rank([real_rates[k] for k in keys])
        s = np.asarray(sim, dtype=float); r = np.asarray(real, dtype=float)
        denom = (s.std() * r.std())
        return float(((s - s.mean()) * (r - r.mean())).mean() / denom) if denom else 0.0


def _rank(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    for rank, (idx, _) in enumerate(indexed, start=1):
        ranks[idx] = float(rank)
    return ranks


# ── 5. Rejection reason alignment ────────────────────────────────────────────

def _embedder():
    """Lazy-load sentence-transformers. Returns model or None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def _tokens(text: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 2}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _tokens(a), _tokens(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def rejection_reason_alignment(
    sim_reasons: Sequence[str],
    real_reasons: Sequence[str],
    threshold: float = 0.45,
    k: int = 3,
) -> int:
    """
    Count how many of the top-k real reasons have a sim counterpart above
    `threshold` similarity. Uses sentence-transformer embeddings when available,
    falls back to Jaccard token overlap.
    """
    if not sim_reasons or not real_reasons:
        return 0

    model = _embedder()
    if model is not None:
        sim_emb = model.encode(list(sim_reasons), normalize_embeddings=True)
        real_emb = model.encode(list(real_reasons[:k]), normalize_embeddings=True)
        # cosine via dot product because normalized
        sims = real_emb @ sim_emb.T
        return int(sum(1 for row in sims if float(row.max()) >= threshold))

    # Fallback: max Jaccard over sim_reasons for each real reason
    matches = 0
    for r in real_reasons[:k]:
        best = max((_jaccard(r, s) for s in sim_reasons), default=0.0)
        if best >= 0.20:   # token overlap is much sparser; lower threshold
            matches += 1
    return matches


# ── Gate evaluation ──────────────────────────────────────────────────────────

@dataclass
class GateResult:
    metric: str
    value: float | int
    gate: str
    passed: bool


@dataclass
class ValidationReport:
    product_id: str
    results: list[GateResult]

    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def summary(self) -> str:
        lines = [f"Validation Report — {self.product_id}", "─" * 60]
        for r in self.results:
            mark = "✓" if r.passed else "✗"
            v = f"{r.value:.4f}" if isinstance(r.value, float) else str(r.value)
            lines.append(f"  {mark} {r.metric:<32} value={v:<10} gate={r.gate}")
        lines.append(f"\n  PASSED: {self.passed_count()}/{len(self.results)}")
        return "\n".join(lines)


def evaluate_all(
    product_id: str,
    sim_rate: float,
    real_rate: float,
    sim_curve: Sequence[float],
    real_curve: Sequence[float],
    sim_top_segments: Sequence[str],
    real_top_segments: Sequence[str],
    sim_district_rates: dict[str, float],
    real_district_rates: dict[str, float],
    sim_reject_reasons: Sequence[str],
    real_reject_reasons: Sequence[str],
) -> ValidationReport:
    rate_dev = adoption_rate_deviation(sim_rate, real_rate)
    dtw_d = dtw_curve_distance(sim_curve, real_curve)
    seg_acc = top_segment_accuracy(sim_top_segments, real_top_segments)
    rho = regional_spearman(sim_district_rates, real_district_rates)
    rej_align = rejection_reason_alignment(sim_reject_reasons, real_reject_reasons)

    results = [
        GateResult("adoption_rate_deviation", rate_dev, "< 0.08", rate_dev < 0.08),
        GateResult("dtw_curve_distance",      dtw_d,    "< 0.15", dtw_d < 0.15),
        GateResult("top_segment_accuracy",    seg_acc,  ">= 2",   seg_acc >= 2),
        GateResult("regional_spearman",       rho,      "> 0.70", rho > 0.70),
        GateResult("rejection_reason_alignment", rej_align, ">= 2", rej_align >= 2),
    ]
    return ValidationReport(product_id=product_id, results=results)
