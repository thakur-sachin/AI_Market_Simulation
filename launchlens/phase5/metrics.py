"""Five-metric validation protocol for Phase 5 calibration.

All metrics return raw values; pass/fail gates live in ``calibration.py`` so
callers can compose or override them.

Gates (defaults, see plan section "Validation Loop"):
  * adoption_rate_deviation       <  0.08
  * dtw_curve_distance            <  0.15
  * top_segment_accuracy          >= 2 of 3
  * regional_spearman             >  0.70
  * rejection_reason_alignment    >= 2 of 3
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def adoption_rate_deviation(sim_final: float, real_final: float) -> float:
    """|sim - real| / real. Returns ``inf`` if ``real_final == 0``."""
    if real_final == 0:
        return float("inf") if sim_final != 0 else 0.0
    return abs(sim_final - real_final) / abs(real_final)


def dtw_curve_distance(sim_curve: Sequence[float], real_curve: Sequence[float]) -> float:
    """Dynamic time warping distance, normalized so identical curves → 0.

    Uses ``dtaidistance`` when available; falls back to a hand-rolled DP
    implementation for environments without it (so the metric is always
    runnable in CI).
    """
    if not sim_curve or not real_curve:
        return float("inf")
    sim_arr = np.asarray(sim_curve, dtype=float)
    real_arr = np.asarray(real_curve, dtype=float)

    try:
        from dtaidistance import dtw  # type: ignore

        raw = dtw.distance(sim_arr, real_arr, use_pruning=True)
    except Exception:
        raw = _dtw_dp(sim_arr.tolist(), real_arr.tolist())

    # Normalize by sequence length × value range so different sims are comparable.
    denom = max(len(sim_arr), len(real_arr)) * max(
        float(np.max(real_arr)), float(np.max(sim_arr)), 1e-6
    )
    return float(raw) / denom


def _dtw_dp(a: list[float], b: list[float]) -> float:
    n, m = len(a), len(b)
    INF = float("inf")
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def top_segment_accuracy(sim_top3: Sequence[str], real_top3: Sequence[str]) -> int:
    """Count of overlapping labels between two top-3 lists (0..3)."""
    sim_set = {s.strip().lower() for s in sim_top3 if s}
    real_set = {r.strip().lower() for r in real_top3 if r}
    return len(sim_set & real_set)


def regional_spearman(
    sim_district_rates: dict[str, float],
    real_district_rates: dict[str, float],
) -> float:
    """Spearman rank correlation of adoption rates across shared districts.

    Returns ``nan`` if fewer than 3 districts overlap (correlation undefined).
    """
    shared = sorted(set(sim_district_rates) & set(real_district_rates))
    if len(shared) < 3:
        return float("nan")
    sim_vals = np.array([sim_district_rates[d] for d in shared])
    real_vals = np.array([real_district_rates[d] for d in shared])
    try:
        from scipy.stats import spearmanr  # type: ignore

        rho, _ = spearmanr(sim_vals, real_vals)
    except Exception:
        rho = _spearman_fallback(sim_vals, real_vals)
    return float(rho) if rho is not None and not np.isnan(rho) else float("nan")


def _spearman_fallback(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata(a)
    rb = _rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _rankdata(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks


def rejection_reason_alignment(
    sim_reasons: Iterable[str],
    real_reasons: Sequence[str],
    similarity_threshold: float = 0.6,
) -> int:
    """Count how many of ``real_reasons[:3]`` have a near-match in ``sim_reasons``.

    Prefers ``sentence-transformers`` cosine similarity when installed.
    Falls back to a lightweight token-Jaccard heuristic for CI / tests.
    """
    sim_list = [s for s in sim_reasons if s]
    real_list = list(real_reasons)[:3]
    if not sim_list or not real_list:
        return 0

    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore

        model = SentenceTransformer("all-MiniLM-L6-v2")
        sim_emb = model.encode(sim_list, convert_to_tensor=True)
        real_emb = model.encode(real_list, convert_to_tensor=True)
        scores = util.cos_sim(real_emb, sim_emb)
        return int((scores.max(dim=1).values >= similarity_threshold).sum().item())
    except Exception:
        return _jaccard_alignment(sim_list, real_list, similarity_threshold)


def _jaccard_alignment(sim_list: list[str], real_list: list[str],
                       threshold: float) -> int:
    def toks(s: str) -> set[str]:
        return {t.lower().strip(".,!?:;\"'") for t in s.split() if len(t) > 2}

    matches = 0
    for r in real_list:
        rt = toks(r)
        if not rt:
            continue
        best = 0.0
        for s in sim_list:
            st = toks(s)
            if not st:
                continue
            jaccard = len(rt & st) / max(len(rt | st), 1)
            best = max(best, jaccard)
        if best >= threshold:
            matches += 1
    return matches


# ── Gate evaluator ────────────────────────────────────────────────────────────

def passes_all_gates(metrics: dict[str, float]) -> dict[str, bool]:
    """Apply default gates to a metric dict; missing metrics yield ``False``."""
    return {
        "adoption_rate_deviation": metrics.get("adoption_rate_deviation", 1.0) < 0.08,
        "dtw_curve_distance": metrics.get("dtw_curve_distance", 1.0) < 0.15,
        "top_segment_accuracy": metrics.get("top_segment_accuracy", 0) >= 2,
        "regional_spearman": metrics.get("regional_spearman", 0.0) > 0.70,
        "rejection_reason_alignment": metrics.get("rejection_reason_alignment", 0) >= 2,
    }
