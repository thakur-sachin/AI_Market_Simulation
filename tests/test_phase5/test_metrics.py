"""Phase 5 metric correctness — known-answer fixture tests."""
from __future__ import annotations

import math

import pytest

from launchlens.phase5.metrics import (
    adoption_rate_deviation,
    dtw_curve_distance,
    passes_all_gates,
    regional_spearman,
    rejection_reason_alignment,
    top_segment_accuracy,
)


# adoption_rate_deviation ────────────────────────────────────────────────────

def test_adoption_rate_deviation_identical():
    assert adoption_rate_deviation(0.15, 0.15) == 0.0


def test_adoption_rate_deviation_proportional():
    # |0.20 - 0.15| / 0.15 = 0.333…
    assert adoption_rate_deviation(0.20, 0.15) == pytest.approx(1 / 3, abs=1e-3)


def test_adoption_rate_deviation_real_zero_inf():
    assert adoption_rate_deviation(0.1, 0.0) == float("inf")


def test_adoption_rate_deviation_both_zero():
    assert adoption_rate_deviation(0.0, 0.0) == 0.0


# dtw_curve_distance ─────────────────────────────────────────────────────────

def test_dtw_identical_curves_zero():
    curve = [0.0, 0.02, 0.05, 0.09, 0.12]
    assert dtw_curve_distance(curve, curve) == pytest.approx(0.0, abs=1e-6)


def test_dtw_increases_with_divergence():
    curve_a = [0.0, 0.02, 0.05, 0.09, 0.12]
    curve_b = [0.0, 0.10, 0.20, 0.30, 0.40]
    d_same = dtw_curve_distance(curve_a, curve_a)
    d_diff = dtw_curve_distance(curve_a, curve_b)
    assert d_diff > d_same


def test_dtw_handles_empty():
    assert math.isinf(dtw_curve_distance([], [0.1]))


# top_segment_accuracy ───────────────────────────────────────────────────────

def test_top_segment_full_overlap():
    assert top_segment_accuracy(["A1", "A2", "B1"], ["A1", "A2", "B1"]) == 3


def test_top_segment_partial_overlap():
    assert top_segment_accuracy(["A1", "A2", "B2"], ["A1", "B1", "B2"]) == 2


def test_top_segment_no_overlap():
    assert top_segment_accuracy(["C1", "C2", "D1"], ["A1", "A2", "B1"]) == 0


def test_top_segment_case_insensitive():
    assert top_segment_accuracy(["a1", "A2"], ["A1", "a2"]) == 2


# regional_spearman ──────────────────────────────────────────────────────────

def test_spearman_perfect_correlation():
    sim = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    real = {"a": 0.05, "b": 0.10, "c": 0.20, "d": 0.40}
    assert regional_spearman(sim, real) == pytest.approx(1.0, abs=1e-6)


def test_spearman_anticorrelation():
    sim = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    real = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    assert regional_spearman(sim, real) == pytest.approx(-1.0, abs=1e-6)


def test_spearman_too_few_districts_nan():
    sim = {"a": 0.1, "b": 0.2}
    real = {"a": 0.05, "b": 0.10}
    rho = regional_spearman(sim, real)
    assert math.isnan(rho)


# rejection_reason_alignment ─────────────────────────────────────────────────

def test_rejection_alignment_exact_match():
    sim = [
        "Too expensive for the value provided",
        "Brand is unfamiliar to me",
        "Quality concerns from peer reviews",
    ]
    real = [
        "Too expensive for the value provided",   # near-identical
        "Brand recognition is low",
        "Quality is questionable",
    ]
    # Jaccard fallback should match at least 1; sentence-transformers would match 3.
    assert rejection_reason_alignment(sim, real) >= 1


def test_rejection_alignment_empty_returns_zero():
    assert rejection_reason_alignment([], ["a", "b"]) == 0
    assert rejection_reason_alignment(["a", "b"], []) == 0


# passes_all_gates ───────────────────────────────────────────────────────────

def test_gates_pass_all_when_thresholds_met():
    metrics = {
        "adoption_rate_deviation": 0.05,
        "dtw_curve_distance": 0.10,
        "top_segment_accuracy": 3,
        "regional_spearman": 0.85,
        "rejection_reason_alignment": 2,
    }
    gates = passes_all_gates(metrics)
    assert all(gates.values())


def test_gates_fail_when_below_threshold():
    metrics = {
        "adoption_rate_deviation": 0.50,   # FAIL
        "dtw_curve_distance": 0.10,
        "top_segment_accuracy": 1,         # FAIL
        "regional_spearman": 0.85,
        "rejection_reason_alignment": 2,
    }
    gates = passes_all_gates(metrics)
    assert gates["adoption_rate_deviation"] is False
    assert gates["top_segment_accuracy"] is False
    assert gates["dtw_curve_distance"] is True
