"""Phase 5 metrics tests."""
from launchlens.phase5.metrics import (
    adoption_rate_deviation,
    dtw_curve_distance,
    top_segment_accuracy,
    regional_spearman,
    rejection_reason_alignment,
    evaluate_all,
)


def test_adoption_rate_deviation_exact_match():
    assert adoption_rate_deviation(0.05, 0.05) == 0.0


def test_adoption_rate_deviation_absolute():
    assert abs(adoption_rate_deviation(0.10, 0.06) - 0.04) < 1e-9


def test_dtw_zero_for_identical_curves():
    curve = [0.01, 0.02, 0.04, 0.07, 0.10]
    assert dtw_curve_distance(curve, curve) < 0.01


def test_dtw_small_for_shape_match_different_scale():
    sim = [0.02, 0.04, 0.08, 0.14, 0.20]
    real = [0.01, 0.02, 0.04, 0.07, 0.10]
    d = dtw_curve_distance(sim, real)
    # Same shape → normalized DTW should be small
    assert d < 0.2


def test_top_segment_accuracy_full_match():
    sim = ["urban_A1-A3", "urban_B1-B2", "urban_C1-C2"]
    real = ["urban_A1-A3", "urban_B1-B2", "urban_C1-C2"]
    assert top_segment_accuracy(sim, real) == 3


def test_top_segment_accuracy_partial():
    sim = ["urban_A1-A3", "rural_E1-E3", "rural_D1-D2"]
    real = ["urban_A1-A3", "urban_B1-B2", "urban_C1-C2"]
    assert top_segment_accuracy(sim, real) == 1


def test_top_segment_accuracy_case_insensitive():
    sim = ["URBAN_A1-A3"]
    real = ["urban_a1-a3"]
    assert top_segment_accuracy(sim, real) == 1


def test_regional_spearman_perfect_correlation():
    sim = {"d1": 0.10, "d2": 0.07, "d3": 0.04, "d4": 0.01}
    real = {"d1": 0.12, "d2": 0.08, "d3": 0.05, "d4": 0.02}
    assert regional_spearman(sim, real) > 0.99


def test_regional_spearman_inverse():
    sim = {"d1": 0.10, "d2": 0.07, "d3": 0.04, "d4": 0.01}
    real = {"d1": 0.01, "d2": 0.04, "d3": 0.07, "d4": 0.10}
    assert regional_spearman(sim, real) < -0.99


def test_regional_spearman_disjoint_keys_returns_zero():
    assert regional_spearman({"d1": 0.1}, {"d2": 0.1}) == 0.0


def test_rejection_alignment_falls_back_without_embedder(monkeypatch):
    """Force the fallback by stubbing sentence-transformers import."""
    from launchlens.phase5 import metrics as m
    monkeypatch.setattr(m, "_embedder", lambda: None)

    sim = ["price is too high for daily use", "tastes overly sweet", "bottle too small"]
    real = ["too expensive for daily consumption", "too sweet", "small bottle"]
    score = rejection_reason_alignment(sim, real)
    assert score >= 2


def test_evaluate_all_returns_five_gates():
    rep = evaluate_all(
        product_id="x",
        sim_rate=0.05, real_rate=0.06,
        sim_curve=[0.01, 0.03, 0.05], real_curve=[0.01, 0.03, 0.06],
        sim_top_segments=["a", "b", "c"], real_top_segments=["a", "b", "c"],
        sim_district_rates={"d1": 0.1, "d2": 0.05}, real_district_rates={"d1": 0.1, "d2": 0.05},
        sim_reject_reasons=["too expensive"], real_reject_reasons=["price too high"],
    )
    assert len(rep.results) == 5
    metric_names = {r.metric for r in rep.results}
    assert metric_names == {
        "adoption_rate_deviation", "dtw_curve_distance",
        "top_segment_accuracy", "regional_spearman",
        "rejection_reason_alignment",
    }
