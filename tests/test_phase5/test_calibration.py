"""Calibration harness tests using shipped placeholder cases."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from launchlens.phase5.calibration import (
    CalibrationCase,
    CalibrationReport,
    load_calibration_case,
    run_calibration,
    tune_signal,
)


def _shipped_case() -> CalibrationCase:
    return load_calibration_case("paper_boat_aam_panna")


def test_load_shipped_case():
    case = _shipped_case()
    assert case.product_id == "paper_boat_aam_panna"
    assert len(case.real_adoption_curve) == 8
    assert len(case.real_top3_segments) == 3
    assert case.real_adoption_curve[-1] == 0.15


def test_load_nonexistent_case_raises():
    with pytest.raises(FileNotFoundError):
        load_calibration_case("does_not_exist")


def test_run_calibration_perfect_match_passes_adoption_gate():
    case = _shipped_case()
    sim = {
        "engine": "mock",
        "adoption_curve": case.real_adoption_curve,
        "top_segments": case.real_top3_segments,
        "rejection_reasons": case.real_top3_rejections,
    }
    report = run_calibration(case, sim)
    assert isinstance(report, CalibrationReport)
    assert report.metrics["adoption_rate_deviation"] == 0.0
    assert report.metrics["dtw_curve_distance"] == 0.0
    assert report.gates["adoption_rate_deviation"] is True
    assert report.gates["top_segment_accuracy"] is True


def test_run_calibration_zero_sim_yields_failing_gates():
    case = _shipped_case()
    sim = {
        "engine": "mock",
        "adoption_curve": [0.0] * len(case.real_adoption_curve),
        "top_segments": [],
        "rejection_reasons": [],
    }
    report = run_calibration(case, sim)
    assert report.gates["adoption_rate_deviation"] is False
    assert report.gates["top_segment_accuracy"] is False
    assert "adoption_rate" in report.tuning_signals


def test_tune_signal_diagnoses_over_adoption():
    metrics = {"adoption_rate_deviation": 0.5}
    signals = tune_signal(metrics, sim_curve=[0.0, 0.5], real_curve=[0.0, 0.10])
    assert "adoption_rate" in signals
    assert "EXCEEDS" in signals["adoption_rate"] or "exceeds" in signals["adoption_rate"]


def test_tune_signal_diagnoses_under_adoption():
    metrics = {"adoption_rate_deviation": 0.5}
    signals = tune_signal(metrics, sim_curve=[0.0, 0.05], real_curve=[0.0, 0.20])
    assert "adoption_rate" in signals
    assert "UNDERSHOOTS" in signals["adoption_rate"] or "undershoots" in signals["adoption_rate"]


def test_report_round_trip_to_dict():
    case = _shipped_case()
    sim = {"engine": "mock", "adoption_curve": case.real_adoption_curve}
    report = run_calibration(case, sim)
    d = report.to_dict()
    assert d["product_id"] == case.product_id
    # JSON-serializable
    assert json.dumps(d)
