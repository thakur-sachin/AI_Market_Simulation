"""Phase 5 — Validation & Calibration."""
from launchlens.phase5.metrics import (
    adoption_rate_deviation,
    dtw_curve_distance,
    top_segment_accuracy,
    regional_spearman,
    rejection_reason_alignment,
    evaluate_all,
    ValidationReport,
    GateResult,
)
from launchlens.phase5.bias import (
    affluence_bias,
    positivity_bias,
    homogeneity_bias,
    language_audit_sample,
    run_bias_suite,
    BiasFlag,
)
from launchlens.phase5.calibration import (
    CalibrationFixture,
    CalibrationResult,
    calibrate,
    save_fixture,
    load_fixture,
    sim_final_rate,
    sim_top_segments,
    sim_district_rates,
    sim_reject_reasons,
)

__all__ = [
    "adoption_rate_deviation", "dtw_curve_distance", "top_segment_accuracy",
    "regional_spearman", "rejection_reason_alignment",
    "evaluate_all", "ValidationReport", "GateResult",
    "affluence_bias", "positivity_bias", "homogeneity_bias",
    "language_audit_sample", "run_bias_suite", "BiasFlag",
    "CalibrationFixture", "CalibrationResult", "calibrate",
    "save_fixture", "load_fixture",
    "sim_final_rate", "sim_top_segments", "sim_district_rates", "sim_reject_reasons",
]
