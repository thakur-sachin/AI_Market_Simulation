"""Phase 5 — Validation & Calibration."""
from launchlens.phase5.bias import (
    affluence_bias,
    homogeneity_gini,
    language_bias_sample,
    positivity_bias,
    run_bias_suite,
)
from launchlens.phase5.calibration import (
    CalibrationCase,
    CalibrationReport,
    build_sim_summary,
    calibrate_from_sim_log,
    load_calibration_case,
    run_calibration,
    tune_signal,
)
from launchlens.phase5.metrics import (
    adoption_rate_deviation,
    dtw_curve_distance,
    passes_all_gates,
    regional_spearman,
    rejection_reason_alignment,
    top_segment_accuracy,
)

__all__ = [
    "adoption_rate_deviation", "dtw_curve_distance", "top_segment_accuracy",
    "regional_spearman", "rejection_reason_alignment", "passes_all_gates",
    "affluence_bias", "positivity_bias", "homogeneity_gini",
    "language_bias_sample", "run_bias_suite",
    "CalibrationCase", "CalibrationReport",
    "load_calibration_case", "run_calibration", "tune_signal",
    "build_sim_summary", "calibrate_from_sim_log",
]
