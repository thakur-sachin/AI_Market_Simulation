"""Bias detection suite tests."""
from __future__ import annotations

from launchlens.phase1.schemas import AgentPersona, DemographicVector
from launchlens.phase3.schemas import AgentDecision
from launchlens.phase5.bias import (
    affluence_bias,
    homogeneity_gini,
    positivity_bias,
    run_bias_suite,
)


def _persona(i: int, isec: str = "B1", lang: str = "hindi") -> AgentPersona:
    return AgentPersona(
        agent_id=f"a{i}",
        demographic=DemographicVector(
            age=30, sex="male", urban=True, isec_tier=isec,  # type: ignore[arg-type]
            primary_language=lang, occupation="services_formal",
            monthly_hh_income=25000, tech_adoption="early_majority",
            smartphone_owner=True, upi_user=True,
            district_id="MP001", district_name="x", state_name="y",
        ),
        biography="b", llm_route="sarvam",
    )


def _decision(i: int, state: str = "IGNORE") -> AgentDecision:
    return AgentDecision(
        agent_id=f"a{i}", product_id="p", timestep=0,
        internal_reasoning="r", decision=state,  # type: ignore[arg-type]
        primary_reason="r", would_discuss_with="no_one",
        language_of_discussion="N/A",
    )


def test_affluence_bias_flags_when_lower_buys_more():
    personas = [
        _persona(0, "A1"), _persona(1, "A2"),
        _persona(2, "D1"), _persona(3, "D2"), _persona(4, "E1"),
    ]
    # Lower tiers all BUY, upper tiers all IGNORE
    decisions = [
        _decision(0, "IGNORE"), _decision(1, "IGNORE"),
        _decision(2, "BUY"), _decision(3, "BUY"), _decision(4, "BUY"),
    ]
    out = affluence_bias(decisions, personas)
    assert out["upper_tier_buy_rate"] == 0.0
    assert out["lower_tier_buy_rate"] == 1.0
    assert out["flagged"] is True


def test_affluence_bias_ok_when_upper_dominates():
    personas = [
        _persona(0, "A1"), _persona(1, "A2"),
        _persona(2, "D1"), _persona(3, "D2"),
    ]
    decisions = [
        _decision(0, "BUY"), _decision(1, "BUY"),
        _decision(2, "IGNORE"), _decision(3, "IGNORE"),
    ]
    out = affluence_bias(decisions, personas)
    assert out["flagged"] is False


def test_positivity_bias_flags_when_sim_too_positive():
    decisions = [_decision(i, "BUY") for i in range(95)] + [_decision(i, "REJECT") for i in range(5)]
    out = positivity_bias(decisions, category_benchmark_reject_rate=0.40)
    assert out["sim_reject_rate"] == 0.05
    assert out["flagged"] is True


def test_positivity_bias_no_benchmark_no_flag():
    decisions = [_decision(0, "REJECT")]
    out = positivity_bias(decisions, category_benchmark_reject_rate=None)
    assert out["flagged"] is False


def test_homogeneity_gini_flags_uniform_cohort():
    """If every B1 agent picks IGNORE, the Gini should be ~0 → flag."""
    personas = [_persona(i, "B1") for i in range(10)]
    decisions = [_decision(i, "IGNORE") for i in range(10)]
    out = homogeneity_gini(decisions, personas, flag_threshold=0.30)
    assert out["mean_gini"] == 0.0
    assert out["flagged"] is True


def test_homogeneity_gini_ok_when_diverse():
    personas = [_persona(i, "B1") for i in range(8)]
    decisions = [
        _decision(0, "IGNORE"), _decision(1, "AWARE"),
        _decision(2, "RESEARCH"), _decision(3, "CONSIDER"),
        _decision(4, "BUY"), _decision(5, "REJECT"),
        _decision(6, "SHARE_POSITIVE"), _decision(7, "COMPLAIN"),
    ]
    out = homogeneity_gini(decisions, personas, flag_threshold=0.30)
    assert out["mean_gini"] > 0.7
    assert out["flagged"] is False


def test_run_bias_suite_returns_all_sections():
    personas = [_persona(0, "A1"), _persona(1, "D1")]
    decisions = [_decision(0, "BUY"), _decision(1, "IGNORE")]
    out = run_bias_suite(decisions, personas, category_benchmark_reject_rate=0.4)
    assert set(out.keys()) == {"affluence", "positivity", "homogeneity"}
