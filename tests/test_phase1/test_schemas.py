"""Offline schema and sampling tests — no LLM calls."""
import pytest
from launchlens.phase1.schemas import AgeDistribution, DistrictProfile, DemographicVector
from launchlens.phase1.data_pipeline import (
    _disaggregate_isec,
    _estimate_smartphone_penetration,
)
from launchlens.phase1.persona_gen import sample_demographic_vectors


@pytest.fixture
def sample_profile() -> DistrictProfile:
    return DistrictProfile(
        district_id="MP001",
        district_name="Indore",
        state_name="Madhya Pradesh",
        population=3_276_697,
        age_distribution=AgeDistribution(
            bucket_0_4=0.09, bucket_5_14=0.18, bucket_15_24=0.18,
            bucket_25_34=0.17, bucket_35_44=0.14, bucket_45_54=0.10,
            bucket_55_64=0.07, bucket_65_plus=0.07,
        ),
        sex_ratio=920,
        urban_share=0.70,
        literacy_rate=0.82,
        language_distribution={"hindi": 0.85, "urdu": 0.08, "english": 0.07},
        isec_distribution={
            "A1": 0.03, "A2": 0.05, "A3": 0.07, "B1": 0.09, "B2": 0.10,
            "C1": 0.12, "C2": 0.12, "D1": 0.13, "D2": 0.12,
            "E1": 0.08, "E2": 0.05, "E3": 0.04,
        },
        median_monthly_hh_expenditure=18000,
        smartphone_penetration=0.62,
        internet_penetration=0.50,
        upi_adoption=0.38,
    )


def test_age_distribution_validation(sample_profile):
    assert sample_profile.age_distribution.validate_sums_to_one()


def test_isec_disaggregation_sums_to_one():
    result = _disaggregate_isec(0.7, 0.8, "Maharashtra")
    total = sum(result.values())
    assert abs(total - 1.0) < 0.01


def test_disaggregation_urban_skews_upper():
    urban = _disaggregate_isec(0.95, 0.95, "Test")
    rural = _disaggregate_isec(0.05, 0.30, "Test")
    urban_upper = sum(urban.get(t, 0) for t in ("A1","A2","A3","B1","B2"))
    rural_upper = sum(rural.get(t, 0) for t in ("A1","A2","A3","B1","B2"))
    assert urban_upper > rural_upper


def test_smartphone_penetration_bounds():
    pen = _estimate_smartphone_penetration(0.7, None)
    assert 0.0 <= pen <= 1.0


def test_stratified_sampling_count(sample_profile):
    vectors = sample_demographic_vectors(sample_profile, n=100, seed=42)
    assert len(vectors) == 100


def test_stratified_sampling_adult_only(sample_profile):
    vectors = sample_demographic_vectors(sample_profile, n=200, seed=99)
    assert all(v.age >= 15 for v in vectors)


def test_stratified_sampling_diversity(sample_profile):
    vectors = sample_demographic_vectors(sample_profile, n=500, seed=7)
    isec_tiers = {v.isec_tier for v in vectors}
    # Should sample at least 6 different ISEC tiers from a diverse district
    assert len(isec_tiers) >= 6


def test_urban_share_approximate(sample_profile):
    vectors = sample_demographic_vectors(sample_profile, n=1000, seed=0)
    urban_rate = sum(1 for v in vectors if v.urban) / len(vectors)
    # Allow ±10% from target 0.70
    assert abs(urban_rate - sample_profile.urban_share) < 0.10


def test_income_within_tier_range(sample_profile):
    from launchlens.phase1.persona_gen import _ISEC_INCOME_RANGE
    vectors = sample_demographic_vectors(sample_profile, n=200, seed=1)
    for v in vectors:
        lo, hi = _ISEC_INCOME_RANGE[v.isec_tier]
        # Allow ±15% stochastic variation
        assert v.monthly_hh_income >= lo * 0.85
        assert v.monthly_hh_income <= hi * 1.15
