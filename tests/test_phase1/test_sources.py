"""Phase 1 data source chain tests — synthetic CSVs only, no real downloads."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from launchlens.phase1.schemas import DistrictProfile
from launchlens.phase1.sources import (
    IncompleteProfileError,
    load_district_profile_chain,
)


def _write_census(census_dir: Path, *, district="MP001",
                  name="Indore", state="Madhya Pradesh") -> None:
    census_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "district_code": district,
        "district_name": name,
        "state_name": state,
        "total_population": 3_300_000,
        "male_population": 1_700_000,
        "female_population": 1_600_000,
        "urban_population": 2_300_000,
        "literate_population": 2_700_000,
        "age_0_4": 9, "age_5_14": 18, "age_15_24": 18,
        "age_25_34": 17, "age_35_44": 14, "age_45_54": 10,
        "age_55_64": 7, "age_65_plus": 7,
    }]).to_csv(census_dir / "pca_district.csv", index=False)


def _write_nfhs(nfhs_dir: Path, district="MP001") -> None:
    nfhs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "district_code": district,
        "wealth_lowest": 12, "wealth_lower": 18,
        "wealth_middle": 22, "wealth_higher": 24, "wealth_highest": 24,
        "mobile_internet_men": 65, "mobile_internet_women": 42,
    }]).to_csv(nfhs_dir / "nfhs5_district.csv", index=False)


def _write_trai(trai_dir: Path, state="Madhya Pradesh") -> None:
    trai_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "state_name": state, "quarter": "Q4-2024",
        "urban_internet_pct": 80, "rural_internet_pct": 40,
    }]).to_csv(trai_dir / "trai_state_quarterly.csv", index=False)


def test_chain_with_all_sources_present(tmp_path: Path):
    census = tmp_path / "census"
    nfhs = tmp_path / "nfhs"
    trai = tmp_path / "trai"
    nsso = tmp_path / "nsso"
    _write_census(census)
    _write_nfhs(nfhs)
    _write_trai(trai)

    profile = load_district_profile_chain(
        district_id="MP001",
        district_name="Indore",
        state_name="Madhya Pradesh",
        census_dir=census, nfhs_dir=nfhs, trai_dir=trai, nsso_dir=nsso,
    )

    assert profile.district_id == "MP001"
    assert profile.district_name == "Indore"
    assert profile.population == 3_300_000
    assert profile.provenance["population"] == "census"
    assert profile.provenance["isec_distribution"] == "nfhs"
    # smartphone may be 'trai' or 'nfhs' depending on which value won (nfhs floor)
    assert profile.provenance["smartphone_penetration"] in ("trai", "nfhs")
    # UPI is always derived for now
    assert profile.provenance["upi_adoption"] == "fallback"


def test_chain_falls_back_when_all_sources_missing(tmp_path: Path):
    profile = load_district_profile_chain(
        district_id="ZZ999",
        district_name="Nowhere",
        state_name="Test State",
        census_dir=tmp_path / "census",
        nfhs_dir=tmp_path / "nfhs",
        trai_dir=tmp_path / "trai",
        nsso_dir=tmp_path / "nsso",
    )
    # Profile validates (Pydantic checks bounds + sums)
    assert isinstance(profile, DistrictProfile)
    # Every required field is marked fallback
    fallback_fields = [k for k, v in profile.provenance.items() if v == "fallback"]
    assert "population" in fallback_fields
    assert "isec_distribution" in fallback_fields


def test_chain_strict_raises_on_fallback(tmp_path: Path):
    with pytest.raises(IncompleteProfileError):
        load_district_profile_chain(
            district_id="ZZ999",
            district_name="Nowhere",
            state_name="Test State",
            census_dir=tmp_path / "census",
            nfhs_dir=tmp_path / "nfhs",
            trai_dir=tmp_path / "trai",
            nsso_dir=tmp_path / "nsso",
            strict=True,
        )


def test_provenance_field_completeness(tmp_path: Path):
    """Every required field must have a provenance entry."""
    profile = load_district_profile_chain(
        district_id="ZZ999",
        district_name="Nowhere",
        state_name="Test State",
        census_dir=tmp_path / "c",
        nfhs_dir=tmp_path / "n",
        trai_dir=tmp_path / "t",
        nsso_dir=tmp_path / "s",
    )
    required = {
        "population", "age_distribution", "sex_ratio", "urban_share",
        "literacy_rate", "language_distribution", "isec_distribution",
        "median_monthly_hh_expenditure", "smartphone_penetration",
        "internet_penetration", "upi_adoption",
    }
    assert required <= set(profile.provenance.keys())


def test_schema_validators_reject_bad_sums():
    """A DistrictProfile with garbage isec_distribution must not validate."""
    with pytest.raises(ValueError, match="isec_distribution"):
        DistrictProfile(
            district_id="x", district_name="x", state_name="x",
            population=1, age_distribution={"bucket_25_34": 1.0},   # type: ignore[arg-type]
            sex_ratio=950.0, urban_share=0.5, literacy_rate=0.7,
            language_distribution={"hindi": 1.0},
            isec_distribution={"A1": 0.1, "A2": 0.2},  # sums to 0.3
            median_monthly_hh_expenditure=10000,
            smartphone_penetration=0.5, internet_penetration=0.4,
            upi_adoption=0.3,
        )


def test_schema_validators_reject_out_of_range():
    with pytest.raises(ValueError):
        DistrictProfile(
            district_id="x", district_name="x", state_name="x", population=1,
            age_distribution={"bucket_25_34": 1.0},  # type: ignore[arg-type]
            sex_ratio=950.0,
            urban_share=1.5,   # invalid
            literacy_rate=0.7,
            language_distribution={"hindi": 1.0},
            isec_distribution={
                "A1": 0.05, "A2": 0.05, "A3": 0.10, "B1": 0.10, "B2": 0.10,
                "C1": 0.10, "C2": 0.10, "D1": 0.10, "D2": 0.10,
                "E1": 0.10, "E2": 0.05, "E3": 0.05,
            },
            median_monthly_hh_expenditure=10000,
            smartphone_penetration=0.5, internet_penetration=0.4,
            upi_adoption=0.3,
        )
