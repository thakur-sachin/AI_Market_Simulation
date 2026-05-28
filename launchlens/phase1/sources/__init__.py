"""District profile orchestrator: Census -> NFHS -> TRAI -> NSSO -> baseline.

Each source contributes whichever fields it can. Per-field provenance is
recorded on ``DistrictProfile.provenance`` so consumers can see exactly which
numbers came from real data and which fell back to a baseline.

In ``strict=True`` mode the loader raises ``IncompleteProfileError`` if any
required field is ``"fallback"``. By default it returns the partial profile
so the dev hardware can keep running while data files are being collected.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog

from launchlens.config import get_settings
from launchlens.phase1.data_pipeline import (
    _disaggregate_isec,
)
from launchlens.phase1.data_pipeline import (
    load_district_profile as _load_local_district_profile_legacy,
)
from launchlens.phase1.schemas import AgeDistribution, DataSource, DistrictProfile
from launchlens.phase1.sources import census_pca, nfhs, nsso_datagovindia, trai

log = structlog.get_logger()
_cfg = get_settings()


class IncompleteProfileError(RuntimeError):
    """Raised by ``load_district_profile_chain(strict=True)`` when required fields fall back."""


_BASELINE_AGE = AgeDistribution(
    bucket_0_4=0.09, bucket_5_14=0.18, bucket_15_24=0.18,
    bucket_25_34=0.17, bucket_35_44=0.14, bucket_45_54=0.10,
    bucket_55_64=0.07, bucket_65_plus=0.07,
)

_BASELINE_LANGUAGE = {"hindi": 0.85, "english": 0.10, "other": 0.05}

_REQUIRED_FIELDS = (
    "population", "age_distribution", "sex_ratio", "urban_share",
    "literacy_rate", "language_distribution", "isec_distribution",
    "median_monthly_hh_expenditure", "smartphone_penetration",
    "internet_penetration", "upi_adoption",
)


def load_district_profile_chain(
    district_id: str,
    district_name: str,
    state_name: str = "",
    *,
    census_dir: Path | None = None,
    nfhs_dir: Path | None = None,
    trai_dir: Path | None = None,
    nsso_dir: Path | None = None,
    strict: bool = False,
) -> DistrictProfile:
    census_dir = census_dir or _cfg.census_data_dir
    nfhs_dir = nfhs_dir or _cfg.nfhs_data_dir
    trai_dir = trai_dir or _cfg.trai_data_dir
    nsso_dir = nsso_dir or _cfg.nsso_data_dir

    provenance: dict[str, DataSource] = {}

    # 1. Census ------------------------------------------------------------
    crow = census_pca.load(census_dir, district_id)
    if crow is None and district_name:
        crow = census_pca.load_by_name(census_dir, district_name)

    if crow is not None:
        population = crow.population
        sex_ratio = crow.sex_ratio
        urban_share = crow.urban_share
        literacy_rate = crow.literacy_rate
        age_dist = crow.age_distribution or _BASELINE_AGE
        lang_dist = crow.language_distribution or _BASELINE_LANGUAGE
        provenance.update({
            "population": "census",
            "sex_ratio": "census",
            "urban_share": "census",
            "literacy_rate": "census",
            "age_distribution": "census" if crow.age_distribution else "fallback",
            "language_distribution": "census" if crow.language_distribution else "fallback",
        })
        state_name = state_name or crow.state_name
        district_name = crow.district_name
    else:
        log.warning("census_chain_fallback", district=district_id)
        population = 1_000_000
        sex_ratio = 930.0
        urban_share = 0.35
        literacy_rate = 0.74
        age_dist = _BASELINE_AGE
        lang_dist = _BASELINE_LANGUAGE
        provenance.update({
            "population": "fallback",
            "sex_ratio": "fallback",
            "urban_share": "fallback",
            "literacy_rate": "fallback",
            "age_distribution": "fallback",
            "language_distribution": "fallback",
        })

    # 2. NFHS-5 ------------------------------------------------------------
    nrow = nfhs.load(nfhs_dir, district_id)
    isec_dist = None
    nfhs_internet = None
    if nrow is not None:
        if nrow.wealth_quintiles is not None:
            isec_dist = nfhs.wealth_quintiles_to_isec(nrow.wealth_quintiles)
            provenance["isec_distribution"] = "nfhs"
        nfhs_internet = nrow.mobile_internet

    if isec_dist is None:
        isec_dist = _disaggregate_isec(urban_share, literacy_rate, state_name)
        provenance["isec_distribution"] = "fallback"

    # 3. TRAI --------------------------------------------------------------
    tri = trai.load(trai_dir, state_name) if state_name else None
    if tri is not None:
        smartphone_pen = round(
            urban_share * tri.urban_internet + (1 - urban_share) * tri.rural_internet, 4
        )
        internet_pen = smartphone_pen
        provenance["smartphone_penetration"] = "trai"
        provenance["internet_penetration"] = "trai"
    else:
        fb = trai.fallback()
        smartphone_pen = round(
            urban_share * fb.urban_internet + (1 - urban_share) * fb.rural_internet, 4
        )
        internet_pen = round(smartphone_pen * 0.95, 4)
        provenance["smartphone_penetration"] = "fallback"
        provenance["internet_penetration"] = "fallback"

    if nfhs_internet is not None:
        smartphone_pen = max(smartphone_pen, round(nfhs_internet, 4))
        provenance["smartphone_penetration"] = "nfhs"

    upi_adoption = round(
        urban_share * smartphone_pen * 0.60 + (1 - urban_share) * smartphone_pen * 0.35,
        4,
    )
    provenance["upi_adoption"] = "fallback"

    # 4. NSSO CES ----------------------------------------------------------
    ns = nsso_datagovindia.load(nsso_dir, state_name) if state_name else None
    if ns is not None:
        median_hh_exp = ns.median_mpce_inr * 4
        provenance["median_monthly_hh_expenditure"] = "nsso"
    else:
        upper_share = sum(isec_dist.get(t, 0) for t in ("A1", "A2", "A3", "B1", "B2"))
        median_hh_exp = int(upper_share * 28000 + (1 - upper_share) * 9000)
        provenance["median_monthly_hh_expenditure"] = "fallback"

    profile = DistrictProfile(
        district_id=district_id,
        district_name=district_name,
        state_name=state_name or "Unknown",
        population=population,
        age_distribution=age_dist,
        sex_ratio=sex_ratio,
        urban_share=urban_share,
        literacy_rate=literacy_rate,
        language_distribution=lang_dist,
        isec_distribution=isec_dist,
        median_monthly_hh_expenditure=median_hh_exp,
        smartphone_penetration=smartphone_pen,
        internet_penetration=internet_pen,
        upi_adoption=upi_adoption,
        provenance=provenance,
    )

    fallbacks = [k for k, v in provenance.items() if v == "fallback"]
    if fallbacks:
        log.warning("district_profile_partial",
                    district=district_id, fallback_fields=fallbacks)
        if strict:
            raise IncompleteProfileError(
                f"District {district_id} profile has fallback fields: {fallbacks}. "
                f"Download missing data files into data/raw/ and re-run."
            )

    return profile


def load_local_district_profile(district_id_or_name: str) -> DistrictProfile:
    """Load a previously saved profile from ``data/processed/districts/``."""
    try:
        return _load_local_district_profile_legacy(district_id_or_name)
    except FileNotFoundError:
        pass

    profiles_dir = _cfg.district_profiles_dir
    for p in profiles_dir.glob("*.json"):
        prof = DistrictProfile.model_validate_json(p.read_text())
        if prof.district_name.lower() == district_id_or_name.lower():
            return prof
    raise FileNotFoundError(
        f"No saved profile for '{district_id_or_name}'. "
        f"Run: python -m launchlens.cli fetch-data --district {district_id_or_name}"
    )


__all__ = [
    "IncompleteProfileError",
    "load_district_profile_chain",
    "load_local_district_profile",
]
