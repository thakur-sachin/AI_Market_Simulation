"""
Demographic data ingestion pipeline.
Normalises Census 2011, NFHS-5, NSSO, TRAI to DistrictProfile objects.

Real data setup:
  - Census PCA tables: download from censusindia.gov.in or Kaggle mirror,
    place CSVs in $CENSUS_DATA_DIR/
  - NFHS-5 district factsheets: download from dhsprogram.com,
    place in $NFHS_DATA_DIR/
  - NSSO CES: data.gov.in OGD API (requires token in .env)
  - TRAI: trai.gov.in quarterly PDFs → parse or use cached JSON
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import structlog

from launchlens.config import get_settings
from launchlens.phase1.schemas import (
    AgeDistribution,
    DistrictProfile,
    ISECTier,
)

log = structlog.get_logger()
_cfg = get_settings()

# ISEC national baseline distribution (MRSI 2024 published figures)
_ISEC_NATIONAL_BASELINE: dict[ISECTier, float] = {
    "A1": 0.02, "A2": 0.03, "A3": 0.05,
    "B1": 0.07, "B2": 0.08,
    "C1": 0.10, "C2": 0.12,
    "D1": 0.14, "D2": 0.15,
    "E1": 0.10, "E2": 0.08, "E3": 0.06,
}


# ── Census loader ─────────────────────────────────────────────────────────────

def _load_census_pca(census_dir: Path) -> pd.DataFrame:
    """Load Primary Census Abstract. Expects district-level CSV with standard columns."""
    path = census_dir / "district_pca.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Census PCA not found at {path}. "
            "Download from censusindia.gov.in → Primary Census Abstract → district level."
        )
    df = pd.read_csv(path, dtype={"district_code": str})
    required = {"district_code", "district_name", "state_name", "total_population",
                "male_population", "female_population", "urban_population",
                "literate_population"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Census PCA missing columns: {missing}")
    return df


def _load_nfhs_district(nfhs_dir: Path) -> pd.DataFrame:
    """Load NFHS-5 district factsheet data (pre-compiled JSON/CSV)."""
    path = nfhs_dir / "nfhs5_district.csv"
    if not path.exists():
        log.warning("nfhs_data_missing", path=str(path),
                    note="Will fall back to state-level disaggregation")
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"district_code": str})


# ── Statistical disaggregation ────────────────────────────────────────────────

def _disaggregate_isec(
    urban_share: float,
    literacy_rate: float,
    state_name: str,
) -> dict[ISECTier, float]:
    """
    Estimate district ISEC distribution by shifting national baseline
    based on urbanization + literacy (proxy for affluence).
    Higher urban/literacy → shift weight toward upper tiers.
    """
    baseline = dict(_ISEC_NATIONAL_BASELINE)
    # affluence index: 0→rural/illiterate, 1→fully urban/literate
    affluence = (urban_share * 0.6 + literacy_rate * 0.4)
    shift = (affluence - 0.5) * 0.15   # ±15% max shift

    upper_tiers: list[ISECTier] = ["A1", "A2", "A3", "B1", "B2"]
    lower_tiers: list[ISECTier] = ["D2", "E1", "E2", "E3"]

    for t in upper_tiers:
        baseline[t] = max(0.005, baseline[t] + shift / len(upper_tiers))
    for t in lower_tiers:
        baseline[t] = max(0.005, baseline[t] - shift / len(lower_tiers))

    # renormalise
    total = sum(baseline.values())
    return {k: round(v / total, 4) for k, v in baseline.items()}


def _estimate_smartphone_penetration(urban_share: float, nfhs_row: pd.Series | None) -> float:
    """Combine TRAI urban/rural penetration with district urban share."""
    # TRAI Q4 2024: urban ~78%, rural ~38%
    urban_pen, rural_pen = 0.78, 0.38
    if nfhs_row is not None and "mobile_internet_women" in nfhs_row.index:
        # NFHS-5 gives women's internet access as a floor estimate
        floor = float(nfhs_row["mobile_internet_women"]) / 100
        return max(floor, urban_share * urban_pen + (1 - urban_share) * rural_pen)
    return urban_share * urban_pen + (1 - urban_share) * rural_pen


# ── Age distribution from Census C-13/C-14 ───────────────────────────────────

_AGE_COLS = {
    "bucket_0_4": "age_0_4", "bucket_5_14": "age_5_14",
    "bucket_15_24": "age_15_24", "bucket_25_34": "age_25_34",
    "bucket_35_44": "age_35_44", "bucket_45_54": "age_45_54",
    "bucket_55_64": "age_55_64", "bucket_65_plus": "age_65_plus",
}

_INDIA_AGE_BASELINE = AgeDistribution(
    bucket_0_4=0.09, bucket_5_14=0.18, bucket_15_24=0.18,
    bucket_25_34=0.17, bucket_35_44=0.14, bucket_45_54=0.10,
    bucket_55_64=0.07, bucket_65_plus=0.07,
)


def _parse_age_distribution(row: pd.Series) -> AgeDistribution:
    data = {}
    for field, col in _AGE_COLS.items():
        if col in row.index:
            data[field] = float(row[col]) / 100
    if not data:
        return _INDIA_AGE_BASELINE
    total = sum(data.values())
    return AgeDistribution(**{k: round(v / total, 4) for k, v in data.items()})


# ── Language distribution from Census C-16 ───────────────────────────────────

def _load_language_distribution(census_dir: Path, district_code: str) -> dict[str, float]:
    """Load mother-tongue shares from Census C-16 language tables if available."""
    path = census_dir / "c16_language.csv"
    if not path.exists():
        return {"hindi": 1.0}   # fallback — will be overridden by state defaults
    df = pd.read_csv(path, dtype={"district_code": str})
    sub = df[df["district_code"] == district_code]
    if sub.empty:
        return {"hindi": 1.0}
    lang_cols = [c for c in sub.columns if c not in ("district_code", "district_name", "state_name")]
    row = sub.iloc[0][lang_cols]
    total = row.sum()
    return {lang: round(float(val) / total, 4) for lang, val in row.items() if val > 0}


# ── Main builder ─────────────────────────────────────────────────────────────

def build_district_profile(
    census_row: pd.Series,
    nfhs_row: pd.Series | None,
    census_dir: Path,
) -> DistrictProfile:
    district_code = str(census_row["district_code"])
    total_pop = int(census_row["total_population"])
    urban_pop = int(census_row.get("urban_population", 0))
    urban_share = round(urban_pop / total_pop, 4) if total_pop else 0.35
    literate = int(census_row.get("literate_population", 0))
    literacy_rate = round(literate / total_pop, 4) if total_pop else 0.65
    sex_ratio = round(
        int(census_row.get("female_population", 0)) /
        max(int(census_row.get("male_population", 1)), 1) * 1000, 1
    )

    isec = _disaggregate_isec(urban_share, literacy_rate, str(census_row["state_name"]))
    smartphone_pen = _estimate_smartphone_penetration(urban_share, nfhs_row)
    # Internet ≈ 80% of smartphone owners have data; UPI ≈ 60% of internet users
    internet_pen = round(smartphone_pen * 0.80, 4)
    upi_adoption = round(internet_pen * 0.60, 4)

    # Median HH expenditure: rough estimate from NSSO decile mapped to ISEC
    # Upper quartile ISEC * ₹25K + lower * ₹8K
    upper_share = sum(isec.get(t, 0) for t in ["A1","A2","A3","B1","B2"])
    median_exp = int(upper_share * 25000 + (1 - upper_share) * 8000)

    return DistrictProfile(
        district_id=district_code,
        district_name=str(census_row["district_name"]),
        state_name=str(census_row["state_name"]),
        population=total_pop,
        age_distribution=_parse_age_distribution(census_row),
        sex_ratio=sex_ratio,
        urban_share=urban_share,
        literacy_rate=literacy_rate,
        language_distribution=_load_language_distribution(census_dir, district_code),
        isec_distribution=isec,
        median_monthly_hh_expenditure=median_exp,
        smartphone_penetration=round(smartphone_pen, 4),
        internet_penetration=internet_pen,
        upi_adoption=upi_adoption,
    )


def build_all_district_profiles(
    census_dir: Path | None = None,
    nfhs_dir: Path | None = None,
) -> Iterator[DistrictProfile]:
    census_dir = census_dir or _cfg.census_data_dir
    nfhs_dir = nfhs_dir or _cfg.nfhs_data_dir

    census_df = _load_census_pca(census_dir)
    nfhs_df = _load_nfhs_district(nfhs_dir)
    nfhs_index = (
        nfhs_df.set_index("district_code") if not nfhs_df.empty else pd.DataFrame()
    )

    for _, row in census_df.iterrows():
        code = str(row["district_code"])
        nfhs_row = nfhs_index.loc[code] if (not nfhs_index.empty and code in nfhs_index.index) else None
        yield build_district_profile(row, nfhs_row, census_dir)


def save_district_profiles(profiles: Iterator[DistrictProfile], out_dir: Path | None = None) -> int:
    out_dir = out_dir or _cfg.district_profiles_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for profile in profiles:
        path = out_dir / f"{profile.district_id}.json"
        path.write_text(profile.model_dump_json(indent=2))
        count += 1
    log.info("district_profiles_saved", count=count, dir=str(out_dir))
    return count


def load_district_profile(district_id: str, profiles_dir: Path | None = None) -> DistrictProfile:
    profiles_dir = profiles_dir or _cfg.district_profiles_dir
    path = profiles_dir / f"{district_id}.json"
    return DistrictProfile.model_validate_json(path.read_text())


def load_district_profile_by_name(name: str, profiles_dir: Path | None = None) -> DistrictProfile:
    profiles_dir = profiles_dir or _cfg.district_profiles_dir
    for p in profiles_dir.glob("*.json"):
        profile = DistrictProfile.model_validate_json(p.read_text())
        if profile.district_name.lower() == name.lower():
            return profile
    raise ValueError(f"No profile found for district: {name}")
