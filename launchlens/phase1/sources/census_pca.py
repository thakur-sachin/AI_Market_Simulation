"""Census 2011 Primary Census Abstract loader.

Expected input: ``data/raw/census/pca_district.csv`` with columns:
  district_code, district_name, state_name,
  total_population, male_population, female_population,
  urban_population, literate_population
Optional columns: age_0_4, age_5_14, ..., age_65_plus (counts).

The file is *not* shipped with the repo; download instructions are in
``ROADMAP.md``. If the file is absent, ``load`` returns ``None`` so the
chain can fall back to the next source.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog

from launchlens.phase1.schemas import AgeDistribution

log = structlog.get_logger()


@dataclass
class CensusRow:
    district_code: str
    district_name: str
    state_name: str
    population: int
    sex_ratio: float
    urban_share: float
    literacy_rate: float
    age_distribution: AgeDistribution | None
    language_distribution: dict[str, float] | None


_AGE_COLS = {
    "bucket_0_4": "age_0_4",
    "bucket_5_14": "age_5_14",
    "bucket_15_24": "age_15_24",
    "bucket_25_34": "age_25_34",
    "bucket_35_44": "age_35_44",
    "bucket_45_54": "age_45_54",
    "bucket_55_64": "age_55_64",
    "bucket_65_plus": "age_65_plus",
}

_REQUIRED = {
    "district_code", "district_name", "state_name",
    "total_population", "male_population", "female_population",
    "urban_population", "literate_population",
}


def _parse_age(row: pd.Series) -> AgeDistribution | None:
    counts: dict[str, float] = {}
    for field, col in _AGE_COLS.items():
        if col in row.index and pd.notna(row[col]):
            counts[field] = float(row[col])
    total = sum(counts.values())
    if total <= 0:
        return None
    return AgeDistribution(**{k: round(v / total, 4) for k, v in counts.items()})


def _parse_language(census_dir: Path, district_code: str) -> dict[str, float] | None:
    """Parse Census C-16 mother-tongue table if present."""
    path = census_dir / "c16_language.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"district_code": str})
    sub = df[df["district_code"] == district_code]
    if sub.empty:
        return None
    lang_cols = [c for c in sub.columns
                 if c not in ("district_code", "district_name", "state_name")]
    row = sub.iloc[0][lang_cols]
    total = float(row.sum())
    if total <= 0:
        return None
    return {
        lang.lower(): round(float(val) / total, 4)
        for lang, val in row.items() if float(val) > 0
    }


def load(census_dir: Path, district_id: str) -> CensusRow | None:
    """Return the Census row for ``district_id``, or None if unavailable."""
    path = census_dir / "pca_district.csv"
    if not path.exists():
        log.info("census_pca_unavailable", path=str(path))
        return None

    df = pd.read_csv(path, dtype={"district_code": str})
    missing = _REQUIRED - set(df.columns)
    if missing:
        log.warning("census_pca_missing_columns", missing=sorted(missing))
        return None

    sub = df[df["district_code"] == district_id]
    if sub.empty:
        # try case-insensitive name match if id missed
        return None
    row = sub.iloc[0]

    pop = int(row["total_population"])
    male = max(int(row["male_population"]), 1)
    female = int(row["female_population"])
    urban = int(row["urban_population"])
    literate = int(row["literate_population"])

    return CensusRow(
        district_code=str(row["district_code"]),
        district_name=str(row["district_name"]),
        state_name=str(row["state_name"]),
        population=pop,
        sex_ratio=round(female / male * 1000, 1),
        urban_share=round(urban / pop, 4) if pop else 0.35,
        literacy_rate=round(literate / pop, 4) if pop else 0.65,
        age_distribution=_parse_age(row),
        language_distribution=_parse_language(census_dir, district_id),
    )


def load_by_name(census_dir: Path, name: str) -> CensusRow | None:
    """Lookup district by human-readable name."""
    path = census_dir / "pca_district.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"district_code": str})
    sub = df[df["district_name"].str.lower() == name.lower()]
    if sub.empty:
        return None
    return load(census_dir, str(sub.iloc[0]["district_code"]))
