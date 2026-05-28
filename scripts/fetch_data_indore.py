"""One-shot script: build a full DistrictProfile for Indore (MP001).

Walks Census → NFHS → TRAI → NSSO, records per-field provenance, and writes
``data/processed/districts/MP001.json``. Prints clear instructions for any
data source that's missing so the user can fill the gap.

Usage:
    python -m scripts.fetch_data_indore
"""
from __future__ import annotations

import json
from pathlib import Path

from launchlens.config import get_settings
from launchlens.phase1.sources import load_district_profile_chain

INDORE_ID = "MP001"
INDORE_NAME = "Indore"
INDORE_STATE = "Madhya Pradesh"


def main() -> int:
    cfg = get_settings()
    print(f"Building district profile for {INDORE_NAME} ({INDORE_ID}, {INDORE_STATE})...")

    profile = load_district_profile_chain(
        district_id=INDORE_ID,
        district_name=INDORE_NAME,
        state_name=INDORE_STATE,
        strict=False,
    )

    out_dir = cfg.district_profiles_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{profile.district_id}.json"
    out_path.write_text(profile.model_dump_json(indent=2))

    print()
    print(f"✓ Wrote {out_path}")
    print()
    print("Per-field provenance:")
    for field, source in profile.provenance.items():
        marker = "✓" if source not in ("fallback",) else "⚠"
        print(f"  {marker} {field:35s} <- {source}")

    fallbacks = [k for k, v in profile.provenance.items() if v == "fallback"]
    if fallbacks:
        print()
        print("Some fields fell back to baselines. To replace them with real data:")
        print(f"  Census PCA   →  {cfg.census_data_dir}/pca_district.csv "
              "(censusindia.gov.in → Primary Census Abstract)")
        print(f"  Census C-16  →  {cfg.census_data_dir}/c16_language.csv "
              "(censusindia.gov.in → C-Series → C-16 Mother Tongue)")
        print(f"  NFHS-5       →  {cfg.nfhs_data_dir}/nfhs5_district.csv "
              "(dhsprogram.com or rchiips.org/nfhs/factsheet_NFHS-5.shtml)")
        print(f"  TRAI         →  {cfg.trai_data_dir}/trai_state_quarterly.csv "
              "(trai.gov.in quarterly reports)")
        print(f"  NSSO CES     →  set DATAGOVINDIA_API_KEY and NSSO_RESOURCE_ID env vars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
