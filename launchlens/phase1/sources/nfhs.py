"""NFHS-5 (2019-21) district factsheet loader.

Provides wealth-index quintiles (proxy for ISEC) and mobile-internet penetration
when available. Expected input: ``data/raw/nfhs/nfhs5_district.csv``.

Useful columns (any subset may be present):
  district_code, district_name, state_name,
  wealth_lowest, wealth_lower, wealth_middle, wealth_higher, wealth_highest,
  mobile_internet_men, mobile_internet_women,
  bank_account_women
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger()


@dataclass
class NFHSRow:
    wealth_quintiles: dict[str, float] | None  # 5 quintiles summing to 1
    mobile_internet: float | None              # combined estimate, 0-1
    raw: pd.Series


_WEALTH_COLS = ("wealth_lowest", "wealth_lower", "wealth_middle",
                "wealth_higher", "wealth_highest")


def load(nfhs_dir: Path, district_id: str) -> NFHSRow | None:
    path = nfhs_dir / "nfhs5_district.csv"
    if not path.exists():
        log.info("nfhs5_unavailable", path=str(path))
        return None
    df = pd.read_csv(path, dtype={"district_code": str})
    sub = df[df["district_code"] == district_id]
    if sub.empty:
        return None
    row = sub.iloc[0]

    wealth = None
    if all(c in row.index for c in _WEALTH_COLS):
        vals = [float(row[c]) for c in _WEALTH_COLS]
        total = sum(vals)
        if total > 0:
            wealth = {c: v / total for c, v in zip(_WEALTH_COLS, vals)}

    mobile_int = None
    pieces = []
    if "mobile_internet_men" in row.index and pd.notna(row["mobile_internet_men"]):
        pieces.append(float(row["mobile_internet_men"]) / 100)
    if "mobile_internet_women" in row.index and pd.notna(row["mobile_internet_women"]):
        pieces.append(float(row["mobile_internet_women"]) / 100)
    if pieces:
        mobile_int = sum(pieces) / len(pieces)

    return NFHSRow(wealth_quintiles=wealth, mobile_internet=mobile_int, raw=row)


def wealth_quintiles_to_isec(quintiles: dict[str, float]) -> dict[str, float]:
    """Map NFHS-5 wealth quintiles → 12-tier ISEC distribution.

    NFHS quintiles (Q1 lowest .. Q5 highest) are coarser than ISEC tiers.
    We split each quintile across two-to-three adjacent ISEC tiers using a
    fixed, MRSI-aligned split. This is an approximation — use NSSO CES when
    available for finer resolution.
    """
    q1, q2, q3, q4, q5 = (
        quintiles["wealth_lowest"], quintiles["wealth_lower"],
        quintiles["wealth_middle"], quintiles["wealth_higher"],
        quintiles["wealth_highest"],
    )
    isec = {
        "A1": q5 * 0.15, "A2": q5 * 0.25, "A3": q5 * 0.35, "B1": q5 * 0.25,
        "B2": q4 * 0.40, "C1": q4 * 0.35, "C2": q4 * 0.25,
        "D1": q3 * 0.55, "D2": q3 * 0.45,
        "E1": q2 * 0.50, "E2": q2 * 0.30 + q1 * 0.20,
        "E3": q2 * 0.20 + q1 * 0.80,
    }
    # Renormalize for safety
    total = sum(isec.values())
    return {k: round(v / total, 4) for k, v in isec.items()}
