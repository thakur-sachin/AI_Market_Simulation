"""TRAI state-level penetration loader.

Expected input: ``data/raw/trai/trai_state_quarterly.csv`` with columns:
  state_name, quarter (e.g. "Q4-2024"),
  urban_teledensity_pct, rural_teledensity_pct,
  urban_internet_pct, rural_internet_pct

The most recent quarter present is used. Falls back to a national constant
pair when unavailable (clearly marked as ``fallback`` provenance by the
orchestrator).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger()


@dataclass
class TRAIPenetration:
    state_name: str
    quarter: str
    urban_internet: float   # 0-1
    rural_internet: float   # 0-1


# National Q4 2024 estimates used only when the CSV is absent.
_FALLBACK = TRAIPenetration(
    state_name="ALL_INDIA",
    quarter="fallback",
    urban_internet=0.78,
    rural_internet=0.38,
)


def load(trai_dir: Path, state_name: str) -> TRAIPenetration | None:
    path = trai_dir / "trai_state_quarterly.csv"
    if not path.exists():
        log.info("trai_unavailable", path=str(path))
        return None
    df = pd.read_csv(path)
    sub = df[df["state_name"].str.lower() == state_name.lower()]
    if sub.empty:
        return None
    # Pick the lexicographically last quarter (works for "Q1-2024" < "Q4-2024")
    sub = sub.sort_values("quarter").iloc[-1]
    return TRAIPenetration(
        state_name=state_name,
        quarter=str(sub["quarter"]),
        urban_internet=float(sub.get("urban_internet_pct", 78)) / 100,
        rural_internet=float(sub.get("rural_internet_pct", 38)) / 100,
    )


def fallback() -> TRAIPenetration:
    return _FALLBACK
