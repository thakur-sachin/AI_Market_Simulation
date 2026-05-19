"""NSSO Consumption Expenditure (CES) data fetched via the ``datagovindia`` API.

We pull state-level expenditure deciles (NSSO 68th round or later, depending
on what's published on data.gov.in). Results are cached on disk so a single
fetch is enough per state.

The ``datagovindia`` package requires an API token (data.gov.in API key)
exposed via ``DATAGOVINDIA_API_KEY``. If the env var is missing or the
package errors out, ``load`` returns ``None`` and the chain continues.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import structlog

log = structlog.get_logger()


@dataclass
class NSSOExpenditure:
    state_name: str
    median_mpce_inr: int     # monthly per-capita expenditure (₹)
    decile_breakdown: dict[str, int] | None   # decile label → MPCE


def _cache_path(nsso_dir: Path, state_name: str) -> Path:
    safe = state_name.lower().replace(" ", "_")
    return nsso_dir / f"nsso_state_{safe}.json"


def load(nsso_dir: Path, state_name: str) -> NSSOExpenditure | None:
    nsso_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(nsso_dir, state_name)
    if cache.exists():
        data = json.loads(cache.read_text())
        return NSSOExpenditure(**data)

    api_key = os.environ.get("DATAGOVINDIA_API_KEY")
    if not api_key:
        log.info("nsso_unavailable", reason="DATAGOVINDIA_API_KEY not set")
        return None

    try:
        import datagovindia  # type: ignore
    except ImportError:
        log.warning("nsso_unavailable", reason="datagovindia package missing")
        return None

    # Best-effort search for NSSO CES resources on data.gov.in.
    try:
        client = datagovindia.DataGovIndia(api_key=api_key)  # type: ignore[attr-defined]
        resources = client.search_api(query="NSSO consumer expenditure", limit=5)
    except Exception as exc:
        log.warning("nsso_search_failed", error=str(exc))
        return None

    # Without a stable resource id we cannot deterministically pull this; in
    # production set NSSO_RESOURCE_ID env var to a known dataset and use
    # client.get_data(...) directly. Stub returns None when no usable
    # dataset matched.
    resource_id = os.environ.get("NSSO_RESOURCE_ID")
    if not resource_id:
        log.info(
            "nsso_unavailable",
            reason="NSSO_RESOURCE_ID not set",
            candidates=[r.get("title", "?") for r in resources or []][:3],
        )
        return None

    try:
        records = client.get_data(resource_id, filters={"state_name": state_name})
    except Exception as exc:
        log.warning("nsso_fetch_failed", error=str(exc))
        return None

    if not records:
        return None

    # Expect records to carry a 'monthly_per_capita_expenditure' field.
    try:
        mpce = sorted(float(r["monthly_per_capita_expenditure"]) for r in records)
        median = int(mpce[len(mpce) // 2])
        deciles = {
            f"D{i + 1}": int(mpce[int(len(mpce) * (i + 1) / 10) - 1])
            for i in range(10)
        }
    except (KeyError, ValueError, IndexError) as exc:
        log.warning("nsso_parse_failed", error=str(exc))
        return None

    out = NSSOExpenditure(
        state_name=state_name,
        median_mpce_inr=median,
        decile_breakdown=deciles,
    )
    cache.write_text(json.dumps(out.__dict__, indent=2))
    return out
