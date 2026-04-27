"""
CLI entry points.
Usage:
  python -m launchlens.cli ingest-census          # build DistrictProfile JSONs
  python -m launchlens.cli generate-personas --district MP001 --n 1000
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import structlog

log = structlog.get_logger()


def cmd_ingest_census(args: list[str]) -> None:
    from launchlens.phase1 import build_all_district_profiles, save_district_profiles
    n = save_district_profiles(build_all_district_profiles())
    print(f"Saved {n} district profiles.")


def cmd_generate_personas(args: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--district", required=True, help="District ID or name")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("./data/processed/personas"))
    opts = p.parse_args(args)

    from launchlens.phase1 import (
        load_district_profile,
        load_district_profile_by_name,
        sample_demographic_vectors,
        generate_personas,
        validate_population_diversity,
        run_qa_batch,
    )

    try:
        profile = load_district_profile(opts.district)
    except FileNotFoundError:
        profile = load_district_profile_by_name(opts.district)

    print(f"Sampling {opts.n} agents from {profile.district_name}...")
    vectors = sample_demographic_vectors(profile, n=opts.n, seed=opts.seed)

    print("Generating persona biographies (LLM calls)...")
    personas = asyncio.run(generate_personas(vectors))

    div_flags = validate_population_diversity(personas, profile)
    if div_flags:
        print("⚠ Diversity flags:")
        for dim, issues in div_flags.items():
            for issue in issues:
                print(f"  [{dim}] {issue}")
    else:
        print("✓ Diversity check passed.")

    print("Running persona QA...")
    passed, failed = asyncio.run(run_qa_batch(personas))
    print(f"QA: {len(passed)} passed, {len(failed)} failed")

    opts.out.mkdir(parents=True, exist_ok=True)
    out_path = opts.out / f"{profile.district_id}_personas.jsonl"
    with open(out_path, "w") as f:
        for persona in passed:
            f.write(persona.model_dump_json() + "\n")
    print(f"Saved {len(passed)} personas → {out_path}")


_COMMANDS = {
    "ingest-census": cmd_ingest_census,
    "generate-personas": cmd_generate_personas,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print(f"Usage: python -m launchlens.cli [{' | '.join(_COMMANDS)}] [args]")
        sys.exit(1)
    _COMMANDS[sys.argv[1]](sys.argv[2:])
