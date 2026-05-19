"""
CLI entry points.

Usage examples:
  python -m launchlens.cli ingest-census
  python -m launchlens.cli generate-personas --district MP001 --n 1000 [--qa]
  python -m launchlens.cli run-sim --district indore --agents 50 --timesteps 4 [--engine local]
  python -m launchlens.cli calibrate --product paper_boat_aam_panna --district indore
  python -m launchlens.cli fetch-data --district indore
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import structlog

from launchlens.config import get_settings
from launchlens.llm import (
    LLMError,
    LLMRoute,
    estimate_cost,
    get_usage_tracker,
    route_for_agent,
    select_provider,
)

log = structlog.get_logger()
_cfg = get_settings()


# ── ingest-census ─────────────────────────────────────────────────────────────

def cmd_ingest_census(args: list[str]) -> int:
    from launchlens.phase1 import build_all_district_profiles, save_district_profiles

    try:
        n = save_district_profiles(build_all_district_profiles())
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Saved {n} district profiles.")
    return 0


# ── fetch-data (W1 orchestrator) ──────────────────────────────────────────────

def cmd_fetch_data(args: list[str]) -> int:
    p = argparse.ArgumentParser(description="Fetch real demographic data for a district.")
    p.add_argument("--district", required=True, help="Census district code (e.g. MP001)")
    p.add_argument("--name", help="Human-readable name (defaults to district code)")
    p.add_argument("--state", help="State name (helps NSSO lookups)")
    p.add_argument("--out", type=Path, default=None)
    opts = p.parse_args(args)

    from launchlens.phase1.sources import load_district_profile_chain

    profile = load_district_profile_chain(
        district_id=opts.district,
        district_name=opts.name or opts.district,
        state_name=opts.state or "",
    )
    out = opts.out or (_cfg.district_profiles_dir / f"{profile.district_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(profile.model_dump_json(indent=2))
    print(f"✓ Saved {out}")
    print(f"  Provenance: {profile.provenance}")
    return 0


# ── generate-personas ─────────────────────────────────────────────────────────

def cmd_generate_personas(args: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--district", required=True, help="District ID or name")
    p.add_argument("--n", type=int, default=_cfg.default_agent_count)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("./data/processed/personas"))
    p.add_argument("--qa", action="store_true", help="Run persona QA after generation")
    p.add_argument("--engine", choices=["auto", "mock", "local", "sarvam", "claude"],
                   default=None, help="Override LAUNCHLENS_ENGINE for this run")
    opts = p.parse_args(args)

    if opts.engine:
        os.environ["LAUNCHLENS_ENGINE"] = opts.engine

    from launchlens.phase1 import (
        generate_personas,
        load_district_profile,
        load_district_profile_by_name,
        run_qa_batch,
        sample_demographic_vectors,
        validate_population_diversity,
    )

    try:
        profile = load_district_profile(opts.district)
    except FileNotFoundError:
        profile = load_district_profile_by_name(opts.district)

    print(f"Sampling {opts.n} agents from {profile.district_name}...")
    vectors = sample_demographic_vectors(profile, n=opts.n, seed=opts.seed)

    provider = select_provider(LLMRoute.SARVAM)
    print(f"Using LLM provider: {provider.name} (model={provider.model})")
    personas = asyncio.run(generate_personas(vectors))

    div_flags = validate_population_diversity(personas, profile)
    if div_flags:
        print("⚠ Diversity flags:")
        for dim, issues in div_flags.items():
            for issue in issues:
                print(f"  [{dim}] {issue}")
    else:
        print("✓ Diversity check passed.")

    if opts.qa:
        print("Running persona QA...")
        passed, failed = asyncio.run(run_qa_batch(personas))
        print(f"QA: {len(passed)} passed, {len(failed)} failed")
        personas = passed
    else:
        print("(persona QA skipped — pass --qa to enable)")

    opts.out.mkdir(parents=True, exist_ok=True)
    out_path = opts.out / f"{profile.district_id}_personas.jsonl"
    with open(out_path, "w") as f:
        for persona in personas:
            f.write(persona.model_dump_json() + "\n")
    print(f"Saved {len(personas)} personas → {out_path}")
    return 0


# ── run-sim (mock | local | remote) ──────────────────────────────────────────

def cmd_run_sim(args: list[str]) -> int:
    p = argparse.ArgumentParser(description="Run an end-to-end LaunchLens simulation.")
    p.add_argument("--district", default="indore")
    p.add_argument("--agents", type=int, default=50)
    p.add_argument("--timesteps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--engine",
                   choices=["auto", "mock", "local", "sarvam", "claude"],
                   default="auto")
    p.add_argument("--model", default=None,
                   help="Override Ollama model tag for the local engine")
    p.add_argument("--dry-run", action="store_true",
                   help="Force mock engine (sim_lite heuristic decisions, no LLM)")
    p.add_argument("--confirm-cost", action="store_true",
                   help="Acknowledge projected remote-provider cost above threshold")
    p.add_argument("--out", type=Path, default=Path("./outputs"))
    opts = p.parse_args(args)

    if opts.dry_run:
        # --dry-run uses the stochastic mock heuristic engine (sim_lite) end-to-end.
        # It does NOT call any LLM. Useful for shape-validating runs on the dev hardware.
        from launchlens.sim_lite import run_lite

        sim_log = asyncio.run(run_lite(
            n_agents=opts.agents,
            n_timesteps=opts.timesteps,
            seed=opts.seed,
        ))
        opts.out.mkdir(parents=True, exist_ok=True)
        path = opts.out / f"sim_dryrun_{opts.district}_{opts.agents}a_{opts.timesteps}t.json"
        path.write_text(json.dumps({
            "engine": "dry-run (sim_lite heuristic)",
            "adoption_curve": sim_log.adoption_curve(),
        }, indent=2))
        print(f"Saved {path}")
        return 0

    if opts.model:
        os.environ["LAUNCHLENS_OLLAMA_DEFAULT_MODEL"] = opts.model
    os.environ["LAUNCHLENS_ENGINE"] = opts.engine

    # Cost pre-flight: only meaningful for remote providers.
    try:
        projected, provider_name = estimate_cost(
            n_agents=opts.agents, n_timesteps=opts.timesteps,
            engine_override=opts.engine,
        )
    except LLMError as exc:
        print(f"ERROR resolving engine '{opts.engine}': {exc}")
        return 2

    print(f"Projected cost: ${projected:.4f} on provider '{provider_name}'")
    if projected > _cfg.cost_confirm_threshold_usd and not opts.confirm_cost:
        print(f"Projected cost exceeds ${_cfg.cost_confirm_threshold_usd:.2f}. "
              f"Re-run with --confirm-cost to proceed.")
        return 3

    return asyncio.run(_run_real_sim(opts))


async def _run_real_sim(opts: argparse.Namespace) -> int:
    from launchlens.phase1 import sample_demographic_vectors
    from launchlens.phase1.schemas import AgentPersona
    from launchlens.phase1.sources import load_local_district_profile
    from launchlens.phase2.graph import build_graph, to_sim_graph, validate_small_world
    from launchlens.phase2.influencers import inject_influencers
    from launchlens.phase3.memory import MemoryStore
    from launchlens.phase3.schemas import ProductStimulus
    from launchlens.phase4.loop import run_simulation

    try:
        profile = load_local_district_profile(opts.district)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Run `python -m launchlens.cli fetch-data --district <id>` first.")
        return 1

    vectors = sample_demographic_vectors(profile, n=opts.agents, seed=opts.seed)
    personas = [
        AgentPersona(
            agent_id=f"agent_{i:04d}",
            demographic=v,
            biography=(
                f"{v.sex.title()}, age {v.age}, {v.occupation.replace('_', ' ')}, "
                f"ISEC {v.isec_tier}, monthly household income ₹{v.monthly_hh_income:,}, "
                f"{v.primary_language}-speaking, {v.tech_adoption.replace('_', ' ')}. "
                f"Lives in {profile.district_name}, {profile.state_name}."
            ),
            llm_route=route_for_agent(v.primary_language, v.isec_tier).value,
        )
        for i, v in enumerate(vectors)
    ]

    G = build_graph(personas, k=6, beta=0.15, seed=opts.seed)
    node_meta = inject_influencers(G, personas, seed=opts.seed)
    sim_graph = to_sim_graph(G, personas, node_meta, k=6, beta=0.15)

    sw = validate_small_world(G, k=6)
    print(f"Graph σ={sw['small_world_sigma']:.2f}  CC={sw['clustering_coefficient']:.3f}")

    store = MemoryStore()
    await store.init_all({p.agent_id: p.biography for p in personas})

    product = ProductStimulus(
        product_id="prod_demo",
        product_name="FreshBite Protein Bar",
        category="Health & Nutrition",
        price_mrp=99, price_launch=79,
        key_features=["20g protein", "No added sugar", "Mango / Chocolate / Peanut"],
        distribution_channels=["Amazon India", "BigBasket"],
        marketing_copy="Fuel your grind. India's first truly tasty protein bar.",
        competitor_context="Yoga Bar (₹50-80), RiteBite Max (₹80-120)",
        target_segment="Health-conscious millennials, 22-35, SEC A/B",
    )

    routes = {p.agent_id: LLMRoute(p.llm_route) for p in personas}

    sim_log = await run_simulation(
        product=product,
        graph=sim_graph,
        memory_store=store,
        agent_llm_routes=routes,
        n_timesteps=opts.timesteps,
        seed=opts.seed,
        engine_override=opts.engine,
    )

    opts.out.mkdir(parents=True, exist_ok=True)
    path = opts.out / f"sim_{opts.district}_{opts.agents}a_{opts.timesteps}t.json"
    summary = {
        "engine": sim_log.engine,
        "agents": sim_log.n_agents,
        "adoption_curve": sim_log.adoption_curve(),
        "parse_failures": sim_log.total_parse_failures,
        "usage": get_usage_tracker().summary(),
    }
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {path}")
    return 0


# ── calibrate ────────────────────────────────────────────────────────────────

def cmd_calibrate(args: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--product", required=True,
                   help="Calibration case id (file under data/calibration/<id>.json)")
    p.add_argument("--district", default="indore")
    p.add_argument("--agents", type=int, default=200)
    p.add_argument("--timesteps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--engine", choices=["auto", "mock", "local", "sarvam", "claude"],
                   default="auto")
    opts = p.parse_args(args)

    from launchlens.phase5.calibration import (
        CalibrationCase,
        load_calibration_case,
        run_calibration,
    )

    try:
        case: CalibrationCase = load_calibration_case(opts.product)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    # Re-use run-sim path to generate the SimulationLog.
    fake_args = [
        "--district", opts.district,
        "--agents", str(opts.agents),
        "--timesteps", str(opts.timesteps),
        "--seed", str(opts.seed),
        "--engine", opts.engine,
        "--out", "./outputs",
    ]
    rc = cmd_run_sim(fake_args)
    if rc != 0:
        return rc

    sim_path = Path("./outputs") / f"sim_{opts.district}_{opts.agents}a_{opts.timesteps}t.json"
    sim_summary = json.loads(sim_path.read_text())
    report = run_calibration(case, sim_summary)
    out = Path("./outputs") / f"calibration_{opts.product}.json"
    out.write_text(json.dumps(report.to_dict(), indent=2))
    print(json.dumps(report.to_dict(), indent=2))
    print(f"Saved {out}")
    return 0


# ── dispatcher ───────────────────────────────────────────────────────────────

_COMMANDS = {
    "ingest-census": cmd_ingest_census,
    "fetch-data": cmd_fetch_data,
    "generate-personas": cmd_generate_personas,
    "run-sim": cmd_run_sim,
    "calibrate": cmd_calibrate,
}


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] not in _COMMANDS:
        print(f"Usage: python -m launchlens.cli [{ ' | '.join(_COMMANDS) }] [args]")
        return 1
    return _COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
