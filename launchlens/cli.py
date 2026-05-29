"""
CLI entry points.

Usage examples:
  python -m launchlens.cli ingest-census
  python -m launchlens.cli fetch-data --district MP001 --name Indore --state "Madhya Pradesh"
  python -m launchlens.cli generate-personas --district MP001 --n 100 --engine local [--qa]
  python -m launchlens.cli run-sim --district MP001 --agents 30 --timesteps 5 --engine local
  python -m launchlens.cli run-sim ... --calibrate paper_boat_aam_panna   # also emits report
  python -m launchlens.cli calibrate --product paper_boat_aam_panna --sim-log <file>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import structlog

from launchlens.config import get_settings, reload_settings
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


def _apply_engine_overrides(engine: str | None, model: str | None) -> None:
    """Mutate env + invalidate settings cache so overrides take effect this process."""
    if engine:
        os.environ["LAUNCHLENS_ENGINE"] = engine
    if model:
        os.environ["OLLAMA_DEFAULT_MODEL"] = model
        os.environ["OLLAMA_INDIC_MODEL"] = model
    if engine or model:
        reload_settings()


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


# ── fetch-data (real-data orchestrator) ──────────────────────────────────────

def cmd_fetch_data(args: list[str]) -> int:
    p = argparse.ArgumentParser(description="Fetch real demographic data for a district.")
    p.add_argument("--district", required=True, help="Census district code (e.g. MP001)")
    p.add_argument("--name", help="Human-readable name (defaults to district code)")
    p.add_argument("--state", help="State name (helps NSSO lookups)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--strict", action="store_true",
                   help="Fail if any required field falls back to baseline")
    opts = p.parse_args(args)

    from launchlens.phase1.sources import load_district_profile_chain

    profile = load_district_profile_chain(
        district_id=opts.district,
        district_name=opts.name or opts.district,
        state_name=opts.state or "",
        strict=opts.strict,
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
    p.add_argument("--engine", choices=["auto", "mock", "local", "sarvam", "claude"], default=None)
    p.add_argument("--model", default=None, help="Override Ollama model tag for local engine")
    opts = p.parse_args(args)

    _apply_engine_overrides(opts.engine, opts.model)

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

    print(f"Sampling {opts.n} agents from {profile.district_name}…")
    vectors = sample_demographic_vectors(profile, n=opts.n, seed=opts.seed)

    provider = select_provider(LLMRoute.SARVAM, engine_override=opts.engine, model_override=opts.model)
    print(f"Using LLM provider: {provider.name} (model={provider.model})")
    personas = asyncio.run(generate_personas(vectors, engine_override=opts.engine))

    div_flags = validate_population_diversity(personas, profile)
    if div_flags:
        print("⚠ Diversity flags:")
        for dim, issues in div_flags.items():
            for issue in issues:
                print(f"  [{dim}] {issue}")
    else:
        print("✓ Diversity check passed.")

    if opts.qa:
        print("Running persona QA…")
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


# ── run-sim ──────────────────────────────────────────────────────────────────

def cmd_run_sim(args: list[str]) -> int:
    p = argparse.ArgumentParser(description="Run an end-to-end LaunchLens simulation.")
    p.add_argument("--district", default="indore")
    p.add_argument("--agents", type=int, default=30)
    p.add_argument("--timesteps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--engine", choices=["auto", "mock", "local", "sarvam", "claude"], default="auto")
    p.add_argument("--model", default=None, help="Override Ollama model tag for the local engine")
    p.add_argument("--dry-run", action="store_true",
                   help="Force the sim_lite heuristic engine (no LLM)")
    p.add_argument("--confirm-cost", action="store_true",
                   help="Acknowledge projected remote-provider cost above threshold")
    p.add_argument("--skip-personas-llm", action="store_true",
                   help="Use template-only persona biographies (skip LLM bio generation)")
    p.add_argument("--calibrate", default=None,
                   help="Calibration case id (file under data/calibration/<id>.json)")
    p.add_argument("--out", type=Path, default=Path("./outputs"))
    opts = p.parse_args(args)

    _apply_engine_overrides(opts.engine, opts.model)

    if opts.dry_run:
        from launchlens.sim_lite import run_lite
        sim_log = asyncio.run(run_lite(
            n_agents=opts.agents, n_timesteps=opts.timesteps, seed=opts.seed,
        ))
        opts.out.mkdir(parents=True, exist_ok=True)
        path = opts.out / f"sim_dryrun_{opts.district}_{opts.agents}a_{opts.timesteps}t.json"
        path.write_text(json.dumps({
            "engine": "dry-run (sim_lite heuristic)",
            "adoption_curve": sim_log.adoption_curve(),
        }, indent=2))
        print(f"Saved {path}")
        return 0

    # Cost preflight (no-op for local/mock)
    try:
        projected, provider_name = estimate_cost(
            n_agents=opts.agents, n_timesteps=opts.timesteps,
            engine_override=opts.engine, model_override=opts.model,
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
    from launchlens.phase1 import sample_demographic_vectors, generate_personas
    from launchlens.phase1.schemas import AgentPersona
    from launchlens.phase1.sources import load_local_district_profile
    from launchlens.phase2.graph import build_graph, to_sim_graph, validate_small_world
    from launchlens.phase2.influencers import inject_influencers
    from launchlens.phase3.memory import MemoryStore
    from launchlens.phase3.schemas import ProductStimulus
    from launchlens.phase4.loop import run_simulation
    from launchlens.sim_lite import _indore_profile

    # District profile
    if opts.district.lower() in ("indore", "mp001"):
        try:
            profile = load_local_district_profile(opts.district)
        except FileNotFoundError:
            profile = _indore_profile()
            print(f"Note: using hardcoded Indore profile (no saved profile at "
                  f"{_cfg.district_profiles_dir}). Run `fetch-data --district MP001` "
                  f"to materialize real one.")
    else:
        try:
            profile = load_local_district_profile(opts.district)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            print("Run `python -m launchlens.cli fetch-data --district <id>` first.")
            return 1

    # Personas
    print(f"Sampling {opts.agents} agents from {profile.district_name}…")
    vectors = sample_demographic_vectors(profile, n=opts.agents, seed=opts.seed)

    if opts.skip_personas_llm:
        print("Skipping LLM bio generation (using template stubs).")
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
    else:
        print("Generating persona biographies via LLM…")
        personas = await generate_personas(vectors, engine_override=opts.engine)
        for i, persona in enumerate(personas):
            persona.agent_id = f"agent_{i:04d}"

    # Graph
    G = build_graph(personas, k=6, beta=0.15, seed=opts.seed)
    node_meta = inject_influencers(G, personas, seed=opts.seed)
    sim_graph = to_sim_graph(G, personas, node_meta, k=6, beta=0.15)
    sw = validate_small_world(G, k=6)
    print(f"Graph σ={sw['small_world_sigma']:.2f}  CC={sw['clustering_coefficient']:.3f}")

    # Memory
    store = MemoryStore()
    await store.init_all({p.agent_id: p.biography for p in personas})

    # Product. If --calibrate is set and the fixture carries a 'product' block,
    # use that so the sim and the ground truth are about the same product.
    product: ProductStimulus | None = None
    if opts.calibrate:
        from launchlens.phase5 import load_calibration_case
        try:
            case = load_calibration_case(opts.calibrate)
        except FileNotFoundError:
            case = None
        if case is not None and case.product:
            product = ProductStimulus.model_validate(case.product)
            print(f"Using product from calibration fixture: {product.product_name} "
                  f"@ ₹{product.price_launch}")
    if product is None:
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

    # Run
    sim_log = await run_simulation(
        product=product, graph=sim_graph, memory_store=store,
        agent_llm_routes=routes, n_timesteps=opts.timesteps, seed=opts.seed,
        engine_override=opts.engine, model_override=opts.model,
    )

    # Persist
    opts.out.mkdir(parents=True, exist_ok=True)
    sim_path = opts.out / f"sim_{opts.district}_{opts.agents}a_{opts.timesteps}t.json"
    sim_payload = {
        "product_id": sim_log.product_id,
        "engine": sim_log.engine,
        "n_agents": sim_log.n_agents,
        "adoption_curve": sim_log.adoption_curve(),
        "parse_failures": sim_log.total_parse_failures,
        "llm_errors": sim_log.total_llm_errors,
        "usage": get_usage_tracker().summary(),
        "timesteps": [
            {
                "t": t.timestep,
                "counts": t.decision_counts(),
                "parse_failures": t.parse_failures,
                "llm_errors": t.llm_errors,
                "missing_memory": t.missing_memory,
                "decisions": [d.model_dump() for d in t.decisions],
            }
            for t in sim_log.timesteps
        ],
    }
    sim_path.write_text(json.dumps(sim_payload, indent=2))
    print(f"Sim log → {sim_path}")

    # Optional calibration + report
    from launchlens.phase5 import calibrate_from_sim_log, load_calibration_case
    from launchlens.phase6 import write_report

    calibration = None
    if opts.calibrate:
        try:
            case = load_calibration_case(opts.calibrate)
        except FileNotFoundError as exc:
            print(f"⚠ Calibration case not found: {exc}")
        else:
            calibration = calibrate_from_sim_log(sim_log, personas, case)
            calib_path = opts.out / f"calibration_{opts.calibrate}.json"
            calib_path.write_text(json.dumps({
                "product_id": calibration.product_id,
                "engine": calibration.engine,
                "metrics": calibration.metrics,
                "gates": calibration.gates,
                "tuning_signals": calibration.tuning_signals,
                "passed_all_gates": calibration.passed,
            }, indent=2))
            print(f"Calibration → {calib_path}")
            passed = sum(1 for v in calibration.gates.values() if v)
            print(f"\nValidation gates: {passed}/{len(calibration.gates)} passed")
            for metric, val in calibration.metrics.items():
                mark = "✓" if calibration.gates.get(metric) else "✗"
                v = f"{val:.4f}" if isinstance(val, float) else str(val)
                print(f"  {mark} {metric:<30} {v}")

    report_path = opts.out / f"report_{opts.district}_{opts.agents}a_{opts.timesteps}t.md"
    write_report(report_path, product, sim_log, personas, calibration)
    print(f"Report → {report_path}")

    curve = sim_log.adoption_curve()
    print(f"\nFinal cumulative adoption: {curve[-1]:.1%}" if curve else "No decisions.")
    return 0


# ── calibrate (re-score an existing sim log) ─────────────────────────────────

def cmd_calibrate(args: list[str]) -> int:
    p = argparse.ArgumentParser(description="Score an existing sim log against a calibration case.")
    p.add_argument("--product", required=True, help="Calibration case id (data/calibration/<id>.json)")
    p.add_argument("--sim-log", type=Path, required=True, help="Path to sim JSON written by run-sim")
    p.add_argument("--out", type=Path, default=Path("./outputs"))
    opts = p.parse_args(args)

    from launchlens.phase5 import load_calibration_case, run_calibration

    case = load_calibration_case(opts.product)
    sim_summary = json.loads(opts.sim_log.read_text())
    report = run_calibration(case, sim_summary)

    opts.out.mkdir(parents=True, exist_ok=True)
    out = opts.out / f"calibration_{opts.product}.json"
    out.write_text(json.dumps({
        "product_id": report.product_id,
        "engine": report.engine,
        "metrics": report.metrics,
        "gates": report.gates,
        "tuning_signals": report.tuning_signals,
        "passed_all_gates": report.passed,
    }, indent=2))
    passed = sum(1 for v in report.gates.values() if v)
    print(f"\nValidation gates: {passed}/{len(report.gates)} passed")
    for metric, val in report.metrics.items():
        mark = "✓" if report.gates.get(metric) else "✗"
        v = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"  {mark} {metric:<30} {v}")
    print(f"\nSaved {out}")
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
