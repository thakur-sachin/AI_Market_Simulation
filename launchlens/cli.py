"""
CLI entry points.

Usage:
  python -m launchlens.cli ingest-census                          # build DistrictProfile JSONs
  python -m launchlens.cli generate-personas --district MP001 --n 1000 [--local]
  python -m launchlens.cli run-sim --district Indore --agents 30 --timesteps 5 [--local]
  python -m launchlens.cli calibrate --sim-log path/to/log.json --fixture path/to/fixture.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import structlog

log = structlog.get_logger()


def _enable_local_if_requested(local: bool) -> None:
    if local:
        from launchlens.llm import use_local_llm
        use_local_llm(True)
        print("→ Local LLM override active")


# ── ingest-census ─────────────────────────────────────────────────────────────

def cmd_ingest_census(args: list[str]) -> None:
    from launchlens.phase1 import build_all_district_profiles, save_district_profiles
    n = save_district_profiles(build_all_district_profiles())
    print(f"Saved {n} district profiles.")


# ── generate-personas ─────────────────────────────────────────────────────────

def cmd_generate_personas(args: list[str]) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--district", required=True, help="District ID or name")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("./data/processed/personas"))
    p.add_argument("--local", action="store_true", help="Use local LLM (Ollama) for all calls")
    p.add_argument("--skip-qa", action="store_true", help="Skip persona QA pass")
    opts = p.parse_args(args)

    _enable_local_if_requested(opts.local)

    from launchlens.phase1 import (
        load_district_profile, load_district_profile_by_name,
        sample_demographic_vectors, generate_personas,
        validate_population_diversity, run_qa_batch,
    )

    try:
        profile = load_district_profile(opts.district)
    except FileNotFoundError:
        profile = load_district_profile_by_name(opts.district)

    print(f"Sampling {opts.n} agents from {profile.district_name}…")
    vectors = sample_demographic_vectors(profile, n=opts.n, seed=opts.seed)

    print("Generating persona biographies (LLM calls)…")
    personas = asyncio.run(generate_personas(vectors))

    div_flags = validate_population_diversity(personas, profile)
    if div_flags:
        print("⚠ Diversity flags:")
        for dim, issues in div_flags.items():
            for issue in issues:
                print(f"  [{dim}] {issue}")
    else:
        print("✓ Diversity check passed.")

    if opts.skip_qa:
        passed, failed = personas, []
        print("Skipping persona QA.")
    else:
        print("Running persona QA…")
        passed, failed = asyncio.run(run_qa_batch(personas))
        print(f"QA: {len(passed)} passed, {len(failed)} failed")

    opts.out.mkdir(parents=True, exist_ok=True)
    out_path = opts.out / f"{profile.district_id}_personas.jsonl"
    with open(out_path, "w") as f:
        for persona in passed:
            f.write(persona.model_dump_json() + "\n")
    print(f"Saved {len(passed)} personas → {out_path}")


# ── run-sim ───────────────────────────────────────────────────────────────────

def cmd_run_sim(args: list[str]) -> None:
    """Full LLM-driven simulation loop (Phase 4). Sensible defaults for local runs."""
    p = argparse.ArgumentParser()
    p.add_argument("--district", default="Indore", help="District name or ID")
    p.add_argument("--agents", type=int, default=20)
    p.add_argument("--timesteps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local", action="store_true", help="Use local LLM (Ollama) for all calls")
    p.add_argument("--product-name", default="FreshBite Protein Bar")
    p.add_argument("--price", type=int, default=79)
    p.add_argument("--price-mrp", type=int, default=99)
    p.add_argument("--out", type=Path, default=Path("./data/processed/sim_logs"))
    p.add_argument("--skip-personas-llm", action="store_true",
                   help="Use template-only personas (no LLM bio generation) — fastest path")
    p.add_argument("--fixture", type=Path, default=None,
                   help="Optional calibration fixture path; runs phase5 + writes report")
    opts = p.parse_args(args)

    _enable_local_if_requested(opts.local)

    from launchlens.llm import LLMRoute, route_for_agent, effective_max_concurrent
    from launchlens.phase1.persona_gen import sample_demographic_vectors, generate_personas
    from launchlens.phase1.schemas import AgentPersona
    from launchlens.phase2.graph import build_graph
    from launchlens.phase2.influencers import inject_influencers
    from launchlens.phase2.schemas import SimGraph
    from launchlens.phase3.schemas import ProductStimulus
    from launchlens.phase3.memory import MemoryStore
    from launchlens.phase4.loop import run_simulation
    from launchlens.sim_lite import _indore_profile

    # 1. District profile (Indore default; falls back if real profiles aren't on disk)
    if opts.district.lower() in ("indore", "mp001"):
        profile = _indore_profile()
    else:
        from launchlens.phase1 import load_district_profile, load_district_profile_by_name
        try:
            profile = load_district_profile(opts.district)
        except FileNotFoundError:
            profile = load_district_profile_by_name(opts.district)

    # 2. Population
    print(f"Sampling {opts.agents} agents from {profile.district_name}…")
    vectors = sample_demographic_vectors(profile, n=opts.agents, seed=opts.seed)

    if opts.skip_personas_llm:
        print("Skipping LLM bio generation (using template stubs).")
        personas = [
            AgentPersona(
                agent_id=f"agent_{i:04d}",
                demographic=v,
                biography=(
                    f"{v.sex.title()}, age {v.age}, {v.occupation.replace('_',' ')}, "
                    f"ISEC {v.isec_tier}, income ₹{v.monthly_hh_income:,}, "
                    f"{v.primary_language}-speaking, {v.tech_adoption.replace('_',' ')} "
                    f"in {v.district_name}."
                ),
                llm_route=route_for_agent(v.primary_language, v.isec_tier).value,
            )
            for i, v in enumerate(vectors)
        ]
    else:
        print("Generating persona biographies via LLM…")
        personas = asyncio.run(generate_personas(vectors))
        for i, p in enumerate(personas):
            p.agent_id = f"agent_{i:04d}"

    # 3. Graph
    print("Building social graph…")
    G = build_graph(personas, k=6, beta=0.15, seed=opts.seed)
    node_meta = inject_influencers(G, personas, seed=opts.seed)
    graph = SimGraph(
        node_ids=[p.agent_id for p in personas],
        adjacency={n: list(G.neighbors(n)) for n in G.nodes()},
        node_meta=node_meta,
        k=6, beta=0.15, n_agents=opts.agents,
    )

    # 4. Product
    product = ProductStimulus(
        product_id="prod_001",
        product_name=opts.product_name,
        category="Health & Nutrition",
        price_mrp=opts.price_mrp,
        price_launch=opts.price,
        key_features=["20g protein", "No added sugar", "Mango / Chocolate flavors"],
        distribution_channels=["Amazon India", "BigBasket", "Modern Trade"],
        marketing_copy="Fuel your grind. India's first truly tasty protein bar.",
        competitor_context="Yoga Bar (₹50-80), RiteBite Max (₹80-120)",
        target_segment="Health-conscious urban millennials, 22-35, SEC A/B",
    )

    # 5. Memory
    store = MemoryStore()
    asyncio.run(store.init_all({p.agent_id: p.biography for p in personas}))

    # 6. Routes
    routes = {p.agent_id: route_for_agent(p.demographic.primary_language, p.demographic.isec_tier)
              for p in personas}

    # 7. Run
    print(f"Running {opts.timesteps}-timestep simulation "
          f"(concurrency={effective_max_concurrent()})…")
    sim_log = asyncio.run(run_simulation(
        product=product, graph=graph, memory_store=store,
        agent_llm_routes=routes, n_timesteps=opts.timesteps, seed=opts.seed,
    ))

    # 8. Persist
    opts.out.mkdir(parents=True, exist_ok=True)
    sim_path = opts.out / f"{product.product_id}_sim.json"
    sim_path.write_text(json.dumps({
        "product_id": sim_log.product_id,
        "n_agents": sim_log.n_agents,
        "adoption_curve": sim_log.adoption_curve(),
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
    }, indent=2))
    print(f"Sim log → {sim_path}")

    # 9. Optional report + calibration
    if opts.fixture:
        from launchlens.phase5 import calibrate, load_fixture
        from launchlens.phase6 import write_report
        fixture = load_fixture(opts.fixture)
        result = calibrate(sim_log, personas, fixture)
        print("\n" + result.summary())
        report_path = opts.out / f"{product.product_id}_report.md"
        write_report(report_path, product, sim_log, personas, result)
        print(f"\nReport → {report_path}")
    else:
        from launchlens.phase6 import write_report
        report_path = opts.out / f"{product.product_id}_report.md"
        write_report(report_path, product, sim_log, personas)
        print(f"Report → {report_path}")

    # 10. Headline
    curve = sim_log.adoption_curve()
    print(f"\nFinal cumulative adoption: {curve[-1]:.1%}" if curve else "No decisions.")


# ── calibrate ─────────────────────────────────────────────────────────────────

def cmd_calibrate(args: list[str]) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sim-log", type=Path, required=True)
    p.add_argument("--personas", type=Path, required=True, help="JSONL persona dump")
    p.add_argument("--fixture", type=Path, required=True)
    opts = p.parse_args(args)

    from launchlens.phase1.schemas import AgentPersona
    from launchlens.phase3.schemas import AgentDecision
    from launchlens.phase4.loop import SimulationLog, TimestepLog
    from launchlens.phase5 import calibrate, load_fixture

    personas = [AgentPersona.model_validate_json(line) for line in opts.personas.read_text().splitlines() if line.strip()]
    payload = json.loads(opts.sim_log.read_text())
    sim_log = SimulationLog(product_id=payload["product_id"], n_agents=payload["n_agents"])
    for t_entry in payload["timesteps"]:
        ts = TimestepLog(timestep=t_entry["t"])
        ts.parse_failures = t_entry.get("parse_failures", 0)
        ts.llm_errors = t_entry.get("llm_errors", 0)
        ts.missing_memory = t_entry.get("missing_memory", 0)
        ts.decisions = [AgentDecision.model_validate(d) for d in t_entry["decisions"]]
        sim_log.timesteps.append(ts)

    fixture = load_fixture(opts.fixture)
    result = calibrate(sim_log, personas, fixture)
    print(result.summary())


# ── dispatch ──────────────────────────────────────────────────────────────────

_COMMANDS = {
    "ingest-census": cmd_ingest_census,
    "generate-personas": cmd_generate_personas,
    "run-sim": cmd_run_sim,
    "calibrate": cmd_calibrate,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print(f"Usage: python -m launchlens.cli [{' | '.join(_COMMANDS)}] [args]")
        sys.exit(1)
    _COMMANDS[sys.argv[1]](sys.argv[2:])
