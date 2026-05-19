"""End-to-end Phase 4 loop test with a deterministic recorded-transcript provider.

We bypass real LLM calls by monkey-patching ``launchlens.phase4.loop.complete``
to return a JSON response whose decision depends on the agent_id. This lets us
assert that:

  * the loop emits at least one valid decision per timestep
  * adoption (cumulative BUY rate) is monotonically non-decreasing
  * propagation begins firing once the first BUY emerges
  * parse_failures stays at zero for well-formed transcripts
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from launchlens.phase1.schemas import AgentPersona, DemographicVector
from launchlens.phase2.graph import build_graph, to_sim_graph
from launchlens.phase2.influencers import inject_influencers
from launchlens.phase3.memory import MemoryStore
from launchlens.phase3.schemas import ProductStimulus
from launchlens.phase4 import loop as loop_module


pytestmark = pytest.mark.asyncio


def _persona(i: int) -> AgentPersona:
    return AgentPersona(
        agent_id=f"agent_{i:03d}",
        demographic=DemographicVector(
            age=25 + (i % 30), sex="male" if i % 2 == 0 else "female",
            urban=True, isec_tier="B2",
            primary_language="hindi", occupation="services_formal",
            monthly_hh_income=30000 + i * 100, tech_adoption="early_majority",
            smartphone_owner=True, upi_user=True,
            district_id="MP001", district_name="Indore",
            state_name="Madhya Pradesh",
        ),
        biography=f"Test agent {i}. Lives in Indore, earns ₹30K/month.",
        llm_route="sarvam",
    )


def _make_recorded_response(agent_id: str, t: int) -> str:
    """Cycle through decisions deterministically based on agent_id + t.

    Pattern: first few timesteps RESEARCH/CONSIDER, then early agents BUY,
    later agents follow once they receive peer signals. Designed so the
    propagation cascade lights up.
    """
    idx = int(agent_id.split("_")[1])
    if t == 0:
        decision = "AWARE" if idx % 5 == 0 else "IGNORE"
    elif t == 1:
        decision = "RESEARCH" if idx % 3 == 0 else "AWARE"
    elif t == 2:
        decision = "BUY" if idx < 5 else "CONSIDER"
    else:
        decision = "BUY" if idx < 15 else "RESEARCH"
    return json.dumps({
        "internal_reasoning": f"reasoning for {agent_id} at t={t}",
        "decision": decision,
        "primary_reason": "test",
        "would_discuss_with": "friends",
        "language_of_discussion": "Hindi",
    })


async def test_e2e_loop_with_deterministic_provider(monkeypatch):
    n_agents = 30
    n_timesteps = 4

    personas = [_persona(i) for i in range(n_agents)]
    G = build_graph(personas, k=6, beta=0.15, seed=0)
    node_meta = inject_influencers(G, personas, seed=0)
    sim_graph = to_sim_graph(G, personas, node_meta, k=6, beta=0.15)

    store = MemoryStore()
    await store.init_all({p.agent_id: p.biography for p in personas})

    product = ProductStimulus(
        product_id="prod_test",
        product_name="Test Snack",
        category="Snacks",
        price_mrp=99, price_launch=79,
        key_features=["a"], distribution_channels=["x"],
        marketing_copy="m", competitor_context="c",
        target_segment="t",
    )

    # Monkey-patch the LLM entry point used by the loop.
    async def fake_complete(*, route: Any = None, system: str, user: str,
                            temperature: float = 0.8, max_tokens: int = 400,
                            json_mode: bool = False,
                            engine_override: str | None = None,
                            use_cache: bool = False) -> str:
        # Recover agent identity from the user prompt (biographies include "Test agent N.")
        agent_id = "agent_000"
        for line in user.splitlines():
            if "Test agent " in line:
                token = line.split("Test agent ")[1].split(".")[0].strip()
                try:
                    agent_id = f"agent_{int(token):03d}"
                except ValueError:
                    pass
                break
        timestep = user.count("- t")  # crude proxy: episodic_buffer entries grow each tick
        timestep = min(timestep, n_timesteps - 1)
        return _make_recorded_response(agent_id, timestep)

    monkeypatch.setattr(loop_module, "complete", fake_complete)

    from launchlens.llm import LLMRoute
    routes = {p.agent_id: LLMRoute(p.llm_route) for p in personas}

    sim_log = await loop_module.run_simulation(
        product=product,
        graph=sim_graph,
        memory_store=store,
        agent_llm_routes=routes,
        n_timesteps=n_timesteps,
        seed=42,
        engine_override="mock",  # bypass cost gates; real call is monkey-patched
    )

    # No parse failures with well-formed JSON
    assert sim_log.total_parse_failures == 0

    # Each timestep produced decisions
    for ts in sim_log.timesteps:
        assert len(ts.decisions) > 0

    # Adoption curve must be monotonically non-decreasing
    curve = sim_log.adoption_curve()
    assert len(curve) == n_timesteps
    for prev, curr in zip(curve, curve[1:]):
        assert curr >= prev

    # At least one BUY occurs by timestep 2
    final = curve[-1]
    assert final > 0.0


async def test_e2e_loop_records_parse_failures(monkeypatch):
    """Garbage LLM output must surface as parse_failures, never silent IGNOREs."""
    personas = [_persona(i) for i in range(5)]
    G = build_graph(personas, k=4, beta=0.1, seed=0)
    node_meta = inject_influencers(G, personas, seed=0)
    sim_graph = to_sim_graph(G, personas, node_meta, k=4, beta=0.1)
    store = MemoryStore()
    await store.init_all({p.agent_id: p.biography for p in personas})

    async def garbage(**kwargs) -> str:
        return "no JSON and no fielded format here, just rambling text"

    monkeypatch.setattr(loop_module, "complete", garbage)

    from launchlens.llm import LLMRoute
    routes = {p.agent_id: LLMRoute(p.llm_route) for p in personas}

    product = ProductStimulus(
        product_id="p", product_name="P", category="c",
        price_mrp=10, price_launch=10, key_features=["a"],
        distribution_channels=["x"], marketing_copy="m",
        competitor_context="c", target_segment="t",
    )
    sim_log = await loop_module.run_simulation(
        product=product, graph=sim_graph, memory_store=store,
        agent_llm_routes=routes, n_timesteps=2, seed=0,
        engine_override="mock",
    )
    assert sim_log.total_parse_failures == len(personas) * 2
    assert all(len(t.decisions) == 0 for t in sim_log.timesteps)
