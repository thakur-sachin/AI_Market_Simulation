"""
Main simulation loop — Phase 4.
Async batch processing: build feeds → LLM decisions → update memory → propagate.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field

import structlog

from launchlens.config import get_settings
from launchlens.llm import LLMRoute, complete
from launchlens.phase2.schemas import SimGraph
from launchlens.phase3.schemas import AgentDecision, AgentMemory, ProductStimulus
from launchlens.phase3.memory import MemoryStore
from launchlens.phase3.feed import build_feed
from launchlens.phase4.prompts import build_decision_prompt
from launchlens.phase4.decisions import parse_decision
from launchlens.phase4.propagation import propagate_decisions

log = structlog.get_logger()
_cfg = get_settings()


@dataclass
class TimestepLog:
    timestep: int
    decisions: list[AgentDecision] = field(default_factory=list)
    parse_failures: int = 0

    def decision_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.decisions:
            counts[d.decision] = counts.get(d.decision, 0) + 1
        return counts


@dataclass
class SimulationLog:
    product_id: str
    n_agents: int
    timesteps: list[TimestepLog] = field(default_factory=list)

    def adoption_curve(self) -> list[float]:
        """Cumulative BUY rate per timestep."""
        buyers: set[str] = set()
        curve = []
        for t in self.timesteps:
            for d in t.decisions:
                if d.decision == "BUY":
                    buyers.add(d.agent_id)
            curve.append(len(buyers) / self.n_agents if self.n_agents else 0.0)
        return curve


# ── Single-agent decision ─────────────────────────────────────────────────────

async def _agent_step(
    agent_id: str,
    product: ProductStimulus,
    graph: SimGraph,
    all_memories: dict[str, AgentMemory],
    timestep: int,
    semaphore: asyncio.Semaphore,
    rng: random.Random,
    llm_route: LLMRoute,
) -> AgentDecision | None:
    memory = all_memories.get(agent_id)
    if memory is None:
        return None

    feed = build_feed(agent_id, memory, product, graph, all_memories, timestep, rng)
    system, user = build_decision_prompt(
        memory, feed, product.product_name, product.price_launch, product.product_id
    )

    async with semaphore:
        try:
            raw = await complete(
                route=llm_route,
                system=system,
                user=user,
                temperature=0.85,
                max_tokens=400,
            )
        except Exception as e:
            log.warning("llm_call_failed", agent_id=agent_id, error=str(e))
            return None

    return parse_decision(raw, agent_id, product.product_id, timestep)


# ── Memory update after decision ──────────────────────────────────────────────

def _apply_decision(decision: AgentDecision, memory: AgentMemory) -> None:
    memory.current_decision[decision.product_id] = decision.decision
    memory.product_opinion[decision.product_id] = decision.internal_reasoning
    memory.add_event(
        f"t{decision.timestep}: {decision.decision} — {decision.primary_reason[:80]}"
    )
    if decision.decision == "BUY":
        memory.purchase_history.append({
            "timestep": decision.timestep,
            "product_id": decision.product_id,
            "reason": decision.primary_reason,
        })
    # Clear consumed signals
    memory.peer_signals = []


# ── Main loop ─────────────────────────────────────────────────────────────────

async def run_simulation(
    product: ProductStimulus,
    graph: SimGraph,
    memory_store: MemoryStore,
    agent_llm_routes: dict[str, LLMRoute],
    n_timesteps: int = 12,
    batch_size: int | None = None,
    max_concurrent: int | None = None,
    seed: int | None = None,
) -> SimulationLog:
    batch_size = batch_size or _cfg.llm_batch_size
    max_concurrent = max_concurrent or _cfg.llm_max_concurrent
    rng = random.Random(seed)
    sem = asyncio.Semaphore(max_concurrent)

    agent_ids = graph.node_ids
    sim_log = SimulationLog(product_id=product.product_id, n_agents=len(agent_ids))

    for t in range(n_timesteps):
        log.info("timestep_start", t=t, agents=len(agent_ids))
        ts_log = TimestepLog(timestep=t)

        # Load all memories
        all_memories = await memory_store.get_many(agent_ids)

        # Process in batches
        for batch_start in range(0, len(agent_ids), batch_size):
            batch = agent_ids[batch_start: batch_start + batch_size]
            tasks = [
                _agent_step(
                    aid, product, graph, all_memories, t, sem, rng,
                    agent_llm_routes.get(aid, LLMRoute.SARVAM),
                )
                for aid in batch
            ]
            results = await asyncio.gather(*tasks)

            for aid, decision in zip(batch, results):
                if decision is None:
                    ts_log.parse_failures += 1
                    continue
                ts_log.decisions.append(decision)
                mem = all_memories[aid]
                _apply_decision(decision, mem)
                await memory_store.update(mem)

        # Propagate social signals
        propagate_decisions(ts_log.decisions, graph, all_memories, t)

        # Persist updated memories after propagation
        for mem in all_memories.values():
            await memory_store.update(mem)

        sim_log.timesteps.append(ts_log)
        counts = ts_log.decision_counts()
        log.info("timestep_done", t=t, **counts, failures=ts_log.parse_failures)

    return sim_log
