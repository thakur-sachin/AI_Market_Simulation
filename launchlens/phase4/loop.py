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
from launchlens.llm import LLMRoute, complete, effective_max_concurrent
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
    llm_errors: int = 0
    missing_memory: int = 0

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

    def all_decisions(self) -> list[AgentDecision]:
        return [d for t in self.timesteps for d in t.decisions]


# ── Single-agent decision ─────────────────────────────────────────────────────

@dataclass
class _StepResult:
    decision: AgentDecision | None
    error: str | None = None   # "missing_memory" | "llm_error" | "parse_failure" | None


async def _agent_step(
    agent_id: str,
    product: ProductStimulus,
    graph: SimGraph,
    all_memories: dict[str, AgentMemory],
    timestep: int,
    semaphore: asyncio.Semaphore,
    rng: random.Random,
    llm_route: LLMRoute,
) -> _StepResult:
    memory = all_memories.get(agent_id)
    if memory is None:
        return _StepResult(None, "missing_memory")

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
            return _StepResult(None, "llm_error")

    decision = parse_decision(raw, agent_id, product.product_id, timestep)
    if decision is None:
        return _StepResult(None, "parse_failure")
    return _StepResult(decision, None)


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
    max_concurrent = max_concurrent or effective_max_concurrent()
    rng = random.Random(seed)
    sem = asyncio.Semaphore(max_concurrent)

    agent_ids = graph.node_ids
    sim_log = SimulationLog(product_id=product.product_id, n_agents=len(agent_ids))

    for t in range(n_timesteps):
        log.info("timestep_start", t=t, agents=len(agent_ids))
        ts_log = TimestepLog(timestep=t)

        all_memories = await memory_store.get_many(agent_ids)

        for batch_start in range(0, len(agent_ids), batch_size):
            batch = agent_ids[batch_start: batch_start + batch_size]
            tasks = [
                _agent_step(
                    aid, product, graph, all_memories, t, sem, rng,
                    agent_llm_routes.get(aid, LLMRoute.SARVAM),
                )
                for aid in batch
            ]
            results: list[_StepResult] = await asyncio.gather(*tasks)

            for aid, res in zip(batch, results):
                if res.decision is None:
                    if res.error == "missing_memory":
                        ts_log.missing_memory += 1
                    elif res.error == "llm_error":
                        ts_log.llm_errors += 1
                    else:
                        ts_log.parse_failures += 1
                    continue
                ts_log.decisions.append(res.decision)
                mem = all_memories[aid]
                _apply_decision(res.decision, mem)
                await memory_store.update(mem)

        # Propagate social signals: decays carried-over signals AND writes fresh ones
        propagate_decisions(ts_log.decisions, graph, all_memories, t)

        # Persist updated memories after propagation
        for mem in all_memories.values():
            await memory_store.update(mem)

        sim_log.timesteps.append(ts_log)
        counts = ts_log.decision_counts()
        log.info(
            "timestep_done", t=t, **counts,
            parse_failures=ts_log.parse_failures,
            llm_errors=ts_log.llm_errors,
            missing_memory=ts_log.missing_memory,
        )

    return sim_log
