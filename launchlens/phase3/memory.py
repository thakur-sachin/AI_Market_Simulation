"""
Two-tier agent memory.
Tier 1 (episodic): Redis with TTL — fast rolling buffer.
Tier 2 (semantic): PostgreSQL + pgvector — structured K/V + similarity search.
Falls back to in-memory dicts when REDIS_URL / POSTGRES_URL are not configured,
enabling the lightweight sim to run without infrastructure.
"""
from __future__ import annotations

from typing import Protocol

import structlog

from launchlens.phase3.schemas import AgentMemory

log = structlog.get_logger()


# ── Storage backend protocol ──────────────────────────────────────────────────

class MemoryBackend(Protocol):
    async def save(self, memory: AgentMemory) -> None: ...
    async def load(self, agent_id: str) -> AgentMemory | None: ...
    async def load_many(self, agent_ids: list[str]) -> dict[str, AgentMemory]: ...


# ── In-memory backend (dev / lightweight sim) ─────────────────────────────────

class InMemoryBackend:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def save(self, memory: AgentMemory) -> None:
        self._store[memory.agent_id] = memory.model_dump_json()

    async def load(self, agent_id: str) -> AgentMemory | None:
        raw = self._store.get(agent_id)
        return AgentMemory.model_validate_json(raw) if raw else None

    async def load_many(self, agent_ids: list[str]) -> dict[str, AgentMemory]:
        result = {}
        for aid in agent_ids:
            m = await self.load(aid)
            if m:
                result[aid] = m
        return result


# ── Redis backend (production) ────────────────────────────────────────────────

class RedisMemoryBackend:
    """Tier-1 episodic memory via Redis. TTL keeps only recent context."""

    def __init__(self, redis_url: str, ttl_seconds: int = 86400 * 7) -> None:
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(redis_url, decode_responses=True)
            self._ttl = ttl_seconds
            self._ok = True
        except ImportError:
            log.warning("redis_not_installed", fallback="in-memory")
            self._ok = False
            self._fallback = InMemoryBackend()

    async def save(self, memory: AgentMemory) -> None:
        if not self._ok:
            await self._fallback.save(memory)
            return
        key = f"agent:{memory.agent_id}"
        await self._redis.set(key, memory.model_dump_json(), ex=self._ttl)

    async def load(self, agent_id: str) -> AgentMemory | None:
        if not self._ok:
            return await self._fallback.load(agent_id)
        raw = await self._redis.get(f"agent:{agent_id}")
        return AgentMemory.model_validate_json(raw) if raw else None

    async def load_many(self, agent_ids: list[str]) -> dict[str, AgentMemory]:
        if not self._ok:
            return await self._fallback.load_many(agent_ids)
        keys = [f"agent:{aid}" for aid in agent_ids]
        values = await self._redis.mget(*keys)
        result = {}
        for aid, raw in zip(agent_ids, values):
            if raw:
                result[aid] = AgentMemory.model_validate_json(raw)
        return result


# ── Factory ───────────────────────────────────────────────────────────────────

def make_backend(redis_url: str | None = None) -> MemoryBackend:
    if redis_url:
        return RedisMemoryBackend(redis_url)
    log.info("memory_backend", mode="in_memory")
    return InMemoryBackend()


# ── MemoryStore — high-level interface ────────────────────────────────────────

class MemoryStore:
    def __init__(self, backend: MemoryBackend | None = None) -> None:
        self._backend = backend or InMemoryBackend()

    async def init_agent(self, agent_id: str, biography: str) -> AgentMemory:
        memory = AgentMemory(agent_id=agent_id, biography=biography)
        await self._backend.save(memory)
        return memory

    async def get(self, agent_id: str) -> AgentMemory | None:
        return await self._backend.load(agent_id)

    async def get_many(self, agent_ids: list[str]) -> dict[str, AgentMemory]:
        return await self._backend.load_many(agent_ids)

    async def update(self, memory: AgentMemory) -> None:
        await self._backend.save(memory)

    async def init_all(self, agent_map: dict[str, str]) -> None:
        """agent_map: {agent_id: biography}"""
        for agent_id, bio in agent_map.items():
            await self.init_agent(agent_id, bio)
        log.info("memory_initialized", agents=len(agent_map))
