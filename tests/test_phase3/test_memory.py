"""MemoryStore + InMemoryBackend behaviour tests."""
from __future__ import annotations

import pytest

from launchlens.phase3.memory import InMemoryBackend, MemoryStore
from launchlens.phase3.schemas import AgentMemory, PeerSignal


pytestmark = pytest.mark.asyncio


async def test_init_agent_round_trips():
    store = MemoryStore()
    mem = await store.init_agent("agent_1", "A 30-year-old engineer.")
    assert mem.agent_id == "agent_1"
    assert mem.biography == "A 30-year-old engineer."

    loaded = await store.get("agent_1")
    assert loaded is not None
    assert loaded.biography == "A 30-year-old engineer."


async def test_init_all_loads_many():
    store = MemoryStore()
    bios = {f"a{i}": f"bio {i}" for i in range(5)}
    await store.init_all(bios)
    loaded = await store.get_many(list(bios.keys()))
    assert set(loaded.keys()) == set(bios.keys())
    for aid, mem in loaded.items():
        assert mem.biography == bios[aid]


async def test_update_persists_episodic_buffer():
    store = MemoryStore()
    await store.init_agent("a1", "bio")
    mem = await store.get("a1")
    assert mem is not None
    mem.add_event("t0: AWARE")
    mem.add_event("t1: RESEARCH")
    await store.update(mem)

    loaded = await store.get("a1")
    assert loaded is not None
    assert loaded.episodic_buffer == ["t0: AWARE", "t1: RESEARCH"]


async def test_episodic_buffer_capped_at_ten():
    mem = AgentMemory(agent_id="a", biography="b")
    for i in range(15):
        mem.add_event(f"event {i}")
    assert len(mem.episodic_buffer) == 10
    assert mem.episodic_buffer[0] == "event 5"   # oldest 5 dropped
    assert mem.episodic_buffer[-1] == "event 14"


async def test_pending_peer_signals_drops_below_floor():
    mem = AgentMemory(agent_id="a", biography="b")
    mem.peer_signals = [
        PeerSignal(from_agent_id="x", decision="BUY", reason="r", salience=0.5),
        PeerSignal(from_agent_id="y", decision="BUY", reason="r", salience=0.04),
        PeerSignal(from_agent_id="z", decision="BUY", reason="r", salience=0.10),
    ]
    pending = mem.pending_peer_signals("p")
    sources = {s.from_agent_id for s in pending}
    assert "x" in sources
    assert "z" in sources
    assert "y" not in sources  # below 0.05


async def test_missing_agent_returns_none():
    store = MemoryStore()
    assert await store.get("never_initialized") is None


async def test_in_memory_backend_isolated_per_instance():
    """Two stores should not share state."""
    s1 = MemoryStore(InMemoryBackend())
    s2 = MemoryStore(InMemoryBackend())
    await s1.init_agent("a", "bio1")
    assert await s2.get("a") is None
