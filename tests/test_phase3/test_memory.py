"""MemoryStore + AgentMemory tests — no LLM calls."""
import pytest

from launchlens.phase3.memory import InMemoryBackend, MemoryStore
from launchlens.phase3.schemas import AgentMemory, PeerSignal


@pytest.mark.asyncio
async def test_init_and_load_roundtrip():
    store = MemoryStore()
    mem = await store.init_agent("a1", "biography text")
    assert mem.agent_id == "a1"
    loaded = await store.get("a1")
    assert loaded is not None
    assert loaded.biography == "biography text"


@pytest.mark.asyncio
async def test_get_many_skips_missing():
    store = MemoryStore()
    await store.init_agent("a1", "bio1")
    await store.init_agent("a2", "bio2")
    result = await store.get_many(["a1", "a2", "missing"])
    assert set(result.keys()) == {"a1", "a2"}


@pytest.mark.asyncio
async def test_update_persists_changes():
    store = MemoryStore()
    await store.init_agent("a1", "bio")
    mem = await store.get("a1")
    mem.product_opinion["p1"] = "I like it"
    mem.current_decision["p1"] = "CONSIDER"
    await store.update(mem)

    reloaded = await store.get("a1")
    assert reloaded.product_opinion["p1"] == "I like it"
    assert reloaded.current_decision["p1"] == "CONSIDER"


def test_episodic_buffer_caps_at_ten():
    mem = AgentMemory(agent_id="a1", biography="bio")
    for i in range(15):
        mem.add_event(f"event_{i}")
    assert len(mem.episodic_buffer) == 10
    # Oldest events dropped first
    assert mem.episodic_buffer[0] == "event_5"
    assert mem.episodic_buffer[-1] == "event_14"


def test_pending_peer_signals_floor():
    mem = AgentMemory(agent_id="a1", biography="bio")
    mem.peer_signals = [
        PeerSignal(from_agent_id="x", decision="BUY", reason="r", salience=0.8),
        PeerSignal(from_agent_id="y", decision="BUY", reason="r", salience=0.04),
        PeerSignal(from_agent_id="z", decision="BUY", reason="r", salience=0.06),
    ]
    pending = mem.pending_peer_signals("any_product")
    assert {s.from_agent_id for s in pending} == {"x", "z"}
