"""Prompt construction: ensure anti-positivity prior and JSON instruction are present."""
from __future__ import annotations

from launchlens.phase3.schemas import AgentMemory, MarketplaceFeed
from launchlens.phase4.prompts import build_decision_prompt


def _feed() -> MarketplaceFeed:
    return MarketplaceFeed(
        product_stimulus="Product: Foo @ ₹50",
        peer_reviews=[],
        peer_purchases=[],
    )


def test_system_prompt_contains_price_skepticism_prior():
    mem = AgentMemory(agent_id="a", biography="bio")
    system, _ = build_decision_prompt(mem, _feed(), "Foo", 50, "p1")
    s = system.lower()
    # Prior is now advisory + archetype-aware, not "default to X" prescriptive
    assert "skeptical" in s
    assert "archetype" in s
    # Explicitly instructs REJECT for clear mismatches (not just BUY/not-BUY)
    assert "reject" in s


def test_system_prompt_lists_enums_without_pipe_separator():
    """Pipe-separated enums confuse small models — they echo `'a | b'` as a value."""
    mem = AgentMemory(agent_id="a", biography="bio")
    system, _ = build_decision_prompt(mem, _feed(), "Foo", 50, "p1")
    # The enum lines must not list options separated by ``|``
    for line in system.splitlines():
        if "decision" in line.lower() and ("BUY" in line or "buy" in line):
            assert " | " not in line, f"pipe-separated enum found: {line!r}"


def test_system_prompt_requests_json_output():
    mem = AgentMemory(agent_id="a", biography="bio")
    system, _ = build_decision_prompt(mem, _feed(), "Foo", 50, "p1")
    assert "json" in system.lower()
    for field in (
        "internal_reasoning", "decision", "primary_reason",
        "would_discuss_with", "language_of_discussion",
    ):
        assert field in system


def test_user_prompt_embeds_product_and_price():
    mem = AgentMemory(agent_id="a", biography="bio")
    _, user = build_decision_prompt(mem, _feed(), "Foo", 50, "p1")
    assert "Foo" in user
    assert "₹50" in user


def test_user_prompt_includes_recent_episodic():
    mem = AgentMemory(agent_id="a", biography="bio")
    mem.add_event("t0: AWARE")
    mem.add_event("t1: RESEARCH")
    _, user = build_decision_prompt(mem, _feed(), "Foo", 50, "p1")
    assert "t0: AWARE" in user
    assert "t1: RESEARCH" in user
