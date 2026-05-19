"""Provider selection + tracker tests (no real network calls)."""
from __future__ import annotations

import os

import pytest

from launchlens import llm as llm_module
from launchlens.llm import (
    CompletionResult,
    LLMRoute,
    MissingAPIKey,
    MockProvider,
    OllamaProvider,
    SarvamProvider,
    ClaudeProvider,
    LLMUsageTracker,
    get_usage_tracker,
    select_provider,
)


def test_mock_provider_returns_parseable_json():
    p = MockProvider()
    import asyncio
    res = asyncio.run(p.complete(
        system="s", user="u", temperature=0.5, max_tokens=100, json_mode=True,
    ))
    assert res.provider == "mock"
    assert res.cost_usd == 0.0
    import json
    obj = json.loads(res.text)
    assert obj["decision"] == "IGNORE"


def test_select_provider_explicit_mock():
    p = select_provider(LLMRoute.SARVAM, engine_override="mock")
    assert isinstance(p, MockProvider)


def test_select_provider_local_without_ollama_raises(monkeypatch):
    monkeypatch.setattr(llm_module, "_ollama_available", lambda _: False)
    llm_module.reset_provider_cache()
    with pytest.raises(llm_module.LocalModelUnavailable):
        select_provider(LLMRoute.SARVAM, engine_override="local")


def test_select_provider_sarvam_without_key_raises(monkeypatch):
    monkeypatch.setattr(llm_module._settings, "sarvam_api_key", None)
    with pytest.raises(MissingAPIKey):
        select_provider(LLMRoute.SARVAM, engine_override="sarvam")


def test_select_provider_claude_without_key_raises(monkeypatch):
    monkeypatch.setattr(llm_module._settings, "anthropic_api_key", None)
    with pytest.raises(MissingAPIKey):
        select_provider(LLMRoute.CLAUDE, engine_override="claude")


def test_select_provider_auto_falls_back_to_mock(monkeypatch):
    """When no Ollama and no remote keys, auto must return MockProvider."""
    monkeypatch.setattr(llm_module, "_ollama_available", lambda _: False)
    monkeypatch.setattr(llm_module._settings, "sarvam_api_key", None)
    monkeypatch.setattr(llm_module._settings, "anthropic_api_key", None)
    monkeypatch.delenv("LAUNCHLENS_ENGINE", raising=False)
    llm_module.reset_provider_cache()
    p = select_provider(LLMRoute.SARVAM, engine_override="auto")
    assert isinstance(p, MockProvider)


def test_usage_tracker_aggregates_cost():
    t = LLMUsageTracker()
    t.record(CompletionResult(
        text="x", provider="claude", model="m",
        input_tokens=1000, output_tokens=500, cost_usd=0.012,
    ))
    t.record(CompletionResult(
        text="y", provider="claude", model="m",
        input_tokens=2000, output_tokens=1000, cost_usd=0.030,
    ))
    t.record(CompletionResult(
        text="z", provider="ollama", model="qwen", cost_usd=0.0,
    ))
    assert t.total_calls() == 3
    assert t.total_cost_usd() == pytest.approx(0.042, abs=1e-6)
    assert t.by_provider["claude"].calls == 2
    assert t.by_provider["ollama"].calls == 1


def test_estimate_cost_mock_is_zero():
    cost, name = llm_module.estimate_cost(
        n_agents=100, n_timesteps=10, engine_override="mock",
    )
    assert cost == 0.0
    assert name == "mock"


def test_route_for_agent_routes_english_premium_to_claude():
    assert llm_module.route_for_agent("english", "A1") == LLMRoute.CLAUDE
    assert llm_module.route_for_agent("hindi", "A1") == LLMRoute.SARVAM
    assert llm_module.route_for_agent("english", "D2") == LLMRoute.SARVAM
