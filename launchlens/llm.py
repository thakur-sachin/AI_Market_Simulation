"""Provider-agnostic LLM dispatch.

Layered so the same agent decision loop runs across:
  * Ollama (local, free, default on this hardware)
  * Sarvam-105B (remote, OpenAI-compatible) — placeholder until key arrives
  * Claude (remote, Anthropic SDK) — placeholder until key arrives
  * Mock (deterministic IGNORE responses for offline tests)

Selection order at runtime:

    explicit override > settings.launchlens_engine > "auto"

In ``auto`` mode we probe a reachable Ollama first, then fall back to whichever
remote provider has an API key, and finally to ``MockProvider``. The intent is
that the system is *always* runnable end-to-end without external network or
secrets — degrading gracefully.

Every successful completion is recorded in ``LLMUsageTracker`` (tokens + cost,
where cost is $0 for local). Results are cached by content hash via ``diskcache``
so repeated runs with identical prompts cost nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

import structlog

from launchlens.config import get_settings

log = structlog.get_logger()
_settings = get_settings()


# ── Routing hint (kept for backward compatibility with persona generation) ───

class LLMRoute(str, Enum):
    """Per-agent routing hint. Determines which model is preferred for an agent.

    Kept as ``(str, Enum)`` (not ``StrEnum``) because pydantic stores the value as a
    plain string in ``AgentPersona.llm_route`` and existing serialized data uses
    the bare string form.
    """
    CLAUDE = "claude"
    SARVAM = "sarvam"


def route_for_agent(language: str, isec_tier: str) -> LLMRoute:
    """Pick a routing hint based on agent language + socioeconomic tier."""
    english_only = language.lower() == "english"
    premium_sec = isec_tier.upper() in ("A1", "A2", "A3", "B1", "B2")
    if english_only and premium_sec:
        return LLMRoute.CLAUDE
    return LLMRoute.SARVAM


# ── Errors ───────────────────────────────────────────────────────────────────

class LLMError(RuntimeError):
    """Base class for LLM-layer failures."""


class MissingAPIKey(LLMError):
    """A remote provider was requested but its API key is not configured."""


class LocalModelUnavailable(LLMError):
    """Ollama is not running or the requested model is not pulled."""


# ── Result envelope ──────────────────────────────────────────────────────────

@dataclass
class CompletionResult:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False


# ── Provider protocol ────────────────────────────────────────────────────────

class LLMProvider(Protocol):
    name: str
    model: str

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> CompletionResult: ...


# ── Usage tracker (process-singleton) ────────────────────────────────────────

@dataclass
class _ProviderUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_hits: int = 0


@dataclass
class LLMUsageTracker:
    by_provider: dict[str, _ProviderUsage] = field(default_factory=dict)

    def record(self, result: CompletionResult) -> None:
        bucket = self.by_provider.setdefault(result.provider, _ProviderUsage())
        bucket.calls += 1
        if result.cached:
            bucket.cache_hits += 1
        bucket.input_tokens += result.input_tokens
        bucket.output_tokens += result.output_tokens
        bucket.cost_usd += result.cost_usd

    def total_cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.by_provider.values())

    def total_calls(self) -> int:
        return sum(u.calls for u in self.by_provider.values())

    def reset(self) -> None:
        self.by_provider.clear()

    def summary(self) -> dict[str, dict]:
        return {
            p: {
                "calls": u.calls,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cost_usd": round(u.cost_usd, 4),
                "cache_hits": u.cache_hits,
            }
            for p, u in self.by_provider.items()
        }


_TRACKER = LLMUsageTracker()


def get_usage_tracker() -> LLMUsageTracker:
    return _TRACKER


# ── Response cache (disk-backed, content-addressed) ──────────────────────────

_cache = None


def _get_cache():
    global _cache
    if _cache is not None:
        return _cache
    try:
        from diskcache import Cache

        cache_dir = Path(_settings.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache = Cache(
            str(cache_dir),
            size_limit=int(_settings.cache_size_limit_gb * 1024**3),
        )
    except ImportError:
        log.warning("diskcache_not_installed", note="response cache disabled")
        _cache = {}  # in-memory fallback
    return _cache


def _cache_key(provider: str, model: str, system: str, user: str,
               temperature: float, json_mode: bool) -> str:
    blob = json.dumps(
        {"p": provider, "m": model, "s": system, "u": user,
         "t": round(temperature, 3), "j": json_mode},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()


# ── Concrete providers ──────────────────────────────────────────────────────

class MockProvider:
    """Deterministic placeholder. Emits a parseable JSON IGNORE response.

    Used for offline tests, dry-run flows, and as the final auto-mode fallback
    when no real provider is reachable. Tests that need richer behaviour should
    inject their own ``LLMProvider`` rather than rely on this.
    """

    name = "mock"
    model = "mock"

    async def complete(self, *, system: str, user: str, temperature: float,
                       max_tokens: int, json_mode: bool) -> CompletionResult:
        text = json.dumps({
            "internal_reasoning": "Mock provider response — no real model invoked.",
            "decision": "IGNORE",
            "primary_reason": "Mock fallback; not engaged.",
            "would_discuss_with": "no_one",
            "language_of_discussion": "N/A",
        })
        return CompletionResult(
            text=text, provider=self.name, model=self.model,
            input_tokens=len(user) // 4, output_tokens=len(text) // 4,
            cost_usd=0.0,
        )


class OllamaProvider:
    """Local inference via Ollama's OpenAI-compatible HTTP endpoint.

    No API key required. Cost is recorded as $0 regardless of token volume.
    Probes ``GET {base}/api/tags`` on init and raises ``LocalModelUnavailable``
    if the server is unreachable or the requested model is not pulled.
    """

    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or _settings.ollama_default_model
        self.base_url = base_url or _settings.ollama_base_url

    @classmethod
    def probe(cls, base_url: str | None = None) -> list[str]:
        """Return the list of model tags Ollama has pulled, or raise."""
        import httpx

        url = (base_url or _settings.ollama_base_url).rstrip("/v1").rstrip("/") + "/api/tags"
        try:
            resp = httpx.get(url, timeout=2.0)
            resp.raise_for_status()
        except Exception as exc:
            raise LocalModelUnavailable(f"Ollama not reachable at {url}: {exc}") from exc
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    async def complete(self, *, system: str, user: str, temperature: float,
                       max_tokens: int, json_mode: bool) -> CompletionResult:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=self.base_url, api_key="ollama")  # placeholder key
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}
        try:
            resp = await client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:
            raise LocalModelUnavailable(
                f"Ollama call failed for model {self.model}: {exc}"
            ) from exc

        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        input_toks = getattr(usage, "prompt_tokens", 0) if usage else len(user) // 4
        output_toks = getattr(usage, "completion_tokens", 0) if usage else len(text) // 4
        return CompletionResult(
            text=text, provider=self.name, model=self.model,
            input_tokens=input_toks, output_tokens=output_toks,
            cost_usd=0.0,
        )


class SarvamProvider:
    """Remote Sarvam-105B (OpenAI-compatible). Activated when API key is set."""

    name = "sarvam"
    model = "sarvam-105b"

    def __init__(self) -> None:
        if not _settings.sarvam_api_key:
            raise MissingAPIKey(
                "SARVAM_API_KEY not configured. Set it in .env or environment "
                "to enable the Sarvam provider."
            )

    async def complete(self, *, system: str, user: str, temperature: float,
                       max_tokens: int, json_mode: bool) -> CompletionResult:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=_settings.sarvam_api_key,
            base_url=_settings.sarvam_base_url,
        )
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}
        resp = await client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        input_toks = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_toks = getattr(usage, "completion_tokens", 0) if usage else 0
        cost = (
            input_toks * _settings.sarvam_input_price_per_mtok / 1_000_000
            + output_toks * _settings.sarvam_output_price_per_mtok / 1_000_000
        )
        return CompletionResult(
            text=text, provider=self.name, model=self.model,
            input_tokens=input_toks, output_tokens=output_toks, cost_usd=cost,
        )


class ClaudeProvider:
    """Remote Claude (Anthropic Messages API). Activated when API key is set."""

    name = "claude"
    model = "claude-sonnet-4-6"

    def __init__(self) -> None:
        if not _settings.anthropic_api_key:
            raise MissingAPIKey(
                "ANTHROPIC_API_KEY not configured. Set it in .env or environment "
                "to enable the Claude provider."
            )

    async def complete(self, *, system: str, user: str, temperature: float,
                       max_tokens: int, json_mode: bool) -> CompletionResult:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=_settings.anthropic_api_key)
        sys_prompt = system
        if json_mode:
            sys_prompt = system + "\n\nRespond with a single JSON object and nothing else."
        msg = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=sys_prompt,
            messages=[{"role": "user", "content": user}],
        )
        text = msg.content[0].text if msg.content else ""
        in_toks = getattr(msg.usage, "input_tokens", 0) if hasattr(msg, "usage") else 0
        out_toks = getattr(msg.usage, "output_tokens", 0) if hasattr(msg, "usage") else 0
        cost = (
            in_toks * _settings.claude_input_price_per_mtok / 1_000_000
            + out_toks * _settings.claude_output_price_per_mtok / 1_000_000
        )
        return CompletionResult(
            text=text, provider=self.name, model=self.model,
            input_tokens=in_toks, output_tokens=out_toks, cost_usd=cost,
        )


# ── Selection ────────────────────────────────────────────────────────────────

_VALID_ENGINES = {"auto", "mock", "local", "sarvam", "claude"}


def _engine_setting() -> str:
    env = os.environ.get("LAUNCHLENS_ENGINE", "").lower().strip()
    if env in _VALID_ENGINES:
        return env
    cfg = (_settings.launchlens_engine or "auto").lower().strip()
    return cfg if cfg in _VALID_ENGINES else "auto"


def _model_for_route(route: LLMRoute) -> str:
    if route == LLMRoute.SARVAM:
        return _settings.ollama_indic_model
    return _settings.ollama_default_model


_AVAILABILITY_CACHE: dict[str, bool] = {}


def _ollama_available(model: str) -> bool:
    """Cheap probe (cached per process) of Ollama + model presence."""
    key = f"ollama:{model}"
    if key in _AVAILABILITY_CACHE:
        return _AVAILABILITY_CACHE[key]
    try:
        tags = OllamaProvider.probe()
    except LocalModelUnavailable:
        _AVAILABILITY_CACHE[key] = False
        return False
    # Ollama tag may be exact match or prefix (e.g. "qwen2.5:3b-instruct-q4_K_M").
    present = any(t == model or t.split(":")[0] == model.split(":")[0] for t in tags)
    _AVAILABILITY_CACHE[key] = present
    if not present:
        log.warning("ollama_model_missing", requested=model, available=tags)
    return present


def reset_provider_cache() -> None:
    """Clear cached availability probes (use after `ollama pull`)."""
    _AVAILABILITY_CACHE.clear()


def select_provider(
    route: LLMRoute = LLMRoute.SARVAM,
    *,
    engine_override: str | None = None,
) -> LLMProvider:
    """Resolve the active provider for one agent call."""
    requested = (engine_override or _engine_setting()).lower()
    if requested not in _VALID_ENGINES:
        raise LLMError(f"unknown engine {requested!r}; valid: {_VALID_ENGINES}")

    if requested == "mock":
        return MockProvider()

    if requested == "local":
        model = _model_for_route(route)
        if not _ollama_available(model):
            raise LocalModelUnavailable(
                f"Ollama or model {model!r} unavailable. "
                f"Run `ollama serve` and `ollama pull {model}`."
            )
        return OllamaProvider(model=model)

    if requested == "sarvam":
        return SarvamProvider()  # raises MissingAPIKey if no key

    if requested == "claude":
        return ClaudeProvider()  # raises MissingAPIKey if no key

    # AUTO: local first, then remote, then mock.
    model = _model_for_route(route)
    if _ollama_available(model):
        return OllamaProvider(model=model)
    if route == LLMRoute.SARVAM and _settings.sarvam_api_key:
        return SarvamProvider()
    if route == LLMRoute.CLAUDE and _settings.anthropic_api_key:
        return ClaudeProvider()
    if _settings.sarvam_api_key:
        return SarvamProvider()
    if _settings.anthropic_api_key:
        return ClaudeProvider()
    log.warning("no_real_provider_available", note="falling back to MockProvider")
    return MockProvider()


# ── Public entry point ───────────────────────────────────────────────────────

async def complete(
    route: LLMRoute,
    system: str,
    user: str,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    json_mode: bool = False,
    engine_override: str | None = None,
    use_cache: bool = True,
) -> str:
    """Run one LLM call, recording usage and (optionally) caching the result."""
    provider = select_provider(route, engine_override=engine_override)

    cache = _get_cache() if use_cache else None
    key = _cache_key(provider.name, provider.model, system, user, temperature, json_mode)
    if cache is not None and key in cache:
        cached_text = cache[key]
        result = CompletionResult(
            text=cached_text, provider=provider.name, model=provider.model,
            input_tokens=0, output_tokens=0, cost_usd=0.0, cached=True,
        )
        _TRACKER.record(result)
        return cached_text

    result = await provider.complete(
        system=system, user=user, temperature=temperature,
        max_tokens=max_tokens, json_mode=json_mode,
    )
    _TRACKER.record(result)
    if cache is not None:
        import contextlib
        # diskcache transient errors must not break the run
        with contextlib.suppress(Exception):
            cache[key] = result.text
    return result.text


# ── Cost estimation (used by phase4/loop preflight) ──────────────────────────

def estimate_cost(
    n_agents: int,
    n_timesteps: int,
    avg_input_tokens: int = 800,
    avg_output_tokens: int = 200,
    engine_override: str | None = None,
) -> tuple[float, str]:
    """Rough USD estimate for a sim run, plus the resolved provider name."""
    provider = select_provider(LLMRoute.SARVAM, engine_override=engine_override)
    pname = provider.name
    calls = n_agents * n_timesteps
    if pname == "ollama" or pname == "mock":
        return 0.0, pname
    if pname == "sarvam":
        cost = (
            calls * avg_input_tokens * _settings.sarvam_input_price_per_mtok / 1_000_000
            + calls * avg_output_tokens * _settings.sarvam_output_price_per_mtok / 1_000_000
        )
        return round(cost, 4), pname
    if pname == "claude":
        cost = (
            calls * avg_input_tokens * _settings.claude_input_price_per_mtok / 1_000_000
            + calls * avg_output_tokens * _settings.claude_output_price_per_mtok / 1_000_000
        )
        return round(cost, 4), pname
    return 0.0, pname


# Backward-compat aliases (some older code imports these directly).
__all__ = [
    "LLMRoute",
    "LLMError",
    "MissingAPIKey",
    "LocalModelUnavailable",
    "CompletionResult",
    "LLMProvider",
    "MockProvider",
    "OllamaProvider",
    "SarvamProvider",
    "ClaudeProvider",
    "LLMUsageTracker",
    "get_usage_tracker",
    "select_provider",
    "reset_provider_cache",
    "route_for_agent",
    "complete",
    "estimate_cost",
]
