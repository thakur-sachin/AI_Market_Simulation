"""LLM client factory with routing logic.

Supports three routes:
  - CLAUDE  — Anthropic API (English / SEC A-B urban + persona gen + analysis)
  - SARVAM  — Sarvam-105B OpenAI-compatible (Hindi / regional / rural)
  - LOCAL   — Any OpenAI-compatible local server (Ollama, llama.cpp, LM Studio)

When `LOCAL_LLM_ENABLED=true` (or `use_local_llm()` is called), ALL routes
collapse to LOCAL — useful for feasibility testing without API costs.
"""
from __future__ import annotations

from enum import Enum

import anthropic
from openai import AsyncOpenAI

from launchlens.config import get_settings


class LLMRoute(str, Enum):
    CLAUDE = "claude"
    SARVAM = "sarvam"
    LOCAL = "local"


CLAUDE_MODEL = "claude-sonnet-4-6"
SARVAM_MODEL = "sarvam-105b"


# ── Local-mode override ───────────────────────────────────────────────────────
# Set via env (LOCAL_LLM_ENABLED) or at runtime via use_local_llm().
_LOCAL_OVERRIDE: bool = False


def use_local_llm(enabled: bool = True) -> None:
    """Override all LLM routing to use the local model. Idempotent."""
    global _LOCAL_OVERRIDE
    _LOCAL_OVERRIDE = enabled


def _local_active() -> bool:
    return _LOCAL_OVERRIDE or get_settings().local_llm_enabled


# ── Routing ───────────────────────────────────────────────────────────────────

def route_for_agent(language: str, isec_tier: str) -> LLMRoute:
    """Select model based on agent language + socioeconomic tier."""
    if _local_active():
        return LLMRoute.LOCAL
    english_only = language.lower() in ("english",)
    premium_sec = isec_tier.upper() in ("A1", "A2", "A3", "B1", "B2")
    if english_only and premium_sec:
        return LLMRoute.CLAUDE
    return LLMRoute.SARVAM


# ── Clients ───────────────────────────────────────────────────────────────────

def get_anthropic() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=get_settings().anthropic_api_key)


def get_sarvam() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(api_key=s.sarvam_api_key, base_url=s.sarvam_base_url)


def get_local() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(api_key=s.local_llm_api_key, base_url=s.local_llm_url)


# ── Unified completion ────────────────────────────────────────────────────────

async def complete(
    route: LLMRoute,
    system: str,
    user: str,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    json_mode: bool = False,
) -> str:
    """Unified completion call; returns assistant text."""
    # Honor runtime override regardless of caller's choice
    if _local_active():
        route = LLMRoute.LOCAL

    if route == LLMRoute.CLAUDE:
        client = get_anthropic()
        if json_mode:
            system = system + "\n\nRespond with valid JSON only."
        msg = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text

    if route == LLMRoute.LOCAL:
        client = get_local()
        model = get_settings().local_llm_model
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    # SARVAM
    client = get_sarvam()
    response_format = {"type": "json_object"} if json_mode else {"type": "text"}
    resp = await client.chat.completions.create(
        model=SARVAM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def effective_max_concurrent() -> int:
    """Return the appropriate concurrency limit for the active backend."""
    s = get_settings()
    return s.local_llm_max_concurrent if _local_active() else s.llm_max_concurrent
