"""LLM client factory with routing logic per implementation plan Section 8."""
from enum import Enum
from typing import Any

import anthropic
from openai import AsyncOpenAI

from launchlens.config import get_settings

_settings = get_settings()


class LLMRoute(str, Enum):
    CLAUDE = "claude"       # English/SEC-A/B urban + persona gen + post-hoc analysis
    SARVAM = "sarvam"       # Hindi/Hinglish/regional/rural


def route_for_agent(language: str, isec_tier: str) -> LLMRoute:
    """Select model based on agent language + socioeconomic tier."""
    english_only = language.lower() in ("english",)
    premium_sec = isec_tier.upper() in ("A1", "A2", "A3", "B1", "B2")
    if english_only and premium_sec:
        return LLMRoute.CLAUDE
    return LLMRoute.SARVAM


def get_anthropic() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=_settings.anthropic_api_key)


def get_sarvam() -> AsyncOpenAI:
    # Sarvam-105B exposes an OpenAI-compatible endpoint
    return AsyncOpenAI(
        api_key=_settings.sarvam_api_key,
        base_url=_settings.sarvam_base_url,
    )


CLAUDE_MODEL = "claude-sonnet-4-6"
SARVAM_MODEL = "sarvam-105b"


async def complete(
    route: LLMRoute,
    system: str,
    user: str,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    json_mode: bool = False,
) -> str:
    """Unified completion call; returns assistant text."""
    if route == LLMRoute.CLAUDE:
        client = get_anthropic()
        kwargs: dict[str, Any] = {}
        if json_mode:
            # instruct model via system prompt; Claude doesn't have a native json_mode param yet
            system = system + "\n\nRespond with valid JSON only."
        msg = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        return msg.content[0].text

    else:  # SARVAM
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
