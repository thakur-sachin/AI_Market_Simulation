"""
Persona QA module — Phase 1c.
5 sanity-check prompts per agent; flag for regeneration if ≥2 fail.
"""
from __future__ import annotations

import asyncio
import json
import re

import structlog

from launchlens.llm import LLMRoute, complete
from launchlens.phase1.schemas import AgentPersona

log = structlog.get_logger()

# ── QA prompts ────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are evaluating whether a consumer persona behaves consistently with their profile. "
    "You will be given a biography and a question. Reply ONLY with valid JSON: "
    '{"answer": "<short answer>", "consistent": true|false, "reason": "<one sentence>"}'
)


def _qa_prompts(persona: AgentPersona, product_price: int = 2000) -> list[tuple[str, str]]:
    bio = persona.biography
    income = persona.demographic.monthly_hh_income
    archetype = persona.demographic.tech_adoption
    media_hint = "online (YouTube/Instagram)" if persona.demographic.smartphone_owner else "radio/TV"
    lang = persona.demographic.primary_language
    isec = persona.demographic.isec_tier
    # rough spend threshold: upper ISEC can spend up to 10% disposable on a single item
    affordable = product_price < income * 0.10

    return [
        # 1 — price sensitivity
        (
            bio,
            f"Given your income of ₹{income:,}/month, would you seriously consider buying "
            f"a product priced at ₹{product_price:,}? "
            "Is this consistent with someone of your financial profile? "
            f"Expected: {'yes if affordable' if affordable else 'no, too expensive for this tier'}",
        ),
        # 2 — media channel
        (
            bio,
            f"How would you first hear about and research a new product? "
            f"Is {media_hint} the primary channel consistent with your profile?",
        ),
        # 3 — recommendation propensity
        (
            bio,
            f"If you liked this product, would you tell others about it? "
            f"Expected propensity for a '{archetype}' archetype: "
            f"{'high' if archetype in ('innovator','early_adopter') else 'moderate to low'}.",
        ),
        # 4 — language consistency
        (
            bio,
            f"In what language would you naturally discuss this product with friends or family? "
            f"Expected: primarily {lang}. Does your biography support this?",
        ),
        # 5 — concern type
        (
            bio,
            f"What is your PRIMARY concern when evaluating a new product? "
            f"Expected for ISEC {isec}: "
            f"{'quality and brand trust' if isec in ('A1','A2','A3','B1','B2') else 'price and value for money'}.",
        ),
    ]


# ── Consistency judge ─────────────────────────────────────────────────────────

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


async def _run_check(
    bio: str,
    question: str,
    route: LLMRoute,
    semaphore: asyncio.Semaphore,
    check_name: str,
) -> tuple[str, bool]:
    async with semaphore:
        raw = await complete(
            route=route,
            system=_SYSTEM,
            user=f"BIOGRAPHY:\n{bio}\n\nQUESTION:\n{question}",
            temperature=0.2,
            max_tokens=150,
            json_mode=True,
        )
    match = _JSON_RE.search(raw)
    if not match:
        return check_name, False
    try:
        data = json.loads(match.group())
        return check_name, bool(data.get("consistent", False))
    except json.JSONDecodeError:
        return check_name, False


async def qa_persona(
    persona: AgentPersona,
    semaphore: asyncio.Semaphore,
    fail_threshold: int = 2,
    product_price: int = 2000,
) -> AgentPersona:
    route = LLMRoute(persona.llm_route)
    checks = _qa_prompts(persona, product_price)
    check_names = ["price_sensitivity", "media_channel", "recommendation", "language", "concern_type"]

    tasks = [
        _run_check(persona.biography, q, route, semaphore, name)
        for (bio, q), name in zip(checks, check_names)
    ]
    results = await asyncio.gather(*tasks)

    failures = [name for name, passed in results if not passed]
    persona.qa_passed = len(failures) < fail_threshold
    persona.qa_failures = failures
    return persona


async def run_qa_batch(
    personas: list[AgentPersona],
    max_concurrent: int = 30,
    fail_threshold: int = 2,
    product_price: int = 2000,
) -> tuple[list[AgentPersona], list[AgentPersona]]:
    """Returns (passed, failed) persona lists."""
    sem = asyncio.Semaphore(max_concurrent)
    results = await asyncio.gather(*[qa_persona(p, sem, fail_threshold, product_price) for p in personas])
    passed = [p for p in results if p.qa_passed]
    failed = [p for p in results if not p.qa_passed]
    log.info("persona_qa_complete", total=len(results), passed=len(passed), failed=len(failed))
    return passed, failed
