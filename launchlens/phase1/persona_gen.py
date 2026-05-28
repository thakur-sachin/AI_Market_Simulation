"""
Persona generation engine — Phase 1b.
Stratified sampling from DistrictProfile → DemographicVector → biography via LLM.
"""
from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import structlog
from jinja2 import Environment, FileSystemLoader

from launchlens.config import get_settings
from launchlens.llm import LLMRoute, complete, effective_max_concurrent, route_for_agent
from launchlens.phase1.schemas import (
    AgentPersona,
    DemographicVector,
    DistrictProfile,
    ISECTier,
    OccupationCategory,
    TechAdoptionArchetype,
)

log = structlog.get_logger()
_cfg = get_settings()

_TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"
_jinja = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)

# ── Name banks ────────────────────────────────────────────────────────────────

_NAMES_MALE = [
    "Rahul","Amit","Vikram","Suresh","Rajesh","Arun","Pradeep","Sanjay",
    "Deepak","Manoj","Rohit","Kiran","Ajay","Vijay","Ramesh","Anand",
    "Mohan","Santosh","Girish","Dinesh","Naveen","Harish","Sunil","Ashok",
]
_NAMES_FEMALE = [
    "Priya","Sunita","Kavita","Meena","Pooja","Anita","Radha","Geeta",
    "Nisha","Rekha","Sonal","Divya","Anjali","Seema","Neha","Aarti",
    "Shweta","Komal","Vandana","Mamta","Ritu","Usha","Lata","Pushpa",
]
_NAMES_BY_LANG: dict[str, tuple[list[str], list[str]]] = {
    "tamil": (["Karthik","Senthil","Vijay","Arjun","Murugan"],
               ["Kavitha","Priya","Lakshmi","Meenakshi","Ananya"]),
    "telugu": (["Venkat","Srinivas","Ravi","Suresh","Krishna"],
                ["Sirisha","Padma","Lalitha","Swapna","Anusha"]),
    "bengali": (["Sourav","Arnab","Debashis","Ayan","Saurav"],
                 ["Piya","Rupa","Mitu","Swati","Debarati"]),
    "marathi": (["Sachin","Vijay","Nitin","Ganesh","Suresh"],
                 ["Sneha","Vaishali","Manasi","Aparna","Shubhangi"]),
    "kannada": (["Suresh","Ravi","Mahesh","Kiran","Lokesh"],
                 ["Savitha","Deepa","Rekha","Nandini","Pallavi"]),
    "malayalam": (["Arun","Sreejith","Nidhin","Vishnu","Ajith"],
                   ["Sreeja","Divya","Reshma","Anuja","Sindhu"]),
    "gujarati": (["Nirav","Hardik","Bhavesh","Dhruv","Jay"],
                  ["Hetal","Pooja","Dimple","Foram","Ekta"]),
    "punjabi": (["Harpreet","Gurpreet","Manpreet","Jaspreet","Ramandeep"],
                 ["Simran","Navneet","Karanpreet","Mandeep","Jasleen"]),
}


def _pick_name(sex: str, language: str) -> str:
    lang = language.lower()
    males, females = _NAMES_BY_LANG.get(lang, (_NAMES_MALE, _NAMES_FEMALE))
    return random.choice(males if sex == "male" else females)


# ── Education + occupation mapping ───────────────────────────────────────────

_ISEC_EDUCATION: dict[ISECTier, str] = {
    "A1": "MBA / Post-Graduate from premier institute",
    "A2": "Graduate from reputed college",
    "A3": "Graduate from local college",
    "B1": "Diploma / Some college",
    "B2": "12th pass (HSC/Intermediate)",
    "C1": "10th pass (SSC/Matriculation)",
    "C2": "8th–9th grade education",
    "D1": "5th–7th grade education",
    "D2": "Below 5th grade",
    "E1": "Literate, no formal schooling",
    "E2": "Partially literate",
    "E3": "Illiterate",
}

_ISEC_INCOME_RANGE: dict[ISECTier, tuple[int, int]] = {
    "A1": (150000, 500000), "A2": (80000, 150000), "A3": (50000, 80000),
    "B1": (30000, 50000),  "B2": (20000, 30000),
    "C1": (15000, 20000),  "C2": (10000, 15000),
    "D1": (7000, 10000),   "D2": (5000, 7000),
    "E1": (3500, 5000),    "E2": (2500, 3500),   "E3": (1500, 2500),
}

_OCC_DESCRIPTIONS: dict[OccupationCategory, str] = {
    "agriculture": "farmer / agricultural labourer",
    "manufacturing": "factory or workshop worker",
    "trade_retail": "shopkeeper / trader",
    "services_formal": "salaried professional in a company",
    "services_informal": "daily-wage service worker",
    "student": "student",
    "homemaker": "homemaker managing household",
    "professional": "doctor / engineer / teacher / lawyer",
    "self_employed": "self-employed / small business owner",
}

_CITY_TIERS: dict[str, str] = {
    "metro": "Metro city (Tier-1)",
    "tier2": "Tier-2 city",
    "tier3": "Tier-3 city / town",
    "rural": "Village / rural area",
}

_ADOPTION_ARCHETYPE_MAP: dict[ISECTier, list[TechAdoptionArchetype]] = {
    "A1": ["innovator", "early_adopter"],
    "A2": ["innovator", "early_adopter"],
    "A3": ["early_adopter", "early_majority"],
    "B1": ["early_majority"],
    "B2": ["early_majority", "late_majority"],
    "C1": ["late_majority"],
    "C2": ["late_majority", "laggard"],
    "D1": ["laggard"], "D2": ["laggard"],
    "E1": ["laggard"], "E2": ["laggard"], "E3": ["laggard"],
}


def _city_tier(urban: bool, isec: ISECTier) -> str:
    if not urban:
        return _CITY_TIERS["rural"]
    upper = isec in ("A1","A2","A3","B1")
    mid = isec in ("B2","C1","C2")
    return _CITY_TIERS["metro"] if upper else (_CITY_TIERS["tier2"] if mid else _CITY_TIERS["tier3"])


# ── Stratified sampler ────────────────────────────────────────────────────────

def _weighted_choice(distribution: dict, rng: random.Random) -> str:
    keys = list(distribution.keys())
    weights = [distribution[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def sample_demographic_vectors(
    profile: DistrictProfile,
    n: int,
    seed: int | None = None,
) -> list[DemographicVector]:
    """Stratified sample respecting all marginal distributions in DistrictProfile."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    vectors: list[DemographicVector] = []

    age_dist = profile.age_distribution.model_dump()
    age_buckets = list(age_dist.keys())
    age_weights = [age_dist[k] for k in age_buckets]

    _BUCKET_RANGES = {
        "bucket_0_4": (0, 4), "bucket_5_14": (5, 14), "bucket_15_24": (15, 24),
        "bucket_25_34": (25, 34), "bucket_35_44": (35, 44), "bucket_45_54": (45, 54),
        "bucket_55_64": (55, 64), "bucket_65_plus": (65, 80),
    }
    # Only sample working-age adults (15+) as consumer agents
    adult_buckets = [b for b in age_buckets if b != "bucket_0_4" and b != "bucket_5_14"]
    adult_weights_raw = [age_dist[b] for b in adult_buckets]
    adult_total = sum(adult_weights_raw)
    adult_weights = [w / adult_total for w in adult_weights_raw]

    for _ in range(n):
        bucket = rng.choices(adult_buckets, weights=adult_weights, k=1)[0]
        lo, hi = _BUCKET_RANGES[bucket]
        age = rng.randint(max(lo, 15), hi)

        sex = "female" if rng.random() < (profile.sex_ratio / (1000 + profile.sex_ratio)) else "male"
        urban = rng.random() < profile.urban_share
        isec: ISECTier = _weighted_choice(profile.isec_distribution, rng)  # type: ignore[assignment]
        language = _weighted_choice(profile.language_distribution, rng)

        lo_inc, hi_inc = _ISEC_INCOME_RANGE[isec]
        # Add ±15% stochastic variation within band
        base = rng.randint(lo_inc, hi_inc)
        income = int(base * (1 + rng.uniform(-0.15, 0.15)))

        smartphone = rng.random() < profile.smartphone_penetration
        upi = rng.random() < profile.upi_adoption if smartphone else False

        occ_pool: list[OccupationCategory]
        if age < 22:
            occ_pool = ["student"]
        elif not urban and isec in ("D1","D2","E1","E2","E3"):
            occ_pool = ["agriculture", "services_informal", "manufacturing"]
        elif isec in ("A1","A2","A3"):
            occ_pool = ["professional", "services_formal", "self_employed"]
        else:
            occ_pool = ["services_formal","services_informal","trade_retail","self_employed","homemaker"]
        occupation: OccupationCategory = rng.choice(occ_pool)  # type: ignore[assignment]

        archetypes = _ADOPTION_ARCHETYPE_MAP.get(isec, ["late_majority"])
        tech_adoption: TechAdoptionArchetype = rng.choice(archetypes)  # type: ignore[assignment]

        vectors.append(DemographicVector(
            age=age,
            sex=sex,
            urban=urban,
            isec_tier=isec,
            primary_language=language,
            occupation=occupation,
            monthly_hh_income=income,
            tech_adoption=tech_adoption,
            smartphone_owner=smartphone,
            upi_user=upi,
            district_id=profile.district_id,
            district_name=profile.district_name,
            state_name=profile.state_name,
        ))
    return vectors


# ── Bio generation ────────────────────────────────────────────────────────────

_SECONDARY_LANG: dict[str, str] = {
    "hindi": "english", "tamil": "english", "telugu": "english",
    "bengali": "hindi", "marathi": "hindi", "kannada": "english",
    "gujarati": "hindi", "punjabi": "hindi",
}


def _render_template(vec: DemographicVector) -> str:
    tmpl = _jinja.get_template("persona_bio.j2")
    return tmpl.render(
        name=_pick_name(vec.sex, vec.primary_language),
        age=vec.age,
        sex=vec.sex,
        district_name=vec.district_name,
        state_name=vec.state_name,
        urban=vec.urban,
        city_tier=_city_tier(vec.urban, vec.isec_tier),
        occupation_desc=_OCC_DESCRIPTIONS[vec.occupation],
        monthly_hh_income=vec.monthly_hh_income,
        education_level=_ISEC_EDUCATION[vec.isec_tier],
        primary_language=vec.primary_language,
        secondary_language=_SECONDARY_LANG.get(vec.primary_language.lower()),
        family_desc="joint family household" if not vec.urban else "nuclear family",
        smartphone_owner=vec.smartphone_owner,
        upi_user=vec.upi_user,
        tech_adoption=vec.tech_adoption,
        isec_tier=vec.isec_tier,
    )


async def _generate_bio(vec: DemographicVector, semaphore: asyncio.Semaphore) -> str:
    route = route_for_agent(vec.primary_language, vec.isec_tier)
    prompt = _render_template(vec)
    async with semaphore:
        return await complete(
            route=route,
            system="You are a skilled market research ethnographer writing consumer personas for India.",
            user=prompt,
            temperature=0.9,
            max_tokens=512,
        )


async def generate_personas(
    vectors: list[DemographicVector],
    max_concurrent: int | None = None,
    engine_override: str | None = None,
) -> list[AgentPersona]:
    max_concurrent = max_concurrent or effective_max_concurrent(engine_override=engine_override)
    sem = asyncio.Semaphore(max_concurrent)
    tasks = [_generate_bio(v, sem) for v in vectors]
    bios = await asyncio.gather(*tasks)

    personas = []
    for vec, bio in zip(vectors, bios):
        route = route_for_agent(vec.primary_language, vec.isec_tier)
        personas.append(AgentPersona(
            demographic=vec,
            biography=bio,
            llm_route=route.value,
        ))
    log.info("personas_generated", count=len(personas))
    return personas


# ── Diversity validation ─────────────────────────────────────────────────────

class DiversityCheckFailure(RuntimeError):
    """Raised when a generated population deviates beyond ``threshold`` on any marginal."""


def enforce_population_diversity(
    personas: Sequence[AgentPersona],
    profile: DistrictProfile,
    threshold: float = 0.05,
) -> None:
    """Hard gate: raise ``DiversityCheckFailure`` if any marginal deviates beyond ``threshold``."""
    flags = validate_population_diversity(personas, profile, threshold=threshold)
    if flags:
        joined = "; ".join(f"{dim}: {issues}" for dim, issues in flags.items())
        raise DiversityCheckFailure(
            f"Population marginals deviate beyond {threshold:.0%}: {joined}"
        )


def validate_population_diversity(
    personas: Sequence[AgentPersona],
    profile: DistrictProfile,
    threshold: float = 0.05,
) -> dict[str, list[str]]:
    """
    Compare simulated population marginals against DistrictProfile.
    Returns dict of dimension → list of flagged deviations (empty = pass).
    """
    n = len(personas)
    flags: dict[str, list[str]] = {}

    def _check(dim: str, simulated: dict, target: dict) -> None:
        issues = []
        for key, target_share in target.items():
            sim_share = simulated.get(str(key), 0.0)
            deviation = abs(sim_share - target_share)
            if deviation > threshold:
                issues.append(f"{key}: sim={sim_share:.3f} target={target_share:.3f} Δ={deviation:.3f}")
        if issues:
            flags[dim] = issues

    # Urban/rural
    urban_sim = sum(1 for p in personas if p.demographic.urban) / n
    urban_dev = abs(urban_sim - profile.urban_share)
    if urban_dev > threshold:
        flags["urban_rural"] = [f"urban: sim={urban_sim:.3f} target={profile.urban_share:.3f}"]

    # ISEC
    isec_sim: dict[str, float] = {}
    for p in personas:
        t = p.demographic.isec_tier
        isec_sim[t] = isec_sim.get(t, 0) + 1 / n
    _check("isec", isec_sim, profile.isec_distribution)  # type: ignore[arg-type]

    # Language (top 3 only)
    lang_sim: dict[str, float] = {}
    for p in personas:
        l = p.demographic.primary_language
        lang_sim[l] = lang_sim.get(l, 0) + 1 / n
    top_langs = dict(sorted(profile.language_distribution.items(), key=lambda x: -x[1])[:3])
    _check("language", lang_sim, top_langs)

    return flags
