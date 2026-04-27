"""Core data models for Phase 1."""
from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


# ── Demographic building blocks ──────────────────────────────────────────────

class AgeDistribution(BaseModel):
    """Population share per 5-year bucket."""
    bucket_0_4: float = 0.0
    bucket_5_14: float = 0.0
    bucket_15_24: float = 0.0
    bucket_25_34: float = 0.0
    bucket_35_44: float = 0.0
    bucket_45_54: float = 0.0
    bucket_55_64: float = 0.0
    bucket_65_plus: float = 0.0

    def validate_sums_to_one(self) -> bool:
        return abs(sum(self.model_dump().values()) - 1.0) < 0.02


ISECTier = Literal["A1","A2","A3","B1","B2","C1","C2","D1","D2","E1","E2","E3"]

OccupationCategory = Literal[
    "agriculture", "manufacturing", "trade_retail", "services_formal",
    "services_informal", "student", "homemaker", "professional", "self_employed",
]

TechAdoptionArchetype = Literal["innovator", "early_adopter", "early_majority", "late_majority", "laggard"]


# ── DistrictProfile ──────────────────────────────────────────────────────────

class DistrictProfile(BaseModel):
    """District-level demographic profile. Normalized to all major data sources."""

    district_id: str           # Census district code
    district_name: str
    state_name: str
    population: int

    # Demographics
    age_distribution: AgeDistribution
    sex_ratio: float           # females per 1000 males
    urban_share: float         # 0–1
    literacy_rate: float       # 0–1

    # Language (mother tongue shares, sum ≈ 1)
    language_distribution: dict[str, float]

    # Socioeconomic
    isec_distribution: dict[ISECTier, float]  # must sum to 1
    median_monthly_hh_expenditure: int         # INR

    # Tech access
    smartphone_penetration: float    # 0–1 (derived from TRAI + NFHS)
    internet_penetration: float      # 0–1
    upi_adoption: float              # 0–1 (proxy: NPCI data)

    # Source metadata
    census_year: int = 2011
    nfhs_round: int = 5


# ── AgentPersona ─────────────────────────────────────────────────────────────

class DemographicVector(BaseModel):
    """Sampled demographic attributes for one agent before bio expansion."""
    age: int
    sex: Literal["male", "female"]
    urban: bool
    isec_tier: ISECTier
    primary_language: str
    occupation: OccupationCategory
    monthly_hh_income: int          # INR
    tech_adoption: TechAdoptionArchetype
    smartphone_owner: bool
    upi_user: bool
    district_id: str
    district_name: str
    state_name: str


class AgentPersona(BaseModel):
    """Fully instantiated agent with biography text ready for simulation."""
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    demographic: DemographicVector
    biography: str          # natural-language bio; injected into every LLM prompt
    llm_route: Literal["claude", "sarvam"]
    qa_passed: bool = False
    qa_failures: list[str] = Field(default_factory=list)
