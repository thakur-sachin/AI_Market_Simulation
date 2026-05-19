"""Core data models for Phase 1."""
from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ── Demographic building blocks ──────────────────────────────────────────────

class AgeDistribution(BaseModel):
    """Population share per Census age bucket."""
    bucket_0_4: float = 0.0
    bucket_5_14: float = 0.0
    bucket_15_24: float = 0.0
    bucket_25_34: float = 0.0
    bucket_35_44: float = 0.0
    bucket_45_54: float = 0.0
    bucket_55_64: float = 0.0
    bucket_65_plus: float = 0.0

    def validate_sums_to_one(self, tolerance: float = 0.02) -> bool:
        return abs(sum(self.model_dump().values()) - 1.0) < tolerance


ISECTier = Literal["A1","A2","A3","B1","B2","C1","C2","D1","D2","E1","E2","E3"]

OccupationCategory = Literal[
    "agriculture", "manufacturing", "trade_retail", "services_formal",
    "services_informal", "student", "homemaker", "professional", "self_employed",
]

TechAdoptionArchetype = Literal["innovator", "early_adopter", "early_majority", "late_majority", "laggard"]

DataSource = Literal["census", "nfhs", "trai", "nsso", "baseline", "manual", "fallback"]


# ── DistrictProfile ──────────────────────────────────────────────────────────

class DistrictProfile(BaseModel):
    """District-level demographic profile. Normalized across data sources."""

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
    smartphone_penetration: float    # 0–1
    internet_penetration: float      # 0–1
    upi_adoption: float              # 0–1

    # Source metadata
    census_year: int = 2011
    nfhs_round: int = 5

    # Per-field provenance. Keys correspond to public attribute names; values name
    # the data source that supplied the value. "fallback" means a baseline was used
    # because the upstream source was unavailable.
    provenance: dict[str, DataSource] = Field(default_factory=dict)

    # ── Validators ───────────────────────────────────────────────────────

    @field_validator("sex_ratio")
    @classmethod
    def _sex_ratio_bounds(cls, v: float) -> float:
        if not 600 <= v <= 1200:
            raise ValueError(f"sex_ratio {v} outside plausible range [600, 1200]")
        return v

    @field_validator("urban_share", "literacy_rate",
                     "smartphone_penetration", "internet_penetration", "upi_adoption")
    @classmethod
    def _zero_to_one(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"value {v} not in [0, 1]")
        return v

    @field_validator("language_distribution")
    @classmethod
    def _language_sums_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("language_distribution cannot be empty")
        total = sum(v.values())
        if abs(total - 1.0) > 0.05:
            raise ValueError(f"language_distribution sums to {total:.3f}, expected ≈ 1")
        return v

    @field_validator("isec_distribution")
    @classmethod
    def _isec_sums_to_one(cls, v: dict) -> dict:
        if not v:
            raise ValueError("isec_distribution cannot be empty")
        total = sum(v.values())
        if abs(total - 1.0) > 0.05:
            raise ValueError(f"isec_distribution sums to {total:.3f}, expected ≈ 1")
        return v


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
