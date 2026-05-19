"""Global pytest fixtures and seeding.

Every test gets a deterministically-seeded random module and numpy generator
so that simulation outputs are reproducible across runs.
"""
from __future__ import annotations

import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_global_rng() -> None:
    """Reset all module-level RNGs before each test."""
    random.seed(42)
    np.random.seed(42)


@pytest.fixture
def rng() -> random.Random:
    """Per-test deterministic random.Random instance."""
    return random.Random(42)


@pytest.fixture
def np_rng() -> np.random.Generator:
    """Per-test deterministic numpy Generator."""
    return np.random.default_rng(42)
