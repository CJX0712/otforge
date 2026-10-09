"""Shared pytest fixtures for otForge."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from otforge.core.config import Config  # noqa: E402
from otforge.data.synthesizer import make_problem  # noqa: E402


@pytest.fixture
def small_config() -> Config:
    """A fast configuration used by most tests."""
    return Config(n_source=24, n_target=24, seeds=(0, 1), anneal_steps=8, final_iters=800)


@pytest.fixture
def tiny_problem():
    """A tiny 1-D problem where the closed-form solution is available."""
    return make_problem("gaussian_mix_1d", 16, 16, seed=0)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
