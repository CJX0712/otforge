"""Configuration object with ``OTFORGE_*`` environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

from otforge.core.errors import ConfigError

_ENV_PREFIX = "OTFORGE_"


@dataclass
class Config:
    """Runtime configuration for otForge benchmarks.

    Attributes:
        n_source: Number of source atoms (default 64).
        n_target: Number of target atoms (default 64).
        cost_p: Minkowski exponent of the ground cost (1, 2, ...).
        eps_init: Initial entropic regularisation for eps-annealing (in units
            of the median ground cost, so it is scale free).
        eps_final: Final (coldest) entropic regularisation.
        anneal_steps: Number of geometric cooling stages.
        sinkhorn_iters: Max Sinkhorn iterations per intermediate stage.
        final_iters: Iteration budget for the coldest stage (it is the one that
            actually decides accuracy, so it gets a much larger budget).
        intermediate_tol: Loose marginal tolerance letting warm stages exit.
        sinkhorn_tol: Marginal-violation tolerance for the final stage.
        max_gap: Pre-registered gate: maximum allowed relative excess over the
            exact Kantorovich cost.
        min_reduction: Pre-registered gate: minimum relative cost reduction
            versus the independent-coupling baseline.
        max_marginal_violation: Pre-registered gate: max marginal violation.
        max_dual_violation: Pre-registered gate: max dual-objective decrease.
        barycenter_iters: Outer iterations of the Sinkhorn barycenter.
        barycenter_weights: Equal weights over the input measures.
        exact_backend: ``"auto"`` | ``"scipy"`` | ``"numpy"``.
        seeds: Seeds for the multi-seed benchmark.
    """

    n_source: int = 64
    n_target: int = 64
    cost_p: int = 2
    eps_init: float = 1.0
    eps_final: float = 1e-2
    anneal_steps: int = 20
    sinkhorn_iters: int = 400
    final_iters: int = 4000
    intermediate_tol: float = 1e-5
    sinkhorn_tol: float = 1e-10
    safeguard_iters: int = 20000
    max_gap: float = 0.02
    min_reduction: float = 0.30
    max_marginal_violation: float = 1e-5
    max_dual_violation: float = 1e-9
    barycenter_iters: int = 60
    barycenter_iters_inner: int = 40
    exact_backend: str = "auto"
    seeds: tuple = (0, 1, 2)

    @classmethod
    def from_env(cls, **overrides) -> Config:
        """Build a config, applying ``OTFORGE_*`` env vars then overrides."""
        values = {}
        for f in fields(cls):
            raw = os.environ.get(f"{_ENV_PREFIX}{f.name.upper()}")
            if raw is None:
                continue
            values[f.name] = _cast(f.type, raw)
        values.update(overrides)
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Raise :class:`ConfigError` when the config is inconsistent."""
        if self.n_source < 2 or self.n_target < 2:
            raise ConfigError("n_source and n_target must be >= 2")
        if self.cost_p < 1:
            raise ConfigError("cost_p must be >= 1")
        if not 0.0 < self.eps_final <= self.eps_init:
            raise ConfigError("require 0 < eps_final <= eps_init")
        if self.anneal_steps < 1:
            raise ConfigError("anneal_steps must be >= 1")
        if self.sinkhorn_iters < 1:
            raise ConfigError("sinkhorn_iters must be >= 1")
        if self.exact_backend not in {"auto", "scipy", "numpy"}:
            raise ConfigError("exact_backend must be one of auto/scipy/numpy")
        if not 0.0 <= self.max_gap <= 1.0:
            raise ConfigError("max_gap must lie in [0, 1]")


def _cast(type_str: str, raw: str):
    """Coerce an environment string into the declared dataclass field type."""
    if type_str in {"int", "float", "str"}:
        return {"int": int, "float": float, "str": str}[type_str](raw)
    if type_str == "tuple":
        return tuple(int(x) for x in raw.split(","))
    return raw
