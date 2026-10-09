"""Core dataclasses shared by solvers, methods, metrics and the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OtProblem:
    """A discrete optimal-transport problem.

    Attributes:
        X: Source support, shape ``(n, d)``.
        Y: Target support, shape ``(m, d)``.
        a: Source marginal, shape ``(n,)``, sums to 1.
        b: Target marginal, shape ``(m,)``, sums to 1.
        preset: Name of the generating preset.
        dim: Ambient dimension ``d``.
    """

    X: np.ndarray
    Y: np.ndarray
    a: np.ndarray
    b: np.ndarray
    preset: str
    dim: int

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.Y = np.asarray(self.Y, dtype=float)
        self.a = np.asarray(self.a, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        if self.X.ndim != 2 or self.Y.ndim != 2:
            raise ValueError("X and Y must be 2-D arrays")
        if self.X.shape[1] != self.Y.shape[1]:
            raise ValueError("X and Y must share the same ambient dimension")
        if self.a.shape[0] != self.X.shape[0]:
            raise ValueError("len(a) must equal n rows of X")
        if self.b.shape[0] != self.Y.shape[0]:
            raise ValueError("len(b) must equal m rows of Y")
        if np.any(self.a < 0) or np.any(self.b < 0):
            raise ValueError("marginals must be non-negative")
        self.a = self.a / self.a.sum()
        self.b = self.b / self.b.sum()


@dataclass
class OtResult:
    """Outcome of running one method on one problem.

    Attributes:
        method: Method name.
        preset: Dataset preset name.
        seed: RNG seed of the run.
        cost: Transport cost ``<C, P>`` of the returned plan.
        exact: Exact Kantorovich LP cost for the same problem.
        marginal_violation: ``max(||P1 - a||_inf, ||P^T 1 - b||_inf)``.
        dual_violation: Largest *decrease* of the entropic dual objective
            across Sinkhorn updates (must be ~0: coordinate ascent on a
            concave objective can never decrease it).
        runtime_s: Wall-clock seconds spent in the solver.
        reduction: Relative cost reduction versus the independent coupling
            baseline ``(cost_indep - cost) / cost_indep``.
        extra: Method-specific diagnostics.
    """

    method: str
    preset: str
    seed: int
    cost: float
    exact: float
    marginal_violation: float
    dual_violation: float
    runtime_s: float
    reduction: float
    extra: dict = field(default_factory=dict)

    @property
    def gap(self) -> float:
        """Relative excess of this cost over the exact optimum (>= 0)."""
        if self.exact <= 0.0:
            return 0.0
        return (self.cost - self.exact) / self.exact


@dataclass
class GateResult:
    """Verdict of the pre-registered quality gate."""

    passed: bool
    per_dataset: dict
    aggregate: dict


@dataclass
class BarycenterResult:
    """Result of a fixed-support Wasserstein barycenter computation."""

    weights: np.ndarray
    objective: float
    history: list
    converged: bool
