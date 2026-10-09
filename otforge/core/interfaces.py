"""Protocol definitions for transport solvers and methods."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from otforge.core.types import OtProblem, OtResult


@runtime_checkable
class TransportMethod(Protocol):
    """Anything that turns an :class:`OtProblem` into an :class:`OtResult`."""

    name: str

    def solve(self, problem: OtProblem) -> OtResult:
        """Solve the transport problem and report diagnostics."""
        ...


@runtime_checkable
class PlanSolver(Protocol):
    """Anything that returns a coupling matrix ``P`` with shape ``(n, m)``."""

    def plan(self, C: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return a feasible coupling for the given cost and marginals."""
        ...
