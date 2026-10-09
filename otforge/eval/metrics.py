"""Evaluation metrics for optimal transport.

Every metric here is *scale free* and every one has a theorem behind it:

* ``relative_gap``   — a feasible coupling can never beat the LP optimum, so
  the gap must be >= 0; small means the solver found the optimum.
* ``marginal_violation`` — 0 means the plan really is a transport plan.
* ``dual_ascent_violation`` — 0 means the Sinkhorn updates really are exact
  block-coordinate ascent on the concave entropic dual.
* ``cost_reduction`` — the whole point of transport: beating the product
  coupling that ignores geometry.
"""

from __future__ import annotations

import numpy as np

from otforge.core.errors import EvalError
from otforge.core.types import OtResult


def relative_gap(cost: float, exact: float) -> float:
    """``(cost - exact) / exact``.  Must be ``>= 0`` for a feasible plan."""
    if exact <= 0.0:
        raise EvalError("exact optimum must be positive to form a relative gap")
    return float((cost - exact) / exact)


def cost_reduction(cost: float, baseline: float) -> float:
    """Fractional saving over the independent (product) coupling."""
    if baseline <= 0.0:
        raise EvalError("baseline cost must be positive")
    return float((baseline - cost) / baseline)


def wasserstein(cost: float, p: float = 2.0) -> float:
    """Convert ``<C, P>`` (with ``C = d**p``) to the ``W_p`` distance."""
    if cost < 0.0:
        raise EvalError("transport cost must be non-negative")
    return float(cost ** (1.0 / p))


def aggregate_across_seeds(values: list[float]) -> dict:
    """Mean and standard deviation of a metric over seeds."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise EvalError("cannot aggregate an empty list")
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


def summarise(results: list[OtResult]) -> dict:
    """Aggregate a list of :class:`OtResult` into mean/std statistics."""
    if not results:
        raise EvalError("cannot summarise an empty result list")
    return {
        "gap": aggregate_across_seeds([r.gap for r in results]),
        "cost": aggregate_across_seeds([r.cost for r in results]),
        "exact": aggregate_across_seeds([r.exact for r in results]),
        "marginal_violation": aggregate_across_seeds([r.marginal_violation for r in results]),
        "dual_violation": aggregate_across_seeds([r.dual_violation for r in results]),
        "reduction": aggregate_across_seeds([r.reduction for r in results]),
        "runtime_s": aggregate_across_seeds([r.runtime_s for r in results]),
    }
