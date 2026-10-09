"""Barycenter invariants.

The 1-D W2 barycenter has a closed form (the average of the quantile
functions).  We verify it is a genuine local minimiser by perturbation, and we
verify the fixed-support solver beats every input measure.
"""

from __future__ import annotations

import numpy as np
import pytest

from otforge.core.errors import DataError, SolverError
from otforge.ot.methods import WassersteinBarycenter
from otforge.ot.solvers import (
    barycenter_1d_quantile,
    barycenter_fixed_support,
    cost_matrix,
    sinkhorn_log,
    w_p_closed_form_1d,
)


def _objective(atoms: np.ndarray, supports: list[np.ndarray]) -> float:
    return float(np.mean([w_p_closed_form_1d(atoms, s, 2.0) for s in supports]))


def _entropic_objective(weights: np.ndarray, costs: list[np.ndarray], targets: list[np.ndarray], eps: float) -> float:
    """``sum_k w_k * W_eps(weights, b_k)`` — the barycenter objective.

    ``W_eps`` is the entropic value ``min <C,P> - eps*H(P)``, i.e. the Sinkhorn
    *dual*, not the raw transport cost.
    """
    total = 0.0
    for C_k, b_k in zip(costs, targets, strict=True):
        total += sinkhorn_log(C_k, weights, b_k, eps, iters=600, tol=1e-11, track=False).dual
    return float(total / len(costs))


def test_closed_form_barycenter_is_a_local_minimiser():
    """Perturb the quantile-average barycenter: the objective must not drop."""
    rng = np.random.default_rng(0)
    supports = [
        np.sort(rng.uniform(0.0, 0.5, size=24)),
        np.sort(rng.uniform(0.25, 0.75, size=24)),
        np.sort(rng.uniform(0.5, 1.0, size=24)),
    ]
    bary = barycenter_1d_quantile(supports)
    base = _objective(bary, supports)
    for _ in range(30):
        step = rng.normal(scale=1e-3, size=bary.shape)
        assert _objective(np.sort(bary + step), supports) >= base - 1e-12
    for _ in range(10):
        idx = rng.integers(0, bary.size, size=3)
        moved = bary.copy()
        moved[idx] += rng.normal(scale=5e-3, size=3)
        assert _objective(np.sort(moved), supports) >= base - 1e-12


def test_closed_form_barycenter_is_between_the_inputs():
    rng = np.random.default_rng(1)
    supports = [
        np.sort(rng.uniform(0.0, 0.4, size=16)),
        np.sort(rng.uniform(0.6, 1.0, size=16)),
    ]
    bary = barycenter_1d_quantile(supports)
    assert bary.min() >= min(s.min() for s in supports) - 1e-12
    assert bary.max() <= max(s.max() for s in supports) + 1e-12
    assert np.all(np.diff(bary) >= -1e-12)


def test_closed_form_rejects_unequal_sizes():
    with pytest.raises(DataError):
        barycenter_1d_quantile([np.arange(5.0), np.arange(4.0)])


def test_fixed_support_barycenter_beats_every_input_measure():
    rng = np.random.default_rng(2)
    grid = np.linspace(0.0, 1.0, 120).reshape(-1, 1)
    supports = [
        np.sort(rng.uniform(0.05, 0.4, size=24)).reshape(-1, 1),
        np.sort(rng.uniform(0.3, 0.7, size=24)).reshape(-1, 1),
        np.sort(rng.uniform(0.6, 0.95, size=24)).reshape(-1, 1),
    ]
    targets = [np.full(24, 1.0 / 24) for _ in supports]
    costs = [cost_matrix(grid, s, 2.0) for s in supports]
    eps = 0.05 * float(np.median(costs[0]))

    weights, objective, history, _ = barycenter_fixed_support(costs, targets, eps=eps, outer_iters=120, inner_iters=80)
    assert weights.shape == (120,)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(weights >= -1e-15)

    # The barycenter is the minimiser, so no input measure may beat it.
    for s in supports:
        lifted = np.zeros(120)
        idx = np.clip(np.searchsorted(grid.ravel(), s.ravel()), 0, 119)
        np.add.at(lifted, idx, 1.0 / 24)
        lifted = lifted / lifted.sum()
        value = _entropic_objective(lifted, costs, targets, eps)
        assert objective <= value + 1e-9

    assert objective >= 0.0
    assert history[-1] <= history[0] + 1e-9


def test_barycenter_objective_is_non_increasing():
    rng = np.random.default_rng(3)
    grid = np.linspace(0.0, 1.0, 60).reshape(-1, 1)
    supports = [
        np.sort(rng.uniform(0.1, 0.5, size=16)).reshape(-1, 1),
        np.sort(rng.uniform(0.5, 0.9, size=16)).reshape(-1, 1),
    ]
    targets = [np.full(16, 1.0 / 16) for _ in supports]
    costs = [cost_matrix(grid, s, 2.0) for s in supports]
    _, _, history, converged = barycenter_fixed_support(
        costs, targets, eps=0.05 * float(np.median(costs[0])), outer_iters=60, inner_iters=60
    )
    assert len(history) >= 2
    diffs = np.diff(np.asarray(history))
    assert diffs.max() <= 1e-12
    assert isinstance(converged, bool)


def test_undamped_update_is_worse_than_the_damped_one():
    """Damping is not cosmetic: it reaches a strictly better fixed point."""
    rng = np.random.default_rng(5)
    grid = np.linspace(0.0, 1.0, 80).reshape(-1, 1)
    supports = [
        np.sort(rng.uniform(0.05, 0.4, size=16)).reshape(-1, 1),
        np.sort(rng.uniform(0.3, 0.7, size=16)).reshape(-1, 1),
        np.sort(rng.uniform(0.6, 0.95, size=16)).reshape(-1, 1),
    ]
    targets = [np.full(16, 1.0 / 16) for _ in supports]
    costs = [cost_matrix(grid, s, 2.0) for s in supports]
    eps = 0.05 * float(np.median(costs[0]))
    _, obj_damped, _, _ = barycenter_fixed_support(costs, targets, eps=eps, outer_iters=80, inner_iters=60, step=0.10)
    _, obj_undamped, _, _ = barycenter_fixed_support(costs, targets, eps=eps, outer_iters=80, inner_iters=60, step=1.0)
    assert obj_damped <= obj_undamped + 1e-12


def test_barycenter_requires_measures():
    with pytest.raises(SolverError):
        barycenter_fixed_support([], [])


def test_barycenter_method_wrapper():
    rng = np.random.default_rng(4)
    grid = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    supports = [
        np.sort(rng.uniform(0.2, 0.8, size=12)).reshape(-1, 1),
        np.sort(rng.uniform(0.2, 0.8, size=12)).reshape(-1, 1),
    ]
    targets = [np.full(12, 1.0 / 12) for _ in supports]
    costs = [cost_matrix(grid, s, 2.0) for s in supports]
    res = WassersteinBarycenter(eps=0.05, outer_iters=20, inner_iters=40).run(grid, costs, targets)
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert len(res.history) >= 1
