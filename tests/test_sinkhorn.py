"""Invariants of the entropic solver: feasibility, monotone dual ascent, bias."""

from __future__ import annotations

import numpy as np
import pytest

from otforge.core.errors import SolverError
from otforge.data.synthesizer import make_problem
from otforge.ot.solvers import (
    annealed_sinkhorn,
    cost_matrix,
    dual_ascent_violation,
    emd_exact,
    marginal_violation,
    sinkhorn_divergence,
    sinkhorn_log,
)


def _normalised_problem(preset: str, n: int, seed: int):
    prob = make_problem(preset, n, n, seed=seed)
    C = cost_matrix(prob.X, prob.Y, 2.0)
    scale = max(float(np.median(C)), 1e-12)
    return prob, C / scale, scale


def test_sinkhorn_satisfies_marginals():
    prob, Cn, _ = _normalised_problem("point_cloud_2d", 24, 0)
    st = sinkhorn_log(Cn, prob.a, prob.b, eps=0.05, iters=2000, tol=1e-12)
    assert st.marginal_violation < 1e-10
    assert np.all(st.plan >= 0.0)
    assert st.plan.sum() == pytest.approx(1.0, abs=1e-10)


def test_dual_objective_is_monotonically_non_decreasing():
    """Sinkhorn = exact block coordinate ascent on a concave dual."""
    prob, Cn, _ = _normalised_problem("domain_shift_2d", 24, 1)
    st = sinkhorn_log(Cn, prob.a, prob.b, eps=0.05, iters=400, tol=0.0)
    assert len(st.dual_history) >= 100
    assert dual_ascent_violation(st.dual_history) < 1e-12
    assert st.dual_history[-1] > st.dual_history[0]


def test_dual_ascent_violation_detects_a_decrease():
    assert dual_ascent_violation([1.0, 2.0, 3.0]) == 0.0
    assert dual_ascent_violation([1.0, 0.5]) == pytest.approx(0.5)
    assert dual_ascent_violation([1.0]) == 0.0


def test_sinkhorn_cost_is_a_feasible_upper_bound_of_the_optimum():
    """A feasible coupling can never be cheaper than the LP optimum."""
    for preset in ("gaussian_mix_1d", "point_cloud_2d", "domain_shift_2d"):
        prob, Cn, scale = _normalised_problem(preset, 24, 0)
        exact, _ = emd_exact(cost_matrix(prob.X, prob.Y, 2.0), prob.a, prob.b)
        st = sinkhorn_log(Cn, prob.a, prob.b, eps=0.05, iters=3000, tol=1e-12)
        cost = st.cost * scale
        assert cost >= exact - 1e-9, f"{preset}: {cost} < {exact}"


def test_entropic_bias_shrinks_as_eps_cools():
    prob, Cn, scale = _normalised_problem("point_cloud_2d", 24, 2)
    exact, _ = emd_exact(cost_matrix(prob.X, prob.Y, 2.0), prob.a, prob.b)
    gaps = []
    for eps in (0.2, 0.05, 0.01):
        st = sinkhorn_log(Cn, prob.a, prob.b, eps=eps, iters=4000, tol=1e-12)
        gaps.append((st.cost * scale - exact) / exact)
    assert all(g >= -1e-9 for g in gaps)
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[-1] < 0.05


def test_annealed_sinkhorn_converges_better_than_a_cold_start():
    """Same cold epsilon, same budget: annealing buys a feasible plan.

    A cold start at eps=3e-3 leaves the iterates outside the transport
    polytope (so its cost silently falls *below* the true optimum), whereas
    the annealed schedule with warm-started potentials lands on it.
    """
    prob, Cn, scale = _normalised_problem("gaussian_mix_1d", 32, 0)
    eps_final = 3e-3
    cold = sinkhorn_log(Cn, prob.a, prob.b, eps=eps_final, iters=2000, tol=1e-12)
    annealed = annealed_sinkhorn(
        Cn,
        prob.a,
        prob.b,
        eps_init=1.0,
        eps_final=eps_final,
        steps=20,
        iters_per_step=300,
        final_iters=2000,
        tol=1e-12,
    )
    assert annealed.marginal_violation <= cold.marginal_violation + 1e-12
    # Both end at the same epsilon, so their optima agree to a few 1e-3.
    assert abs(annealed.cost - cold.cost) / max(cold.cost, 1e-12) < 5e-3


def test_annealing_dual_invariant_measured_per_stage():
    prob, Cn, _ = _normalised_problem("point_cloud_2d", 24, 3)
    st = annealed_sinkhorn(Cn, prob.a, prob.b, eps_init=1.0, eps_final=1e-2, steps=10, iters_per_step=150)
    assert st.dual_violation < 1e-9


def test_safeguard_only_spends_budget_when_needed():
    prob, Cn, _ = _normalised_problem("point_cloud_2d", 24, 4)
    without = annealed_sinkhorn(
        Cn,
        prob.a,
        prob.b,
        eps_init=1.0,
        eps_final=1e-2,
        steps=10,
        iters_per_step=200,
        safeguard_iters=0,
    )
    with_sg = annealed_sinkhorn(
        Cn,
        prob.a,
        prob.b,
        eps_init=1.0,
        eps_final=1e-2,
        steps=10,
        iters_per_step=200,
        safeguard_iters=5000,
    )
    assert with_sg.marginal_violation <= without.marginal_violation + 1e-15
    if without.converged:
        assert with_sg.iterations == without.iterations


def test_sinkhorn_divergence_is_nonnegative_and_zero_on_identical_measures():
    X = np.random.default_rng(8).random((16, 2))
    C_xx = cost_matrix(X, X, 2.0)
    C_xx = C_xx / max(float(np.median(C_xx)), 1e-12)
    a = np.full(16, 1.0 / 16)
    assert sinkhorn_divergence(C_xx, C_xx, C_xx, a, a, eps=0.05) == pytest.approx(0.0, abs=1e-9)

    prob, Cn, _ = _normalised_problem("domain_shift_2d", 16, 5)
    C_yy = cost_matrix(prob.Y, prob.Y, 2.0) / max(float(np.median(cost_matrix(prob.Y, prob.Y, 2.0))), 1e-12)
    div = sinkhorn_divergence(Cn, C_xx, C_yy, prob.a, prob.b, eps=0.05)
    assert div >= -1e-9


def test_sinkhorn_rejects_nonpositive_eps():
    prob, Cn, _ = _normalised_problem("point_cloud_2d", 8, 0)
    with pytest.raises(SolverError):
        sinkhorn_log(Cn, prob.a, prob.b, eps=0.0)


def test_marginal_violation_detects_a_bad_plan():
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    good = np.array([[0.5, 0.0], [0.0, 0.5]])
    bad = np.array([[0.9, 0.0], [0.0, 0.1]])
    assert marginal_violation(good, a, b) == pytest.approx(0.0, abs=1e-15)
    assert marginal_violation(bad, a, b) == pytest.approx(0.4, abs=1e-12)
