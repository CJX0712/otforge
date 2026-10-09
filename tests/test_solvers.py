"""Ground-truth cross-validation of the solvers.

The whole point of otForge is that three independent routes must agree:
pure-NumPy Hungarian, SciPy assignment, and the analytic 1-D closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

from otforge.core.errors import SolverError
from otforge.data.synthesizer import make_problem
from otforge.ot.solvers import (
    _hungarian,
    available_scipy,
    cost_matrix,
    emd_exact,
    logsumexp,
    marginal_violation,
    w_p_closed_form_1d,
)


def test_cost_matrix_is_symmetric_and_zero_on_diagonal():
    X = np.random.default_rng(0).random((7, 3))
    C = cost_matrix(X, X, 2.0)
    assert np.allclose(C, C.T, atol=1e-12)
    assert np.allclose(np.diag(C), 0.0, atol=1e-12)
    assert np.all(C >= 0.0)


def test_cost_matrix_p1_is_euclidean():
    rng = np.random.default_rng(1)
    X = rng.random((5, 2))
    Y = rng.random((6, 2))
    C = cost_matrix(X, Y, 1.0)
    manual = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
    assert np.allclose(C, manual, atol=1e-12)


def test_hungarian_matches_scipy_on_random_costs():
    if not available_scipy():  # pragma: no cover - SciPy present in CI
        pytest.skip("SciPy unavailable")
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(2)
    for n, m in [(5, 5), (12, 12), (9, 15), (15, 9)]:
        C = rng.random((n, m))
        ours = _hungarian(C)
        rows, cols = linear_sum_assignment(C)
        sel = ours >= 0
        assert sel.sum() == min(n, m)
        assert abs(C[np.arange(n)[sel], ours[sel]].sum() - C[rows, cols].sum()) < 1e-9


def test_hungarian_optimal_against_brute_force():
    """On a 5x5 problem, enumerate all 120 permutations."""
    import itertools

    rng = np.random.default_rng(3)
    C = rng.random((5, 5))
    ours = _hungarian(C)
    best = min(C[np.arange(5), p].sum() for p in itertools.permutations(range(5)))
    assert abs(C[np.arange(5), ours].sum() - best) < 1e-12


def test_exact_emd_agrees_with_closed_form_1d():
    """The single most important invariant: LP == analytic W_p."""
    for seed in (0, 1, 2, 3):
        prob = make_problem("gaussian_mix_1d", 20, 20, seed=seed)
        C = cost_matrix(prob.X, prob.Y, 2.0)
        cost, plan = emd_exact(C, prob.a, prob.b, backend="numpy")
        closed = w_p_closed_form_1d(prob.X.ravel(), prob.Y.ravel(), 2.0)
        assert abs(cost - closed) < 1e-10
        assert marginal_violation(plan, prob.a, prob.b) < 1e-12


def test_exact_emd_backends_agree():
    if not available_scipy():  # pragma: no cover
        pytest.skip("SciPy unavailable")
    prob = make_problem("point_cloud_2d", 24, 24, seed=1)
    C = cost_matrix(prob.X, prob.Y, 2.0)
    c_np, _ = emd_exact(C, prob.a, prob.b, backend="numpy")
    c_sp, _ = emd_exact(C, prob.a, prob.b, backend="scipy")
    assert c_np == pytest.approx(c_sp, abs=1e-12)


def test_exact_emd_beats_every_random_coupling():
    """The LP optimum is a minimum: no feasible plan may be cheaper."""
    rng = np.random.default_rng(4)
    prob = make_problem("domain_shift_2d", 16, 16, seed=2)
    C = cost_matrix(prob.X, prob.Y, 2.0)
    optimum, _ = emd_exact(C, prob.a, prob.b)
    for _ in range(20):
        P = rng.random((16, 16))
        P = P / P.sum()
        assert np.sum(P * C) >= optimum - 1e-12


def test_exact_emd_rejects_non_rational_marginals():
    C = np.random.default_rng(5).random((4, 4))
    a = np.array([0.1, 0.2, 0.3, 0.4])
    b = np.array([0.25, 0.25, 0.25, 0.25])
    a = a / 1.0000001  # deliberately non-rational on a small grid
    a = a / a.sum()
    b = b / b.sum()
    with pytest.raises(SolverError):
        emd_exact(C, a, b, backend="numpy")


def test_logsumexp_matches_naive_and_is_stable():
    rng = np.random.default_rng(6)
    A = rng.random((4, 5))
    assert np.allclose(logsumexp(A, axis=1), np.log(np.sum(np.exp(A), axis=1)), atol=1e-12)
    B = np.full((3, 3), -800.0)
    out = logsumexp(B, axis=1)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, -800.0 + np.log(3.0), atol=1e-9)


def test_w_p_closed_form_is_zero_for_identical_samples():
    x = np.random.default_rng(7).random(30)
    assert w_p_closed_form_1d(x, x, 2.0) == pytest.approx(0.0, abs=1e-15)


def test_w_p_closed_form_rejects_unequal_sizes():
    from otforge.core.errors import DataError

    with pytest.raises(DataError):
        w_p_closed_form_1d(np.arange(5.0), np.arange(4.0), 2.0)
