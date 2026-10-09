"""Zero-hard-dependency behaviour.

SciPy is Tier-0 only: when it is missing the pure-NumPy Jonker-Volgenant
assignment solver must still deliver the *exact* Kantorovich optimum.
"""

from __future__ import annotations

import builtins
import sys

import numpy as np
import pytest

from otforge.data.synthesizer import make_problem
from otforge.ot.solvers import available_scipy, cost_matrix, emd_exact, emd_exact_backend_used


def test_backend_reporting():
    assert emd_exact_backend_used("numpy") == "numpy"
    assert emd_exact_backend_used("scipy") == "scipy"
    assert emd_exact_backend_used("auto") == ("scipy" if available_scipy() else "numpy")


def test_exact_emd_works_without_scipy(monkeypatch):
    """Block every scipy import and confirm the NumPy path is still exact."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("scipy blocked by test")
        return real_import(name, *args, **kwargs)

    prob = make_problem("point_cloud_2d", 16, 16, seed=0)
    C = cost_matrix(prob.X, prob.Y, 2.0)
    reference, _ = emd_exact(C, prob.a, prob.b, backend="numpy")

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "scipy", raising=False)
    monkeypatch.delitem(sys.modules, "scipy.optimize", raising=False)
    try:
        assert available_scipy() is False
        cost, plan = emd_exact(C, prob.a, prob.b, backend="auto")
    finally:
        monkeypatch.undo()

    assert cost == pytest.approx(reference, abs=1e-12)
    assert plan.shape == C.shape
    assert np.all(plan >= 0.0)
    assert plan.sum() == pytest.approx(1.0, abs=1e-12)


def test_requesting_scipy_without_scipy_raises(monkeypatch):
    from otforge.core.errors import SolverError

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("scipy blocked by test")
        return real_import(name, *args, **kwargs)

    prob = make_problem("point_cloud_2d", 8, 8, seed=1)
    C = cost_matrix(prob.X, prob.Y, 2.0)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(SolverError, match="SciPy is unavailable"):
        emd_exact(C, prob.a, prob.b, backend="scipy")


def test_numpy_backend_matches_1d_closed_form():
    """Even offline, the exact solver reproduces the analytic answer."""
    from otforge.ot.solvers import w_p_closed_form_1d

    prob = make_problem("gaussian_mix_1d", 20, 20, seed=2)
    C = cost_matrix(prob.X, prob.Y, 2.0)
    cost, _ = emd_exact(C, prob.a, prob.b, backend="numpy")
    closed = w_p_closed_form_1d(prob.X.ravel(), prob.Y.ravel(), 2.0)
    assert abs(cost - closed) < 1e-10
