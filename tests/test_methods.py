"""Method-level behaviour: baselines, the flagship, offline fallback."""

from __future__ import annotations

import numpy as np
import pytest

from otforge.core.config import Config
from otforge.core.errors import MethodError
from otforge.data.synthesizer import make_problem
from otforge.ot.methods import (
    METHODS,
    IndependentMethod,
    OtFuseMethod,
    SinkhornMethod,
    build_method,
    independent_cost,
)
from otforge.ot.solvers import cost_matrix, emd_exact


def test_independent_coupling_is_feasible_and_never_better_than_optimum():
    prob = make_problem("domain_shift_2d", 24, 24, seed=0)
    C = cost_matrix(prob.X, prob.Y, 2.0)
    res = IndependentMethod(Config()).solve(prob)
    assert res.marginal_violation < 1e-12
    assert res.reduction == pytest.approx(0.0, abs=1e-12)
    assert res.cost == pytest.approx(independent_cost(C, prob.a, prob.b), rel=1e-12)
    assert res.cost >= res.exact - 1e-12


def test_independent_cost_equals_outer_product_cost():
    prob = make_problem("point_cloud_2d", 16, 16, seed=1)
    C = cost_matrix(prob.X, prob.Y, 2.0)
    P = np.outer(prob.a, prob.b)
    assert independent_cost(C, prob.a, prob.b) == pytest.approx(float(np.sum(P * C)), rel=1e-12)


def test_flagship_beats_baseline_by_a_wide_margin():
    cfg = Config(n_source=24, n_target=24, anneal_steps=8, final_iters=1500)
    for preset in ("gaussian_mix_1d", "point_cloud_2d", "domain_shift_2d"):
        prob = make_problem(preset, cfg.n_source, cfg.n_target, seed=0)
        res = OtFuseMethod(cfg).solve(prob)
        assert res.reduction > 0.30
        assert res.gap >= -1e-9
        assert res.gap <= cfg.max_gap


def test_sinkhorn_method_runs_and_reports_diagnostics():
    cfg = Config(n_source=16, n_target=16, anneal_steps=4, final_iters=500)
    prob = make_problem("point_cloud_2d", 16, 16, seed=2)
    res = SinkhornMethod(cfg).solve(prob)
    assert res.method == "sinkhorn"
    assert res.extra["eps"] == cfg.eps_final
    assert res.dual_violation < 1e-9


def test_flagship_reports_divergence_and_wasserstein():
    cfg = Config(n_source=16, n_target=16, anneal_steps=4, final_iters=500)
    prob = make_problem("gaussian_mix_1d", 16, 16, seed=3)
    res = OtFuseMethod(cfg).solve(prob)
    assert "divergence" in res.extra
    assert "wasserstein" in res.extra
    assert res.extra["divergence"] >= -1e-9
    assert res.extra["wasserstein"] > 0.0


def test_build_method_registry():
    assert set(METHODS) == {"independent", "sinkhorn", "otfuse"}
    for name in METHODS:
        assert build_method(name).name == name
    with pytest.raises(MethodError):
        build_method("does-not-exist")


def test_exact_backend_selection_is_respected():
    cfg = Config(n_source=12, n_target=12, exact_backend="numpy", anneal_steps=4, final_iters=300)
    prob = make_problem("point_cloud_2d", 12, 12, seed=4)
    res = OtFuseMethod(cfg).solve(prob)
    C = cost_matrix(prob.X, prob.Y, 2.0)
    expected, _ = emd_exact(C, prob.a, prob.b, backend="numpy")
    assert res.exact == pytest.approx(expected, rel=1e-12)


def test_result_gap_property_matches_definition():
    cfg = Config(n_source=12, n_target=12, anneal_steps=4, final_iters=300)
    prob = make_problem("domain_shift_2d", 12, 12, seed=5)
    res = OtFuseMethod(cfg).solve(prob)
    assert res.gap == pytest.approx((res.cost - res.exact) / res.exact, rel=1e-12)
