"""Bit-identical determinism."""

from __future__ import annotations

import numpy as np

from otforge.core.config import Config
from otforge.core.seed import set_all
from otforge.data.synthesizer import make_problem
from otforge.ot.methods import build_method
from otforge.ot.solvers import annealed_sinkhorn, cost_matrix, emd_exact


def _fingerprint(res) -> tuple:
    return (
        res.cost,
        res.exact,
        res.marginal_violation,
        res.dual_violation,
        res.reduction,
    )


def test_same_seed_two_runs_are_bit_identical():
    cfg = Config(n_source=16, n_target=16, anneal_steps=5, final_iters=400)
    for preset in ("gaussian_mix_1d", "point_cloud_2d", "domain_shift_2d"):
        a = build_method("otfuse", cfg).solve(make_problem(preset, 16, 16, seed=7))
        b = build_method("otfuse", cfg).solve(make_problem(preset, 16, 16, seed=7))
        assert _fingerprint(a) == _fingerprint(b)


def test_different_seeds_give_different_problems():
    x = make_problem("point_cloud_2d", 16, 16, seed=0).X
    y = make_problem("point_cloud_2d", 16, 16, seed=1).X
    assert not np.allclose(x, y)


def test_set_all_makes_global_rng_reproducible():
    set_all(42)
    first = np.random.default_rng(42).random(5)
    set_all(42)
    second = np.random.default_rng(42).random(5)
    assert np.array_equal(first, second)


def test_annealed_sinkhorn_is_deterministic():
    prob = make_problem("domain_shift_2d", 16, 16, seed=3)
    C = cost_matrix(prob.X, prob.Y, 2.0)
    C = C / max(float(np.median(C)), 1e-12)
    s1 = annealed_sinkhorn(C, prob.a, prob.b, 1.0, 1e-2, 6, 100)
    s2 = annealed_sinkhorn(C, prob.a, prob.b, 1.0, 1e-2, 6, 100)
    assert s1.cost == s2.cost
    assert np.array_equal(s1.plan, s2.plan)


def test_exact_emd_is_deterministic():
    prob = make_problem("point_cloud_2d", 16, 16, seed=4)
    C = cost_matrix(prob.X, prob.Y, 2.0)
    c1, p1 = emd_exact(C, prob.a, prob.b, backend="numpy")
    c2, p2 = emd_exact(C, prob.a, prob.b, backend="numpy")
    assert c1 == c2
    assert np.array_equal(p1, p2)
