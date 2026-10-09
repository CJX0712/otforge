"""Metric definitions and error handling."""

from __future__ import annotations

import numpy as np
import pytest

from otforge.core.errors import EvalError
from otforge.core.types import OtResult
from otforge.eval.metrics import (
    aggregate_across_seeds,
    cost_reduction,
    relative_gap,
    summarise,
    wasserstein,
)


def _result(cost: float, exact: float, **kw) -> OtResult:
    base = {
        "method": "otfuse",
        "preset": "p",
        "seed": 0,
        "marginal_violation": 1e-12,
        "dual_violation": 1e-16,
        "runtime_s": 0.1,
        "reduction": 0.5,
    }
    base.update(kw)
    return OtResult(cost=cost, exact=exact, **base)


def test_relative_gap():
    assert relative_gap(1.10, 1.00) == pytest.approx(0.10)
    assert relative_gap(1.00, 1.00) == pytest.approx(0.0)
    with pytest.raises(EvalError):
        relative_gap(1.0, 0.0)


def test_cost_reduction():
    assert cost_reduction(5.0, 10.0) == pytest.approx(0.5)
    assert cost_reduction(10.0, 10.0) == pytest.approx(0.0)
    with pytest.raises(EvalError):
        cost_reduction(1.0, 0.0)


def test_wasserstein_is_the_p_th_root():
    assert wasserstein(4.0, 2.0) == pytest.approx(2.0)
    assert wasserstein(8.0, 3.0) == pytest.approx(2.0)
    with pytest.raises(EvalError):
        wasserstein(-1.0)


def test_aggregate_across_seeds():
    stats = aggregate_across_seeds([1.0, 2.0, 3.0])
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["std"] == pytest.approx(np.std([1.0, 2.0, 3.0]))
    assert stats["n"] == 3
    with pytest.raises(EvalError):
        aggregate_across_seeds([])


def test_summarise_matches_result_property():
    results = [_result(1.02, 1.00), _result(1.04, 1.00)]
    stats = summarise(results)
    assert stats["gap"]["mean"] == pytest.approx(0.03)
    assert stats["cost"]["mean"] == pytest.approx(1.03)
    assert stats["exact"]["mean"] == pytest.approx(1.00)
    with pytest.raises(EvalError):
        summarise([])


def test_result_gap_property():
    r = _result(1.05, 1.00)
    assert r.gap == pytest.approx(0.05)
    zero = _result(1.0, 0.0)
    assert zero.gap == 0.0
