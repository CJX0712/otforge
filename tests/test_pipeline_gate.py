"""The pre-registered gate, end to end."""

from __future__ import annotations

import pytest

from otforge.core.config import Config
from otforge.core.errors import ConfigError
from otforge.data.synthesizer import make_benchmark_suite
from otforge.pipeline.pipeline import _evaluate_gate, benchmark, run_one


def test_benchmark_structure():
    cfg = Config(n_source=12, n_target=12, seeds=(0, 1), anneal_steps=4, final_iters=300)
    out = benchmark(cfg, methods=["independent", "otfuse"])
    assert set(out) == set(make_benchmark_suite())
    for preset, by_method in out.items():
        assert set(by_method) == {"independent", "otfuse"}
        for method, rs in by_method.items():
            assert len(rs) == 2
            assert all(r.preset == preset for r in rs)
            assert all(r.method == method for r in rs)


def test_gate_passes_on_every_dataset():
    cfg = Config(n_source=24, n_target=24, seeds=(0, 1), anneal_steps=8, final_iters=1500)
    results = benchmark(cfg)
    gate = _evaluate_gate(results, cfg)
    for preset, info in gate.per_dataset.items():
        assert info["passed"], f"{preset} failed: {info['checks']}"
    assert gate.passed is True
    assert gate.aggregate["max_gap_mean"] <= cfg.max_gap
    assert gate.aggregate["min_reduction_mean"] >= cfg.min_reduction


def test_gate_flags_an_unrealistically_tight_budget_as_failing():
    cfg = Config(n_source=16, n_target=16, seeds=(0,), anneal_steps=3, final_iters=200, max_gap=1e-12)
    results = benchmark(cfg, methods=["independent", "sinkhorn", "otfuse"])
    gate = _evaluate_gate(results, cfg)
    assert gate.passed is False
    assert any(not info["checks"]["gap_within_budget"] for info in gate.per_dataset.values())


def test_gate_flags_a_too_weak_reduction_requirement():
    cfg = Config(n_source=16, n_target=16, seeds=(0,), anneal_steps=4, final_iters=400, min_reduction=0.999)
    results = benchmark(cfg)
    gate = _evaluate_gate(results, cfg)
    assert gate.passed is False
    assert all(not info["checks"]["reduction"] for info in gate.per_dataset.values())


def test_run_one_sets_the_seed_on_the_result():
    cfg = Config(n_source=12, n_target=12, anneal_steps=3, final_iters=200)
    res = run_one("point_cloud_2d", cfg, 9, "otfuse")
    assert res.seed == 9


def test_benchmark_rejects_unknown_method():
    cfg = Config(n_source=8, n_target=8)
    with pytest.raises(ValueError, match="unknown method"):
        benchmark(cfg, methods=["nope"])


def test_config_validation_rejects_bad_values():
    with pytest.raises(ConfigError):
        Config(eps_final=2.0, eps_init=1.0).validate()
    with pytest.raises(ConfigError):
        Config(exact_backend="quantum").validate()
    with pytest.raises(ConfigError):
        Config(n_source=1).validate()


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("OTFORGE_N_SOURCE", "18")
    monkeypatch.setenv("OTFORGE_SEEDS", "3,4")
    cfg = Config.from_env()
    assert cfg.n_source == 18
    assert cfg.seeds == (3, 4)
