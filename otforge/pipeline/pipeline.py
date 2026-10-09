"""Benchmark pipeline and the pre-registered quality gate."""

from __future__ import annotations

import numpy as np

from otforge.core.config import Config
from otforge.core.seed import set_all
from otforge.core.types import GateResult, OtResult
from otforge.data.synthesizer import make_benchmark_suite, make_problem
from otforge.eval.metrics import summarise
from otforge.ot.methods import METHODS, build_method


def run_one(preset: str, config: Config, seed: int, method: str) -> OtResult:
    """Run one method on one problem.

    Args:
        preset: Dataset preset name.
        config: Benchmark configuration.
        seed: RNG seed (determines the problem instance).
        method: Registered method name.

    Returns:
        An :class:`OtResult`.
    """
    set_all(seed)
    problem = make_problem(preset, config.n_source, config.n_target, seed)
    result = build_method(method, config).solve(problem)
    result.seed = int(seed)
    return result


def benchmark(
    config: Config,
    seeds: list[int] | tuple[int, ...] | None = None,
    presets: list[str] | None = None,
    methods: list[str] | None = None,
) -> dict:
    """Run every method on every preset for every seed.

    Returns:
        ``{preset: {method: [OtResult, ...]}}``.
    """
    config.validate()
    seeds = list(config.seeds) if seeds is None else list(seeds)
    presets = make_benchmark_suite() if presets is None else list(presets)
    methods = ["independent", "sinkhorn", "otfuse"] if methods is None else list(methods)
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}")

    out: dict[str, dict[str, list[OtResult]]] = {}
    for preset in presets:
        out[preset] = {}
        for method in methods:
            out[preset][method] = [run_one(preset, config, s, method) for s in seeds]
    return out


def _evaluate_gate(results: dict, config: Config | None = None) -> GateResult:
    """Apply the pre-registered gate, **per dataset**.

    A dataset passes only if all four conditions hold on the *mean* over seeds:

    1. ``0 <= gap <= max_gap`` — the plan is feasible (cannot beat the LP
       optimum) and within ``max_gap`` of it.
    2. ``marginal_violation <= max_marginal_violation`` — it really is a
       transport plan.
    3. ``dual_violation <= max_dual_violation`` — Sinkhorn really is monotone
       coordinate ascent.
    4. ``reduction >= min_reduction`` — transport beats the naive baseline.
    """
    cfg = config if config is not None else Config()
    flag = "otfuse"
    per_dataset: dict[str, dict] = {}
    all_pass = True

    for preset, by_method in results.items():
        stats = {m: summarise(rs) for m, rs in by_method.items()}
        f = stats[flag]
        checks = {
            "gap_nonnegative": bool(f["gap"]["mean"] >= -1e-9),
            "gap_within_budget": bool(f["gap"]["mean"] <= cfg.max_gap),
            "marginals": bool(f["marginal_violation"]["mean"] <= cfg.max_marginal_violation),
            "dual_ascent": bool(f["dual_violation"]["mean"] <= cfg.max_dual_violation),
            "reduction": bool(f["reduction"]["mean"] >= cfg.min_reduction),
        }
        passed = all(checks.values())
        all_pass = all_pass and passed
        per_dataset[preset] = {
            "passed": bool(passed),
            "checks": checks,
            "gap_mean": float(f["gap"]["mean"]),
            "gap_std": float(f["gap"]["std"]),
            "cost_mean": float(f["cost"]["mean"]),
            "exact_mean": float(f["exact"]["mean"]),
            "marginal_violation_mean": float(f["marginal_violation"]["mean"]),
            "dual_violation_mean": float(f["dual_violation"]["mean"]),
            "reduction_mean": float(f["reduction"]["mean"]),
            "runtime_s_mean": float(f["runtime_s"]["mean"]),
            "baseline_gap_mean": (float(stats["independent"]["gap"]["mean"]) if "independent" in stats else None),
            "ablation_gap_mean": (float(stats["sinkhorn"]["gap"]["mean"]) if "sinkhorn" in stats else None),
        }

    gaps = [v["gap_mean"] for v in per_dataset.values()]
    reds = [v["reduction_mean"] for v in per_dataset.values()]
    aggregate = {
        "flagship": flag,
        "max_gap_mean": float(np.max(gaps)) if gaps else float("nan"),
        "mean_gap_mean": float(np.mean(gaps)) if gaps else float("nan"),
        "min_reduction_mean": float(np.min(reds)) if reds else float("nan"),
        "n_datasets": len(per_dataset),
    }
    return GateResult(passed=bool(all_pass), per_dataset=per_dataset, aggregate=aggregate)
