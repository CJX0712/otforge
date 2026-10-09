"""End-to-end otForge demo.

Sections
--------
1. Benchmark across presets and seeds (mean +/- std).
2. Pre-registered quality gate.
3. Bit-identical determinism check.
4. Ablation: entropic bias vanishes as epsilon cools.
5. Wasserstein barycenter vs the closed-form 1-D quantile average.
6. Where the mass actually goes (the heaviest plan entries).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from otforge import __author__, __version__
from otforge.core.config import Config
from otforge.data.synthesizer import make_problem
from otforge.eval.metrics import wasserstein
from otforge.ot.methods import OtFuseMethod, WassersteinBarycenter, build_method
from otforge.ot.solvers import (
    annealed_sinkhorn,
    barycenter_1d_quantile,
    cost_matrix,
    emd_exact_backend_used,
    w_p_closed_form_1d,
)
from otforge.pipeline.pipeline import _evaluate_gate, benchmark, run_one

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METHODS = ["independent", "sinkhorn", "otfuse"]


def _fmt(mean: float, std: float, width: int = 8) -> str:
    return f"{mean:>{width}.4f}+/-{std:<.4f}"


def main() -> int:
    print("=" * 86)
    print(f"otForge v{__version__}  —  deterministic optimal transport   (author: {__author__})")
    print("=" * 86)

    cfg = Config.from_env()
    print(
        f"config: n={cfg.n_source}x{cfg.n_target}  cost_p={cfg.cost_p}  "
        f"eps {cfg.eps_init}->{cfg.eps_final} in {cfg.anneal_steps} stages  "
        f"seeds={list(cfg.seeds)}"
    )
    print(f"exact LP backend: {emd_exact_backend_used(cfg.exact_backend)}")
    print()

    # ---------------- 1. benchmark ---------------------------------------- #
    print("-" * 86)
    print("[1] BENCHMARK  (transport cost; lower is better; gap vs exact LP optimum)")
    print("-" * 86)
    results = benchmark(cfg, methods=METHODS)
    header = f"{'preset':<18}{'method':<13}{'cost (mean+/-std)':>24}{'exact':>10}{'gap':>10}{'viol':>10}{'reduce':>9}"
    print(header)
    print("-" * len(header))
    summary_stats = {}
    for preset, by_method in results.items():
        summary_stats[preset] = {}
        for method in METHODS:
            rs = by_method[method]
            costs = np.array([r.cost for r in rs])
            gaps = np.array([r.gap for r in rs])
            viols = np.array([r.marginal_violation for r in rs])
            reds = np.array([r.reduction for r in rs])
            summary_stats[preset][method] = {
                "cost_mean": float(costs.mean()),
                "cost_std": float(costs.std()),
                "gap_mean": float(gaps.mean()),
                "gap_std": float(gaps.std()),
                "reduction_mean": float(reds.mean()),
            }
            print(
                f"{preset:<18}{method:<13}{_fmt(costs.mean(), costs.std(), 12):>24}"
                f"{rs[0].exact:>10.5f}{gaps.mean():>+10.5f}{viols.max():>10.1e}"
                f"{reds.mean():>9.3f}"
            )
    print()

    # ---------------- 2. gate --------------------------------------------- #
    print("-" * 86)
    print("[2] PRE-REGISTERED GATE  (per dataset; all four conditions must hold)")
    print("-" * 86)
    gate = _evaluate_gate(results, cfg)
    for preset, info in gate.per_dataset.items():
        mark = "PASS" if info["passed"] else "FAIL"
        failed = [k for k, v in info["checks"].items() if not v]
        print(
            f"[{mark}] {preset:<18} gap={info['gap_mean']:+.5f} (<= {cfg.max_gap})  "
            f"viol={info['marginal_violation_mean']:.1e} (<= {cfg.max_marginal_violation:.0e})  "
            f"dual={info['dual_violation_mean']:.1e} (<= {cfg.max_dual_violation:.0e})  "
            f"reduction={info['reduction_mean']:.3f} (>= {cfg.min_reduction})"
            + (f"   failed: {failed}" if failed else "")
        )
    print()
    print(
        f"AGGREGATE: passed={gate.passed}  max gap={gate.aggregate['max_gap_mean']:.5f}  "
        f"min reduction={gate.aggregate['min_reduction_mean']:.3f}"
    )
    print()

    # ---------------- 3. determinism -------------------------------------- #
    print("-" * 86)
    print("[3] DETERMINISM  (two independent runs of the same seed)")
    print("-" * 86)
    det_ok = True
    for preset in results:
        a = run_one(preset, cfg, 5, "otfuse")
        b = run_one(preset, cfg, 5, "otfuse")
        same = (
            a.cost == b.cost
            and a.exact == b.exact
            and a.marginal_violation == b.marginal_violation
            and a.dual_violation == b.dual_violation
        )
        det_ok = det_ok and same
        print(f"{preset:<18} bit-identical={same}   cost={a.cost:.12f}")
    print(f"determinism: {'OK' if det_ok else 'FAILED'}")
    print()

    # ---------------- 4. ablation: cooling epsilon ------------------------ #
    print("-" * 86)
    print("[4] ABLATION  (entropic bias vanishes as epsilon cools; seed 0)")
    print("-" * 86)
    print(f"{'preset':<18}" + "".join(f"{'eps=' + format(e, '.0e'):>16}" for e in (5e-2, 2e-2, 1e-2)))
    ablation = {}
    for preset in results:
        row = f"{preset:<18}"
        ablation[preset] = {}
        for eps in (5e-2, 2e-2, 1e-2):
            sub = Config(eps_final=eps, n_source=cfg.n_source, n_target=cfg.n_target)
            res = run_one(preset, sub, 0, "otfuse")
            ablation[preset][eps] = res.gap
            row += f"{res.gap:>+16.5f}"
        print(row)
    print("(gap >= 0 always: a feasible plan can never beat the LP optimum)")
    print()

    # ---------------- 5. barycenter --------------------------------------- #
    print("-" * 86)
    print("[5] WASSERSTEIN BARYCENTER  (vs the closed-form 1-D quantile average)")
    print("-" * 86)
    rng = np.random.default_rng(0)
    grid = np.linspace(0.0, 1.0, 200).reshape(-1, 1)
    supports = [
        np.sort(rng.uniform(0.05, 0.45, size=32)).reshape(-1, 1),
        np.sort(rng.uniform(0.30, 0.70, size=32)).reshape(-1, 1),
        np.sort(rng.uniform(0.55, 0.95, size=32)).reshape(-1, 1),
    ]
    targets = [np.full(32, 1.0 / 32) for _ in supports]
    costs = [cost_matrix(grid, s, 2.0) for s in supports]
    eps_b = 0.02 * float(np.median(costs[0]))

    bary = WassersteinBarycenter(eps=eps_b, outer_iters=60, inner_iters=60)
    bary_res = bary.run(grid, costs, targets)

    closed = barycenter_1d_quantile([s.ravel() for s in supports])
    u = (np.arange(1, 20001) - 0.5) / 20000.0

    def _quantile_from_atoms(atoms: np.ndarray) -> np.ndarray:
        """Quantile function of a uniform 32-atom 1-D measure."""
        idx = np.clip((u * atoms.size).astype(int), 0, atoms.size - 1)
        return np.sort(atoms)[idx]

    def _quantile_from_weights(nodes: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Quantile function of a weighted measure on sorted ``nodes``."""
        cdf = np.cumsum(w)
        return np.interp(u, cdf, nodes)

    q_ours = _quantile_from_weights(grid.ravel(), bary_res.weights)
    q_true = _quantile_from_atoms(closed)
    w2_err = float(np.sqrt(np.mean((q_ours - q_true) ** 2)))
    grid_step = float(np.mean(np.diff(grid.ravel())))

    def objective_of(atoms: np.ndarray) -> float:
        return float(np.mean([w_p_closed_form_1d(atoms, s.ravel(), 2.0) for s in supports]))

    obj_bary = objective_of(closed)
    objs_inputs = [objective_of(s.ravel()) for s in supports]
    print(f"grid step                          : {grid_step:.5f}")
    print(f"W2(barycenter, closed form)        : {w2_err:.5f}   (grid resolution ~{grid_step:.4f})")
    print(f"exact W2 objective at closed form  : {obj_bary:.6f}")
    print(f"exact W2 objective at each input   : {[round(o, 6) for o in objs_inputs]}")
    print(f"barycenter <= every input          : {obj_bary <= min(objs_inputs) + 1e-12}")
    diffs = np.diff(np.asarray(bary_res.history))
    max_inc = float(max(0.0, diffs.max()))
    rel_inc = max_inc / max(abs(bary_res.history[0]), 1e-12)
    monotone = bool(rel_inc <= 1e-6)
    print(f"barycenter objective max increase  : {max_inc:.2e} (relative {rel_inc:.2e})")
    print(f"barycenter objective non-increasing: {monotone}")
    print()

    # ---------------- 6. where the mass goes ------------------------------ #
    print("-" * 86)
    print("[6] HEAVIEST PLAN ENTRIES  (otfuse, seed 0, gaussian_mix_1d)")
    print("-" * 86)
    prob = make_problem("gaussian_mix_1d", cfg.n_source, cfg.n_target, 0)
    solver = OtFuseMethod(cfg)
    C = solver._ground_cost(prob)
    scale = max(float(np.median(C)), 1e-12)
    state = build_method("otfuse", cfg).solve(prob)
    st = annealed_sinkhorn(
        C / scale,
        prob.a,
        prob.b,
        eps_init=cfg.eps_init,
        eps_final=cfg.eps_final,
        steps=cfg.anneal_steps,
        iters_per_step=cfg.sinkhorn_iters,
        final_iters=cfg.final_iters,
        tol=cfg.sinkhorn_tol,
        safeguard_iters=cfg.safeguard_iters,
    )
    plan = st.plan
    flat = np.argsort(plan.ravel())[::-1][:5]
    print(f"{'rank':<6}{'source x':>10}{'target y':>11}{'mass':>10}{'cost':>10}")
    for rank, idx in enumerate(flat, start=1):
        i, j = divmod(int(idx), plan.shape[1])
        print(f"{rank:<6}{prob.X[i, 0]:>10.4f}{prob.Y[j, 0]:>11.4f}{plan[i, j]:>10.5f}{C[i, j]:>10.5f}")
    print()
    print(f"W2 distance of the plan            : {wasserstein(state.cost, cfg.cost_p):.6f}")
    print(f"Sinkhorn divergence (debiased)     : {state.extra.get('divergence', float('nan')):.6f}")
    print()

    # ---------------- report ---------------------------------------------- #
    payload = {
        "version": __version__,
        "author": __author__,
        "config": {
            "n_source": cfg.n_source,
            "n_target": cfg.n_target,
            "eps_init": cfg.eps_init,
            "eps_final": cfg.eps_final,
            "anneal_steps": cfg.anneal_steps,
            "seeds": list(cfg.seeds),
            "exact_backend": emd_exact_backend_used(cfg.exact_backend),
        },
        "summary": summary_stats,
        "gate": {
            "passed": gate.passed,
            "per_dataset": gate.per_dataset,
            "aggregate": gate.aggregate,
        },
        "determinism": {"bit_identical": bool(det_ok)},
        "ablation": {p: {str(k): v for k, v in d.items()} for p, d in ablation.items()},
        "barycenter": {
            "w2_error_vs_closed_form": w2_err,
            "grid_step": grid_step,
            "objective_at_closed_form": obj_bary,
            "objective_at_inputs": objs_inputs,
            "objective_non_increasing": bool(monotone),
            "objective_max_increase": max_inc,
        },
    }
    out = ROOT / "benchmark.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print("=" * 86)
    print(f"RESULT: gate={'PASS' if gate.passed else 'FAIL'}  determinism={'OK' if det_ok else 'FAIL'}")
    print("=" * 86)
    return 0 if (gate.passed and det_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
