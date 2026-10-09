"""Command-line entry point for otForge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from otforge.core.config import Config
from otforge.core.types import OtResult
from otforge.data.synthesizer import make_benchmark_suite
from otforge.ot.methods import METHODS
from otforge.pipeline.pipeline import _evaluate_gate, benchmark


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="otforge",
        description="otForge — deterministic optimal transport with provable invariants.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the benchmark and evaluate the gate")
    run.add_argument("--n-source", type=int, default=Config.n_source)
    run.add_argument("--n-target", type=int, default=Config.n_target)
    run.add_argument("--seeds", type=str, default="0,1,2", help="comma-separated seeds")
    run.add_argument(
        "--methods",
        type=str,
        default="independent,sinkhorn,otfuse",
        help="comma-separated method names",
    )
    run.add_argument("--presets", type=str, default=None, help="comma-separated presets")
    run.add_argument("--eps-final", type=float, default=Config.eps_final)
    run.add_argument("--anneal-steps", type=int, default=Config.anneal_steps)
    run.add_argument("--exact-backend", type=str, default=Config.exact_backend)
    run.add_argument("--output", type=str, default=None, help="write the report as JSON")

    sub.add_parser("methods", help="list registered transport methods")
    return parser


def _result_to_dict(r: OtResult) -> dict:
    return {
        "method": r.method,
        "preset": r.preset,
        "seed": r.seed,
        "cost": r.cost,
        "exact": r.exact,
        "gap": r.gap,
        "marginal_violation": r.marginal_violation,
        "dual_violation": r.dual_violation,
        "reduction": r.reduction,
        "runtime_s": r.runtime_s,
        "extra": r.extra,
    }


def _print_table(results: dict) -> None:
    header = f"{'preset':<18}{'method':<13}{'cost':>10}{'exact':>10}{'gap':>10}{'viol':>10}{'red':>8}"
    print(header)
    print("-" * len(header))
    for preset, by_method in results.items():
        for method, rs in by_method.items():
            r = rs[0]
            print(
                f"{preset:<18}{method:<13}{r.cost:>10.5f}{r.exact:>10.5f}"
                f"{r.gap:>+10.5f}{r.marginal_violation:>10.1e}{r.reduction:>8.3f}"
            )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)

    if args.command == "methods":
        for name in sorted(METHODS):
            print(name)
        return 0

    config = Config(
        n_source=args.n_source,
        n_target=args.n_target,
        eps_final=args.eps_final,
        anneal_steps=args.anneal_steps,
        exact_backend=args.exact_backend,
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
    )
    config.validate()
    presets = [p.strip() for p in args.presets.split(",") if p.strip()] if args.presets else make_benchmark_suite()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    results = benchmark(config, presets=presets, methods=methods)
    _print_table(results)
    gate = _evaluate_gate(results, config)

    print()
    for preset, info in gate.per_dataset.items():
        mark = "PASS" if info["passed"] else "FAIL"
        print(
            f"[{mark}] {preset:<18} gap={info['gap_mean']:+.5f} "
            f"viol={info['marginal_violation_mean']:.1e} "
            f"dual={info['dual_violation_mean']:.1e} "
            f"reduction={info['reduction_mean']:.3f}"
        )
    print()
    print(
        f"GATE: passed={gate.passed}  max gap={gate.aggregate['max_gap_mean']:.5f} "
        f"(threshold {config.max_gap})  min reduction={gate.aggregate['min_reduction_mean']:.3f} "
        f"(threshold {config.min_reduction})"
    )

    if args.output:
        payload = {
            "config": {
                "n_source": config.n_source,
                "n_target": config.n_target,
                "eps_final": config.eps_final,
                "anneal_steps": config.anneal_steps,
                "seeds": list(config.seeds),
            },
            "runs": {
                preset: {m: [_result_to_dict(r) for r in rs] for m, rs in by_method.items()}
                for preset, by_method in results.items()
            },
            "gate": {
                "passed": gate.passed,
                "per_dataset": gate.per_dataset,
                "aggregate": gate.aggregate,
            },
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")

    return 0 if gate.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
