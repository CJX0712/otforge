"""Transport methods: baselines, the flagship solver, and the barycenter.

Every method returns an :class:`OtResult` that carries its cost *and* the exact
Kantorovich optimum for the same problem, so the gap is always measurable.
"""

from __future__ import annotations

import time

import numpy as np

from otforge.core.config import Config
from otforge.core.errors import MethodError
from otforge.core.types import BarycenterResult, OtProblem, OtResult
from otforge.ot.solvers import (
    annealed_sinkhorn,
    barycenter_fixed_support,
    cost_matrix,
    emd_exact,
    marginal_violation,
    sinkhorn_divergence,
    sinkhorn_log,
    transport_cost,
    wasserstein_from_cost,
)


def independent_cost(C: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Cost of the *unoptimised* product coupling ``P = a (x) b``.

    This is the honest baseline: it is feasible, so its cost is an upper bound
    on the true optimum, and any real transport method must beat it.
    """
    return float(np.asarray(a, dtype=float) @ np.asarray(C, dtype=float) @ np.asarray(b, float))


class _BaseMethod:
    """Shared plumbing: config holding, cost caching, result assembly."""

    name = "base"

    def __init__(self, config: Config | None = None, **overrides) -> None:
        self.config = config if config is not None else Config(**overrides)
        self.config.validate()

    def _ground_cost(self, problem: OtProblem) -> np.ndarray:
        return cost_matrix(problem.X, problem.Y, self.config.cost_p)

    def _exact(self, C: np.ndarray, problem: OtProblem) -> float:
        cost, _plan = emd_exact(C, problem.a, problem.b, backend=self.config.exact_backend)
        return cost

    def _wrap(
        self,
        problem: OtProblem,
        C: np.ndarray,
        cost: float,
        exact: float,
        viol: float,
        dual_viol: float,
        runtime_s: float,
        **extra,
    ) -> OtResult:
        base = independent_cost(C, problem.a, problem.b)
        reduction = 0.0 if base <= 0.0 else (base - cost) / base
        return OtResult(
            method=self.name,
            preset=problem.preset,
            seed=0,
            cost=float(cost),
            exact=float(exact),
            marginal_violation=float(viol),
            dual_violation=float(dual_viol),
            runtime_s=float(runtime_s),
            reduction=float(reduction),
            extra=dict(extra),
        )


class IndependentMethod(_BaseMethod):
    """Baseline: do not optimise at all — ship the product coupling."""

    name = "independent"

    def solve(self, problem: OtProblem) -> OtResult:
        t0 = time.perf_counter()
        C = self._ground_cost(problem)
        P = np.outer(problem.a, problem.b)
        cost = transport_cost(P, C)
        exact = self._exact(C, problem)
        runtime = time.perf_counter() - t0
        return self._wrap(problem, C, cost, exact, marginal_violation(P, problem.a, problem.b), 0.0, runtime)


class SinkhornMethod(_BaseMethod):
    """Single-stage entropic transport at a fixed ``eps`` (the ablation arm)."""

    name = "sinkhorn"

    def solve(self, problem: OtProblem) -> OtResult:
        t0 = time.perf_counter()
        C = self._ground_cost(problem)
        scale = max(float(np.median(C)), 1e-12)
        Cn = C / scale

        # Same total iteration budget as the annealed flagship: this isolates
        # the *value of annealing* rather than the value of more compute.
        budget = self.config.anneal_steps * self.config.sinkhorn_iters + self.config.final_iters
        eps = self.config.eps_final
        state = sinkhorn_log(Cn, problem.a, problem.b, eps, iters=budget, tol=self.config.sinkhorn_tol)
        runtime = time.perf_counter() - t0
        exact = self._exact(C, problem)
        return self._wrap(
            problem,
            C,
            state.cost * scale,
            exact,
            state.marginal_violation,
            state.dual_violation,
            runtime,
            eps=eps,
            iterations=state.iterations,
            converged=state.converged,
        )


class OtFuseMethod(_BaseMethod):
    """Flagship: epsilon-annealed log-domain Sinkhorn + Sinkhorn divergence.

    Three ingredients, each provably doing work:

    * *Log-domain potentials* — no under/overflow even at ``eps -> 0``.
    * *Geometric eps-annealing with warm-started, rescaled potentials* — reaches
      a far colder ``eps`` (hence a far tighter optimum) than a cold start.
    * *Debiased Sinkhorn divergence* — a non-negative discrepancy that is zero
      exactly when the two measures coincide.
    """

    name = "otfuse"

    def solve(self, problem: OtProblem) -> OtResult:
        t0 = time.perf_counter()
        C = self._ground_cost(problem)
        scale = max(float(np.median(C)), 1e-12)
        Cn = C / scale

        eps_init = self.config.eps_init
        eps_final = self.config.eps_final
        state = annealed_sinkhorn(
            Cn,
            problem.a,
            problem.b,
            eps_init=eps_init,
            eps_final=eps_final,
            steps=self.config.anneal_steps,
            iters_per_step=self.config.sinkhorn_iters,
            tol=self.config.sinkhorn_tol,
            final_iters=self.config.final_iters,
            intermediate_tol=self.config.intermediate_tol,
            safeguard_iters=self.config.safeguard_iters,
        )
        divergence = sinkhorn_divergence(
            Cn,
            cost_matrix(problem.X, problem.X, self.config.cost_p) / scale,
            cost_matrix(problem.Y, problem.Y, self.config.cost_p) / scale,
            problem.a,
            problem.b,
            eps=eps_final,
            iters=self.config.sinkhorn_iters,
            tol=self.config.sinkhorn_tol,
        )
        runtime = time.perf_counter() - t0
        exact = self._exact(C, problem)
        return self._wrap(
            problem,
            C,
            state.cost * scale,
            exact,
            state.marginal_violation,
            state.dual_violation,
            runtime,
            eps_final=eps_final,
            iterations=state.iterations,
            converged=state.converged,
            divergence=float(divergence) * scale,
            wasserstein=wasserstein_from_cost(state.cost * scale, self.config.cost_p),
        )


class WassersteinBarycenter:
    """Fixed-support entropic barycenter of several measures."""

    def __init__(
        self,
        eps: float = 0.05,
        outer_iters: int = 60,
        inner_iters: int = 40,
        step: float = 0.05,
    ) -> None:
        self.eps = eps
        self.outer_iters = outer_iters
        self.inner_iters = inner_iters
        self.step = step

    def run(
        self,
        support: np.ndarray,
        cost_matrices: list[np.ndarray],
        targets: list[np.ndarray],
        weights: np.ndarray | None = None,
    ) -> BarycenterResult:
        """Compute the barycenter weights on ``support``.

        Args:
            support: ``(n, d)`` barycenter support.
            cost_matrices: ``[C_k]`` with ``C_k`` of shape ``(n, m_k)``.
            targets: ``[b_k]`` target marginals.
            weights: Convex weights (default uniform).

        Returns:
            A :class:`BarycenterResult`.
        """
        if support is None or len(cost_matrices) == 0:
            raise MethodError("barycenter needs a support and at least one measure")
        a, objective, history, converged = barycenter_fixed_support(
            cost_matrices,
            targets,
            weights=weights,
            eps=self.eps,
            outer_iters=self.outer_iters,
            inner_iters=self.inner_iters,
            step=self.step,
        )
        return BarycenterResult(weights=a, objective=objective, history=history, converged=converged)


METHODS: dict[str, type] = {
    "independent": IndependentMethod,
    "sinkhorn": SinkhornMethod,
    "otfuse": OtFuseMethod,
}


def build_method(name: str, config: Config | None = None) -> _BaseMethod:
    """Instantiate a registered transport method by name."""
    if name not in METHODS:
        raise MethodError(f"unknown method {name!r}; known: {sorted(METHODS)}")
    return METHODS[name](config)
