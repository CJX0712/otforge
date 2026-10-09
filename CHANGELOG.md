# Changelog

All notable changes to otForge are documented here.

## [0.1.0] — 2026-10-10

### Added
- **Exact Kantorovich LP** (`emd_exact`) via linear assignment, with a
  **pure-NumPy Jonker–Volgenant** implementation so the exact optimum is
  available with zero SciPy. SciPy's `linear_sum_assignment` is Tier-0.
- **Analytic 1-D ground truth** (`w_p_closed_form_1d`): for equal-size uniform
  marginals the optimal Monge map is order-preserving, so
  `W_p^p = mean(|x_(i) − y_(i)|^p)` on sorted samples.
- **Log-domain Sinkhorn** (`sinkhorn_log`) with dual potentials, stable at
  arbitrarily cold ε, reporting the entropic dual objective.
- **ε-annealed Sinkhorn** (`annealed_sinkhorn`) with warm-started, rescaled
  potentials, loose tolerances on warm stages, and a lazily-spent convergence
  safeguard.
- **Debiased Sinkhorn divergence** `S = W(a,b) − ½W(a,a) − ½W(b,b)` (≥ 0, and
  = 0 exactly when the measures coincide).
- **Fixed-support entropic Wasserstein barycenter** with a damped,
  log-domain-stable multiplicative update, plus the closed-form 1-D quantile
  barycenter for validation.
- Three methods (`independent`, `sinkhorn`, `otfuse`), four dataset presets,
  metrics, benchmark pipeline, pre-registered gate, CLI, demo, 65 tests,
  Dockerfile, Makefile, CI on Python 3.10 + 3.13 plus an offline (no-SciPy) job.

### Fixed
- **Jonker–Volgenant self-loop hang.** The vectorised relaxation updated
  `minv`/`way` for *all* columns, including the current one `j0`; whenever the
  reduced cost of `(i0, j0)` was strictly below the incumbent it wrote
  `way[j0] = j0`, creating a self-loop that made the augmenting-path
  reconstruction spin forever. The reference algorithm guards the relaxation
  with `if not used[j]`. Now vectorised as `(cur < minv) & ~used`, plus hard
  iteration guards that raise `SolverError` instead of hanging.
- **Wrong monotone potential.** `KL(P‖K)` computed as `(f·r + g·c)/ε` is *not*
  monotone across Sinkhorn half-steps (measured increases up to 3e+03). The
  provably monotone quantity is the concave dual
  `L = ⟨f,a⟩ + ⟨g,b⟩ − ε·ΣP`, of which Sinkhorn performs exact block
  coordinate ascent. Replaced throughout; measured violation is now ~1e-16.
- **Cross-stage dual comparison.** The dual is ε-scaled, so concatenating
  histories across annealing stages compared incomparable numbers. The ascent
  invariant is now measured *within* each stage and the worst stage reported.
- **Barycenter objective / gradient mismatch.** The barycenter minimised the
  raw transport cost `⟨C,P⟩`, whose gradient w.r.t. `a` is *not* the Sinkhorn
  potential `f` (measured ratio 0.64). Switched to the entropic value
  `W_ε = ⟨f,a⟩ + ⟨g,b⟩ − ε`, whose gradient is exactly `f` (verified to 1.5%).
- **Barycenter divergence.** The update divided the gradient by ε, making the
  effective step ~200× too large; the objective climbed 0.082 → 0.312 and
  weights collapsed to zero. Replaced with the log-domain fixed point
  `log a ← Σ_k w_k · logsumexp_j((g_k[j] − C_k[i,j])/ε)`, **damped** by
  `step = 0.05`.
- **Undamped barycenter fixed point is worse.** At `step = 1.0` the iteration
  overshoots and converges to a *higher* objective (0.0515) than the damped one
  (0.0211), and is not monotone. Damping is therefore the default, not a knob.
- **Rectangular assignment semantics.** For `n > m`, unmatched rows were
  previously filled with uninitialised memory; they are now reported as `-1`.

### Validated
- Pure-NumPy Hungarian reproduces SciPy's optimum exactly on 5×5 … 64×64 and
  rectangular 30×50 / 50×30 problems.
- 1-D closed form vs exact LP: agreement to **8.9e-16** across seeds.
- Gate passes on all four presets × three seeds with margin.

Author: 晨星 (Morning Star) — MIT License.
