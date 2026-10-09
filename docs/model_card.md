# Model card — otForge v0.1.0

## Model details

| field | value |
|---|---|
| name | otForge |
| version | 0.1.0 |
| author | 晨星 (Morning Star) |
| license | MIT |
| kind | deterministic numerical library (optimal transport) |
| learned parameters | none — no training, no weights |
| languages | Python 3.10+ |
| dependencies | NumPy (required), SciPy (optional accelerator) |

## Intended use

**In scope**
- Computing transport plans and Wasserstein distances between discrete
  measures, with a verifiable accuracy certificate (gap vs the exact LP).
- Entropic/annealed transport where the exact LP is too expensive.
- Fixed-support Wasserstein barycenters.
- Research and education: every shipped invariant is a theorem, and each is
  asserted in the test suite.

**Out of scope**
- Continuous (non-discrete) measures.
- Problems whose marginals have no small integer common denominator, if the
  exact LP is required (Sinkhorn and the barycenter work regardless).
- Safety-critical decision-making without independent validation.

## Components

| component | algorithm | complexity |
|---|---|---|
| `emd_exact` | linear assignment (SciPy / Jonker–Volgenant) | O(N³) on the expanded N×N problem |
| `w_p_closed_form_1d` | sorting | O(n log n) |
| `sinkhorn_log` | Bregman alternating projections, log-domain | O(nm) per iteration |
| `annealed_sinkhorn` | geometric ε-cooling + warm start | O(stages × iterations × nm) |
| `sinkhorn_divergence` | 3 entropic solves | 3× one solve |
| `barycenter_fixed_support` | damped multiplicative fixed point | O(outer × K × inner × nm) |

## Data

All benchmarks use **synthetic** generators (`data/synthesizer.py`); no personal
or licensed data is involved.

| preset | dim | structure |
|---|---|---|
| `gaussian_mix_1d` | 1 | two-component Gaussian mixture vs a shifted, scaled copy |
| `point_cloud_2d` | 2 | two clusters vs a 40°-rotated, translated copy |
| `grid_warp_2d` | 2 | uniform 8×8 grid vs its radial contraction `r → r^1.5` |
| `domain_shift_2d` | 2 | uniform square vs an anisotropically stretched, shifted copy |

All presets carry **uniform marginals**, which is what makes the exact LP
available as ground truth.

## Quantitative performance

n = 64 source × 64 target, 3 seeds, cost `‖x − y‖²`, ε schedule 1.0 → 1e-2
(20 geometric stages), Python 3.13 / NumPy 2.5.3.

| dataset | gap vs exact LP | marginal violation | dual-ascent violation | reduction vs baseline |
|---|---|---|---|---|
| `gaussian_mix_1d` | +0.00400 | 1.3e-07 | 1.5e-16 | 0.423 |
| `point_cloud_2d` | +0.01244 | 9.9e-11 | 1.1e-16 | 0.500 |
| `grid_warp_2d` | +0.01180 | 1.0e-10 | 1.4e-17 | 0.944 |
| `domain_shift_2d` | +0.00596 | 9.9e-11 | 2.2e-16 | 0.472 |

Cross-validation:

| check | result |
|---|---|
| pure-NumPy Hungarian vs SciPy assignment | identical optimum (5×5 … 64×64, and rectangular) |
| exact LP vs analytic 1-D closed form | agreement to **8.9e-16** |
| Hungarian vs brute-force permutations (5×5) | identical |
| determinism (same seed, two runs) | bit-identical |
| barycenter vs closed-form 1-D quantile average | W2 error 0.0098 on a grid of step 0.0050 |

## Invariants asserted in tests

1. Feasibility: `cost(plan) >= cost(exact LP)` for every method.
2. Marginals: `‖P1 − a‖∞`, `‖Pᵀ1 − b‖∞` below tolerance.
3. Monotone dual ascent: largest decrease of `L` below 1e-12.
4. Bias monotone in ε: the gap shrinks as ε cools.
5. Divergence: `S ≥ 0`, and `S = 0` when the two measures coincide.
6. Barycenter: objective non-increasing (relative increase ≤ 1e-7); damped
   fixed point strictly better than the undamped one; closed-form 1-D
   barycenter is a local minimiser under perturbation.
7. Offline: exact LP still exact with SciPy unimportable.

## Limitations and failure modes

- **Cold ε can fail to converge.** At very small ε the Sinkhorn iteration slows
  and the plan can leave the transport polytope, making the reported cost fall
  *below* the true optimum. otForge detects this via `marginal_violation` and
  spends the safeguard budget; the default ε = 1e-2 (in median-cost units) is
  the validated operating point where all presets converge.
- **Entropic bias is problem-dependent.** It scales like ε·log(nm) *relative to
  the optimal cost*, so presets whose optimum is small compared with the cost
  scale (e.g. `grid_warp_2d`) need a colder ε for the same relative accuracy.
- **Exact LP needs rational marginals.** Otherwise `emd_exact` raises
  `SolverError`; use Sinkhorn.
- **Fixed-support barycenter** is limited by grid resolution and inherits the
  entropic bias of its inner solves.
- Assignment problems above a few thousand atoms are not the target: the exact
  LP is O(N³).

## Ethical considerations

otForge is a numerical library with no data, no labels and no learned
parameters, so it carries no dataset bias. Optimal transport is, however,
commonly used in fairness, domain adaptation and distribution matching — where
the choice of ground cost *is* a normative choice. The cost exponent `cost_p`
and the ε schedule materially change results; both are explicit, configurable
and reported alongside every number.

## Citation

```
otForge v0.1.0 — deterministic optimal transport with provable invariants.
晨星 (Morning Star), 2026. MIT License.
https://github.com/CJX0712/otforge
```

### References

- Kantorovich (1942) — the transportation problem.
- Sinkhorn (1964, 1967) — diagonal scaling of matrices.
- Cuturi (2013), *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*.
- Peyré & Cuturi (2019), *Computational Optimal Transport*.
- Benamou et al. (2015), *Iterative Bregman Projections for Regularized Transportation Problems*.
- Jonker & Volgenant (1987) — shortest augmenting path assignment.
