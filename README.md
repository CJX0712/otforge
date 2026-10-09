# otForge

**World-class, deterministic optimal transport — with the optimum known exactly.**

[![ci](https://github.com/CJX0712/otforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/otforge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Author: **晨星 (Morning Star)**

---

## Why this is different

Most optimal-transport libraries ask you to *trust* their solver. otForge does
not: every approximate solver is scored against a **ground truth it cannot
see**, and three independent routes must agree.

| Route | Nature | Role |
|---|---|---|
| `emd_exact` | Exact Kantorovich LP | ground truth (SciPy Tier-0, **pure-NumPy JV fallback**) |
| `w_p_closed_form_1d` | Analytic 1-D quantile formula | independent ground truth |
| `sinkhorn_log` | Entropic regularisation, log-domain | the scalable solver being scored |

The 1-D closed form agrees with the exact LP to **8.9e-16**, and the pure-NumPy
Jonker–Volgenant assignment solver reproduces SciPy's optimum **bit for bit** on
square and rectangular problems. That is the bar.

## Install

```bash
pip install -r requirements.txt        # numpy only — fully functional
pip install -r requirements-dev.txt    # + scipy (faster LP), pytest, ruff
```

**NumPy is the only hard dependency.** With SciPy absent, `emd_exact` switches
to otForge's own O(n³) Jonker–Volgenant implementation and still returns the
*exact* optimum.

## Quick start

```bash
python -m otforge.cli methods
python -m otforge.cli run --seeds 0,1,2 --output report.json
python -m examples.run_demo
```

```python
import numpy as np
from otforge.core.config import Config
from otforge.data.synthesizer import make_problem
from otforge.ot.methods import build_method

prob = make_problem("point_cloud_2d", 64, 64, seed=0)
res = build_method("otfuse", Config()).solve(prob)

res.cost                # <C, P> of the plan
res.exact               # exact Kantorovich optimum for the same problem
res.gap                 # relative excess over the optimum  (>= 0 always)
res.marginal_violation  # how far P is from the transport polytope
res.dual_violation      # Sinkhorn dual-ascent violation (must be ~0)
res.reduction           # saving vs the naive independent coupling
```

## The mathematics

### Exact Kantorovich problem

```
min_{P >= 0}  <C, P>     s.t.  P 1 = a,  Pᵀ 1 = b
```

Solved as a linear assignment problem after expanding rational marginals to
integer multiplicities. `C[i,j] = ||x_i − y_j||₂^p`.

### Entropic regularisation (Sinkhorn / Bregman projections)

```
min_{P}  <C, P> − ε H(P)     s.t.  P 1 = a,  Pᵀ 1 = b
```

Solved in the **log domain** with dual potentials `P = exp((f ⊕ g − C)/ε)`:

```
f ← f + ε (log a − logsumexp_j((f_i + g_j − C_ij)/ε))
g ← g + ε (log b − logsumexp_i((f_i + g_j − C_ij)/ε))
```

This never under/overflows, so ε can be driven far colder than a scaling-form
implementation allows.

**Provable invariant.** Because `ε·KL(P‖K) = <C,P> − εH(P)` and each half-step
is an *exact* block maximisation of the concave dual

```
L(f, g) = ⟨f,a⟩ + ⟨g,b⟩ − ε Σ_ij exp((f_i + g_j − C_ij)/ε)
```

the recorded `L` is **non-decreasing**. Measured violation: **~1e-16**.

### ε-annealing with warm-started potentials

A warm ε is well conditioned and converges in a handful of iterations; cooling
ε geometrically while warm-starting (and rescaling) `f, g` reaches a far colder
ε than any cold start. A **safeguard** keeps iterating at the coldest ε only if
the plan has not yet reached the marginal constraints — spent lazily, so a
well-conditioned problem pays nothing.

### The feasibility bound

A Sinkhorn plan satisfies its marginals, hence it is *feasible* for the
Kantorovich LP, hence

```
cost(sinkhorn) >= cost(exact)          always
cost(sinkhorn) -> cost(exact)          as ε -> 0
```

otForge asserts both. The first is a hard theorem; the second is measured.

### Wasserstein barycenter (fixed support)

```
min_a  Σ_k w_k · W_ε(a, b_k)
```

Using the *entropic* value makes `∂W_ε/∂a = f` (verified numerically to 1.5%),
so stationarity over the simplex is `Σ_k w_k f_k = const`, giving the
multiplicative fixed point

```
a_i ∝ Π_k (K_k v_k)_i^{w_k},    K_k = exp(−C_k/ε)
```

evaluated as a stable `logsumexp` and **damped** (`step = 0.05`), which makes
the objective monotonically non-increasing *and* reaches a strictly better
fixed point than the undamped iteration.

## Pre-registered quality gate

Declared **before** the numbers were produced. A dataset passes only if all
four hold on the mean over seeds:

| # | Condition | Threshold |
|---|---|---|
| 1 | `gap >= 0` | feasibility — a plan cannot beat the LP optimum |
| 2 | `gap <= max_gap` | within 2% of the optimum |
| 3 | `marginal_violation <= max_marginal_violation` | 1e-5 — it really is a transport plan |
| 4 | `dual_violation <= max_dual_violation` | 1e-9 — coordinate ascent never decreased |
| 5 | `reduction >= min_reduction` | 30% cheaper than the independent coupling |

### Measured (n = 64×64, 3 seeds, Python 3.13)

| dataset | gap (vs exact LP) | marginal viol. | dual viol. | reduction | verdict |
|---|---|---|---|---|---|
| `gaussian_mix_1d` | **+0.00400** | 1.3e-07 | 1.5e-16 | 0.423 | ✅ PASS |
| `point_cloud_2d` | **+0.01244** | 9.9e-11 | 1.1e-16 | 0.500 | ✅ PASS |
| `grid_warp_2d` | **+0.01180** | 1.0e-10 | 1.4e-17 | 0.944 | ✅ PASS |
| `domain_shift_2d` | **+0.00596** | 9.9e-11 | 2.2e-16 | 0.472 | ✅ PASS |

Max gap **0.0124** against a 0.02 budget; min reduction **0.423** against 0.30.
**Bit-identical determinism**: two runs of the same seed agree to the last bit.

### Ablation — the entropic bias vanishes as ε cools

| dataset | ε=5e-2 | ε=2e-2 | ε=1e-2 |
|---|---|---|---|
| `gaussian_mix_1d` | +0.0211 | +0.0086 | **+0.0042** |
| `point_cloud_2d` | +0.0649 | +0.0254 | **+0.0113** |
| `grid_warp_2d` | +0.6083 | +0.1305 | **+0.0118** |
| `domain_shift_2d` | +0.0439 | +0.0148 | **+0.0061** |

Every entry is `>= 0`, as feasibility demands.

## Methods

| name | what it is |
|---|---|
| `independent` | baseline: `P = a ⊗ b`, no geometry at all |
| `sinkhorn` | single-stage entropic transport at a fixed ε (ablation arm) |
| `otfuse` | **flagship**: ε-annealed log-domain Sinkhorn + safeguard + debiased Sinkhorn divergence, scale-free ε |

## Layout

```
otforge/
  core/       seed · errors · types · config · interfaces
  data/       synthetic OT problem presets (uniform marginals ⇒ exact LP)
  ot/         solvers.py (kernels of the library) · methods.py
  eval/       metrics (relative gap, reduction, seed aggregation)
  pipeline/   benchmark + pre-registered gate
  cli.py
examples/run_demo.py   benchmark · gate · determinism · ablation · barycenter
tests/                 65 pytest cases
docs/                  architecture.md · model_card.md
```

## Determinism

* One seeding entry point (`core.seed.set_all`); no module touches a global RNG.
* All solvers are pure arithmetic on NumPy arrays — no hash-order iteration.
* Two runs of the same seed produce **exactly** equal costs, gaps and violations.

## License

MIT — © 2026 晨星 (Morning Star).
