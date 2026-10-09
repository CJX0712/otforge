# otForge architecture

## Design thesis

An approximate solver is only as good as the ground truth you score it
against. otForge is built so that ground truth is **always available**:

1. Every synthetic preset uses **uniform marginals**, so the Kantorovich LP is
   exactly a linear assignment problem.
2. The assignment problem is solved twice — by SciPy (Tier-0) and by otForge's
   own pure-NumPy Jonker–Volgenant implementation — and both are cross-checked
   against the **analytic 1-D closed form**.
3. Every approximate method reports `exact` alongside `cost`, so its gap is a
   first-class field, not an afterthought.

## Layer map

```
                    ┌──────────────────────────────────────────────┐
   CLI / demo  ───► │  pipeline/   benchmark · run_one · gate      │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │  ot/methods   independent · sinkhorn · otfuse│
                    │               WassersteinBarycenter          │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │  ot/solvers   emd_exact · w_p_closed_form_1d │
                    │               sinkhorn_log · annealed_sinkhorn│
                    │               sinkhorn_divergence · barycenter│
                    └───────────────┬──────────────────────────────┘
                                    │
   data/synthesizer ──────────────► │  core/   types · config · seed│
                                    │          errors · interfaces  │
                                    └──────────────────────────────┘
```

Dependencies point strictly downward: `ot/` never imports `pipeline/`,
`core/` imports nothing from the package.

## Core types

| type | role |
|---|---|
| `OtProblem` | `(X, Y, a, b, preset, dim)`; validates shapes and normalises marginals |
| `OtResult` | one method on one problem; carries `cost`, `exact`, `gap`, violations, `reduction` |
| `GateResult` | `passed`, `per_dataset`, `aggregate` |
| `SinkhornState` | plan, dual potentials, dual value, marginal violation, dual history |
| `BarycenterResult` | weights, objective, history, converged |

`OtResult.gap` is a *property*: `(cost − exact) / exact`. There is exactly one
definition of the gap in the codebase.

## Solvers

### `emd_exact(C, a, b, backend)`

1. Recover integer multiplicities for `a` and `b` (bounded denominator search).
2. Expand rows/columns into an `N × N` cost matrix.
3. Solve with `scipy.optimize.linear_sum_assignment` when available, else
   `_hungarian`.
4. Return `cost = Σ C[i, σ(i)] / N` and the corresponding sparse plan.

`_hungarian` is the Jonker–Volgenant shortest-augmenting-path algorithm,
O(n³) worst case, with the O(m) inner scan **vectorised** (so the Python-level
loop count is O(n²)). Two details matter:

* the relaxation must skip *visited* columns (`(cur < minv) & ~used`);
* hard iteration guards raise `SolverError` rather than spinning.

See CHANGELOG 0.1.0 for the hang this guarded against.

### `w_p_closed_form_1d(x, y, p)`

For equal-size uniform 1-D measures the optimal map is monotone, so
`W_p^p = mean(|x₍ᵢ₎ − y₍ᵢ₎|^p)`. No solver is involved — it is the reference
that proves the solver right.

### `sinkhorn_log(C, a, b, eps, ...)`

Log-domain potentials `P = exp((f ⊕ g − C)/ε)`. All reductions are
`logsumexp`-stabilised, so ε may be driven far below what a scaling-form
implementation survives.

Tracked quantity: the **dual**

```
L(f, g) = ⟨f,a⟩ + ⟨g,b⟩ − ε Σ_ij exp((f_i + g_j − C_ij)/ε)
```

Each half-step is an exact block maximisation of `L`, so `L` never decreases.
`dual_ascent_violation` measures the largest decrease; it is ~1e-16 in
practice.

### `annealed_sinkhorn(...)`

Geometric cooling `ε_k = ε_0 · decay^k` with potentials warm-started and
rescaled by `decay` (they scale like ε). Warm stages use a loose tolerance and
exit early. The coldest stage gets its own larger budget, and a safeguard
continues it only if the marginals are still violated.

The dual invariant is measured **per stage** — the dual is ε-scaled, so values
from different stages are not comparable.

### `barycenter_fixed_support(...)`

Minimises `Σ_k w_k · W_ε(a, b_k)` over `a` on a shared support. Because
`∂W_ε/∂a = f`, stationarity is `Σ_k w_k f_k = const`, i.e.

```
log a_i  ∝  Σ_k w_k · logsumexp_j( (g_k[j] − C_k[i,j]) / ε )
```

which is exactly `log Π_k (K_k v_k)_i^{w_k}` in a numerically safe form. The
update is **damped**: `log a ← (1 − s) log a + s·log target` with `s = 0.05`.
Damping is not cosmetic — the undamped iteration overshoots and converges to a
strictly worse fixed point.

## Methods

| method | solver path | purpose |
|---|---|---|
| `independent` | `P = a ⊗ b` | honest baseline: feasible, so its cost upper-bounds the optimum |
| `sinkhorn` | one `sinkhorn_log` at ε_final | ablation: isolates the value of annealing |
| `otfuse` | `annealed_sinkhorn` + safeguard + divergence | flagship |
| `WassersteinBarycenter` | `barycenter_fixed_support` | multi-measure averaging |

Every method returns `OtResult` through a single `_wrap` helper, so `reduction`
and the gap are computed identically everywhere.

## Determinism

* `core.seed.set_all(seed)` is the only seeding entry point.
* Solvers are pure array arithmetic; no dict/set iteration order affects output.
* `PYTHONHASHSEED` is pinned.

## Extending

1. New preset → `data/synthesizer.py` (`_PRESETS` + a generator). Keep
   marginals uniform so the exact LP stays available.
2. New method → subclass `_BaseMethod` in `ot/methods.py`, register in
   `METHODS`; the pipeline, CLI and gate pick it up automatically.
3. New invariant → assert it in `tests/`; the gate is deliberately narrow so a
   new invariant does not silently change pass/fail semantics.
