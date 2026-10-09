# Delivered — otForge

| field | value |
|---|---|
| system | **otForge** — deterministic optimal transport with provable invariants |
| repository | https://github.com/CJX0712/otforge |
| version / tag | v0.1.0 |
| author | 晨星 (Morning Star) |
| license | MIT |
| domain | optimal transport (Kantorovich LP · entropic Sinkhorn · Wasserstein barycenter) |
| delivered | 2026-10-10 |

## Why this domain

The forge fleet already covered robustness, fairness, causality and Gaussian
processes, but **nothing covered optimal transport** — and OT is one of the
highest-density mathematical areas available: Kantorovich duality, Bregman
projections, entropic regularisation, quantile closed forms. It also admits a
rare property for a benchmark: **the exact optimum is computable**, so every
approximate solver can be scored rather than trusted.

## Headline results (n = 64×64, 3 seeds)

| dataset | gap vs exact LP | marginal viol. | dual viol. | reduction |
|---|---|---|---|---|
| `gaussian_mix_1d` | +0.00400 | 1.3e-07 | 1.5e-16 | 0.423 |
| `point_cloud_2d` | +0.01244 | 9.9e-11 | 1.1e-16 | 0.500 |
| `grid_warp_2d` | +0.01180 | 1.0e-10 | 1.4e-17 | 0.944 |
| `domain_shift_2d` | +0.00596 | 9.9e-11 | 2.2e-16 | 0.472 |

Pre-registered gate: `gap ∈ [0, 0.02]` ∧ `marginal_violation ≤ 1e-5` ∧
`dual_violation ≤ 1e-9` ∧ `reduction ≥ 0.30`. **4/4 datasets pass with margin.**
Determinism: bit-identical across repeated runs.

## Ground-truth cross-validation (the core claim)

- pure-NumPy Jonker–Volgenant == SciPy `linear_sum_assignment` (square and
  rectangular, 5×5 … 64×64)
- exact LP == analytic 1-D closed form `W_p^p = mean(|x₍ᵢ₎ − y₍ᵢ₎|^p)` to
  **8.9e-16**
- Hungarian == brute-force enumeration on 5×5

## Bugs found and fixed during the build

1. **Jonker–Volgenant self-loop hang** — the vectorised relaxation updated
   `minv`/`way` for the current column `j0` too, creating `way[j0] = j0` and an
   infinite augmenting-path loop. Fixed with the reference algorithm's
   `~used` guard plus hard iteration guards.
2. **Wrong monotone potential** — `KL(P‖K)` as `(f·r + g·c)/ε` is *not*
   monotone (measured increases up to 3e+03). The provable invariant is the
   concave dual `L = ⟨f,a⟩ + ⟨g,b⟩ − εΣP`; violation is now ~1e-16.
3. **Cross-stage dual comparison** — the dual is ε-scaled; concatenating
   annealing histories compared incomparable values. Now measured per stage.
4. **Barycenter objective/gradient mismatch** — minimising `⟨C,P⟩` when the
   available gradient is that of `W_ε` (ratio 0.64). Switched to `W_ε`.
5. **Barycenter divergence** — dividing the gradient by ε made the step ~200×
   too large (objective climbed 0.082 → 0.312, weights collapsed). Replaced with
   the log-domain fixed point, damped by `step = 0.05`.
6. **Undamped barycenter is worse** — 0.0515 vs 0.0211 and non-monotone.
7. **Rectangular assignment** — unmatched rows were uninitialised; now `-1`.

## Deliverables

- `otforge/` — core, data, ot (solvers + methods), eval, pipeline, cli
- `examples/run_demo.py` — benchmark, gate, determinism, ablation, barycenter
- `tests/` — 65 pytest cases
- `docs/architecture.md`, `docs/model_card.md`
- CI: `quality-and-tests` (py3.10 + py3.13), `offline` (no SciPy), `secret-scan`
- Dockerfile, Makefile, pyproject, requirements (+ lock), README, CHANGELOG

## Environment note

Direct `git` egress to github.com:443 remains blocked in this sandbox (only
`gh` reaches GitHub). The repository was created and populated through `gh` and
the GitHub Contents / Git Data APIs; API-created commits still trigger the push
workflow, so CI was verified green remotely.
