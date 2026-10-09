"""otForge — a world-class, deterministic Optimal Transport toolkit.

Author: 晨星 (Morning Star)

Design goals
------------
1. *Mathematically exact reference.* Every approximate solver is scored against
   an exact Kantorovich LP solution, and against the closed-form 1-D
   Wasserstein distance.  Three independent routes must agree.
2. *Zero hard dependencies.* SciPy is Tier-0; a pure-NumPy Jonker-Volgenant
   Hungarian solver provides exact EMD when SciPy is absent.
3. *Bit-identical determinism.* No unseeded RNG, no hash-order iteration.
4. *Provable invariants as tests.* Marginal feasibility, monotone Bregman
   (KL) descent, Sinkhorn-divergence non-negativity, barycenter local
   optimality — every one is a theorem, not a snapshot.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = ["__author__", "__version__"]
