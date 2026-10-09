"""Global determinism control.

Every stochastic consumer obtains its randomness from the generator returned by
:func:`set_all` (or from an explicitly seeded ``np.random.default_rng``).  No
module may call a global RNG without going through this helper.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_all(seed: int) -> np.random.Generator:
    """Seed every randomness source and return the canonical generator.

    Args:
        seed: Non-negative integer seed.

    Returns:
        A ``np.random.Generator`` that callers should thread explicitly.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    return np.random.default_rng(seed)
