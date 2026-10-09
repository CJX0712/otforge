"""Synthetic optimal-transport problems.

Every preset produces *uniform* marginals on both sides, which makes the
Kantorovich LP exactly solvable by linear assignment.  That is deliberate: it
gives us a ground-truth optimum to score every approximate solver against.
"""

from __future__ import annotations

import numpy as np

from otforge.core.errors import DataError
from otforge.core.types import OtProblem

# dim -> ambient dimension of the preset.
_PRESETS: dict[str, dict] = {
    "gaussian_mix_1d": {"dim": 1},
    "point_cloud_2d": {"dim": 2},
    "grid_warp_2d": {"dim": 2},
    "domain_shift_2d": {"dim": 2},
}


def make_benchmark_suite() -> list[str]:
    """Return the preset names used by the default benchmark."""
    return ["gaussian_mix_1d", "point_cloud_2d", "grid_warp_2d", "domain_shift_2d"]


def preset_dim(preset: str) -> int:
    """Ambient dimension of a preset."""
    if preset not in _PRESETS:
        raise DataError(f"unknown preset {preset!r}; known: {sorted(_PRESETS)}")
    return int(_PRESETS[preset]["dim"])


def make_problem(preset: str, n_source: int, n_target: int, seed: int) -> OtProblem:
    """Build one OT problem.

    Args:
        preset: One of :func:`make_benchmark_suite`.
        n_source: Number of source atoms.
        n_target: Number of target atoms.
        seed: RNG seed (fully determines the problem).

    Returns:
        An :class:`OtProblem` with uniform marginals.
    """
    if preset not in _PRESETS:
        raise DataError(f"unknown preset {preset!r}; known: {sorted(_PRESETS)}")
    rng = np.random.default_rng(seed)
    dim = preset_dim(preset)

    if preset == "gaussian_mix_1d":
        X = _mixture_1d(rng, n_source)
        Y = 1.3 * _mixture_1d(rng, n_target) + 2.0
    elif preset == "point_cloud_2d":
        X = _two_clusters(rng, n_source)
        ang = np.deg2rad(40.0)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        Y = _two_clusters(rng, n_target) @ rot.T + np.array([1.5, -0.5])
    elif preset == "grid_warp_2d":
        X = _grid(n_source)
        Y = _radial_warp(X)
    elif preset == "domain_shift_2d":
        X = rng.uniform(-1.0, 1.0, size=(n_source, 2))
        Y = rng.uniform(-1.0, 1.0, size=(n_target, 2)) * np.array([2.0, 0.5]) + np.array([1.0, 0.5])
    else:  # pragma: no cover - guarded above
        raise DataError(f"unreachable preset {preset!r}")

    a = np.full(n_source, 1.0 / n_source)
    b = np.full(n_target, 1.0 / n_target)
    return OtProblem(X=X, Y=Y, a=a, b=b, preset=preset, dim=dim)


def _mixture_1d(rng: np.random.Generator, n: int) -> np.ndarray:
    """Two-component 1-D Gaussian mixture, returned as ``(n, 1)``."""
    labels = rng.integers(0, 2, size=n)
    mu = np.where(labels == 0, -1.0, 1.0)
    return (mu + 0.35 * rng.standard_normal(n)).reshape(-1, 1)


def _two_clusters(rng: np.random.Generator, n: int) -> np.ndarray:
    """Two isotropic 2-D Gaussian clusters."""
    centers = np.array([[-1.0, -1.0], [1.0, 1.0]])
    labels = rng.integers(0, 2, size=n)
    return centers[labels] + 0.4 * rng.standard_normal((n, 2))


def _grid(n: int) -> np.ndarray:
    """Roughly-square uniform grid with ``n`` points in ``[0, 1]^2``."""
    side = int(np.ceil(np.sqrt(n)))
    coords = np.linspace(0.0, 1.0, side)
    gx, gy = np.meshgrid(coords, coords, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return pts[:n]


def _radial_warp(X: np.ndarray) -> np.ndarray:
    """Contract the grid radially: ``r -> r**1.5`` about the centre."""
    centre = np.array([0.5, 0.5])
    delta = X - centre
    r = np.linalg.norm(delta, axis=1)
    safe = np.where(r > 1e-12, r, 1.0)
    direction = delta / safe[:, None]
    return centre + (r**1.5)[:, None] * direction
