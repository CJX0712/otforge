"""Evaluation metrics for optimal transport."""

from otforge.eval.metrics import (
    aggregate_across_seeds,
    cost_reduction,
    relative_gap,
    summarise,
    wasserstein,
)

__all__ = [
    "aggregate_across_seeds",
    "cost_reduction",
    "relative_gap",
    "summarise",
    "wasserstein",
]
