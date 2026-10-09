"""Core utilities: seeding, typed errors, dataclasses, config, protocols."""

from otforge.core.config import Config
from otforge.core.errors import (
    ConfigError,
    DataError,
    EvalError,
    MethodError,
    OtForgeError,
    SolverError,
)
from otforge.core.interfaces import PlanSolver, TransportMethod
from otforge.core.seed import set_all
from otforge.core.types import BarycenterResult, GateResult, OtProblem, OtResult

__all__ = [
    "BarycenterResult",
    "Config",
    "ConfigError",
    "DataError",
    "EvalError",
    "GateResult",
    "MethodError",
    "OtForgeError",
    "OtProblem",
    "OtResult",
    "PlanSolver",
    "SolverError",
    "TransportMethod",
    "set_all",
]
