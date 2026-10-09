"""Typed error hierarchy (E1xx configuration ... E5xx evaluation)."""


class OtForgeError(Exception):
    """Base class for all otForge errors."""


class ConfigError(OtForgeError):
    """E1xx — invalid configuration."""


class DataError(OtForgeError):
    """E2xx — malformed problem data (marginals, supports)."""


class SolverError(OtForgeError):
    """E3xx — numerical / algorithmic failure inside a solver."""


class MethodError(OtForgeError):
    """E4xx — method misuse (fit before predict, unknown method, ...)."""


class EvalError(OtForgeError):
    """E5xx — metric evaluation failure."""
