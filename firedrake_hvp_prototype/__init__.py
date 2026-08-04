"""Isolated dual-native Firedrake exact-HVP prototype."""

from .core import (
    CLASSICAL_RK4,
    EULER,
    ButcherTableau,
    ExplicitRungeKutta,
    GradientResult,
    HVPResult,
    WeakStageModel,
    dual_pairing,
    terminal_least_squares_gradient,
    terminal_least_squares_hvp,
    terminal_least_squares_objective,
)

__all__ = [
    "CLASSICAL_RK4",
    "EULER",
    "ButcherTableau",
    "ExplicitRungeKutta",
    "GradientResult",
    "HVPResult",
    "WeakStageModel",
    "dual_pairing",
    "terminal_least_squares_gradient",
    "terminal_least_squares_hvp",
    "terminal_least_squares_objective",
]
