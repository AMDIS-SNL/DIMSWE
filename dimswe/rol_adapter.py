"""Serial, one-scalar PyROL access to the existing DIMSWE objective/adjoint."""

import numpy as np
from pyrol import Bounds, Objective, ParameterList
from pyrol.vectors import NumPyVector


def normalized_c0_bounds(model):
    """Return scalar ROL bounds after the map ``c0 = d_c0 * z``."""
    names = model.get_coeff_list()
    if names != ["s", "c0"]:
        raise ValueError(f"expected DIMSWE coefficient order ['s', 'c0'], got {names}")
    lower, upper = model.get_coeff_bounds()
    scales = model.get_coeff_scaling_factors()
    if scales.shape != (2,) or scales[1] <= 0.0:
        raise ValueError("expected two positive DIMSWE coefficient scales")
    return float(lower[1] / scales[1]), float(upper[1] / scales[1])


def numpy_c0_bounds(model):
    """Construct serial one-element PyROL bounds in normalized coordinates."""
    lower, upper = normalized_c0_bounds(model)
    return Bounds(
        NumPyVector(np.array([lower], dtype=np.float64)),
        NumPyVector(np.array([upper], dtype=np.float64)),
    )


def bound_constrained_lbfgs_parameters(
    gradient_tolerance=1.0e-10, step_tolerance=1.0e-14, iteration_limit=50
):
    """Parameters for a bounded first-order ROL solve with L-BFGS secants."""
    parameters = ParameterList()

    step = ParameterList("Step")
    step.set("Type", "Line Search")
    line_search = ParameterList("Line Search")
    descent = ParameterList("Descent Method")
    descent.set("Type", "Quasi-Newton Method")
    line_search.set("Descent Method", descent)
    step.set("Line Search", line_search)
    parameters.set("Step", step)

    general = ParameterList("General")
    secant = ParameterList("Secant")
    secant.set("Type", "Limited-Memory BFGS")
    general.set("Secant", secant)
    parameters.set("General", general)

    status = ParameterList("Status Test")
    status.set("Gradient Tolerance", float(gradient_tolerance))
    status.set("Step Tolerance", float(step_tolerance))
    status.set("Iteration Limit", int(iteration_limit))
    parameters.set("Status Test", status)
    return parameters


class ScalarC0Objective(Objective):
    """PyROL Objective that exposes only normalized hyperviscosity ``c0``.

    ``reduced_objective`` is the existing
    :class:`dimswe.optimize.Lagrangian_ODEConstrainedOptimization`.  It owns
    both the forward objective and discrete-adjoint gradient.  No forward,
    adjoint, or second-order implementation is duplicated here.
    """

    def __init__(self, reduced_objective, fixed_s_normalized):
        super().__init__()
        self.reduced_objective = reduced_objective
        self.fixed_s_normalized = float(fixed_s_normalized)
        names = reduced_objective.model.get_coeff_list()
        if names != ["s", "c0"]:
            raise ValueError(
                f"expected DIMSWE coefficient order ['s', 'c0'], got {names}"
            )
        scales = np.asarray(
            reduced_objective.coeff_scaling_factors, dtype=np.float64
        )
        if scales.shape != (2,) or np.any(scales <= 0.0):
            raise ValueError("expected two positive DIMSWE coefficient scales")
        self.coefficient_scales = scales
        self.value_evaluations = 0
        self.gradient_evaluations = 0
        self.value_history = []
        self.gradient_history = []

    def pack_normalized_coefficients(self, z):
        """Pack scalar ``z`` into the established normalized ``[s, c0]`` order."""
        if not isinstance(z, NumPyVector) or z.dimension() != 1:
            raise TypeError("c0 control must be a one-element NumPyVector")
        return np.array(
            [self.fixed_s_normalized, float(z.array[0])], dtype=np.float64
        )

    def value(self, z, tol):
        full_coefficient = self.pack_normalized_coefficients(z)
        result = float(self.reduced_objective.obj(full_coefficient, None))
        self.value_evaluations += 1
        self.value_history.append((float(z.array[0]), result))
        return result

    def gradient(self, g, z, tol):
        if not isinstance(g, NumPyVector) or g.dimension() != 1:
            raise TypeError("c0 gradient must be a one-element NumPyVector")
        full_coefficient = self.pack_normalized_coefficients(z)
        full_gradient = np.asarray(
            self.reduced_objective.jac(full_coefficient, None), dtype=np.float64
        )
        if full_gradient.shape != (2,):
            raise ValueError("existing DIMSWE adjoint did not return [s, c0]")
        g.array[:] = 0.0
        g.array[0] = full_gradient[1]
        self.gradient_evaluations += 1
        self.gradient_history.append((float(z.array[0]), float(g.array[0])))
