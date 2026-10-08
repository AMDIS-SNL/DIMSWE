"""Cheap analytic checks for scale-invariant scalar Newton decisions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from dimswe.hidden_c0 import (
    HiddenC0Objective,
    OfflineC0Objective,
    OfflineObservation,
    ScalarOptimizerConfiguration,
    optimize_hidden_c0,
)
from dimswe.learned_physics import TrainingMode


SCALES = (1.0, 1.0e-12, 1.0e12)


class _ScaledAnalyticObjective(HiddenC0Objective):
    """Analytic scalar objective with exact scaled gradient and HVP."""

    def __init__(
        self,
        scale: float,
        value: Callable[[float], float],
        gradient: Callable[[float], float],
        hessian: Callable[[float], float],
    ):
        super().__init__(TrainingMode.APRIORI_OFFLINE, 0.07)
        self.scale = float(scale)
        self._value = value
        self._gradient = gradient
        self._hessian = hessian
        self.value_points = []
        self.hessian_points = []

    def value(self, normalized_z):
        z = float(normalized_z)
        self.objective_evaluations += 1
        self.value_points.append(z)
        return self.scale * self._value(z)

    def value_and_gradient(self, normalized_z):
        z = float(normalized_z)
        self.objective_evaluations += 1
        self.gradient_evaluations += 1
        return self.scale * self._value(z), self.scale * self._gradient(z)

    def gradient(self, normalized_z):
        self.gradient_evaluations += 1
        return self.scale * self._gradient(float(normalized_z))

    def hess_vec(self, normalized_z, direction):
        z = float(normalized_z)
        self.hvp_evaluations += 1
        self.hessian_points.append(z)
        return self.scale * self._hessian(z) * float(direction)


def _fit_family(
    value,
    gradient,
    hessian,
    *,
    initial_c0=0.07,
    configuration=ScalarOptimizerConfiguration(),
):
    objectives = tuple(
        _ScaledAnalyticObjective(alpha, value, gradient, hessian)
        for alpha in SCALES
    )
    results = tuple(
        optimize_hidden_c0(
            objective,
            initial_c0=initial_c0,
            configuration=configuration,
        )
        for objective in objectives
    )
    reference = results[0]
    for alpha, result in zip(SCALES, results):
        np.testing.assert_allclose(
            result.normalized_iterates,
            reference.normalized_iterates,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            result.recovered_normalized_z,
            reference.recovered_normalized_z,
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            0.07 * np.asarray(result.normalized_iterates),
            0.07 * np.asarray(reference.normalized_iterates),
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            np.asarray(result.objective_history) / alpha,
            reference.objective_history,
            rtol=3.0e-14,
            atol=3.0e-14,
        )
        assert result.success == reference.success
        assert result.termination_reason == reference.termination_reason
        assert result.failure_reason == reference.failure_reason
        assert result.counts == reference.counts
    return objectives, results


def test_positive_quadratic_has_scale_invariant_newton_iterates():
    objectives, results = _fit_family(
        lambda z: 0.5 * (z - 2.0) ** 2,
        lambda z: z - 2.0,
        lambda _z: 1.0,
    )
    for objective, result in zip(objectives, results):
        assert result.success
        assert result.normalized_iterates == (1.0, 2.0)
        assert result.recovered_c0 == pytest.approx(0.14, abs=1.0e-15)
        assert (
            result.termination_reason
            == "relative gradient tolerance satisfied"
        )
        assert objective.hessian_points == [1.0]


def test_nonlinear_convex_objective_is_scale_invariant():
    _, results = _fit_family(
        lambda z: 0.5 * (z - 2.0) ** 2 + 0.05 * (z - 2.0) ** 4,
        lambda z: (z - 2.0) + 0.2 * (z - 2.0) ** 3,
        lambda z: 1.0 + 0.6 * (z - 2.0) ** 2,
    )
    assert results[0].success
    assert results[0].recovered_normalized_z == pytest.approx(2.0, abs=1.0e-13)


def test_bounded_optimum_uses_scale_invariant_projected_gradient():
    configuration = ScalarOptimizerConfiguration(physical_upper=0.21)
    _, results = _fit_family(
        lambda z: 0.5 * (z - 4.0) ** 2,
        lambda z: z - 4.0,
        lambda _z: 1.0,
        configuration=configuration,
    )
    assert results[0].recovered_normalized_z == pytest.approx(3.0)
    assert (
        results[0].termination_reason
        == "projected gradient satisfies bound constraint"
    )


def test_armijo_backtracking_path_is_scale_invariant():
    objectives, results = _fit_family(
        lambda z: np.exp(z) - 10.0 * z,
        lambda z: np.exp(z) - 10.0,
        lambda z: np.exp(z),
        configuration=ScalarOptimizerConfiguration(max_iterations=12),
    )
    assert results[0].success
    assert results[0].recovered_normalized_z == pytest.approx(
        np.log(10.0), rel=1.0e-11
    )
    # The first full Newton trial overshoots; all scales take the same
    # backtracked line-search path before accepting an iterate.
    assert len(objectives[0].value_points) >= 2
    for objective in objectives[1:]:
        np.testing.assert_allclose(
            objective.value_points,
            objectives[0].value_points,
            rtol=2.0e-14,
            atol=2.0e-14,
        )


def test_negative_curvature_is_rejected_at_every_objective_scale():
    objectives, results = _fit_family(
        lambda z: (z - 2.0) ** 4 - 0.5 * (z - 2.0) ** 2,
        lambda z: 4.0 * (z - 2.0) ** 3 - (z - 2.0),
        lambda z: 12.0 * (z - 2.0) ** 2 - 1.0,
        initial_c0=0.126,
        configuration=ScalarOptimizerConfiguration(max_iterations=12),
    )
    for objective in objectives:
        assert objective._hessian(1.8) < 0.0
    # Rejecting the negative Hessian sends the first accepted point left along
    # the bounded-gradient fallback, not right along -g/h.
    assert results[0].normalized_iterates[1] < 1.8
    assert results[0].success


def test_small_relative_parameter_step_is_scale_invariant_convergence():
    optimum = 1.0 + 5.0e-12
    _, results = _fit_family(
        lambda z: 0.5 * (z - optimum) ** 2,
        lambda z: z - optimum,
        lambda _z: 1.0,
    )
    assert results[0].normalized_iterates == (1.0,)
    assert (
        results[0].termination_reason
        == "relative parameter step tolerance satisfied"
    )


def test_tiny_scale_neither_false_converges_nor_rejects_positive_curvature():
    _, results = _fit_family(
        lambda z: 0.5 * (z - 2.0) ** 2,
        lambda z: z - 2.0,
        lambda _z: 1.0,
    )
    tiny = results[SCALES.index(1.0e-12)]
    assert tiny.normalized_iterates == (1.0, 2.0)
    assert tiny.counts.hvp_evaluations == 1
    assert tiny.termination_reason != "initial gradient is exactly zero"


@pytest.mark.parametrize(
    "mode",
    (TrainingMode.APRIORI_OFFLINE, TrainingMode.DISCRETE_OFFLINE),
)
def test_test1a_exact_offline_objective_still_recovers_c0(mode):
    objective = OfflineC0Objective(
        mode,
        0.07,
        0.14,
        (OfflineObservation(np.array([1.0, -2.0]), "Test-1A check"),),
    )
    result = optimize_hidden_c0(objective, initial_c0=0.07)
    assert result.success
    assert result.normalized_iterates == (1.0, 2.0)
    assert result.recovered_c0 == pytest.approx(0.14, abs=1.0e-15)
    assert result.counts.solver_calls == 0
