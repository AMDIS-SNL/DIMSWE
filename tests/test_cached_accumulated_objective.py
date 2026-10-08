"""Tiny exact oracles for cached accumulated Test-1B objectives."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pytest

from dimswe.cached_accumulated_objective import (
    AccumulatedTrajectoryTarget,
    AccumulatedTrajectoryWindow,
    CachedAccumulatedC0Objective,
    PrefixAccumulatedC0ObjectiveOracle,
)
from dimswe.hidden_c0 import (
    HiddenC0Objective,
    ScalarOptimizerConfiguration,
    optimize_hidden_c0,
)
from dimswe.learned_physics.objectives import TrainingMode
from dimswe.resolved_hidden_c0 import (
    ObjectiveScanConfiguration,
    ScanDerivativeLevel,
    scan_scalar_objective,
)


@dataclass(frozen=True)
class _ToyCase:
    c0_scale: float = 0.07
    dt: float = 0.25


class _ToyOperations:
    """Exact scalar derivatives of F_c(x)=x+dt*c*x^2."""

    def __init__(self, case):
        self.case = case
        self.physical_c0 = None

    @contextmanager
    def parameter_context(self, physical_c0):
        previous = self.physical_c0
        self.physical_c0 = float(physical_c0)
        try:
            yield
        finally:
            self.physical_c0 = previous

    @staticmethod
    def copy_state(value, _name):
        return float(value)

    @staticmethod
    def copy_dual(value, _name):
        return float(value)

    @staticmethod
    def zero_state(_name):
        return 0.0

    @staticmethod
    def add_duals(left, right, _name):
        return float(left + right)

    def forward_step(self, state, _time, dt):
        x = float(state)
        c = float(self.physical_c0)
        cache = (x, c, float(dt))
        return cache, x + float(dt) * c * x * x

    @staticmethod
    def tangent_step(primal, state_direction, physical_direction):
        x, c, dt = primal
        dx = float(state_direction)
        dc = float(physical_direction)
        a = 1.0 + 2.0 * dt * c * x
        b = dt * x * x
        cache = (primal, dx, dc)
        return cache, a * dx + b * dc

    @staticmethod
    def reverse_step(primal, state_adjoint_out):
        x, c, dt = primal
        adjoint = float(state_adjoint_out)
        a = 1.0 + 2.0 * dt * c * x
        b = dt * x * x
        return a * adjoint, b * adjoint

    @staticmethod
    def incremental_reverse_step(
        tangent, state_adjoint_out, incremental_adjoint_out
    ):
        (x, c, dt), dx, dc = tangent
        adjoint = float(state_adjoint_out)
        incremental = float(incremental_adjoint_out)
        a = 1.0 + 2.0 * dt * c * x
        b = dt * x * x
        da = 2.0 * dt * (dc * x + c * dx)
        db = 2.0 * dt * x * dx
        return (
            a * adjoint,
            da * adjoint + a * incremental,
            b * adjoint,
            db * adjoint + b * incremental,
        )

    @staticmethod
    def local_loss(state, target, scale, _name):
        residual = float(state - target)
        return 0.5 * scale * residual * residual, scale * residual

    @staticmethod
    def local_loss_value(state, target, scale, _name):
        residual = float(state - target)
        return 0.5 * scale * residual * residual

    @staticmethod
    def local_loss_dual(state, target, scale, _name):
        return scale * float(state - target)

    @staticmethod
    def local_loss_hessian_action(state_direction, scale, _name):
        return scale * float(state_direction)


def _truth(case, c0=0.14, nsteps=80):
    values = [0.01]
    for _ in range(nsteps):
        x = values[-1]
        values.append(x + case.dt * c0 * x * x)
    return tuple(values)


def _target(offset, target_step, truth):
    value = truth[target_step]
    return AccumulatedTrajectoryTarget(
        offset=offset,
        target=value,
        normalizer=value * value,
        target_step=target_step,
    )


def _windows(mode):
    case = _ToyCase()
    truth = _truth(case)
    if mode is TrainingMode.ROLLOUT:
        windows = (
            AccumulatedTrajectoryWindow(
                initial_state=truth[0],
                start_time=0.0,
                targets=tuple(_target(step, step, truth) for step in range(1, 81)),
            ),
        )
    else:
        windows = tuple(
            AccumulatedTrajectoryWindow(
                initial_state=truth[start],
                start_time=start * case.dt,
                targets=tuple(
                    _target(offset, start + offset, truth)
                    for offset in range(1, 6)
                ),
            )
            for start in range(0, 80, 5)
        )
    return case, windows


def _pair(mode):
    case, windows = _windows(mode)
    cached = CachedAccumulatedC0Objective(
        mode, case, windows, operations=_ToyOperations(case)
    )
    oracle = PrefixAccumulatedC0ObjectiveOracle(
        mode, case, windows, operations=_ToyOperations(case)
    )
    return cached, oracle


def test_cached_rollout_objective_equals_prefix_sum_oracle():
    cached, oracle = _pair(TrainingMode.ROLLOUT)
    np.testing.assert_allclose(cached.value(1.3), oracle.value(1.3), rtol=2e-13)


def test_cached_rollout_gradient_equals_prefix_sum_oracle():
    cached, oracle = _pair(TrainingMode.ROLLOUT)
    np.testing.assert_allclose(cached.gradient(1.3), oracle.gradient(1.3), rtol=2e-13)


def test_cached_rollout_hvp_equals_prefix_sum_oracle():
    cached, oracle = _pair(TrainingMode.ROLLOUT)
    np.testing.assert_allclose(
        cached.hess_vec(1.3, 0.4), oracle.hess_vec(1.3, 0.4), rtol=4e-13
    )


def test_cached_reset_objective_equals_prefix_sum_oracle():
    cached, oracle = _pair(TrainingMode.TRUTH_RESET)
    np.testing.assert_allclose(cached.value(1.3), oracle.value(1.3), rtol=2e-13)


def test_cached_reset_gradient_equals_prefix_sum_oracle():
    cached, oracle = _pair(TrainingMode.TRUTH_RESET)
    np.testing.assert_allclose(cached.gradient(1.3), oracle.gradient(1.3), rtol=2e-13)


def test_cached_reset_hvp_equals_prefix_sum_oracle():
    cached, oracle = _pair(TrainingMode.TRUTH_RESET)
    np.testing.assert_allclose(
        cached.hess_vec(1.3, 0.4), oracle.hess_vec(1.3, 0.4), rtol=4e-13
    )


def test_canonical_objective_only_forward_counts_are_intrinsic():
    reset, _ = _pair(TrainingMode.TRUTH_RESET)
    rollout, _ = _pair(TrainingMode.ROLLOUT)
    reset.value(1.3)
    rollout.value(1.3)
    assert reset.counts().solver_calls == 80
    assert rollout.counts().solver_calls == 80
    assert reset.work_counts().forward_steps == 80
    assert rollout.work_counts().forward_steps == 80
    assert reset.work_counts().reverse_steps == 0
    assert rollout.work_counts().reverse_steps == 0


def test_cached_target_coverage_has_no_duplication_or_heldout_leakage():
    reset, _ = _pair(TrainingMode.TRUTH_RESET)
    rollout, _ = _pair(TrainingMode.ROLLOUT)
    assert reset.target_steps == tuple(range(1, 81))
    assert rollout.target_steps == tuple(range(1, 81))
    assert len(set(reset.target_steps)) == 80
    assert len(set(rollout.target_steps)) == 80
    assert set(reset.target_steps).isdisjoint(range(81, 161))
    assert set(rollout.target_steps).isdisjoint(range(81, 161))


def test_gradient_and_hvp_work_use_one_reverse_family_traversal_per_step():
    rollout_gradient, _ = _pair(TrainingMode.ROLLOUT)
    rollout_gradient.gradient(1.3)
    assert rollout_gradient.work_counts().forward_steps == 80
    assert rollout_gradient.work_counts().reverse_steps == 80
    assert rollout_gradient.work_counts().tangent_steps == 0

    rollout_hvp, _ = _pair(TrainingMode.ROLLOUT)
    rollout_hvp.hess_vec(1.3, 0.4)
    assert rollout_hvp.work_counts().forward_steps == 80
    assert rollout_hvp.work_counts().tangent_steps == 80
    assert rollout_hvp.work_counts().incremental_reverse_steps == 80


@pytest.mark.parametrize("mode", (TrainingMode.TRUTH_RESET, TrainingMode.ROLLOUT))
def test_canonical_objective_only_scan_point_uses_only_80_forward_steps(mode):
    objective, _ = _pair(mode)
    points = scan_scalar_objective(
        objective,
        ObjectiveScanConfiguration(
            physical_lower=0.04,
            physical_upper=0.20,
            points=3,
            derivative_level=ScanDerivativeLevel.OBJECTIVE_ONLY,
        ),
    )
    assert all(point.forward_steps == 80 for point in points)
    assert all(point.solver_calls == 80 for point in points)
    assert all(point.reverse_steps == 0 for point in points)
    assert all(point.tangent_steps == 0 for point in points)
    assert all(point.incremental_reverse_steps == 0 for point in points)


class _ExactOptimizerQuadratic(HiddenC0Objective):
    def __init__(self):
        super().__init__(TrainingMode.ROLLOUT, 0.07)

    def value(self, normalized_z):
        self.objective_evaluations += 1
        return 0.5 * (float(normalized_z) - 2.0) ** 2

    def value_and_gradient(self, normalized_z):
        self.objective_evaluations += 1
        self.gradient_evaluations += 1
        residual = float(normalized_z) - 2.0
        return 0.5 * residual * residual, residual

    def gradient(self, normalized_z):
        self.gradient_evaluations += 1
        return float(normalized_z) - 2.0

    def hess_vec(self, _normalized_z, direction):
        self.hvp_evaluations += 1
        return float(direction)


def test_gate3_optimizer_default_still_requests_exact_gradient_and_hvp():
    configuration = ScalarOptimizerConfiguration()
    assert configuration.use_hvp
    objective = _ExactOptimizerQuadratic()
    result = optimize_hidden_c0(
        objective,
        initial_c0=0.07,
        configuration=configuration,
    )
    assert result.success
    np.testing.assert_allclose(result.recovered_c0, 0.14, rtol=0.0, atol=1e-15)
    assert result.counts.gradient_evaluations >= 1
    assert result.counts.hvp_evaluations >= 1
