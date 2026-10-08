"""Cheap semantic tests for the shared Test-2A Method-3/4 trajectory API."""

import pytest

from dimswe.test2a_trajectory import (
    NeuralWindowSpec,
    TrajectoryLossMode,
    continuous_rollout,
    reset_windows,
)


def test_endpoint_and_accumulated_modes_remain_explicit_and_distinct():
    endpoint = NeuralWindowSpec(4, 3, TrajectoryLossMode.ENDPOINT, (1.0,))
    accumulated = NeuralWindowSpec(
        4, 3, TrajectoryLossMode.ACCUMULATED, (1.0, 2.0, 3.0)
    )
    assert endpoint.target_steps == (7,)
    assert accumulated.target_steps == (5, 6, 7)
    assert endpoint.to_record()["loss_mode"] == "endpoint"
    assert accumulated.to_record()["loss_mode"] == "accumulated"


def test_reset_windows_are_independent_and_rollout_begins_once_at_zero():
    reset = reset_windows((0, 5), 2, "accumulated", (1.0, 1.0))
    rollout = continuous_rollout(5, "accumulated", (1.0,) * 5)
    assert tuple(window.start_step for window in reset) == (0, 5)
    assert tuple(window.target_steps for window in reset) == ((1, 2), (6, 7))
    assert len(rollout) == 1
    assert rollout[0].start_step == 0
    assert rollout[0].target_steps == (1, 2, 3, 4, 5)


@pytest.mark.parametrize("horizon", (1, 2, 3, 5, 10))
def test_required_horizons_are_supported_without_future_state_access(horizon):
    window = continuous_rollout(horizon, "accumulated", (1.0,) * horizon)[0]
    assert window.horizon == horizon
    assert max(window.target_steps) <= 80


def test_state_81_is_rejected_at_construction():
    with pytest.raises(ValueError, match="after 80"):
        NeuralWindowSpec(80, 1, TrajectoryLossMode.ENDPOINT, (1.0,))


def test_loss_weights_are_never_implicit():
    with pytest.raises(ValueError, match="explicit weights"):
        NeuralWindowSpec(0, 3, TrajectoryLossMode.ACCUMULATED, (1.0,))

