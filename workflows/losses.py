"""Build data blocks and loss objectives for the optimization workflow.

The three loss types differ in how often the model output is compared with the truth:

- ``final``: integrate over the whole training window and compare once, at its end.
- ``windowed``: split the window into blocks of ``window_steps`` steps. Each block starts from the
  truth state at its start and is compared at its end, so errors cannot build up across windows.
- ``trajectory``: integrate over the whole window from one start state and compare every
  ``obs_every`` steps (``obs_every`` equal to the output frequency compares at every truth output).

All of them use dimswe.optimize.ObservationL2Objective; the schedule
(workflows.common.build_observation_schedule) decides the blocks and observation steps.
"""

from __future__ import annotations

from typing import Any


def needed_output_indices(schedule: dict[str, Any]) -> list[int]:
    """Return every truth output index needed by a schedule (block starts and observations)."""
    indices = set(schedule["block_start_indices"])
    for block in schedule["block_obs_indices"]:
        indices.update(block)
    return sorted(indices)


def build_data_blocks(
    schedule: dict[str, Any],
    truth_states_per_run: list[dict[int, list[Any]]],
    start_times_per_run: list[list[float]],
) -> tuple[list[list[list[Any]]], list[list[Any]]]:
    """Assemble data blocks and block start times for ObservationL2Objective.

    truth_states_per_run[r] maps a truth output index to an [x, None] state for truth run r, and
    start_times_per_run[r][k] is the model time at the start of block k of run r. Blocks from all
    truth runs are concatenated, so a joint optimization sums the loss over all of them.
    """
    from firedrake import Constant

    data_blocks = []
    t_blocks = []
    for states, start_times in zip(truth_states_per_run, start_times_per_run, strict=True):
        for block_start, obs_indices, start_time in zip(
            schedule["block_start_indices"], schedule["block_obs_indices"], start_times, strict=True
        ):
            data_blocks.append([states[block_start]] + [states[index] for index in obs_indices])
            t_blocks.append([Constant(start_time)])
    return data_blocks, t_blocks


def build_objective(
    model: Any,
    schedule: dict[str, Any],
    data_blocks: list[list[list[Any]]],
    t_blocks: list[list[Any]],
    normalize: bool,
) -> Any:
    """Create the ObservationL2Objective for a schedule.

    With normalize enabled, each variable is weighted by 1/max||truth||^2 over the observed truth
    states, and the loss is averaged over all observations, so different variables and different
    loss types give losses of comparable size.
    """
    from dimswe.optimize import ObservationL2Objective
    from workflows.opt_helpers import variable_weights

    var_weights = None
    loss_scale = 1.0
    if normalize:
        observed = [state for block in data_blocks for state in block[1:]]
        var_weights = variable_weights(model, observed)
        loss_scale = 1.0 / len(observed)
    return ObservationL2Objective(
        data_blocks,
        t_blocks,
        schedule["block_steps"],
        schedule["obs_steps"],
        model.spaces.dx,
        var_weights=var_weights,
        loss_scale=loss_scale,
    )
