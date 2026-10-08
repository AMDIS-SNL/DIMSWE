"""Pure selected-Test-1B plan validation and completed-truth audit.

This module imports neither Firedrake nor a timestepper.  It never generates a
trajectory, objective, or fit.  Its truth audit reads completed output only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .resolved_hidden_c0 import (
    LateTimeGrowthConfiguration,
    ResolvedInferenceConfiguration,
    ResolvedPilotConfiguration,
    RolloutLoss,
    ScanDerivativeLevel,
    SolverLossNormalization,
    build_inference_index_plan,
    late_time_growth_indicator,
    read_json_record,
    write_json_record,
)


DEFAULT_SELECTED_PLAN = (
    Path(__file__).resolve().parent / "configs" / "test1b_selected_plan.json"
)
FORMAT_VERSION = 1


@dataclass(frozen=True)
class SelectedTest1BConfiguration:
    case: str
    nx: int
    ny: int
    dt: float
    nsteps: int
    final_time: float
    output_stride: int
    s: float
    moist_backend: str
    seed: int
    c0_truth: float
    c0_initial: float
    c0_scale: float
    training_start_step: int
    training_stop_step: int
    heldout_stop_step: int
    observation_stride: int
    reset_window_length: int
    reset_window_stride: int
    reset_window_starts: tuple[int, ...]
    rollout_training_length: int
    loss: RolloutLoss
    solver_loss_normalization: SolverLossNormalization

    def __post_init__(self):
        if self.case != "doublevortex":
            raise ValueError("selected Test 1B case must be doublevortex")
        if (self.nx, self.ny) != (16, 16):
            raise ValueError("selected Test 1B mesh must be 16x16")
        if self.dt != 100.0 or self.nsteps != 160:
            raise ValueError("selected Test 1B time grid must be dt=100, nsteps=160")
        if self.final_time != self.dt * self.nsteps:
            raise ValueError("selected Test 1B final_time is inconsistent")
        if self.output_stride != 1:
            raise ValueError("canonical Test 1B truth requires output_stride=1")
        if self.s != 3.2 or self.moist_backend != "ufl" or self.seed != 0:
            raise ValueError("selected Test 1B physics/backend/seed changed")
        if (self.c0_truth, self.c0_initial, self.c0_scale) != (0.14, 0.07, 0.07):
            raise ValueError("selected Test 1B c0 convention changed")
        if (
            self.training_start_step,
            self.training_stop_step,
            self.heldout_stop_step,
        ) != (0, 80, 160):
            raise ValueError("selected Test 1B train/held-out split changed")
        if (
            self.observation_stride,
            self.reset_window_length,
            self.reset_window_stride,
            self.rollout_training_length,
        ) != (1, 5, 5, 80):
            raise ValueError("selected Test 1B canonical coverage changed")
        expected_starts = tuple(range(0, 80, 5))
        object.__setattr__(
            self, "reset_window_starts", tuple(self.reset_window_starts)
        )
        if self.reset_window_starts != expected_starts:
            raise ValueError("canonical reset starts must be 0,5,...,75")
        if RolloutLoss(self.loss) is not RolloutLoss.ACCUMULATED:
            raise ValueError("canonical solver-in-loop loss must be accumulated")
        if (
            SolverLossNormalization(self.solver_loss_normalization)
            is not SolverLossNormalization.TRUTH_TARGET_MASS
        ):
            raise ValueError(
                "canonical reset and rollout must share truth-target-mass "
                "normalization"
            )
        object.__setattr__(self, "loss", RolloutLoss(self.loss))
        object.__setattr__(
            self,
            "solver_loss_normalization",
            SolverLossNormalization(self.solver_loss_normalization),
        )

    @property
    def training_state_indices(self):
        return tuple(range(self.training_start_step, self.training_stop_step + 1))

    @property
    def training_transition_start_indices(self):
        return tuple(range(self.training_start_step, self.training_stop_step))

    @property
    def heldout_target_state_indices(self):
        return tuple(range(self.training_stop_step + 1, self.heldout_stop_step + 1))

    @property
    def heldout_transition_start_indices(self):
        return tuple(range(self.training_stop_step, self.heldout_stop_step))

    def pilot_configuration(self, output_directory: str | Path):
        return ResolvedPilotConfiguration(
            case=self.case,
            nx=self.nx,
            ny=self.ny,
            dt=self.dt,
            nsteps=self.nsteps,
            output_stride=self.output_stride,
            c0=self.c0_truth,
            s=self.s,
            moist_backend=self.moist_backend,
            seed=self.seed,
            output_directory=str(output_directory),
        )

    def inference_configuration(self, truth_run_directory: str | Path):
        return ResolvedInferenceConfiguration(
            truth_run_directory=str(truth_run_directory),
            c0_truth=self.c0_truth,
            c0_initial=self.c0_initial,
            training_start_step=self.training_start_step,
            training_stop_step=self.training_stop_step,
            heldout_stop_step=self.heldout_stop_step,
            observation_stride=self.observation_stride,
            truth_reset_horizon=self.reset_window_length,
            truth_reset_window_stride=self.reset_window_stride,
            rollout_horizon=self.rollout_training_length,
            truth_reset_loss=self.loss,
            rollout_loss=self.loss,
            solver_loss_normalization=self.solver_loss_normalization,
            c0_scale=self.c0_scale,
        )


def _configuration_from_plan(plan: Mapping[str, Any]):
    selected = dict(plan["selected_configuration"])
    control = dict(plan["control"])
    split = dict(plan["data_split"])
    comparison = dict(plan["canonical_comparison"])
    return SelectedTest1BConfiguration(
        **selected,
        c0_truth=control["c0_truth"],
        c0_initial=control["c0_initial"],
        c0_scale=control["c0_scale"],
        training_start_step=split["training"]["state_indices"][0],
        training_stop_step=split["boundary_state_index"],
        heldout_stop_step=split["heldout"]["target_state_indices"][1],
        observation_stride=comparison["observation_stride"],
        reset_window_length=comparison["reset_window_length"],
        reset_window_stride=comparison["reset_window_stride"],
        reset_window_starts=tuple(comparison["reset_window_starts"]),
        rollout_training_length=comparison["rollout_training_length"],
        loss=comparison["loss"],
        solver_loss_normalization=comparison["solver_loss_normalization"],
    )


def load_selected_test1b_plan(path: str | Path = DEFAULT_SELECTED_PLAN):
    plan = read_json_record(path)
    if plan.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported selected Test 1B plan format")
    configuration = _configuration_from_plan(plan)
    scan = plan["objective_scan"]
    if (
        ScanDerivativeLevel(scan["derivative_level"])
        is not ScanDerivativeLevel.OBJECTIVE_ONLY
    ):
        raise ValueError("selected Test 1B Gate 2 must be objective-only")
    expected_scan = np.linspace(
        scan["physical_lower"], scan["physical_upper"], scan["points"]
    )
    if not np.allclose(
        expected_scan,
        np.asarray(scan["values"]),
        rtol=0.0,
        atol=2.0e-16,
    ):
        raise ValueError("selected Test 1B objective scan values are inconsistent")
    if configuration.c0_truth not in scan["values"]:
        raise ValueError("selected objective scan does not include c0_truth")
    return plan, configuration


def fitting_and_heldout_index_record(configuration):
    inference = configuration.inference_configuration("selected_truth")
    plan = build_inference_index_plan(inference)
    return {
        "training_state_indices": (
            configuration.training_state_indices[0],
            configuration.training_state_indices[-1],
        ),
        "training_transition_start_indices": (
            inference.training_transition_steps[0],
            inference.training_transition_steps[-1],
        ),
        "heldout_boundary_initial_state_index": configuration.training_stop_step,
        "heldout_target_state_indices": (
            configuration.heldout_target_state_indices[0],
            configuration.heldout_target_state_indices[-1],
        ),
        "heldout_transition_start_indices": (
            inference.heldout_transition_steps[0],
            inference.heldout_transition_steps[-1],
        ),
        "heldout_states_available_during_fitting": False,
        "operator_offline_transition_starts": (
            plan.offline_transition_starts[0],
            plan.offline_transition_starts[-1],
        ),
        "deployed_discrete_transition_starts": (
            plan.offline_transition_starts[0],
            plan.offline_transition_starts[-1],
        ),
        "reset_window_starts": plan.truth_reset_windows
        and tuple(start for start, _ in plan.truth_reset_windows),
        "reset_target_state_indices": plan.truth_reset_target_steps,
        "rollout_start_state_index": plan.rollout_start_step,
        "rollout_target_state_indices": plan.rollout_target_steps,
        "reset_recursion_depth": configuration.reset_window_length,
        "rollout_recursion_depth": configuration.rollout_training_length,
    }


def _fingerprint(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _growth_diagnostics(times, diagnostics, configuration):
    common = {
        "baseline_fraction": configuration.baseline_fraction,
        "tail_fraction": configuration.tail_fraction,
    }
    specifications = {
        "kinetic_energy": (
            "kinetic_energy",
            configuration.kinetic_energy_factor,
            configuration.absolute_floor,
        ),
        "projected_enstrophy": (
            "projected_enstrophy",
            configuration.projected_enstrophy_factor,
            configuration.absolute_floor,
        ),
        "hyperviscosity_tendency_mass_norm": (
            "hyperviscosity_tendency_mass_norm",
            configuration.hyperviscosity_tendency_factor,
            configuration.absolute_floor,
        ),
        "velocity_high_wavenumber_energy_fraction": (
            "velocity_high_wavenumber_energy_fraction",
            configuration.high_wavenumber_fraction_factor,
            configuration.high_wavenumber_absolute_floor,
        ),
    }
    return {
        name: late_time_growth_indicator(
            times,
            [record[key] for record in diagnostics],
            growth_factor=factor,
            absolute_floor=floor,
            **common,
        )
        for name, (key, factor, floor) in specifications.items()
    }


def audit_selected_truth(
    truth_run_directory: str | Path,
    *,
    plan_path: str | Path = DEFAULT_SELECTED_PLAN,
):
    """Audit a completed canonical truth directory without importing a solver."""
    plan, selected = load_selected_test1b_plan(plan_path)
    run_directory = Path(truth_run_directory).resolve()
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    metadata_path = run_directory / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing truth metadata {metadata_path}")
    metadata = read_json_record(metadata_path)
    require(metadata.get("status") == "complete", "truth status is not complete")
    actual_configuration = metadata.get("configuration", {})
    expected_configuration = {
        "case": selected.case,
        "nx": selected.nx,
        "ny": selected.ny,
        "dt": selected.dt,
        "nsteps": selected.nsteps,
        "output_stride": selected.output_stride,
        "c0": selected.c0_truth,
        "s": selected.s,
        "moist_backend": selected.moist_backend,
        "seed": selected.seed,
    }
    for name, expected in expected_configuration.items():
        require(
            actual_configuration.get(name) == expected,
            f"truth configuration {name} does not match selection",
        )
    require(
        metadata.get("physical_parameters", {}).get("c0") == selected.c0_truth,
        "physical truth c0 is not exactly 0.14",
    )
    require(
        metadata.get("physical_parameters", {}).get("s") == selected.s,
        "physical hyperviscosity exponent is not 3.2",
    )
    require(metadata.get("moist_backend") == "ufl", "truth moist backend is not UFL")
    require(metadata.get("random_seed") == 0, "truth seed is not zero")

    try:
        pilot = ResolvedPilotConfiguration.from_dict(actual_configuration)
        expected_fingerprint = _fingerprint(
            pilot.physics_configuration(include_c0=False)
        )
        require(
            metadata.get("paired_non_c0_physics_sha256") == expected_fingerprint,
            "truth non-c0 physics fingerprint is invalid",
        )
    except (TypeError, ValueError) as exc:
        failures.append(f"invalid truth pilot configuration: {exc}")

    expected_steps = tuple(range(selected.nsteps + 1))
    completed_steps = tuple(int(value) for value in metadata.get("completed_output_steps", ()))
    require(completed_steps == expected_steps, "truth does not contain steps 0 through 160")
    time_metadata = metadata.get("time", {})
    require(time_metadata.get("dt") == selected.dt, "truth metadata dt is not 100")
    require(time_metadata.get("nsteps") == selected.nsteps, "truth nsteps is not 160")
    require(
        time_metadata.get("final_time") == selected.final_time,
        "truth final_time is not 16000",
    )
    expected_times = tuple(float(step * selected.dt) for step in expected_steps)
    output_times = tuple(float(value) for value in time_metadata.get("output_times", ()))
    require(output_times == expected_times, "truth output times are not 0,100,...,16000")

    state_convention = metadata.get("state_convention", {})
    field_sizes = tuple(int(value) for value in state_convention.get("field_sizes", ()))
    expected_size = sum(field_sizes)
    h_slice = state_convention.get("field_slices", {}).get("h")
    require(len(field_sizes) == 6 and expected_size > 0, "truth field layout is invalid")
    require(h_slice is not None and len(h_slice) == 2, "truth h field slice is missing")
    minimum_height = np.inf
    finite_states = True
    existing = {"restart": 0, "checkpoint": 0, "diagnostic": 0, "spectrum": 0}
    if expected_size > 0 and h_slice is not None and len(h_slice) == 2:
        h_start, h_stop = (int(value) for value in h_slice)
        for step in expected_steps:
            paths = {
                "restart": run_directory / "restart" / f"step_{step:08d}.npy",
                "checkpoint": run_directory / "checkpoints" / f"step_{step:08d}.h5",
                "diagnostic": run_directory / "diagnostics" / f"step_{step:08d}.json",
                "spectrum": run_directory / "spectra" / f"step_{step:08d}.npz",
            }
            for kind, path in paths.items():
                existing[kind] += int(path.exists())
            if not paths["restart"].exists():
                finite_states = False
                continue
            try:
                values = np.asarray(
                    np.load(paths["restart"], allow_pickle=False), dtype=np.float64
                )
            except (OSError, ValueError):
                finite_states = False
                continue
            if values.shape != (expected_size,) or not np.all(np.isfinite(values)):
                finite_states = False
                continue
            minimum_height = min(minimum_height, float(np.min(values[h_start:h_stop])))
    for kind, count in existing.items():
        require(count == len(expected_steps), f"truth has only {count}/161 {kind} snapshots")
    require(finite_states, "one or more truth state arrays are invalid or nonfinite")
    height_floor = float(
        plan["truth_audit"]["minimum_height_coefficient_strictly_greater_than"]
    )
    height_admissible = bool(
        np.isfinite(minimum_height) and minimum_height > height_floor
    )
    require(height_admissible, "truth minimum h is not admissible")

    diagnostics_by_step = {
        int(record["step"]): record for record in metadata.get("diagnostics", ())
    }
    require(
        tuple(sorted(diagnostics_by_step)) == expected_steps,
        "truth metadata does not contain 161 ordered diagnostics",
    )
    growth = {}
    if tuple(sorted(diagnostics_by_step)) == expected_steps:
        ordered_diagnostics = tuple(diagnostics_by_step[step] for step in expected_steps)
        diagnostic_times = tuple(float(record["time"]) for record in ordered_diagnostics)
        require(diagnostic_times == expected_times, "diagnostic times do not match truth grid")
        require(
            all(record["all_state_coefficients_finite"] for record in ordered_diagnostics),
            "truth diagnostics contain a nonfinite-state flag",
        )
        growth_configuration = LateTimeGrowthConfiguration(
            **plan["truth_audit"]["growth_configuration"]
        )
        growth = _growth_diagnostics(
            diagnostic_times, ordered_diagnostics, growth_configuration
        )
        require(
            not any(
                diagnostic["suspicious_late_time_growth"]
                for diagnostic in growth.values()
            ),
            "truth has a late-time numerical-stability heuristic warning",
        )

    index_record = fitting_and_heldout_index_record(selected)
    require(
        index_record["training_state_indices"] == (0, 80),
        "training state indexing is invalid",
    )
    require(
        index_record["heldout_target_state_indices"] == (81, 160),
        "held-out target indexing is invalid",
    )
    return {
        "audit": "selected Test 1B production truth",
        "plan": str(Path(plan_path).resolve()),
        "truth_run": str(run_directory),
        "passed": not failures,
        "failure_reasons": tuple(failures),
        "selected_configuration": asdict(selected),
        "state_count": existing["restart"],
        "expected_state_count": len(expected_steps),
        "output_times": (expected_times[0], expected_times[-1], selected.dt),
        "snapshot_file_counts": existing,
        "all_states_finite": finite_states,
        "minimum_height_coefficient": float(minimum_height),
        "minimum_height_admissible": height_admissible,
        "late_time_growth_heuristics": growth,
        "any_numerical_stability_heuristic_warning": any(
            value["suspicious_late_time_growth"] for value in growth.values()
        ),
        "indexing": index_record,
        "heldout_data_used_for_fitting": False,
    }


def _parser():
    parser = argparse.ArgumentParser(
        description="Validate the selected Test 1B plan or audit completed truth"
    )
    parser.add_argument("command", choices=("validate-plan", "audit-truth"))
    parser.add_argument("--plan", default=str(DEFAULT_SELECTED_PLAN))
    parser.add_argument("--truth-run")
    parser.add_argument("--output")
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    plan, selected = load_selected_test1b_plan(arguments.plan)
    if arguments.command == "validate-plan":
        result = {
            "passed": True,
            "plan": str(Path(arguments.plan).resolve()),
            "selected_configuration": asdict(selected),
            "indexing": fitting_and_heldout_index_record(selected),
            "canonical_comparison": plan["canonical_comparison"],
        }
    else:
        if not arguments.truth_run or not arguments.output:
            raise ValueError("audit-truth requires --truth-run and --output")
        result = audit_selected_truth(
            arguments.truth_run, plan_path=arguments.plan
        )
    if arguments.output:
        write_json_record(arguments.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DEFAULT_SELECTED_PLAN",
    "SelectedTest1BConfiguration",
    "audit_selected_truth",
    "fitting_and_heldout_index_record",
    "load_selected_test1b_plan",
)
