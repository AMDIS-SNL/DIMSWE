"""Opt-in serial execution driver for the J4B resolved hidden-c0 pilot.

The driver reuses the unchanged production MTSWE split and J3 helper.  It is
never imported by :mod:`dimswe.resolved_hidden_c0`, so configuration and
analysis tests remain inexpensive and Firedrake-free.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import numpy as np
from firedrake import (
    CheckpointFile,
    Function,
    LinearVariationalProblem,
    LinearVariationalSolver,
    TestFunction,
    TrialFunction,
    VTKFile,
    assemble,
    inner,
)

from .hidden_c0 import (
    C0_SCALE,
    HiddenC0Case,
    STATE_FIELDS,
    _copy_function,
    _flat_values,
    _serial_solver_parameters,
    _state_squared_difference,
)
from .logger import EmptyLogger
from .models import get_model
from .parameters import get_parameters
from .resolved_hidden_c0 import (
    CANDIDATE_CASES,
    ResolvedPilotConfiguration,
    paired_pilot_configurations,
    shell_averaged_vector_spectrum,
    write_json_record,
)
from .timestepping import get_timestepper
from .ufl_helpers import curl2D


def _git_value(arguments: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(
            ("git",) + arguments,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _configuration_fingerprint(configuration: dict[str, Any]) -> str:
    payload = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolved_hidden_c0_parameters(
    configuration: ResolvedPilotConfiguration,
) -> dict[str, Any]:
    """Return owned production parameters for one resolved pilot configuration."""
    if not isinstance(configuration, ResolvedPilotConfiguration):
        raise TypeError("configuration must be ResolvedPilotConfiguration")
    config_path = configuration.base_config
    if config_path is None:
        config_path = files("dimswe").joinpath(
            "configs", "resolved_hidden_c0_pilot.cfg"
        )
    parameters = get_parameters(str(config_path))
    parameters["initial-conditions"]["name"] = configuration.case
    parameters["initial-conditions"]["zeta"] = configuration.initial_moisture_zeta
    parameters["mesh"]["nx"] = configuration.nx
    parameters["mesh"]["ny"] = configuration.ny
    parameters["timestepping"]["dt"] = configuration.dt
    parameters["timestepping"]["num_steps"] = configuration.nsteps
    parameters["timestepping"]["subcycle_list"] = [2, 1, 2, 1]
    parameters["hyperviscosity"]["c0"] = configuration.c0
    parameters["hyperviscosity"]["s"] = configuration.s
    parameters["hyperviscosity"]["treat_as_coeffs"] = True
    parameters["threewayphysics"]["treat_as_coeffs"] = False
    return parameters


def build_resolved_hidden_c0_case(
    configuration: ResolvedPilotConfiguration,
    *,
    jax_moist_local_physics=None,
) -> HiddenC0Case:
    """Construct a native repository IC with the complete production split."""
    if jax_moist_local_physics is not None and configuration.moist_backend != "jax":
        raise ValueError(
            "an opt-in JAX moist local-physics provider requires moist_backend='jax'"
        )
    parameters = resolved_hidden_c0_parameters(configuration)

    logger = EmptyLogger()
    model = get_model(parameters, logger, has_dynamics_statistics=False)
    if model.mesh.comm.size != 1:
        raise RuntimeError("J4B-PREP resolved pilot is serial CPU only")
    if tuple(model.get_x_var_list()) != STATE_FIELDS:
        raise RuntimeError("resolved pilot requires the complete six-field MTSWE")

    coefficient, coefficient_sub, _ = model.get_coeff_var(
        "resolved_hidden_c0_coefficient"
    )
    state_container, state_sub, _ = model.get_full_var(
        "resolved_hidden_c0_state", split_x_and_aux=True
    )
    time = model.get_t_var()
    model.initialize(state_sub, time)
    model.set_coeffs(parameters, coefficient_sub)
    timestepper = get_timestepper(
        parameters,
        model,
        logger,
        _serial_solver_parameters(),
        moist_backend=configuration.moist_backend,
        jax_moist_local_physics=jax_moist_local_physics,
    )
    timestepper.set_coeff(coefficient)
    helper = timestepper._get_mtswe_split_hvp_helper()
    scales = np.asarray(model.get_coeff_scaling_factors(), dtype=np.float64)
    lower, upper = model.get_coeff_bounds()
    if model.get_coeff_list() != ["s", "c0"] or scales.shape != (2,):
        raise RuntimeError("production coefficient convention changed")
    if float(scales[1]) != C0_SCALE:
        raise RuntimeError("certified c0 = 0.07 z scaling changed")
    field_sizes = tuple(int(block.size) for block in state_container[0].dat.data)
    return HiddenC0Case(
        parameters=parameters,
        model=model,
        timestepper=timestepper,
        helper=helper,
        coefficient_template=coefficient.copy(deepcopy=True),
        initial_state=_copy_function(
            state_container[0], "resolved_hidden_c0_initial_owned"
        ),
        t0=float(time),
        dt=float(configuration.dt),
        moist_backend=configuration.moist_backend,
        c0_lower=float(lower[1]),
        c0_upper=float(upper[1]),
        c0_scale=float(scales[1]),
        field_sizes=field_sizes,
    )


def _kinetic_energy(case: HiddenC0Case, state: Function) -> float:
    velocity = state.sub(0)
    height = state.sub(1)
    return float(
        0.5
        * assemble(
            height * inner(velocity, velocity) * case.model.spaces.dx
        )
    )


class ResolvedDiagnosticEvaluator:
    """Read-only state diagnostics plus an explicit hyper-child probe."""

    def __init__(self, case: HiddenC0Case, configuration):
        self.case = case
        self.configuration = configuration
        self.workspace = case.new_state("resolved_diagnostic_workspace")
        space = case.model.spaces.CG
        test = TestFunction(space)
        trial = TrialFunction(space)
        self.vorticity = Function(space, name="projected_relative_vorticity")
        problem = LinearVariationalProblem(
            inner(test, trial) * case.model.spaces.dx,
            inner(test, curl2D(self.workspace.sub(0))) * case.model.spaces.dx,
            self.vorticity,
        )
        self.vorticity_solver = LinearVariationalSolver(
            problem,
            solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
            options_prefix="resolved_c0_vorticity",
        )

    def sample_velocity(self, state: Function) -> np.ndarray:
        nx, ny = self.configuration.sampling_shape
        lx = float(self.case.model.initcond.Lx)
        ly = float(self.case.model.initcond.Ly)
        # Cell-centred samples avoid duplicated periodic endpoints.  The phase
        # shift does not alter modal energy.
        xs = (np.arange(nx, dtype=np.float64) + 0.5) * lx / nx
        ys = (np.arange(ny, dtype=np.float64) + 0.5) * ly / ny
        points = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
        sampled = np.asarray(state.sub(0).at(points), dtype=np.float64)
        return sampled.reshape(ny, nx, -1)

    def evaluate(self, state: Function, step: int, time: float):
        self.workspace.assign(state)
        self.vorticity_solver.solve()
        dx = self.case.model.spaces.dx
        vorticity_squared = float(assemble(self.vorticity * self.vorticity * dx))
        state_zero = self.case.new_state("resolved_diagnostic_state_zero")
        state_zero.assign(0)
        state_squared = _state_squared_difference(
            self.case,
            state,
            state_zero,
            "resolved_diagnostic_state_norm",
        )
        hyper_cache = self.case.helper.hyper_helper.take_forward_step_cached(
            state, time, self.case.dt
        )
        zero = self.case.new_state("resolved_diagnostic_hyper_zero")
        zero.assign(0)
        tendency_squared = _state_squared_difference(
            self.case,
            hyper_cache.tendency,
            zero,
            "resolved_diagnostic_hyper_tendency_norm",
        )
        update_squared = _state_squared_difference(
            self.case,
            hyper_cache.state_out,
            state,
            "resolved_diagnostic_hyper_update_norm",
        )
        velocity_samples = self.sample_velocity(state)
        spectrum = shell_averaged_vector_spectrum(
            velocity_samples,
            lx=float(self.case.model.initcond.Lx),
            ly=float(self.case.model.initcond.Ly),
            high_wavenumber_fraction=(
                self.configuration.high_wavenumber_fraction
            ),
        )
        values = _flat_values(state)
        diagnostic = {
            "step": int(step),
            "time": float(time),
            "kinetic_energy": _kinetic_energy(self.case, state),
            "projected_vorticity_l2": float(np.sqrt(vorticity_squared)),
            "projected_enstrophy": 0.5 * vorticity_squared,
            "mixed_state_mass_norm": float(np.sqrt(state_squared)),
            "hyperviscosity_tendency_mass_norm": float(
                np.sqrt(tendency_squared)
            ),
            "hyperviscosity_child_update_mass_norm": float(
                np.sqrt(update_squared)
            ),
            "hyperviscosity_child_kinetic_energy_change": (
                _kinetic_energy(self.case, hyper_cache.state_out)
                - _kinetic_energy(self.case, state)
            ),
            "velocity_high_wavenumber_energy_fraction": float(
                spectrum.high_wavenumber_fraction
            ),
            "all_state_coefficients_finite": bool(np.all(np.isfinite(values))),
            "minimum_height_coefficient": float(np.min(state.dat.data[1])),
        }
        return diagnostic, spectrum


def _atomic_numpy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values, dtype=np.float64), allow_pickle=False)
    temporary.replace(path)


def _snapshot_paths(output_directory: Path, step: int):
    stem = f"step_{step:08d}"
    return (
        output_directory / "restart" / f"{stem}.npy",
        output_directory / "checkpoints" / f"{stem}.h5",
        output_directory / "diagnostics" / f"{stem}.json",
        output_directory / "spectra" / f"{stem}.npz",
    )


def _write_checkpoint(
    case: HiddenC0Case,
    state: Function,
    destination: Path,
) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    with CheckpointFile(str(temporary), "w") as checkpoint:
        checkpoint.save_mesh(case.model.mesh)
        checkpoint.save_function(state, name="mixed_state", idx=0)
        for index, name in enumerate(STATE_FIELDS):
            checkpoint.save_function(state.sub(index), name=name, idx=0)
    temporary.replace(destination)


def _write_spectrum(path: Path, spectrum) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            shell=spectrum.shell,
            shell_mode_count=spectrum.shell_mode_count,
            shell_energy_sum=spectrum.shell_energy_sum,
            shell_energy_mean=spectrum.shell_energy_mean,
            physical_shell_wavenumber=spectrum.physical_shell_wavenumber,
            high_wavenumber_fraction=np.float64(
                spectrum.high_wavenumber_fraction
            ),
            parseval_mean_kinetic_energy=np.float64(
                spectrum.parseval_mean_kinetic_energy
            ),
        )
    temporary.replace(path)


def _write_snapshot(
    case,
    state,
    configuration,
    evaluator,
    output_directory,
    step,
):
    restart_path, checkpoint_path, diagnostic_path, spectrum_path = (
        _snapshot_paths(output_directory, step)
    )
    time = case.t0 + step * case.dt
    diagnostic, spectrum = evaluator.evaluate(state, step, time)
    _atomic_numpy(restart_path, _flat_values(state))
    _write_checkpoint(case, state, checkpoint_path)
    write_json_record(diagnostic_path, diagnostic)
    _write_spectrum(spectrum_path, spectrum)
    if configuration.write_vtk:
        vtk_path = output_directory / "vtk" / f"step_{step:08d}.pvd"
        writer = VTKFile(str(vtk_path))
        writer.write(
            *(state.sub(index) for index in range(len(STATE_FIELDS))),
            time=time,
        )
    return diagnostic


def _new_metadata(case, configuration):
    offsets = np.cumsum((0,) + case.field_sizes)
    config = configuration.to_dict()
    pair_physics = configuration.physics_configuration(include_c0=False)
    return {
        "format_version": configuration.format_version,
        "benchmark_stage": "Test 1B-0 resolved identifiability pilot",
        "status": "initialized",
        "configuration": config,
        "configuration_sha256": _configuration_fingerprint(config),
        "paired_non_c0_physics_sha256": _configuration_fingerprint(pair_physics),
        "case": configuration.case,
        "domain": {
            "type": "doubly_periodic_rectangle",
            "Lx": float(case.model.initcond.Lx),
            "Ly": float(case.model.initcond.Ly),
        },
        "boundary_conditions": "periodic in x and y",
        "mesh": {
            "nx": configuration.nx,
            "ny": configuration.ny,
            "quadrilateral": True,
        },
        "finite_element_spaces": {
            "v": "vector CG(3), spectral variant",
            "h": "scalar CG(3), spectral variant",
            "S": "scalar CG(3), spectral variant",
            "Qv": "scalar DG(1), spectral variant",
            "Qc": "scalar DG(1), spectral variant",
            "Qr": "scalar DG(1), spectral variant",
            "mass_measure": "configured GLL-lumped volume quadrature",
        },
        "time": {
            "t0": case.t0,
            "dt": case.dt,
            "nsteps": configuration.nsteps,
            "final_time": configuration.final_time,
            "output_steps": configuration.output_steps,
            "output_times": tuple(
                case.t0 + step * case.dt for step in configuration.output_steps
            ),
        },
        "physical_parameters": {
            "c0": configuration.c0,
            "s": configuration.s,
            "c0_normalization": "c0 = 0.07 z",
            "threewayphysics": deepcopy(
                case.parameters["threewayphysics"]
            ),
        },
        "moist_backend": configuration.moist_backend,
        "initial_condition": {
            "identity": configuration.case,
            "source": CANDIDATE_CASES[configuration.case].source,
            "repository_parameters": deepcopy(
                case.parameters["initial-conditions"]
            ),
        },
        "forcing_configuration": tuple(case.parameters["model"]["forcing_terms"]),
        "solver": {
            "identity": "production_firedrake_mtswe_lie_split",
            "timestepper_list": tuple(case.timestepper.timestepper_list),
            "termlist": tuple(tuple(value) for value in case.timestepper.termlist),
            "subcycle_list": tuple(case.timestepper.subcycle_list),
            "six_child_order": (
                "dry_rk4_0",
                "dry_rk4_1",
                "hyperviscosity_euler",
                "dg_ssprk43_0",
                "dg_ssprk43_1",
                "moist_euler",
            ),
            "serial_cpu_only": True,
        },
        "state_convention": {
            "fields": STATE_FIELDS,
            "field_sizes": case.field_sizes,
            "field_slices": {
                name: (int(offsets[index]), int(offsets[index + 1]))
                for index, name in enumerate(STATE_FIELDS)
            },
            "restart_array": "mixed dat blocks in field order",
        },
        "spectral_diagnostic": {
            "sampling": (
                "physical velocity evaluated at cell-centred points on a "
                "uniform periodic grid"
            ),
            "sampling_shape_nx_ny": configuration.sampling_shape,
            "fft_normalization": "numpy fft2 norm='forward'",
            "wavenumber": "integer cycles and physical 2*pi*k/L",
            "shells": "nearest-integer radial mode; sum and mean retained",
            "vector_treatment": "0.5 times sum of component modal magnitudes",
            "solver_role": "diagnostic only; never fed back into the solver",
        },
        "hyperviscosity_diagnostic": {
            "primary_proxy": (
                "mixed mass norm of the actual deployed hyperviscosity-child "
                "tendency at the saved state"
            ),
            "secondary_proxy": (
                "mixed mass norm of the child update and child-only kinetic "
                "energy change"
            ),
        },
        "vorticity_diagnostic": (
            "CG(3) L2 projection of repository curl2D(v), matching the native "
            "diagnostic construction"
        ),
        "restartability": {
            "kind": "experiment/state restart snapshots",
            "adjoint_checkpointing": False,
            "revolve": False,
            "incomplete_runs_resume_from": "latest valid output-stride state array",
        },
        "random_seed": configuration.seed,
        "git": {
            "checkpoint": _git_value(("rev-parse", "HEAD")),
            "branch": _git_value(("branch", "--show-current")),
        },
        "completed_output_steps": (),
        "diagnostics": (),
        "failure_reason": None,
        "wall_time_seconds": 0.0,
    }


def _latest_restart(output_directory: Path, expected_size: int):
    candidates = sorted((output_directory / "restart").glob("step_*.npy"))
    for path in reversed(candidates):
        try:
            step = int(path.stem.split("_")[1])
            values = np.load(path, allow_pickle=False)
        except (OSError, ValueError, IndexError):
            continue
        if values.shape == (expected_size,) and np.all(np.isfinite(values)):
            return step, np.asarray(values, dtype=np.float64)
    return None


def run_resolved_hidden_c0(
    configuration: ResolvedPilotConfiguration,
    *,
    skip_completed: bool = True,
):
    """Run or resume one deterministic complete-production pilot trajectory."""
    if not isinstance(configuration, ResolvedPilotConfiguration):
        raise TypeError("configuration must be ResolvedPilotConfiguration")
    output_directory = Path(configuration.output_directory).resolve()
    metadata_path = output_directory / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = _configuration_fingerprint(configuration.to_dict())
        if existing.get("configuration_sha256") != expected:
            raise RuntimeError(
                f"existing output {output_directory} belongs to another configuration"
            )
        if existing.get("status") == "complete" and skip_completed:
            required = _snapshot_paths(output_directory, configuration.nsteps)
            expected_size = sum(existing["state_convention"]["field_sizes"])
            restart = _latest_restart(output_directory, expected_size)
            if all(path.exists() for path in required) and (
                restart is not None and restart[0] == configuration.nsteps
            ):
                return existing

    for subdirectory in (
        "restart",
        "checkpoints",
        "diagnostics",
        "spectra",
        "vtk",
    ):
        (output_directory / subdirectory).mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    case = build_resolved_hidden_c0_case(configuration)
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else _new_metadata(case, configuration)
    )
    metadata["status"] = "running"
    metadata["failure_reason"] = None
    write_json_record(metadata_path, metadata)
    evaluator = ResolvedDiagnosticEvaluator(case, configuration)
    restart = _latest_restart(output_directory, sum(case.field_sizes))
    if restart is None:
        current_step = 0
        current = _copy_function(case.initial_state, "resolved_pilot_state_0")
    else:
        current_step, values = restart
        current = case.state_from_values(values, f"resolved_pilot_restart_{current_step}")
    diagnostics = {}
    for item in metadata.get("diagnostics", ()):
        step = int(item["step"])
        if all(path.exists() for path in _snapshot_paths(output_directory, step)):
            diagnostics[step] = item

    try:
        with case.physical_c0(configuration.c0):
            if current_step in configuration.output_steps and current_step not in diagnostics:
                diagnostics[current_step] = _write_snapshot(
                    case,
                    current,
                    configuration,
                    evaluator,
                    output_directory,
                    current_step,
                )
            for step in range(current_step + 1, configuration.nsteps + 1):
                cache = case.helper.take_forward_step_cached(
                    current,
                    case.t0 + (step - 1) * case.dt,
                    case.dt,
                )
                current = _copy_function(
                    cache.state_out, f"resolved_pilot_state_{step}"
                )
                if step in configuration.output_steps:
                    diagnostic = _write_snapshot(
                        case,
                        current,
                        configuration,
                        evaluator,
                        output_directory,
                        step,
                    )
                    diagnostics[step] = diagnostic
                    metadata["completed_output_steps"] = tuple(sorted(diagnostics))
                    metadata["diagnostics"] = tuple(
                        diagnostics[index] for index in sorted(diagnostics)
                    )
                    metadata["wall_time_seconds"] = (
                        float(metadata.get("wall_time_seconds", 0.0))
                        + perf_counter()
                        - started
                    )
                    write_json_record(metadata_path, metadata)
                    started = perf_counter()
                    if not diagnostic["all_state_coefficients_finite"]:
                        raise FloatingPointError(
                            f"nonfinite state detected at step {step}"
                        )
        if tuple(sorted(diagnostics)) != configuration.output_steps:
            raise RuntimeError("run ended without every configured output snapshot")
        metadata["status"] = "complete"
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        metadata["failure_reason"] = "KeyboardInterrupt"
        raise
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["failure_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["completed_output_steps"] = tuple(sorted(diagnostics))
        metadata["diagnostics"] = tuple(
            diagnostics[index] for index in sorted(diagnostics)
        )
        metadata["wall_time_seconds"] = (
            float(metadata.get("wall_time_seconds", 0.0))
            + perf_counter()
            - started
        )
        write_json_record(metadata_path, metadata)
    return metadata


def run_paired_resolved_hidden_c0(
    base: ResolvedPilotConfiguration,
    *,
    c0_a: float = 0.07,
    c0_b: float = 0.14,
    parent_directory: str | Path | None = None,
):
    """Run the identical-physics pair sequentially for predictable CPU use."""
    left, right = paired_pilot_configurations(
        base,
        c0_a,
        c0_b,
        parent_directory=parent_directory,
    )
    return (
        run_resolved_hidden_c0(left),
        run_resolved_hidden_c0(right),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run serial complete-production resolved hidden-c0 pilots"
    )
    parser.add_argument("command", choices=("run", "run-pair"))
    parser.add_argument("--case", default="doublevortex", choices=CANDIDATE_CASES)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--dt", type=float, default=400.0)
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument("--final-time", type=float)
    parser.add_argument("--output-stride", type=int, default=2)
    parser.add_argument("--c0", type=float, default=0.07)
    parser.add_argument("--c0-a", type=float, default=0.07)
    parser.add_argument("--c0-b", type=float, default=0.14)
    parser.add_argument("--s", type=float, default=3.2)
    parser.add_argument("--moist-backend", choices=("ufl", "jax"), default="ufl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--initial-moisture-zeta",
        type=float,
        default=0.0,
        help=(
            "signed initial saturation deficit in Qv=h*(1-zeta)*qsat; "
            "negative values are supersaturated"
        ),
    )
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--base-config")
    parser.add_argument("--spectral-nx", type=int)
    parser.add_argument("--spectral-ny", type=int)
    parser.add_argument("--high-wavenumber-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--write-vtk", action="store_true")
    return parser


def main(argv=None) -> int:
    arguments = _parser().parse_args(argv)
    nsteps = arguments.nsteps
    if arguments.final_time is not None:
        ratio = arguments.final_time / arguments.dt
        rounded = int(round(ratio))
        if not np.isclose(ratio, rounded, rtol=0.0, atol=1.0e-12):
            raise ValueError("final-time must be an integer multiple of dt")
        nsteps = rounded
    configuration = ResolvedPilotConfiguration(
        case=arguments.case,
        nx=arguments.nx,
        ny=arguments.ny if arguments.ny is not None else arguments.nx,
        dt=arguments.dt,
        nsteps=nsteps,
        output_stride=arguments.output_stride,
        c0=arguments.c0,
        s=arguments.s,
        moist_backend=arguments.moist_backend,
        seed=arguments.seed,
        initial_moisture_zeta=arguments.initial_moisture_zeta,
        output_directory=arguments.output_directory,
        base_config=arguments.base_config,
        spectral_nx=arguments.spectral_nx,
        spectral_ny=arguments.spectral_ny,
        high_wavenumber_fraction=arguments.high_wavenumber_fraction,
        write_vtk=arguments.write_vtk,
    )
    if arguments.command == "run":
        metadata = run_resolved_hidden_c0(configuration)
        print(json.dumps({"status": metadata["status"], "output": arguments.output_directory}))
    else:
        left, right = run_paired_resolved_hidden_c0(
            configuration,
            c0_a=arguments.c0_a,
            c0_b=arguments.c0_b,
            parent_directory=arguments.output_directory,
        )
        print(json.dumps({"left": left["status"], "right": right["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ResolvedDiagnosticEvaluator",
    "build_resolved_hidden_c0_case",
    "resolved_hidden_c0_parameters",
    "run_paired_resolved_hidden_c0",
    "run_resolved_hidden_c0",
)
