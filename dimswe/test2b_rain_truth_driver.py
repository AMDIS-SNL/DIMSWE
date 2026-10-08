"""Read-only Firedrake audit of an existing Test2B/Test2A truth run.

This command reconstructs the accepted production case, loads only saved
restart arrays, and replays one complete step per saved state solely to expose
the exact post-children-1..5 state at which the analytical moist child acts.
It never writes a state or changes a model parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from firedrake import assemble

from .hidden_c0 import STATE_FIELDS, _serial_solver_parameters
from .jax_moist_adapter import JAXMoistEulerPrimal
from .resolved_hidden_c0 import ResolvedPilotConfiguration, write_json_record
from .resolved_hidden_c0_driver import build_resolved_hidden_c0_case
from .test2a_apriori_autonomous import rain_activity_diagnostic
from .test2b_rain_truth import source_invariant_residuals, summarize_activity_records


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mass(case, field) -> float:
    return float(assemble(field * case.model.spaces.dx))


def _field_mass(case, state, index: int) -> float:
    return _mass(case, state.sub(index))


def _configuration_from_metadata(metadata, run_directory: Path):
    values = dict(metadata["configuration"])
    values["output_directory"] = str(run_directory)
    return ResolvedPilotConfiguration(**values)


def audit_existing_truth_run(run_directory, *, output, use_jit=True):
    """Audit existing restart snapshots; no trajectory is integrated or saved."""
    started = perf_counter()
    root = Path(run_directory).resolve()
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise RuntimeError("truth audit requires a completed source run")
    configuration = _configuration_from_metadata(metadata, root)
    if tuple(metadata["solver"]["six_child_order"]) != (
        "dry_rk4_0",
        "dry_rk4_1",
        "hyperviscosity_euler",
        "dg_ssprk43_0",
        "dg_ssprk43_1",
        "moist_euler",
    ):
        raise RuntimeError("truth run does not use the accepted six-child split")
    case = build_resolved_hidden_c0_case(configuration)
    adapter = JAXMoistEulerPrimal(
        case.model, _serial_solver_parameters(), use_jit=use_jit
    )
    field_index = {name: index for index, name in enumerate(STATE_FIELDS)}
    diagnostic_by_step = {
        int(record["step"]): record for record in metadata["diagnostics"]
    }
    raw = []
    rate_arrays = []
    a_scale = 0.0
    beta2 = None
    source_residual_max = {
        "water_maximum_absolute": 0.0,
        "water_rms": 0.0,
        "S_minus_beta2_Qv_maximum_absolute": 0.0,
        "S_minus_beta2_Qv_rms": 0.0,
    }

    for step in configuration.output_steps:
        restart_path = root / "restart" / f"step_{step:08d}.npy"
        values = np.load(restart_path, allow_pickle=False)
        state = case.state_from_values(values, f"test2b_truth_audit_{step}")
        time = float(case.t0 + step * case.dt)

        # The full production replay is used only to recover boundary 5.  The
        # analytical local diagnostic below uses the certified JAX/UFL-equivalent
        # law at exactly that state.
        complete = case.helper.take_forward_step_cached(state, time, case.dt)
        post_prefix = complete.boundary_states[-2]
        moist = adapter.evaluate(post_prefix, case.dt)
        parameters = {
            name: float(np.asarray(value))
            for name, value in moist.parameters.items()
        }
        current_beta2 = parameters["g"] * parameters["L"]
        if beta2 is None:
            beta2 = current_beta2
        elif beta2 != current_beta2:
            raise RuntimeError("beta2 changed within a truth run")

        h = np.asarray(moist.packed_state["h"], dtype=np.float64)
        entropy = np.asarray(moist.packed_state["S"], dtype=np.float64) / h
        qv_density = np.asarray(moist.packed_state["Qv"], dtype=np.float64)
        qc_density = np.asarray(moist.packed_state["Qc"], dtype=np.float64)
        _, qr = adapter.interpolate_and_pack(
            post_prefix.sub(field_index["Qr"]),
            f"test2b_truth_audit_Qr_{step}",
        )
        qr_density = np.asarray(qr, dtype=np.float64)
        qv_specific = qv_density / h
        specific_qc = qc_density / h
        qr_specific = qr_density / h
        qsat = (
            parameters["q0"]
            * parameters["H0"]
            / (h + np.asarray(moist.packed_fields["B"], dtype=np.float64))
            * np.exp(20.0 * (1.0 - entropy / parameters["g"]))
        )
        saturation_ratio = qv_specific / qsat
        a_rate = np.asarray(moist.rates["A"], dtype=np.float64)
        r_rate = np.asarray(moist.rates["R"], dtype=np.float64)
        source = {
            name: np.asarray(moist.source_density[name], dtype=np.float64)
            for name in ("S", "Qv", "Qc", "Qr")
        }
        invariants = source_invariant_residuals(source, current_beta2)
        for name, value in invariants.items():
            source_residual_max[name] = max(source_residual_max[name], value)

        water_mass = sum(
            _field_mass(case, state, field_index[name])
            for name in ("Qv", "Qc", "Qr")
        )
        rain_mass = _field_mass(case, state, field_index["Qr"])
        cloud_mass = _field_mass(case, state, field_index["Qc"])
        rain_source_mass_rate = _field_mass(
            case, moist.tendency, field_index["Qr"]
        )
        moist_water_increment = sum(
            _field_mass(case, moist.state_out, field_index[name])
            - _field_mass(case, post_prefix, field_index[name])
            for name in ("Qv", "Qc", "Qr")
        )
        moist_thermo_increment = (
            _field_mass(case, moist.state_out, field_index["S"])
            - _field_mass(case, post_prefix, field_index["S"])
            - current_beta2
            * (
                _field_mass(case, moist.state_out, field_index["Qv"])
                - _field_mass(case, post_prefix, field_index["Qv"])
            )
        )
        a_scale = max(a_scale, float(np.max(np.abs(a_rate))))
        rate_arrays.append((r_rate.copy(), h.copy(), np.asarray(qr).copy()))
        native = diagnostic_by_step[int(step)]
        raw.append(
            {
                "step": int(step),
                "time": time,
                "specific_Qc_maximum": float(np.max(specific_qc)),
                "specific_Qc_rms": float(
                    np.sqrt(np.mean(specific_qc * specific_qc))
                ),
                "specific_Qc_threshold_margin": float(
                    parameters["qprecip"] - np.max(specific_qc)
                ),
                "specific_Qv_minimum": float(np.min(qv_specific)),
                "specific_Qv_maximum": float(np.max(qv_specific)),
                "specific_Qr_minimum": float(np.min(qr_specific)),
                "specific_Qr_maximum": float(np.max(qr_specific)),
                "saturation_ratio_minimum": float(
                    np.min(saturation_ratio)
                ),
                "saturation_ratio_maximum": float(
                    np.max(saturation_ratio)
                ),
                "supersaturation_maximum": float(
                    np.max(saturation_ratio - 1.0)
                ),
                "supersaturation_rms": float(
                    np.sqrt(np.mean((saturation_ratio - 1.0) ** 2))
                ),
                "Qc_mass": cloud_mass,
                "Qc_maximum_at_post_prefix_GLL": float(np.max(qc_density)),
                "Qc_rms_at_post_prefix_GLL": float(
                    np.sqrt(np.mean(qc_density * qc_density))
                ),
                "A_maximum_absolute": float(np.max(np.abs(a_rate))),
                "A_rms": float(np.sqrt(np.mean(a_rate * a_rate))),
                "R_maximum_absolute": float(np.max(np.abs(r_rate))),
                "R_rms": float(np.sqrt(np.mean(r_rate * r_rate))),
                "R_exact_nonzero_fraction": float(np.mean(r_rate != 0.0)),
                "rain_water_mass": rain_mass,
                "rain_source_mass_rate": rain_source_mass_rate,
                "rain_source_mass_increment": float(
                    case.dt * rain_source_mass_rate
                ),
                "applied_to_saved_trajectory": bool(
                    step < configuration.nsteps
                ),
                "total_water_mass": water_mass,
                "moist_child_total_water_mass_increment": moist_water_increment,
                "moist_child_S_minus_beta2_Qv_mass_increment": (
                    moist_thermo_increment
                ),
                "minimum_h_GLL": float(np.min(h)),
                "minimum_Qv_GLL": float(np.min(moist.packed_state["Qv"])),
                "minimum_Qc_GLL": float(np.min(qc_density)),
                "minimum_Qr_GLL": float(np.min(qr)),
                "maximum_h_GLL": float(np.max(h)),
                "maximum_Qv_GLL": float(np.max(qv_density)),
                "maximum_Qc_GLL": float(np.max(qc_density)),
                "maximum_Qr_GLL": float(np.max(qr_density)),
                "kinetic_energy": float(native["kinetic_energy"]),
                "projected_enstrophy": float(native["projected_enstrophy"]),
                "all_state_coefficients_finite": bool(
                    native["all_state_coefficients_finite"]
                ),
                "source_invariants": invariants,
            }
        )

    comparison_scale = max(a_scale, np.finfo(np.float64).tiny)
    for record, (r_rate, h, qr) in zip(raw, rate_arrays):
        rain = rain_activity_diagnostic(
            r_rate,
            h,
            qr,
            case.dt,
            comparison_scale,
            float64_scale_multiplier=64.0,
            physical_increment_relative_threshold=1.0e-12,
        )
        record.update(
            {
                "R_float64_scale_tolerance": rain[
                    "float64_scale_tolerance"
                ],
                "R_above_float64_scale_fraction": rain[
                    "above_float64_scale_fraction"
                ],
                "physically_meaningful_R_fraction": rain[
                    "physically_meaningful_fraction"
                ],
                "maximum_absolute_Qr_increment": rain[
                    "maximum_absolute_Qr_increment"
                ],
            }
        )

    summary = summarize_activity_records(
        raw, qprecip=float(parameters["qprecip"])
    )
    result = {
        "status": "complete",
        "diagnostic": "Test2B rain-active truth read-only preparation audit",
        "source_run": str(root),
        "source_metadata_sha256": _file_sha256(metadata_path),
        "source_run_status": metadata["status"],
        "source_configuration_sha256": metadata["configuration_sha256"],
        "state_access": {
            "saved_steps": list(configuration.output_steps),
            "restart_arrays_only": True,
            "parameters_modified": False,
            "output_states_written": False,
            "post_prefix_replay": (
                "one independent accepted six-child step per saved state; only "
                "boundary 5 and analytical local rates are retained"
            ),
        },
        "mesh": metadata["mesh"],
        "time": metadata["time"],
        "moist_parameters": {**parameters, "beta2": beta2},
        "deployed_GLL_samples_per_state": int(
            adapter.layout.owned_cell_count * adapter.layout.points_per_cell
        ),
        "activity_tolerance": {
            "comparison_rate_scale": comparison_scale,
            "float64_scale_multiplier": 64.0,
            "physical_increment_relative_threshold": 1.0e-12,
            "source": "accepted Test2A autonomous-rain diagnostic contract",
        },
        "summary": summary,
        "maximum_source_invariant_residuals": source_residual_max,
        "records": raw,
        "wall_time_seconds": float(perf_counter() - started),
    }
    write_json_record(output, result)
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit-run",))
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-jit", action="store_true")
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    result = audit_existing_truth_run(
        arguments.run, output=arguments.output, use_jit=not arguments.no_jit
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("audit_existing_truth_run",)
