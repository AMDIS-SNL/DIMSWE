"""Quantitative Test-2 audit on the exact deployed moist GLL representation.

The numerical summaries and plotting helpers are Firedrake-free.  Firedrake
and the certified J1 adapter are imported only by the explicitly invoked audit
driver.  No trajectory is integrated: existing truth snapshots 0 through 80
are loaded read-only and evaluated independently.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .resolved_hidden_c0 import resolved_truth_state_indices, write_json_record
from .selected_test1b import load_selected_test1b_plan


PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)
ACTIVITY_SCALES = (1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3)
SOURCE_FIELDS = ("Qv", "Qc", "Qr", "S")
INPUT_FIELDS = ("h", "S", "Qv", "Qc", "B")
TRAINING_START_STEP = 0
TRAINING_STOP_STEP = 80


def _finite_array(name, values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite array")
    return array


def _percentile_record(values, *, absolute=False):
    array = _finite_array("percentile values", values).reshape(-1)
    if absolute:
        array = np.abs(array)
    quantiles = np.percentile(array, PERCENTILES)
    return {
        str(percentile): float(value)
        for percentile, value in zip(PERCENTILES, quantiles)
    }


def rate_statistics(values):
    """Return deterministic signed and scale-relative activity statistics."""
    array = _finite_array("rate values", values).reshape(-1)
    absolute = np.abs(array)
    maximum_absolute = float(np.max(absolute))
    if maximum_absolute == 0.0:
        activity = {f"{scale:.0e}": 0.0 for scale in ACTIVITY_SCALES}
        activity_contract = "max_abs_rate_is_exactly_zero"
    else:
        activity = {
            f"{scale:.0e}": float(
                np.mean(absolute > scale * maximum_absolute)
            )
            for scale in ACTIVITY_SCALES
        }
        activity_contract = "fraction(abs(rate) > scale * max_abs_rate)"
    return {
        "sample_count": int(array.size),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "maximum_absolute": maximum_absolute,
        "mean": float(np.mean(array)),
        "mean_absolute": float(np.mean(absolute)),
        "rms": float(np.sqrt(np.mean(array * array))),
        "standard_deviation": float(np.std(array)),
        "exact_zero_fraction": float(np.mean(array == 0.0)),
        "positive_fraction": float(np.mean(array > 0.0)),
        "negative_fraction": float(np.mean(array < 0.0)),
        "percentiles": _percentile_record(array),
        "absolute_percentiles": _percentile_record(array, absolute=True),
        "scale_relative_activity_fractions": activity,
        "scale_relative_activity_contract": activity_contract,
    }


def time_rate_statistics(values, global_maximum_absolute):
    """Return one-state rate statistics using global-scale activity levels."""
    array = _finite_array("one-state rate values", values).reshape(-1)
    absolute = np.abs(array)
    maximum = float(global_maximum_absolute)
    if maximum < 0.0 or not np.isfinite(maximum):
        raise ValueError("global maximum absolute rate must be nonnegative")
    if maximum == 0.0:
        activity = {"1e-06": 0.0, "1e-03": 0.0}
    else:
        activity = {
            "1e-06": float(np.mean(absolute > 1.0e-6 * maximum)),
            "1e-03": float(np.mean(absolute > 1.0e-3 * maximum)),
        }
    return {
        "maximum_absolute": float(np.max(absolute)),
        "mean_absolute": float(np.mean(absolute)),
        "rms": float(np.sqrt(np.mean(array * array))),
        "exact_zero_fraction": float(np.mean(array == 0.0)),
        "positive_fraction": float(np.mean(array > 0.0)),
        "negative_fraction": float(np.mean(array < 0.0)),
        "global_scale_relative_activity_fractions": activity,
    }


def magnitude_statistics(values):
    array = _finite_array("source values", values).reshape(-1)
    absolute = np.abs(array)
    return {
        "sample_count": int(array.size),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean_absolute": float(np.mean(absolute)),
        "rms": float(np.sqrt(np.mean(array * array))),
        "absolute_percentiles": _percentile_record(array, absolute=True),
    }


def input_support_statistics(values):
    array = _finite_array("input values", values).reshape(-1)
    return {
        "sample_count": int(array.size),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "percentiles": _percentile_record(array),
    }


def moist_source_terms(h, a_rate, r_rate, beta2):
    """Mirror only the certified structural A/R-to-source identities."""
    h_array = _finite_array("h", h)
    a_array = _finite_array("A", a_rate)
    r_array = _finite_array("R", r_rate)
    if h_array.shape != a_array.shape or h_array.shape != r_array.shape:
        raise ValueError("h, A, and R must have identical deployed shapes")
    beta = float(beta2)
    if not np.isfinite(beta):
        raise ValueError("beta2 must be finite")
    return {
        "Qv": h_array * a_array,
        "Qc": -h_array * (a_array + r_array),
        "Qr": h_array * r_array,
        "S": h_array * beta * a_array,
    }


def deployed_sample_accounting(
    *, states, cells, points_per_cell, shared_points_repeated=True
):
    for name, value in (
        ("states", states),
        ("cells", cells),
        ("points_per_cell", points_per_cell),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    samples_per_state = cells * points_per_cell
    return {
        "stored_states_examined": states,
        "number_of_cells": cells,
        "gll_points_per_cell": points_per_cell,
        "gll_tensor_shape_per_cell": (4, 4),
        "samples_per_state": samples_per_state,
        "total_space_time_samples": states * samples_per_state,
        "shared_cg_boundary_points_repeated": bool(shared_points_repeated),
        "deduplicated": False,
        "representation": (
            "cell-major broken-CG3 carrier with the first physical coordinate "
            "varying fastest inside each 4x4 GLL cell block"
        ),
    }


def _explicit_ratio(numerator, denominator):
    left = float(numerator)
    right = float(denominator)
    if not np.isfinite(left) or not np.isfinite(right) or left < 0.0 or right < 0.0:
        raise ValueError("ratio magnitudes must be finite and nonnegative")
    if right == 0.0:
        return {
            "value": 0.0 if left == 0.0 else None,
            "reference_scale_zero": True,
        }
    return {"value": left / right, "reference_scale_zero": False}


def _require_training_state_keys(states):
    actual = tuple(int(step) for step in states)
    expected = tuple(range(TRAINING_START_STEP, TRAINING_STOP_STEP + 1))
    if actual != expected:
        raise ValueError(
            "Test-2 audit requires exactly truth states 0..80 in order; "
            f"received {actual[:1]}..{actual[-1:] if actual else ()}"
        )
    return expected


def classify_moist_activity(rate_summaries, time_summaries, source_effects):
    """Apply transparent screening criteria, not a universal scientific pass."""
    evidence = {}
    for rate, exclusive_field in (("A", "Qv"), ("R", "Qr")):
        summary = rate_summaries[rate]
        times = time_summaries[rate]
        maximum = summary["maximum_absolute"]
        active_time_fraction = (
            0.0
            if maximum == 0.0
            else float(
                np.mean(
                    [
                        record["maximum_absolute"] > 1.0e-6 * maximum
                        for record in times
                    ]
                )
            )
        )
        sample_fraction = summary["scale_relative_activity_fractions"]["1e-06"]
        increment_ratio = source_effects[exclusive_field][
            "global_rms_increment_over_truth_rms"
        ]["value"]
        nondegenerate = (
            maximum > 0.0
            and sample_fraction >= 1.0e-4
            and active_time_fraction >= 0.05
        )
        measurable_increment = (
            increment_ratio is not None
            and increment_ratio > 100.0 * np.finfo(np.float64).eps
        )
        evidence[rate] = {
            "maximum_absolute_nonzero": maximum > 0.0,
            "fraction_above_1e-6_global_max": sample_fraction,
            "fraction_of_states_active_at_1e-6_global_max": active_time_fraction,
            "exclusive_field_global_rms_increment_ratio": increment_ratio,
            "nondegenerate_space_time_support": nondegenerate,
            "increment_above_100_float64_eps": measurable_increment,
            "screened_active": nondegenerate and measurable_increment,
        }
    a_active = evidence["A"]["screened_active"]
    r_active = evidence["R"]["screened_active"]
    if a_active and r_active:
        classification = "RICH_TWO_RATE_SIGNAL"
    elif a_active and not r_active:
        classification = "A_ACTIVE_R_WEAK"
    elif r_active and not a_active:
        classification = "R_ACTIVE_A_WEAK"
    elif not a_active and not r_active:
        classification = "BOTH_WEAK_OR_DEGENERATE"
    else:
        classification = "NEEDS_SCIENTIFIC_REVIEW"
    return {
        "classification": classification,
        "evidence": evidence,
        "screening_contract": {
            "minimum_space_time_sample_fraction_at_1e-6_max": 1.0e-4,
            "minimum_active_state_fraction_at_1e-6_max": 0.05,
            "minimum_exclusive_increment_ratio": (
                100.0 * np.finfo(np.float64).eps
            ),
            "interpretation": (
                "transparent degeneracy/numerical-effect screen only; not a "
                "universal neural-benchmark pass threshold"
            ),
        },
    }


def _field_mass_update_ratio(case, state_in, state_out, field_index):
    from firedrake import assemble, inner

    difference = state_in.sub(field_index).copy(deepcopy=True)
    difference.assign(state_out.sub(field_index) - state_in.sub(field_index))
    update_squared = float(
        assemble(inner(difference, difference) * case.model.spaces.dx)
    )
    reference_squared = float(
        assemble(
            inner(state_in.sub(field_index), state_in.sub(field_index))
            * case.model.spaces.dx
        )
    )
    return _explicit_ratio(
        float(np.sqrt(max(update_squared, 0.0))),
        float(np.sqrt(max(reference_squared, 0.0))),
    )


def _source_effect_summary(
    source_values,
    reference_values,
    per_time_ratios,
    exact_mass_ratios,
    dt,
):
    source = _finite_array("source values", source_values).reshape(-1)
    reference = _finite_array("truth field values", reference_values).reshape(-1)
    if source.shape != reference.shape:
        raise ValueError("source and truth-field samples must align")
    increment = float(dt) * source
    increment_rms = float(np.sqrt(np.mean(increment * increment)))
    reference_rms = float(np.sqrt(np.mean(reference * reference)))
    increment_maximum = float(np.max(np.abs(increment)))
    reference_maximum = float(np.max(np.abs(reference)))
    return {
        "euler_increment_statistics": magnitude_statistics(increment),
        "truth_field_scale": {
            "rms": reference_rms,
            "mean_absolute": float(np.mean(np.abs(reference))),
            "maximum_absolute": reference_maximum,
        },
        "global_rms_increment_over_truth_rms": _explicit_ratio(
            increment_rms, reference_rms
        ),
        "global_max_increment_over_truth_max": _explicit_ratio(
            increment_maximum, reference_maximum
        ),
        "per_state_gll_rms_increment_over_truth_rms": tuple(per_time_ratios),
        "per_state_exact_child_mass_update_over_truth_mass_norm": tuple(
            exact_mass_ratios
        ),
        "normalization_contract": (
            "dt times the exact GLL source is compared with the same field at "
            "the same GLL samples; the second record uses the certified weak "
            "assembly and mixed mass solve"
        ),
    }


def run_moist_activity_audit(
    truth_run,
    selected_plan,
    *,
    output,
    plot_directory=None,
    use_jit=True,
):
    """Evaluate existing truth states 0..80 without advancing the PDE."""
    from firedrake import SpatialCoordinate

    from .hidden_c0 import STATE_FIELDS, _serial_solver_parameters
    from .jax_moist_adapter import JAXMoistEulerPrimal
    from .resolved_hidden_c0_inference import load_resolved_truth

    started = perf_counter()
    truth_directory = Path(truth_run).resolve()
    plan_path = Path(selected_plan).resolve()
    _, selected = load_selected_test1b_plan(plan_path)
    configuration = selected.inference_configuration(truth_directory)
    if (
        configuration.training_start_step,
        configuration.training_stop_step,
    ) != (TRAINING_START_STEP, TRAINING_STOP_STEP):
        raise ValueError("Test-2 moist audit is fixed to truth states 0..80")
    required_steps = resolved_truth_state_indices(
        configuration, include_heldout=False
    )
    if required_steps != tuple(range(81)):
        raise AssertionError("training-only truth loader requested a future state")
    case, trajectory = load_resolved_truth(
        configuration, include_heldout=False
    )
    steps = _require_training_state_keys(trajectory.states)

    adapter = JAXMoistEulerPrimal(
        case.model,
        _serial_solver_parameters(),
        use_jit=use_jit,
    )
    layout = adapter.layout
    accounting = deployed_sample_accounting(
        states=len(steps),
        cells=layout.owned_cell_count,
        points_per_cell=layout.points_per_cell,
    )
    if accounting["samples_per_state"] != selected.nx * selected.ny * 16:
        raise AssertionError("deployed GLL sample count disagrees with selected mesh")

    coordinates = SpatialCoordinate(case.model.mesh)
    _, packed_x = adapter.interpolate_and_pack(
        coordinates[0], "test2_moist_activity_x"
    )
    _, packed_y = adapter.interpolate_and_pack(
        coordinates[1], "test2_moist_activity_y"
    )

    rate_by_step = {"A": {}, "R": {}}
    source_by_field = {name: [] for name in SOURCE_FIELDS}
    reference_by_field = {name: [] for name in SOURCE_FIELDS}
    input_by_field = {name: [] for name in INPUT_FIELDS}
    gll_time_ratios = {name: [] for name in SOURCE_FIELDS}
    mass_time_ratios = {name: [] for name in SOURCE_FIELDS}
    identity_maximum_errors = {name: 0.0 for name in SOURCE_FIELDS}
    parameters = None
    field_indices = {
        name: index for index, name in enumerate(STATE_FIELDS)
    }

    for step in steps:
        state = trajectory.states[step]
        cache = adapter.evaluate(state, case.dt)
        if cache.rates["A"].shape != (
            layout.owned_cell_count,
            layout.points_per_cell,
        ):
            raise AssertionError("J1 moist cache returned the wrong deployed shape")
        current_parameters = {
            key: float(np.asarray(value))
            for key, value in cache.parameters.items()
        }
        if parameters is None:
            parameters = current_parameters
        elif current_parameters != parameters:
            raise ValueError("moist parameters changed across the truth trajectory")
        beta2 = parameters["g"] * parameters["L"]

        h_values = np.asarray(cache.packed_state["h"], dtype=np.float64)
        a_values = np.asarray(cache.rates["A"], dtype=np.float64)
        r_values = np.asarray(cache.rates["R"], dtype=np.float64)
        expected_sources = moist_source_terms(
            h_values, a_values, r_values, beta2
        )
        for name in SOURCE_FIELDS:
            actual = np.asarray(cache.source_density[name], dtype=np.float64)
            discrepancy = float(np.max(np.abs(actual - expected_sources[name])))
            identity_maximum_errors[name] = max(
                identity_maximum_errors[name], discrepancy
            )
            scale = max(float(np.max(np.abs(actual))), 1.0e-300)
            if discrepancy > 64.0 * np.finfo(np.float64).eps * scale:
                raise AssertionError(f"certified source identity failed for {name}")
            source_by_field[name].append(actual.copy())

        rate_by_step["A"][step] = a_values.copy()
        rate_by_step["R"][step] = r_values.copy()
        for name in ("h", "S", "Qv", "Qc"):
            input_by_field[name].append(
                np.asarray(cache.packed_state[name], dtype=np.float64).copy()
            )
        input_by_field["B"].append(
            np.asarray(cache.packed_fields["B"], dtype=np.float64).copy()
        )

        packed_reference = {
            "S": np.asarray(cache.packed_state["S"], dtype=np.float64),
            "Qv": np.asarray(cache.packed_state["Qv"], dtype=np.float64),
            "Qc": np.asarray(cache.packed_state["Qc"], dtype=np.float64),
        }
        _, packed_reference["Qr"] = adapter.interpolate_and_pack(
            state.sub(field_indices["Qr"]),
            f"test2_moist_activity_Qr_{step}",
        )
        for name in SOURCE_FIELDS:
            reference = np.asarray(packed_reference[name], dtype=np.float64)
            reference_by_field[name].append(reference.copy())
            increment = case.dt * np.asarray(
                cache.source_density[name], dtype=np.float64
            )
            gll_time_ratios[name].append(
                {
                    "step": int(step),
                    "time": float(case.t0 + step * case.dt),
                    **_explicit_ratio(
                        float(np.sqrt(np.mean(increment * increment))),
                        float(np.sqrt(np.mean(reference * reference))),
                    ),
                }
            )
            mass_time_ratios[name].append(
                {
                    "step": int(step),
                    "time": float(case.t0 + step * case.dt),
                    **_field_mass_update_ratio(
                        case,
                        state,
                        cache.state_out,
                        field_indices[name],
                    ),
                }
            )

    rate_arrays = {
        rate: np.concatenate(
            tuple(rate_by_step[rate][step].reshape(-1) for step in steps)
        )
        for rate in ("A", "R")
    }
    rate_summaries = {
        rate: rate_statistics(values) for rate, values in rate_arrays.items()
    }
    time_summaries = {"A": [], "R": []}
    for rate in ("A", "R"):
        maximum = rate_summaries[rate]["maximum_absolute"]
        for step in steps:
            time_summaries[rate].append(
                {
                    "step": int(step),
                    "time": float(case.t0 + step * case.dt),
                    **time_rate_statistics(rate_by_step[rate][step], maximum),
                }
            )

    sources = {
        name: np.concatenate(tuple(value.reshape(-1) for value in values))
        for name, values in source_by_field.items()
    }
    references = {
        name: np.concatenate(tuple(value.reshape(-1) for value in values))
        for name, values in reference_by_field.items()
    }
    source_summaries = {
        name: magnitude_statistics(values) for name, values in sources.items()
    }
    source_effects = {
        name: _source_effect_summary(
            sources[name],
            references[name],
            gll_time_ratios[name],
            mass_time_ratios[name],
            case.dt,
        )
        for name in SOURCE_FIELDS
    }
    inputs = {
        name: input_support_statistics(
            np.concatenate(tuple(value.reshape(-1) for value in values))
        )
        for name, values in input_by_field.items()
    }

    h_all = np.concatenate(tuple(input_by_field["h"])).reshape(-1)
    a_all = rate_arrays["A"]
    r_all = rate_arrays["R"]
    cloud_a = -h_all * a_all
    cloud_r = -h_all * r_all
    process_contributions = {
        "Qc_from_A": magnitude_statistics(cloud_a),
        "Qc_from_R": magnitude_statistics(cloud_r),
        "R_over_A_rms_contribution": _explicit_ratio(
            float(np.sqrt(np.mean(cloud_r * cloud_r))),
            float(np.sqrt(np.mean(cloud_a * cloud_a))),
        ),
        "R_over_A_mean_absolute_contribution": _explicit_ratio(
            float(np.mean(np.abs(cloud_r))),
            float(np.mean(np.abs(cloud_a))),
        ),
        "fraction_abs_R_contribution_exceeds_abs_A_contribution": float(
            np.mean(np.abs(cloud_r) > np.abs(cloud_a))
        ),
        "identity": "Qc_t = (-h*A) + (-h*R)",
    }

    representative = {}
    for rate in ("A", "R"):
        rms_values = np.asarray(
            [record["rms"] for record in time_summaries[rate]],
            dtype=np.float64,
        )
        index = int(np.argmax(rms_values))
        step = steps[index]
        representative[rate] = {
            "selection_criterion": f"maximum domain RMS abs({rate})",
            "step": int(step),
            "time": float(case.t0 + step * case.dt),
            "rate_values_cell_major": rate_by_step[rate][step].tolist(),
        }

    decision = classify_moist_activity(
        rate_summaries, time_summaries, source_effects
    )
    result = {
        "status": "complete",
        "benchmark_stage": "Test 2 moist-physics activity audit",
        "truth_run": str(truth_directory),
        "selected_plan": str(plan_path),
        "truth_state_access": {
            "first_step": 0,
            "last_step": 80,
            "steps": steps,
            "states_after_80_loaded": False,
            "contract": (
                "existing restart snapshots 0..80 only; no PDE integration and "
                "no inspection of future Test-2 deployment states"
            ),
        },
        "sample_accounting": accounting,
        "deployed_representation": {
            "adapter": "dimswe.jax_moist_adapter.JAXMoistEulerPrimal",
            "kernel": "certified float64 J1 JAX moist primal",
            "truth_moist_backend": case.moist_backend,
            "parity_contract": (
                "J1 certifies exact GLL local A/R/source parity with the "
                "deployed UFL moist child"
            ),
            "reference_points": layout.reference_points.tolist(),
            "physical_coordinates_cell_major": {
                "x": packed_x.tolist(),
                "y": packed_y.tolist(),
            },
        },
        "moist_parameters": {
            **parameters,
            "beta2": parameters["g"] * parameters["L"],
        },
        "rate_statistics": rate_summaries,
        "time_resolved_rate_activity": time_summaries,
        "source_statistics": source_summaries,
        "source_identity_maximum_absolute_errors": identity_maximum_errors,
        "source_increment_effects": source_effects,
        "cloud_water_process_contributions": process_contributions,
        "input_space_support": inputs,
        "representative_spatial_activity": representative,
        "decision": decision,
        "wall_time_seconds": float(perf_counter() - started),
    }
    write_json_record(output, result)
    if plot_directory is not None:
        plot_moist_activity(result, plot_directory)
    return result


def plot_moist_activity(summary, output_directory):
    """Write the four required histories and two criterion-selected maps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    histories = summary["time_resolved_rate_activity"]
    for rate in ("A", "R"):
        records = histories[rate]
        times = [record["time"] for record in records]

        figure, axis = plt.subplots()
        axis.plot(
            times,
            [record["maximum_absolute"] for record in records],
            label=f"max |{rate}|",
        )
        axis.plot(
            times,
            [record["rms"] for record in records],
            label=f"RMS {rate}",
        )
        axis.set(xlabel="time (s)", ylabel=f"{rate} rate magnitude")
        axis.legend()
        figure.tight_layout()
        figure.savefig(destination / f"{rate}_maximum_rms_vs_time.png", dpi=150)
        plt.close(figure)

        figure, axis = plt.subplots()
        for threshold in ("1e-06", "1e-03"):
            axis.plot(
                times,
                [
                    record["global_scale_relative_activity_fractions"][
                        threshold
                    ]
                    for record in records
                ],
                label=f"|{rate}| > {threshold} global max",
            )
        axis.set(xlabel="time (s)", ylabel="cell-local GLL active fraction")
        axis.legend()
        figure.tight_layout()
        figure.savefig(destination / f"{rate}_active_fraction_vs_time.png", dpi=150)
        plt.close(figure)

    coordinates = summary["deployed_representation"][
        "physical_coordinates_cell_major"
    ]
    x = np.asarray(coordinates["x"], dtype=np.float64).reshape(-1)
    y = np.asarray(coordinates["y"], dtype=np.float64).reshape(-1)
    for rate in ("A", "R"):
        representative = summary["representative_spatial_activity"][rate]
        values = np.asarray(
            representative["rate_values_cell_major"], dtype=np.float64
        ).reshape(-1)
        figure, axis = plt.subplots()
        points = axis.scatter(x, y, c=values, s=5, cmap="coolwarm")
        figure.colorbar(points, ax=axis, label=rate)
        axis.set(
            xlabel="x",
            ylabel="y",
            title=f"{rate} at RMS-selected t={representative['time']:g}",
        )
        axis.set_aspect("equal")
        figure.tight_layout()
        figure.savefig(destination / f"{rate}_rms_selected_spatial_map.png", dpi=150)
        plt.close(figure)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-run", required=True)
    parser.add_argument("--selected-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot-directory")
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="use the certified unjitted local JAX kernel for diagnostics",
    )
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    run_moist_activity_audit(
        arguments.truth_run,
        arguments.selected_plan,
        output=arguments.output,
        plot_directory=arguments.plot_directory,
        use_jit=not arguments.no_jit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ACTIVITY_SCALES",
    "PERCENTILES",
    "classify_moist_activity",
    "deployed_sample_accounting",
    "input_support_statistics",
    "magnitude_statistics",
    "moist_source_terms",
    "plot_moist_activity",
    "rate_statistics",
    "run_moist_activity_audit",
    "time_rate_statistics",
)
