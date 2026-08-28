"""Test 2A-3B autonomous training-support deployment diagnostics.

The driver loads only trusted states 0..80. It constructs all 80 neural-model
steps before consulting truth targets, and it accepts any parameter artifact
compatible with the frozen Test-2A architecture and normalization.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import numpy as np

from .jax_moist import moist_rates_jax
from .resolved_hidden_c0 import ResolvedPilotConfiguration, write_json_record
from .test2a_embedded_moist import (
    FrozenNeuralAMoistPhysics,
    load_frozen_neural_a_physics,
    parameter_pytree_sha256,
)
from .test2a_operator import load_mlp_parameters


STATE_FIELDS = ("v", "h", "S", "Qv", "Qc", "Qr")
MOISTURE_FIELDS = ("Qv", "Qc", "Qr")
ALLOWED_CLASSIFICATIONS = (
    "APRIORI_DEPLOYMENT_STABLE_AND_ACCURATE",
    "APRIORI_DEPLOYMENT_STABLE_WITH_DRIFT",
    "APRIORI_DEPLOYMENT_SPURIOUS_RAIN",
    "APRIORI_DEPLOYMENT_UNSTABLE",
)


def load_apriori_autonomous_configuration(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if record.get("benchmark_stage") != (
        "Test 2A-3B a-priori autonomous training-support evaluation"
    ):
        raise ValueError("not a selected Test 2A-3B configuration")
    truth = record["truth"]
    if truth["state_indices"] != [0, 80] or not truth[
        "states_after_80_forbidden"
    ]:
        raise ValueError("Test 2A-3B may access only truth states 0..80")
    deployment = record["deployment"]
    if (
        int(deployment["initial_truth_state"]) != 0
        or int(deployment["complete_steps"]) != 80
        or int(deployment["truth_resets_after_initialization"]) != 0
        or deployment["moist_backend"] != "jax"
        or deployment["R"] != "original deployed analytical R"
        or float(deployment["c0"]) != 0.14
    ):
        raise ValueError("Test 2A-3B autonomous deployment contract changed")
    return record


def autonomous_states(initial_state, step, *, nsteps=80):
    """Construct one recursive trajectory without accepting truth targets."""
    if int(nsteps) != nsteps or int(nsteps) < 1:
        raise ValueError("nsteps must be a positive integer")
    current = initial_state
    states = [current]
    for index in range(int(nsteps)):
        current = step(current, index)
        states.append(current)
    return tuple(states)


def _explicit_relative_rms(predicted, reference):
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    reference = np.asarray(reference, dtype=np.float64)
    numerator = float(np.sqrt(np.mean(error * error)))
    denominator = float(np.sqrt(np.mean(reference * reference)))
    return {
        "absolute_rms_error": numerator,
        "reference_rms": denominator,
        "relative_rms_error": None if denominator == 0.0 else numerator / denominator,
        "reference_rms_zero": denominator == 0.0,
    }


def local_a_diagnostic(neural_a, analytical_a, thresholds=(1.0e-3, 1.0e-2, 1.0e-1)):
    neural = np.asarray(neural_a, dtype=np.float64).reshape(-1)
    analytical = np.asarray(analytical_a, dtype=np.float64).reshape(-1)
    if neural.shape != analytical.shape or not np.all(np.isfinite(neural)) or not np.all(
        np.isfinite(analytical)
    ):
        raise ValueError("A diagnostic arrays must be matching and finite")
    result = _explicit_relative_rms(neural, analytical)
    maximum = float(np.max(np.abs(analytical)))
    strata = {}
    for threshold in thresholds:
        mask = np.abs(analytical) > float(threshold) * maximum
        key = f"abs_A_gt_{float(threshold):.0e}_max_abs_A"
        strata[key] = {
            "sample_count": int(np.count_nonzero(mask)),
            "fraction": float(np.mean(mask)),
            "sign_agreement": (
                None if not np.any(mask) else float(np.mean(np.sign(neural[mask]) == np.sign(analytical[mask])))
            ),
            "relative_rms_error": (
                None
                if not np.any(mask)
                else _explicit_relative_rms(neural[mask], analytical[mask])[
                    "relative_rms_error"
                ]
            ),
        }
    result.update(
        {
            "maximum_absolute_analytical_A": maximum,
            "maximum_absolute_error": float(np.max(np.abs(neural - analytical))),
            "active_strata": strata,
        }
    )
    return result


def rain_activity_diagnostic(
    r_rate,
    h,
    qr,
    dt,
    comparison_rate_scale,
    *,
    float64_scale_multiplier=64.0,
    physical_increment_relative_threshold=1.0e-12,
):
    rain = np.asarray(r_rate, dtype=np.float64).reshape(-1)
    depth = np.asarray(h, dtype=np.float64).reshape(-1)
    rain_water = np.asarray(qr, dtype=np.float64).reshape(-1)
    if rain.shape != depth.shape or rain.shape != rain_water.shape:
        raise ValueError("R, h, and Qr must share the deployed GLL shape")
    if not all(np.all(np.isfinite(value)) for value in (rain, depth, rain_water)):
        raise ValueError("rain diagnostics require finite arrays")
    scale = max(float(comparison_rate_scale), np.finfo(np.float64).tiny)
    numerical_tolerance = (
        float(float64_scale_multiplier) * np.finfo(np.float64).eps * scale
    )
    qr_rms = float(np.sqrt(np.mean(rain_water * rain_water)))
    physical_tolerance = float(physical_increment_relative_threshold) * qr_rms
    increments = np.abs(float(dt) * depth * rain)
    physically_meaningful = (np.abs(rain) > numerical_tolerance) & (
        increments > physical_tolerance
    )
    return {
        "maximum_absolute_R": float(np.max(np.abs(rain))),
        "rms_R": float(np.sqrt(np.mean(rain * rain))),
        "exact_nonzero_fraction": float(np.mean(rain != 0.0)),
        "float64_scale_tolerance": numerical_tolerance,
        "above_float64_scale_fraction": float(np.mean(np.abs(rain) > numerical_tolerance)),
        "physical_Qr_increment_tolerance": physical_tolerance,
        "physically_meaningful_fraction": float(np.mean(physically_meaningful)),
        "maximum_absolute_Qr_increment": float(np.max(increments)),
    }


def source_invariant_diagnostic(source, beta2):
    values = {
        key: np.asarray(source[key], dtype=np.float64)
        for key in ("S", "Qv", "Qc", "Qr")
    }
    water = values["Qv"] + values["Qc"] + values["Qr"]
    entropy = values["S"] - float(beta2) * values["Qv"]
    return {
        "water_source_maximum_absolute_residual": float(np.max(np.abs(water))),
        "water_source_rms_residual": float(np.sqrt(np.mean(water * water))),
        "S_minus_beta2_Qv_maximum_absolute_residual": float(
            np.max(np.abs(entropy))
        ),
        "S_minus_beta2_Qv_rms_residual": float(np.sqrt(np.mean(entropy * entropy))),
    }


def load_compatible_neural_physics(
    embedding_configuration,
    parameter_file,
    *,
    expected_pytree_sha256=None,
    use_jit=True,
):
    """Reuse frozen normalization while accepting any compatible pytree."""
    baseline = load_frozen_neural_a_physics(embedding_configuration, use_jit=use_jit)
    parameters, configuration = load_mlp_parameters(parameter_file)
    if configuration != baseline.model_configuration:
        raise ValueError("neural parameter artifact architecture is incompatible")
    fingerprint = parameter_pytree_sha256(parameters)
    if expected_pytree_sha256 is not None and fingerprint != expected_pytree_sha256:
        raise ValueError("neural parameter pytree fingerprint mismatch")
    provenance = {
        "parameter_file": str(Path(parameter_file).resolve()),
        "parameter_pytree_sha256": fingerprint,
        "embedding_configuration": str(Path(embedding_configuration).resolve()),
        "normalization_reused_from_training_states": [0, 80],
        "normalization_refitted": False,
    }
    return FrozenNeuralAMoistPhysics(
        parameters,
        configuration,
        baseline.normalization,
        provenance=provenance,
        use_jit=use_jit,
    )


def _plot_result(result, output_directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    times = result["times"]
    plots = (
        (
            "mixed_state_error_vs_time.png",
            ((result["mixed_state_error"]["relative_mass_norm_error"], "mixed"),),
            "relative mixed-state mass error",
        ),
        (
            "moisture_errors_vs_time.png",
            tuple(
                (result["fieldwise_errors"][field]["relative_mass_norm_error"], field)
                for field in MOISTURE_FIELDS
            ),
            "relative field mass error",
        ),
        (
            "off_manifold_A_error_vs_time.png",
            ((tuple(value["relative_rms_error"] for value in result["local_A_diagnostics"]), "A"),),
            "relative RMS A error on neural states",
        ),
        (
            "R_activity_vs_time.png",
            ((tuple(value["maximum_absolute_R"] for value in result["rain_activity"]), "max |R|"),),
            "original-R activity",
        ),
    )
    for filename, histories, ylabel in plots:
        figure, axis = plt.subplots()
        for history, label in histories:
            axis.plot(times, history, label=label)
        axis.set(xlabel="time", ylabel=ylabel)
        axis.legend()
        figure.tight_layout()
        figure.savefig(destination / filename, dpi=150)
        plt.close(figure)
    figure, axes = plt.subplots(2, 1, sharex=True)
    for axis, key, label in (
        (axes[0], "kinetic_energy", "kinetic energy"),
        (axes[1], "projected_enstrophy", "projected enstrophy"),
    ):
        axis.plot(times, result[key]["truth"], label="truth")
        axis.plot(times, result[key]["predicted"], label="neural")
        axis.set_ylabel(label)
        axis.legend()
    axes[-1].set_xlabel("time")
    figure.tight_layout()
    figure.savefig(destination / "KE_enstrophy_vs_time.png", dpi=150)
    plt.close(figure)


def run_apriori_autonomous(
    configuration_path,
    output_directory,
    *,
    parameter_file=None,
    expected_pytree_sha256=None,
):
    """Run 80 neural complete steps, then evaluate against truth states 1..80."""
    from firedrake import assemble

    from .hidden_c0 import _copy_function, _flat_values
    from .resolved_hidden_c0_driver import (
        ResolvedDiagnosticEvaluator,
        build_resolved_hidden_c0_case,
    )
    from .resolved_hidden_c0_inference import (
        _diagnostic_mismatch,
        _field_trajectory_metric,
        _trajectory_metric,
        load_resolved_truth,
    )
    from .selected_test1b import load_selected_test1b_plan

    selected = load_apriori_autonomous_configuration(configuration_path)
    _, plan = load_selected_test1b_plan(selected["truth"]["selected_plan"])
    inference = plan.inference_configuration(Path(selected["truth"]["run_directory"]).resolve())
    if (inference.training_start_step, inference.training_stop_step) != (0, 80):
        raise ValueError("selected plan training support changed")
    truth_case, loaded_truth = load_resolved_truth(inference, include_heldout=False)
    if tuple(loaded_truth.states) != tuple(range(81)):
        raise ValueError("Test 2A-3B requires exactly truth states 0..80")
    parameter_file = parameter_file or selected["model"]["default_parameter_file"]
    expected = expected_pytree_sha256
    if expected is None and parameter_file == selected["model"]["default_parameter_file"]:
        expected = selected["model"]["default_parameter_pytree_sha256"]
    physics = load_compatible_neural_physics(
        selected["model"]["embedding_configuration"],
        parameter_file,
        expected_pytree_sha256=expected,
    )
    pilot = ResolvedPilotConfiguration.from_dict(loaded_truth.metadata["configuration"])
    neural_pilot = replace(pilot, moist_backend="jax", output_directory=str(output_directory))
    case = build_resolved_hidden_c0_case(
        neural_pilot, jax_moist_local_physics=physics
    )
    initial_truth_state = case.state_from_values(
        _flat_values(loaded_truth.states[0]), "test2a3b_truth_0"
    )

    started = perf_counter()
    with case.physical_c0(float(selected["deployment"]["c0"])):
        generated = autonomous_states(
            _copy_function(initial_truth_state, "test2a3b_initial_truth_only"),
            lambda state, index: _copy_function(
                case.helper.take_forward_step_cached(
                    state, case.t0 + index * case.dt, case.dt
                ).state_out,
                f"test2a3b_model_state_{index + 1}",
            ),
            nsteps=80,
        )
    # Truth target values are first consulted here, after the complete trajectory
    # exists.  Before this point, only X*_0 and the available index set were read.
    truth_states = {0: initial_truth_state}
    truth_states.update(
        {
            step: case.state_from_values(
                _flat_values(loaded_truth.states[step]), f"test2a3b_truth_{step}"
            )
            for step in range(1, 81)
        }
    )
    predicted = {step: generated[step] for step in range(1, 81)}
    truth_proxy = type("TruthProxy", (), {"states": truth_states})()
    steps = tuple(range(1, 81))
    mixed = _trajectory_metric(case, predicted, truth_proxy, steps, "test2a3b_mixed")
    fieldwise = _field_trajectory_metric(
        case, predicted, truth_proxy, steps, "test2a3b_field"
    )
    evaluator = ResolvedDiagnosticEvaluator(case, neural_pilot)
    predicted_diagnostics = []
    truth_diagnostics = []
    local_a = []
    rain = []
    invariants = []
    moisture_extrema = []
    total_water = []
    all_local_neural_a = []
    all_local_analytical_a = []
    rain_config = selected["rain_activity"]
    for step in steps:
        state = predicted[step]
        time = case.t0 + step * case.dt
        predicted_diagnostics.append(evaluator.evaluate(state, step, time)[0])
        truth_diagnostics.append(evaluator.evaluate(truth_states[step], step, time)[0])
        cache = case.helper.moist_helper.take_forward_step_cached(state, time, case.dt)
        state_device = {key: jnp.asarray(value) for key, value in cache.packed_state.items()}
        fields_device = {key: jnp.asarray(value) for key, value in cache.packed_fields.items()}
        parameters_device = {key: jnp.asarray(value) for key, value in cache.parameters.items()}
        analytical_rates = moist_rates_jax(state_device, fields_device, parameters_device)
        neural_a = np.asarray(cache.rates["A"], dtype=np.float64)
        analytical_a = np.asarray(analytical_rates["A"], dtype=np.float64)
        all_local_neural_a.append(neural_a.reshape(-1))
        all_local_analytical_a.append(analytical_a.reshape(-1))
        a_record = local_a_diagnostic(neural_a, analytical_a)
        a_record.update({"step": step, "time": time})
        local_a.append(a_record)
        comparison_scale = max(
            float(np.max(np.abs(neural_a))), float(np.max(np.abs(analytical_a)))
        )
        _, packed_qr = (
            case.helper.moist_helper.primal_helper.interpolate_and_pack(
                state.sub(5), f"test2a3b_Qr_gll_{step}"
            )
        )
        rain_record = rain_activity_diagnostic(
            cache.rates["R"],
            cache.packed_state["h"],
            packed_qr,
            case.dt,
            comparison_scale,
            float64_scale_multiplier=rain_config["float64_scale_multiplier"],
            physical_increment_relative_threshold=rain_config[
                "physical_increment_relative_threshold"
            ],
        )
        rain_record.update({"step": step, "time": time})
        rain.append(rain_record)
        invariant = source_invariant_diagnostic(
            cache.source_density, float(cache.parameters["g"] * cache.parameters["L"])
        )
        invariant.update({"step": step, "time": time})
        invariants.append(invariant)
        field_record = {"step": step, "time": time}
        for field, index in zip(MOISTURE_FIELDS, (3, 4, 5)):
            values = np.asarray(state.sub(index).dat.data_ro, dtype=np.float64)
            field_record[field] = {
                "minimum_coefficient": float(np.min(values)),
                "maximum_coefficient": float(np.max(values)),
            }
        moisture_extrema.append(field_record)
        total_water.append(
            float(
                assemble(
                    (state.sub(3) + state.sub(4) + state.sub(5))
                    * case.model.spaces.dx
                )
            )
        )
    times = np.asarray([case.t0 + step * case.dt for step in steps], dtype=np.float64)
    kinetic = _diagnostic_mismatch(
        [value["kinetic_energy"] for value in predicted_diagnostics],
        [value["kinetic_energy"] for value in truth_diagnostics],
        steps,
        times,
    )
    enstrophy = _diagnostic_mismatch(
        [value["projected_enstrophy"] for value in predicted_diagnostics],
        [value["projected_enstrophy"] for value in truth_diagnostics],
        steps,
        times,
    )
    all_states_finite = all(
        np.all(np.isfinite(_flat_values(state))) for state in generated
    )
    exact_rain_steps = [value["step"] for value in rain if value["exact_nonzero_fraction"] > 0.0]
    physical_rain_steps = [
        value["step"] for value in rain if value["physically_meaningful_fraction"] > 0.0
    ]
    aggregate_a = local_a_diagnostic(
        np.concatenate(all_local_neural_a), np.concatenate(all_local_analytical_a)
    )
    result = {
        "status": "complete",
        "benchmark_stage": selected["benchmark_stage"],
        "interpretation": selected["deployment"]["interpretation"],
        "parameter_provenance": physics.provenance,
        "deployment_contract": {
            "trusted_initial_state": 0,
            "generated_steps": [1, 80],
            "complete_production_steps": 80,
            "truth_resets_after_initialization": 0,
            "predicted_state_recursively_reused": True,
            "truth_targets_consulted_only_after_full_prediction": True,
            "truth_states_accessed": [0, 80],
            "states_after_80_accessed": False,
            "c0": selected["deployment"]["c0"],
            "moist_backend": "jax",
            "A": "artifact neural A",
            "R": "original analytical R",
            "child_order": list(case.helper.production_graph_diagnostics()["forward_child_order"]),
        },
        "steps": list(steps),
        "times": times.tolist(),
        "mixed_state_error": mixed,
        "fieldwise_errors": fieldwise,
        "kinetic_energy": kinetic,
        "projected_enstrophy": enstrophy,
        "all_states_finite": bool(all_states_finite),
        "moisture_field_extrema": moisture_extrema,
        "total_water_integral": total_water,
        "local_A_diagnostics": local_a,
        "aggregate_off_manifold_A_diagnostic": aggregate_a,
        "rain_activity": rain,
        "rain_activity_summary": {
            "maximum_absolute_R": max(value["maximum_absolute_R"] for value in rain),
            "maximum_rms_R": max(value["rms_R"] for value in rain),
            "timesteps_with_exact_nonzero_R": len(exact_rain_steps),
            "fraction_timesteps_with_exact_nonzero_R": len(exact_rain_steps) / 80.0,
            "first_exact_nonzero_R_step": exact_rain_steps[0] if exact_rain_steps else None,
            "timesteps_with_physically_meaningful_R": len(physical_rain_steps),
            "fraction_timesteps_with_physically_meaningful_R": len(physical_rain_steps) / 80.0,
            "first_physically_meaningful_R_step": physical_rain_steps[0] if physical_rain_steps else None,
            "tolerance_contract": rain_config,
        },
        "source_structural_invariants": invariants,
        "decision": {
            "classification": None,
            "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
            "requires_post_run_scientific_review": True,
            "automatic_threshold_classification_used": False,
        },
        "wall_time_seconds": float(perf_counter() - started),
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    write_json_record(output / "rollout_summary.json", result)
    write_json_record(
        output / "trajectory_metrics.json",
        {key: result[key] for key in (
            "steps", "times", "mixed_state_error", "fieldwise_errors",
            "kinetic_energy", "projected_enstrophy", "local_A_diagnostics",
            "rain_activity", "source_structural_invariants",
        )},
    )
    _plot_result(result, output)
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--parameter-file")
    parser.add_argument("--expected-pytree-sha256")
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    run_apriori_autonomous(
        arguments.configuration,
        arguments.output_directory,
        parameter_file=arguments.parameter_file,
        expected_pytree_sha256=arguments.expected_pytree_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ALLOWED_CLASSIFICATIONS",
    "autonomous_states",
    "load_apriori_autonomous_configuration",
    "load_compatible_neural_physics",
    "local_a_diagnostic",
    "rain_activity_diagnostic",
    "run_apriori_autonomous",
    "source_invariant_diagnostic",
)
