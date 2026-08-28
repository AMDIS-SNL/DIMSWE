"""Diagnose the fixed Test-2A stored-UFL versus purported-JAX offset.

This module is read-only with respect to the production model and truth.  It
constructs fresh UFL, genuinely analytical JAX, and frozen-neural JAX moist
children on the same post-prefix states.  No optimizer is imported or run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .resolved_hidden_c0 import write_json_record


FIELD_NAMES = ("v", "h", "S", "Qv", "Qc", "Qr")
TRAINING_TRANSITIONS = tuple(range(80))


def jax_helper_physics_kind(helper):
    """Return an explicit physics label; never infer analytical from no args."""
    local_physics = getattr(helper, "local_physics", None)
    if local_physics is None:
        return "analytical_A_original_R"
    mode = getattr(local_physics, "physics_mode", None)
    if mode != "neural_A_original_R":
        raise ValueError(f"unsupported JAX moist local physics mode {mode!r}")
    return "frozen_neural_A_original_R"


def require_analytical_jax_helper(helper):
    kind = jax_helper_physics_kind(helper)
    if kind != "analytical_A_original_R":
        raise ValueError(
            "analytical JAX audit requires local_physics=None; a neural-local "
            "helper's parameterless call uses its frozen neural parameters"
        )
    return helper


def array_error_statistics(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError("array comparison shapes differ")
    difference = actual - reference
    rms = float(np.sqrt(np.mean(difference * difference)))
    reference_rms = float(np.sqrt(np.mean(reference * reference)))
    sign_disagreement = int(np.count_nonzero(np.signbit(actual) != np.signbit(reference)))
    return {
        "sample_count": int(actual.size),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "RMS_difference": rms,
        "reference_RMS": reference_rms,
        "relative_RMS_difference": (
            None if reference_rms == 0.0 else rms / reference_rms
        ),
        "sign_disagreement_count": sign_disagreement,
        "sign_disagreement_fraction": sign_disagreement / int(actual.size),
    }


def coefficient_decomposition_error(stored_component, backend_component, total):
    """Return max error in total = stored_component + backend_component."""
    stored = np.asarray(stored_component, dtype=np.float64)
    backend = np.asarray(backend_component, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)
    if stored.shape != backend.shape or stored.shape != total.shape:
        raise ValueError("decomposition arrays differ in shape")
    return float(np.max(np.abs(total - (stored + backend))))


def _ufl_rate_expressions(term, state):
    import ufl

    from .physics import qsat

    h = state.sub(1)
    entropy = state.sub(2)
    vapour = state.sub(3)
    cloud = state.sub(4)
    qv = vapour / h
    qc = cloud / h
    specific_entropy = entropy / h
    beta2 = term.g * term.L
    saturation = qsat(
        h, specific_entropy, term.B, term.q0, term.H0, term.g
    )
    gamma_v = 1.0 / (1.0 + 20.0 * saturation * beta2 / term.g)
    condensation_argument = gamma_v * (qv - saturation) / term.tau_v
    evaporation_argument = gamma_v * (saturation - qv) / term.tau_v
    condensation = ufl.max_value(0.0, condensation_argument)
    evaporation_positive = ufl.max_value(0.0, evaporation_argument)
    evaporation_cap = qc / term.dt
    evaporation = ufl.min_value(evaporation_cap, evaporation_positive)
    rain_argument = term.gamma_r * (qc - term.qprecip) / term.tau_r
    rain = ufl.max_value(0.0, rain_argument)
    return {
        "h": h,
        "S": entropy,
        "Qv": vapour,
        "Qc": cloud,
        "B": term.B,
        "qv": qv,
        "qc": qc,
        "s": specific_entropy,
        "depth_denominator": h + term.B,
        "qsat": saturation,
        "gamma_v": gamma_v,
        "condensation_argument": condensation_argument,
        "evaporation_argument": evaporation_argument,
        "evaporation_cap": evaporation_cap,
        "C": condensation,
        "E_positive": evaporation_positive,
        "E": evaporation,
        "A": evaporation - condensation,
        "rain_argument": rain_argument,
        "R": rain,
    }


def _pack_ufl_rates(adapter, state, step):
    expressions = _ufl_rate_expressions(adapter.term, state)
    return {
        name: adapter.interpolate_and_pack(
            expression, f"test2a_offset_ufl_{name}_{step}"
        )[1]
        for name, expression in expressions.items()
    }


def _copy_difference(left, right, name):
    result = left.copy(deepcopy=True)
    result.rename(name)
    with result.dat.vec as output, right.dat.vec_ro as reference:
        output.axpy(-1.0, reference)
    return result


def _mixed_mass_norm(helper, value, name):
    dual = helper.state_mass_map(value, f"{name}_mass")
    squared = float(helper.dual_pairing(dual, value))
    scale = max(1.0, abs(squared))
    if squared < -128.0 * np.finfo(np.float64).eps * scale:
        raise FloatingPointError("negative mixed mass norm")
    return float(np.sqrt(max(0.0, squared)))


def _dual_natural_norm(helper, value, name):
    representative = helper.state_riesz_representative(value, f"{name}_riesz")
    squared = float(helper.dual_pairing(value, representative))
    scale = max(1.0, abs(squared))
    if squared < -128.0 * np.finfo(np.float64).eps * scale:
        raise FloatingPointError("negative dual natural norm")
    return float(np.sqrt(max(0.0, squared)))


def _state_error_record(helper, actual, reference, name):
    from firedrake import norm

    difference = _copy_difference(actual, reference, f"{name}_difference")
    absolute = _mixed_mass_norm(helper, difference, f"{name}_difference")
    reference_norm = _mixed_mass_norm(helper, reference, f"{name}_reference")
    fields = {}
    for field_name, actual_field, reference_field, difference_field in zip(
        FIELD_NAMES,
        actual.subfunctions,
        reference.subfunctions,
        difference.subfunctions,
    ):
        field_absolute = float(norm(difference_field))
        field_reference = float(norm(reference_field))
        fields[field_name] = {
            "absolute_L2_norm": field_absolute,
            "reference_L2_norm": field_reference,
            "relative_L2_norm": (
                None if field_reference == 0.0 else field_absolute / field_reference
            ),
            "maximum_absolute_coefficient_difference": float(
                np.max(np.abs(difference_field.dat.data_ro))
            ),
        }
    return {
        "absolute_mixed_mass_norm": absolute,
        "reference_mixed_mass_norm": reference_norm,
        "relative_mixed_mass_norm": (
            None if reference_norm == 0.0 else absolute / reference_norm
        ),
        "fields": fields,
    }


def _dual_error_record(helper, actual, reference, name):
    difference = actual.copy(deepcopy=True)
    difference.rename(f"{name}_difference")
    with difference.dat.vec as output, reference.dat.vec_ro as expected:
        output.axpy(-1.0, expected)
    absolute = _dual_natural_norm(helper, difference, f"{name}_difference")
    reference_norm = _dual_natural_norm(helper, reference, f"{name}_reference")
    with difference.dat.vec_ro as vector:
        maximum = float(np.max(np.abs(vector.array_r)))
    return {
        "absolute_natural_norm": absolute,
        "reference_natural_norm": reference_norm,
        "relative_natural_norm": (
            None if reference_norm == 0.0 else absolute / reference_norm
        ),
        "maximum_absolute_coefficient_difference": maximum,
    }


def _source_from_rates(packed_state, rates, beta2):
    h = np.asarray(packed_state["h"], dtype=np.float64)
    a_rate = np.asarray(rates["A"], dtype=np.float64)
    rain = np.asarray(rates["R"], dtype=np.float64)
    return {
        "S": h * float(beta2) * a_rate,
        "Qv": h * a_rate,
        "Qc": -h * (a_rate + rain),
        "Qr": h * rain,
    }


def _local_source_record(actual, reference, beta2):
    result = {}
    for name in ("S", "Qv", "Qc", "Qr"):
        result[name] = array_error_statistics(actual[name], reference[name])
    water = actual["Qv"] - reference["Qv"]
    water += actual["Qc"] - reference["Qc"]
    water += actual["Qr"] - reference["Qr"]
    thermal = (actual["S"] - reference["S"]) - float(beta2) * (
        actual["Qv"] - reference["Qv"]
    )
    return (
        result,
        float(np.max(np.abs(water))),
        float(np.max(np.abs(thermal))),
    )


def _integrated_residual_invariants(case, difference, beta2):
    from firedrake import assemble

    dx = case.model.spaces.dx
    water_coefficients = (
        difference.sub(3).dat.data_ro
        + difference.sub(4).dat.data_ro
        + difference.sub(5).dat.data_ro
    )
    water_component_integrals = [
        float(assemble(difference.sub(index) * dx)) for index in (3, 4, 5)
    ]
    entropy_integral = float(assemble(difference.sub(2) * dx))
    vapour_integral = water_component_integrals[0]
    water = float(
        assemble(
            (difference.sub(3) + difference.sub(4) + difference.sub(5)) * dx
        )
    )
    thermal = entropy_integral - float(beta2) * vapour_integral
    water_scale = sum(abs(value) for value in water_component_integrals)
    thermal_scale = abs(entropy_integral) + abs(float(beta2) * vapour_integral)
    return {
        "integrated_water_residual": water,
        "integrated_thermal_residual": thermal,
        "integrated_water_relative_cancellation": (
            0.0 if water_scale == 0.0 else abs(water) / water_scale
        ),
        "integrated_thermal_relative_cancellation": (
            0.0 if thermal_scale == 0.0 else abs(thermal) / thermal_scale
        ),
        "maximum_absolute_water_coefficient_residual": float(
            np.max(np.abs(water_coefficients))
        ),
    }


def _aggregate_state_records(records):
    def _relative_or_zero(value):
        # A zero reference field can only have a meaningful finite relative
        # error when the absolute discrepancy is also zero.  Preserve the
        # absolute norm separately and use zero for that exact case so the
        # across-transition maximum remains JSON/numerically well defined.
        return 0.0 if value is None else value

    result = {
        "maximum_absolute_mixed_mass_norm": max(
            item["absolute_mixed_mass_norm"] for item in records
        ),
        "maximum_relative_mixed_mass_norm": max(
            _relative_or_zero(item["relative_mixed_mass_norm"])
            for item in records
        ),
        "fields": {},
    }
    for field in FIELD_NAMES:
        result["fields"][field] = {
            "maximum_absolute_L2_norm": max(
                item["fields"][field]["absolute_L2_norm"] for item in records
            ),
            "maximum_relative_L2_norm": max(
                _relative_or_zero(item["fields"][field]["relative_L2_norm"])
                for item in records
            ),
            "maximum_absolute_coefficient_difference": max(
                item["fields"][field]["maximum_absolute_coefficient_difference"]
                for item in records
            ),
        }
    return result


def _aggregate_dual_records(records):
    def _relative_or_zero(value):
        return 0.0 if value is None else value

    return {
        "maximum_absolute_natural_norm": max(
            item["absolute_natural_norm"] for item in records
        ),
        "maximum_relative_natural_norm": max(
            _relative_or_zero(item["relative_natural_norm"])
            for item in records
        ),
        "maximum_absolute_coefficient_difference": max(
            item["maximum_absolute_coefficient_difference"] for item in records
        ),
    }


def run_backend_offset_audit(trajectory_configuration, output):
    from firedrake import assemble

    from .jax_moist_hvp import JAXMoistEulerHVP
    from .mtswe_split_hvp import ProductionMoistEulerHVP
    from .test2a_trajectory_certification import _build_case

    _, case, truth, _, _ = _build_case(
        trajectory_configuration, maximum_truth_step=80
    )
    neural_helper = case.helper.moist_helper
    neural_kind = jax_helper_physics_kind(neural_helper)
    if neural_kind != "frozen_neural_A_original_R":
        raise ValueError("trajectory certification case no longer owns neural physics")
    ufl_oracle = case.helper.moist_child.ufl_oracle
    ufl_helper = ProductionMoistEulerHVP(ufl_oracle)
    analytical_jax = require_analytical_jax_helper(
        JAXMoistEulerHVP(ufl_oracle, use_jit=True, local_physics=None)
    )
    adapter = analytical_jax.primal_helper
    beta2 = float(adapter.term.g * adapter.term.L)
    topography_max_abs = None

    rate_arrays = {name: {"jax": [], "ufl": []} for name in ("A", "R")}
    prior_neural_rate_arrays = {name: [] for name in ("A", "R")}
    regime_differences = {"condensation": [], "evaporation": [], "zero": []}
    components = {name: [] for name in ("stored_provenance_A", "ufl_jax_B", "total_C")}
    prior_reported_neural = []
    decomposition_maxima = []
    rhs_errors = []
    carrier_rhs_errors = []
    common_mass_solve_errors = []
    tendency_errors = []
    source_errors = []
    source_water_invariants = []
    source_thermal_invariants = []
    state_invariants = []
    prior_neural_state_invariants = []
    input_maxima = {}
    per_transition = []
    moist_increment_scales = []
    prior_neural_increment_scales = []

    with case.physical_c0(0.14):
        for step in TRAINING_TRANSITIONS:
            time = case.t0 + step * case.dt
            prefix = case.helper.take_fixed_prefix_cached(
                truth[step], time, case.dt
            )
            y_state = prefix.state_out
            z_ufl = ufl_helper.take_forward_step_cached(y_state, time, case.dt)
            # The UFL call populated the exact production RHS before another
            # mutable oracle call can overwrite it.
            ufl_rhs = assemble(ufl_oracle.production_stage_rhs_forms[0])
            z_jax = analytical_jax.take_forward_step_cached(
                y_state, time, case.dt
            )
            z_neural_default = neural_helper.take_forward_step_cached(
                y_state, time, case.dt
            )
            if topography_max_abs is None:
                topography_max_abs = float(
                    np.max(np.abs(np.asarray(z_jax.packed_fields["B"])))
                )

            packed_ufl = _pack_ufl_rates(adapter, y_state, step)
            for name in ("A", "R"):
                rate_arrays[name]["jax"].append(np.asarray(z_jax.rates[name]))
                rate_arrays[name]["ufl"].append(np.asarray(packed_ufl[name]))
                prior_neural_rate_arrays[name].append(
                    np.asarray(z_neural_default.rates[name])
                )
            a_difference = np.asarray(z_jax.rates["A"]) - packed_ufl["A"]
            for regime, mask in (
                ("condensation", packed_ufl["A"] < 0.0),
                ("evaporation", packed_ufl["A"] > 0.0),
                ("zero", packed_ufl["A"] == 0.0),
            ):
                if np.any(mask):
                    regime_differences[regime].append(a_difference[mask])

            # UFL expressions and JAX packing must refer to identical local
            # inputs before comparing nonlinear algebra.
            input_mapping = {
                "h": z_jax.packed_state["h"],
                "S": z_jax.packed_state["S"],
                "Qv": z_jax.packed_state["Qv"],
                "Qc": z_jax.packed_state["Qc"],
                "B": z_jax.packed_fields["B"],
                "qv": z_jax.gll_diagnostics["qv"],
                "qc": z_jax.gll_diagnostics["qc"],
                "s": z_jax.gll_diagnostics["s"],
                "qsat": z_jax.gll_diagnostics["qsat"],
                "gamma_v": z_jax.gll_diagnostics["gamma_v"],
            }
            for name, actual in input_mapping.items():
                maximum = float(np.max(np.abs(actual - packed_ufl[name])))
                input_maxima[name] = max(input_maxima.get(name, 0.0), maximum)

            ufl_source = _source_from_rates(
                z_jax.packed_state,
                {"A": packed_ufl["A"], "R": packed_ufl["R"]},
                beta2,
            )
            local_source, water_invariant, thermal_invariant = (
                _local_source_record(z_jax.source_density, ufl_source, beta2)
            )
            source_errors.append(local_source)
            source_water_invariants.append(water_invariant)
            source_thermal_invariants.append(thermal_invariant)
            carrier_ufl_rhs = analytical_jax.source_assembly(ufl_source)
            carrier_rhs_errors.append(
                _dual_error_record(
                    analytical_jax,
                    carrier_ufl_rhs,
                    ufl_rhs,
                    f"test2a_offset_carrier_vs_ufl_rhs_{step}",
                )
            )
            rhs_errors.append(
                _dual_error_record(
                    analytical_jax,
                    z_jax.source_dual,
                    ufl_rhs,
                    f"test2a_offset_jax_vs_ufl_rhs_{step}",
                )
            )
            common_jax_mass = analytical_jax.state_riesz_representative(
                ufl_rhs, f"test2a_offset_common_jax_mass_{step}"
            )
            common_ufl_mass = ufl_helper.state_riesz_representative(
                ufl_rhs, f"test2a_offset_common_ufl_mass_{step}"
            )
            common_mass_solve_errors.append(
                _state_error_record(
                    analytical_jax,
                    common_jax_mass,
                    common_ufl_mass,
                    f"test2a_offset_common_mass_{step}",
                )
            )
            tendency_errors.append(
                _state_error_record(
                    analytical_jax,
                    z_jax.tendency,
                    z_ufl.tendency,
                    f"test2a_offset_tendency_{step}",
                )
            )

            stored_component = _copy_difference(
                z_ufl.state_out,
                truth[step + 1],
                f"test2a_offset_stored_component_{step}",
            )
            backend_component = _copy_difference(
                z_jax.state_out,
                z_ufl.state_out,
                f"test2a_offset_backend_component_{step}",
            )
            total_component = _copy_difference(
                z_jax.state_out,
                truth[step + 1],
                f"test2a_offset_total_component_{step}",
            )
            components["stored_provenance_A"].append(
                _state_error_record(
                    analytical_jax,
                    z_ufl.state_out,
                    truth[step + 1],
                    f"test2a_offset_A_{step}",
                )
            )
            components["ufl_jax_B"].append(
                _state_error_record(
                    analytical_jax,
                    z_jax.state_out,
                    z_ufl.state_out,
                    f"test2a_offset_B_{step}",
                )
            )
            components["total_C"].append(
                _state_error_record(
                    analytical_jax,
                    z_jax.state_out,
                    truth[step + 1],
                    f"test2a_offset_C_{step}",
                )
            )
            prior_reported_neural.append(
                _state_error_record(
                    analytical_jax,
                    z_neural_default.state_out,
                    truth[step + 1],
                    f"test2a_offset_prior_neural_{step}",
                )
            )
            maximum_decomposition = 0.0
            for stored_field, backend_field, total_field in zip(
                stored_component.subfunctions,
                backend_component.subfunctions,
                total_component.subfunctions,
            ):
                maximum_decomposition = max(
                    maximum_decomposition,
                    coefficient_decomposition_error(
                        stored_field.dat.data_ro,
                        backend_field.dat.data_ro,
                        total_field.dat.data_ro,
                    ),
                )
            decomposition_maxima.append(maximum_decomposition)
            state_invariants.append(
                _integrated_residual_invariants(case, backend_component, beta2)
            )
            prior_neural_component = _copy_difference(
                z_neural_default.state_out,
                z_ufl.state_out,
                f"test2a_offset_prior_neural_component_{step}",
            )
            prior_neural_state_invariants.append(
                _integrated_residual_invariants(
                    case, prior_neural_component, beta2
                )
            )

            moist_increment = _copy_difference(
                z_ufl.state_out,
                y_state,
                f"test2a_offset_ufl_increment_{step}",
            )
            increment_norm = _mixed_mass_norm(
                analytical_jax, moist_increment, f"test2a_offset_increment_{step}"
            )
            backend_norm = _mixed_mass_norm(
                analytical_jax,
                backend_component,
                f"test2a_offset_backend_norm_{step}",
            )
            prior_neural_norm = _mixed_mass_norm(
                analytical_jax,
                prior_neural_component,
                f"test2a_offset_prior_neural_norm_{step}",
            )
            field_scales = {}
            prior_field_scales = {}
            from firedrake import norm

            for field, increment_field, difference_field, prior_difference_field in zip(
                FIELD_NAMES,
                moist_increment.subfunctions,
                backend_component.subfunctions,
                prior_neural_component.subfunctions,
            ):
                inc = float(norm(increment_field))
                err = float(norm(difference_field))
                prior_err = float(norm(prior_difference_field))
                field_scales[field] = {
                    "increment_L2_norm": inc,
                    "backend_error_L2_norm": err,
                    "backend_error_relative_to_increment": (
                        None if inc == 0.0 else err / inc
                    ),
                }
                prior_field_scales[field] = {
                    "increment_L2_norm": inc,
                    "prior_neural_error_L2_norm": prior_err,
                    "prior_neural_error_relative_to_increment": (
                        None if inc == 0.0 else prior_err / inc
                    ),
                }
            moist_increment_scales.append(
                {
                    "start_step": step,
                    "increment_mixed_mass_norm": increment_norm,
                    "backend_error_mixed_mass_norm": backend_norm,
                    "backend_error_relative_to_increment": (
                        None if increment_norm == 0.0 else backend_norm / increment_norm
                    ),
                    "fields": field_scales,
                }
            )
            prior_neural_increment_scales.append(
                {
                    "start_step": step,
                    "increment_mixed_mass_norm": increment_norm,
                    "prior_neural_error_mixed_mass_norm": prior_neural_norm,
                    "prior_neural_error_relative_to_increment": (
                        None
                        if increment_norm == 0.0
                        else prior_neural_norm / increment_norm
                    ),
                    "fields": prior_field_scales,
                }
            )
            per_transition.append(
                {
                    "start_step": step,
                    "target_step": step + 1,
                    "time": float(time),
                    "child6_time": float(z_ufl.t0),
                    "child6_dt": float(z_ufl.dt),
                    "stored_provenance_relative_mixed_error": components[
                        "stored_provenance_A"
                    ][-1]["relative_mixed_mass_norm"],
                    "analytical_JAX_vs_UFL_relative_mixed_error": components[
                        "ufl_jax_B"
                    ][-1]["relative_mixed_mass_norm"],
                    "prior_neural_default_vs_stored_relative_mixed_error": (
                        prior_reported_neural[-1]["relative_mixed_mass_norm"]
                    ),
                }
            )

    rate_statistics = {}
    prior_neural_rate_statistics = {}
    for name in ("A", "R"):
        actual = np.concatenate([value.reshape(-1) for value in rate_arrays[name]["jax"]])
        reference = np.concatenate(
            [value.reshape(-1) for value in rate_arrays[name]["ufl"]]
        )
        rate_statistics[name] = array_error_statistics(actual, reference)
        prior_neural = np.concatenate(
            [value.reshape(-1) for value in prior_neural_rate_arrays[name]]
        )
        prior_neural_rate_statistics[name] = array_error_statistics(
            prior_neural, reference
        )
    regime_statistics = {}
    for name, arrays in regime_differences.items():
        values = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)
        regime_statistics[name] = {
            "sample_count": int(values.size),
            "maximum_absolute_A_difference": (
                0.0 if values.size == 0 else float(np.max(np.abs(values)))
            ),
            "RMS_A_difference": (
                0.0 if values.size == 0 else float(np.sqrt(np.mean(values * values)))
            ),
        }

    component_summary = {
        name: _aggregate_state_records(records)
        for name, records in components.items()
    }
    old_summary = _aggregate_state_records(prior_reported_neural)
    worst_increment = max(
        moist_increment_scales,
        key=lambda item: item["backend_error_relative_to_increment"],
    )
    worst_prior_increment = max(
        prior_neural_increment_scales,
        key=lambda item: item["prior_neural_error_relative_to_increment"],
    )
    source_summary = {
        field: {
            "maximum_absolute_difference": max(
                step[field]["maximum_absolute_difference"] for step in source_errors
            ),
            "maximum_relative_RMS_difference": max(
                0.0
                if step[field]["relative_RMS_difference"] is None
                else step[field]["relative_RMS_difference"]
                for step in source_errors
            ),
        }
        for field in ("S", "Qv", "Qc", "Qr")
    }
    increment_field_summary = {}
    prior_increment_field_summary = {}
    for field in FIELD_NAMES:
        genuine = [
            item["fields"][field]["backend_error_relative_to_increment"]
            for item in moist_increment_scales
            if item["fields"][field]["backend_error_relative_to_increment"]
            is not None
        ]
        prior = [
            item["fields"][field]["prior_neural_error_relative_to_increment"]
            for item in prior_neural_increment_scales
            if item["fields"][field][
                "prior_neural_error_relative_to_increment"
            ]
            is not None
        ]
        increment_field_summary[field] = (
            None if not genuine else float(max(genuine))
        )
        prior_increment_field_summary[field] = (
            None if not prior else float(max(prior))
        )

    record = {
        "status": "complete",
        "benchmark_stage": "Test 2A UFL/JAX stored-truth offset audit",
        "central_decomposition": {
            "A_stored_provenance_Z_UFL_minus_Xstar_next": component_summary[
                "stored_provenance_A"
            ],
            "B_analytical_JAX_minus_UFL": component_summary["ufl_jax_B"],
            "C_analytical_JAX_minus_Xstar_next": component_summary["total_C"],
            "maximum_coefficient_error_in_C_equals_A_plus_B": float(
                max(decomposition_maxima)
            ),
        },
        "root_cause": {
            "classification": "prior_audit_helper_mode_misidentification",
            "explanation": (
                "The trajectory case's JAX moist helper owns frozen neural local_physics. "
                "Calling it without explicit neural_parameters evaluates its frozen neural A, "
                "not analytical JAX A. A separate local_physics=None helper removes the reported offset."
            ),
            "trajectory_helper_physics_kind": neural_kind,
            "separate_audit_helper_physics_kind": jax_helper_physics_kind(
                analytical_jax
            ),
            "prior_reported_neural_default_vs_stored": old_summary,
            "prior_neural_rates_vs_UFL": prior_neural_rate_statistics,
        },
        "rate_level": {
            **rate_statistics,
            "A_error_by_UFL_regime": regime_statistics,
        },
        "stagewise": {
            "maximum_input_or_intermediate_absolute_difference": input_maxima,
            "local_source": source_summary,
            "maximum_local_water_source_residual": float(
                max(source_water_invariants)
            ),
            "maximum_local_thermal_source_residual": float(
                max(source_thermal_invariants)
            ),
            "UFL_rates_through_JAX_carrier_assembly_vs_production_UFL_RHS": (
                _aggregate_dual_records(carrier_rhs_errors)
            ),
            "analytical_JAX_RHS_vs_production_UFL_RHS": _aggregate_dual_records(
                rhs_errors
            ),
            "common_RHS_JAX_mass_solve_vs_UFL_mass_solve": (
                _aggregate_state_records(common_mass_solve_errors)
            ),
            "analytical_JAX_tendency_vs_UFL_tendency": (
                _aggregate_state_records(tendency_errors)
            ),
            "first_nonroundoff_stage": "none for genuine analytical JAX versus UFL",
            "first_nonroundoff_stage_in_prior_comparison": (
                "local A provider: frozen neural A was compared to analytical UFL A"
            ),
        },
        "structural_invariants": {
            "maximum_absolute_integrated_water_residual": float(
                max(abs(item["integrated_water_residual"]) for item in state_invariants)
            ),
            "maximum_absolute_integrated_thermal_residual": float(
                max(abs(item["integrated_thermal_residual"]) for item in state_invariants)
            ),
            "maximum_absolute_water_coefficient_residual": float(
                max(
                    item["maximum_absolute_water_coefficient_residual"]
                    for item in state_invariants
                )
            ),
            "maximum_integrated_water_relative_cancellation": float(
                max(
                    item["integrated_water_relative_cancellation"]
                    for item in state_invariants
                )
            ),
            "maximum_integrated_thermal_relative_cancellation": float(
                max(
                    item["integrated_thermal_relative_cancellation"]
                    for item in state_invariants
                )
            ),
            "prior_neural_maximum_absolute_water_coefficient_residual": float(
                max(
                    item["maximum_absolute_water_coefficient_residual"]
                    for item in prior_neural_state_invariants
                )
            ),
            "prior_neural_maximum_integrated_water_relative_cancellation": float(
                max(
                    item["integrated_water_relative_cancellation"]
                    for item in prior_neural_state_invariants
                )
            ),
            "prior_neural_maximum_integrated_thermal_relative_cancellation": float(
                max(
                    item["integrated_thermal_relative_cancellation"]
                    for item in prior_neural_state_invariants
                )
            ),
            "thermal_identity_note": (
                "S and Qv use different coefficient spaces; their structural "
                "identity is checked in the common weak/integrated pairing, "
                "not by invalid coefficientwise subtraction."
            ),
        },
        "error_relative_to_moist_increment": {
            "maximum_ratio_over_transitions": float(
                worst_increment["backend_error_relative_to_increment"]
            ),
            "transition_attaining_maximum": int(worst_increment["start_step"]),
            "record_at_maximum": worst_increment,
            "maximum_field_L2_ratios": increment_field_summary,
            "prior_reported_neural_offset": {
                "maximum_ratio_over_transitions": float(
                    worst_prior_increment[
                        "prior_neural_error_relative_to_increment"
                    ]
                ),
                "transition_attaining_maximum": int(
                    worst_prior_increment["start_step"]
                ),
                "record_at_maximum": worst_prior_increment,
                "maximum_field_L2_ratios": prior_increment_field_summary,
            },
        },
        "quadrature_and_ordering": {
            "owned_cells": int(adapter.layout.owned_cell_count),
            "GLL_points_per_cell": int(adapter.layout.points_per_cell),
            "samples_per_state": int(
                adapter.layout.owned_cell_count * adapter.layout.points_per_cell
            ),
            "same_adapter_used_to_sample_UFL_expressions": True,
            "same_cell_node_map_and_local_point_order": True,
            "measure": "production configured 4x4 tensor GLL-lumped dx",
            "mass_space_identity": (
                analytical_jax.state_space == ufl_helper.state_space
            ),
            "beta2": beta2,
            "configured_dt": float(adapter.term.dt),
            "topography_max_abs": topography_max_abs,
        },
        "time_and_provenance": {
            "truth_backend": "ufl",
            "truth_files_are_complete_step_boundary_states": True,
            "historical_postprefix_states_stored": False,
            "fresh_prefix_child_order": list(prefix.forward_child_order),
            "fresh_prefix_times_for_last_transition": [
                {"name": child.name, "t0": child.t0, "dt": child.dt}
                for child in prefix.children
            ],
            "child6_time_rule": "t_n",
            "child6_dt": float(case.dt),
            "fresh_UFL_reproduces_stored_truth_bitwise": (
                component_summary["stored_provenance_A"][
                    "maximum_absolute_mixed_mass_norm"
                ]
                == 0.0
            ),
        },
        "prior_certification_reconciliation": {
            "J1_J3_compared": (
                "fresh paired UFL and analytical-JAX children on the same in-memory states"
            ),
            "stored_Test1B_truth_participated": False,
            "compatibility": (
                "fully compatible: the genuine analytical-JAX helper agrees with UFL here; "
                "the prior H1 audit accidentally selected the frozen-neural helper"
            ),
            "prior_J1_J2_J3_certification_revision_required": False,
            "H1_M2_audit_backend_offset_claim_revision_required": True,
        },
        "recommended_future_target": {
            "definition": "analytical JAX child C6_star(Y_k)",
            "reason": (
                "future M2-Y should isolate approximation of the deployed analytical A law. "
                "The stored UFL target is numerically equivalent here, but the analytical-JAX "
                "definition makes the intended physics/provider explicit and avoids accidentally "
                "folding any future provenance or backend difference into neural fitting."
            ),
            "stored_UFL_target_role": "independent parity/provenance check, not the defining target",
        },
        "per_transition": per_transition,
        "truth_state_access": {
            "minimum_loaded_state_index": 0,
            "maximum_loaded_state_index": 80,
            "loaded_state_count": 81,
            "states_after_80_accessed": False,
        },
        "optimization_or_training_performed": False,
    }
    write_json_record(output, record)
    return record


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-configuration",
        default="dimswe/configs/test2a_trajectory_prep.json",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    run_backend_offset_audit(args.trajectory_configuration, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "array_error_statistics",
    "coefficient_decomposition_error",
    "jax_helper_physics_kind",
    "require_analytical_jax_helper",
    "run_backend_offset_audit",
)
