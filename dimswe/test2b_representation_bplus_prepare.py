"""Bounded certification for Test2B structured rain-output variants.

This module has no optimizer entry point.  It reads the frozen Test2B cache,
certifies a BTPL/BTP output map and exact derivatives, and refuses to access
held-out truth states.
"""

from __future__ import annotations

import argparse
from gc import collect
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from .learned_physics.parameters import tree_dot
from .resolved_hidden_c0 import write_json_record
from .test2a_embedded_moist import parameter_pytree_sha256
from .test2b_rain_learning import (
    RainActiveNeuralMoistPhysics,
    RainMLPConfiguration,
    bplus_physical_rates,
    btp_physical_rates,
    build_model,
    initial_parameters,
    source_invariant_diagnostics,
)
from .test2b_rain_learning_campaign import (
    FixedObjective,
    OperatorObjective,
    _direction,
    _directional,
    _duality,
    load_bplus_output_map,
    load_btp_output_map,
    load_configuration,
    load_preparation,
    trajectory_objective,
)
from .test2b_rain_learning_prepare import file_sha256


REPRESENTATIONS = ("BPLUS", "BTPL", "BTP")
ACTIVE_WINDOW_START = 60


def _controlled_local_certificate(normalization, representation):
    parameters = initial_parameters(representation)
    q_precip = (
        normalization.btp_q_precip
        if representation == "BTP" else normalization.bplus_q_precip
    )
    threshold_offset = (
        1.0e-6
        if representation == "BTP" else normalization.bplus_delta_q_scale
    )
    shape = (2, 3)
    h = jnp.full(shape, 700.0, dtype=jnp.float64)
    fields = {"B": jnp.zeros(shape, dtype=jnp.float64)}
    moist = {
        "g": jnp.asarray(9.80616),
        "q0": jnp.asarray(0.002),
        "H0": jnp.asarray(750.0),
        "gamma_r": jnp.asarray(0.001),
        "qprecip": jnp.asarray(q_precip),
        "L": jnp.asarray(10.0),
        "configured_dt": jnp.asarray(100.0),
    }
    base = {
        "h": h,
        "S": jnp.full(shape, 6800.0, dtype=jnp.float64),
        "Qv": jnp.full(shape, 1.4, dtype=jnp.float64),
    }
    states = {
        "below": {**base, "Qc": h * (
            q_precip - threshold_offset
        )},
        "threshold": {**base, "Qc": h * q_precip},
        "above": {**base, "Qc": h * (
            q_precip + threshold_offset
        )},
    }
    provider = RainActiveNeuralMoistPhysics(
        representation, parameters, normalization, use_jit=False
    )
    results = {
        name: provider.combined_kernel(state, fields, moist)
        for name, state in states.items()
    }
    raw = jnp.asarray(((0.25, -0.4),), dtype=jnp.float64)
    scalar_h = jnp.asarray((700.0,), dtype=jnp.float64)
    scalar_direction = jnp.ones(1, dtype=jnp.float64)
    rain_map = btp_physical_rates if representation == "BTP" else bplus_physical_rates
    rain = lambda qc: rain_map(
        raw, scalar_h, qc, q_precip, normalization
    )[1]
    below_qc = scalar_h * (q_precip - threshold_offset)
    above_qc = scalar_h * (q_precip + threshold_offset)
    _, tangent_below = jax.jvp(
        rain, (below_qc,), (scalar_direction,)
    )
    _, tangent_above = jax.jvp(
        rain, (above_qc,), (scalar_direction,)
    )
    expected_above = (
        jnp.asarray(0.0)
        if representation == "BTP"
        else normalization.sigma_r_active
        * jax.nn.softplus(raw[0, 1])
        / jnp.log(2.0)
        / normalization.bplus_delta_q_scale
        / scalar_h[0]
    )
    direction = _direction(parameters)
    source, source_tangent = provider.parameter_jvp(
        states["above"], direction, fields, moist
    )
    covector = {
        name: jnp.full_like(value, 0.2 + index)
        for index, (name, value) in enumerate(source.items())
    }
    adjoint = provider.parameter_vjp(
        states["above"], covector, fields, moist
    )
    tangent_pairing = float(sum(
        jnp.vdot(source_tangent[name], covector[name])
        for name in source_tangent
    ))
    adjoint_pairing = float(tree_dot(direction, adjoint))
    epsilon = 1.0e-3
    plus = jax.tree_util.tree_map(
        lambda value, delta: value + epsilon * delta, parameters, direction
    )
    minus = jax.tree_util.tree_map(
        lambda value, delta: value - epsilon * delta, parameters, direction
    )
    scalar = lambda active: sum(
        jnp.vdot(
            provider.combined_with_parameters(
                states["above"], fields, moist, active
            )["source"][name], covector[name]
        )
        for name in covector
    )
    finite_difference = float((scalar(plus) - scalar(minus)) / (2.0 * epsilon))
    b_parameters = initial_parameters("B")
    features = jnp.stack(
        tuple(states["above"][name] for name in ("h", "S", "Qv", "Qc"))
        + (fields["B"],), axis=-1,
    )
    old_physical = build_model("B")(
        b_parameters, normalization.normalize_features(features)
    ) * jnp.asarray(
        (normalization.sigma_a, normalization.sigma_r_active),
        dtype=jnp.float64,
    )
    old_a = old_physical[..., 0]
    bplus_a = np.asarray(results["above"]["rates"]["A"])
    invariant = source_invariant_diagnostics(
        results["above"]["source"], 98.0616
    )
    return {
        "below_threshold_R_maximum_absolute": float(np.max(np.abs(
            np.asarray(results["below"]["rates"]["R"])
        ))),
        "at_threshold_R_maximum_absolute": float(np.max(np.abs(
            np.asarray(results["threshold"]["rates"]["R"])
        ))),
        "above_threshold_R_minimum": float(np.min(
            np.asarray(results["above"]["rates"]["R"])
        )),
        "below_threshold_state_derivative": float(tangent_below[0]),
        "above_threshold_state_derivative": float(tangent_above[0]),
        "above_threshold_expected_state_derivative": float(expected_above),
        "above_threshold_state_derivative_relative_error": float(
            abs(tangent_above[0] - expected_above)
            / max(abs(float(expected_above)), np.finfo(float).tiny)
        ),
        "parameter_tangent_pairing": tangent_pairing,
        "parameter_adjoint_pairing": adjoint_pairing,
        "parameter_duality_relative_error": abs(
            tangent_pairing - adjoint_pairing
        ) / max(abs(tangent_pairing), abs(adjoint_pairing), np.finfo(float).tiny),
        "parameter_directional_finite_difference": finite_difference,
        "parameter_directional_relative_error": abs(
            tangent_pairing - finite_difference
        ) / max(abs(tangent_pairing), abs(finite_difference), np.finfo(float).tiny),
        "A_head_bitwise_equal_to_B": bool(np.array_equal(
            bplus_a, np.asarray(old_a)
        )),
        "source_invariants": invariant,
    }


def certify(
    configuration_path, preparation_path, rain_output_path, output_path,
    representation="BPLUS",
):
    """Run bounded local/fixed/one-window recursive certificates."""
    from .test2a_trajectory import reset_windows

    started = perf_counter()
    configuration = load_configuration(configuration_path)
    metadata, normalization, data, matrices = load_preparation(preparation_path)
    if representation == "BTP":
        normalization, rain_output = load_btp_output_map(
            rain_output_path, preparation_path, normalization
        )
    else:
        normalization, rain_output = load_bplus_output_map(
            rain_output_path, preparation_path, normalization, data
        )
    parameters = initial_parameters(representation)
    b_parameters = initial_parameters("B")
    if parameter_pytree_sha256(parameters) != parameter_pytree_sha256(b_parameters):
        raise RuntimeError("structured-rain initialization parity failed")
    direction = _direction(parameters)
    index = ACTIVE_WINDOW_START
    weights = np.broadcast_to(
        data["carrier_weights"], data["x_A"][index:index + 1].shape
    )
    operator = OperatorObjective(
        representation, data["x_features"][index:index + 1],
        data["x_A"][index:index + 1], data["x_R"][index:index + 1],
        weights, normalization,
    )
    fixed_x = FixedObjective(
        representation, data["x_features"][index:index + 1],
        data["x_A"][index:index + 1], data["x_R"][index:index + 1],
        matrices, metadata["m2x_denominator"], normalization,
    )
    print(f"{representation} certificate: local and fixed objectives", flush=True)
    local_certificate = _controlled_local_certificate(normalization, representation)
    operator_directional = _directional(
        operator, parameters, direction, 2.0e-6
    )
    fixed_x_directional = _directional(
        fixed_x, parameters, direction, 2.0e-6
    )
    del operator, fixed_x
    collect()
    print(f"{representation} certificate: H1 active window", flush=True)
    case_h1, h1 = trajectory_objective(
        configuration, metadata, normalization, representation, parameters, 1,
        windows=reset_windows((index,), 1, "accumulated", (1.0,)),
    )
    print(f"{representation} certificate: H1 trajectory constructed", flush=True)
    fixed_y = FixedObjective(
        representation, data["y_features"][index:index + 1],
        data["y_A"][index:index + 1], data["y_R"][index:index + 1],
        matrices, metadata["common_horizon_denominator"] / 10000.0,
        normalization,
    )
    print(f"{representation} certificate: H1 cache constructed", flush=True)
    cached_value, cached_gradient = fixed_y.value_and_gradient(parameters)
    print(f"{representation} certificate: H1 cache gradient complete", flush=True)
    literal_value, literal_gradient = h1.value_and_gradient(parameters)
    print(f"{representation} certificate: H1 literal gradient complete", flush=True)
    cached_flat, _ = ravel_pytree(cached_gradient)
    literal_flat, _ = ravel_pytree(literal_gradient)
    difference = np.asarray(cached_flat - literal_flat)
    h1_duality = _duality(case_h1, h1, parameters, 1, start=index)
    print(f"{representation} certificate: H1 duality complete", flush=True)
    h1_directional = _directional(
        h1, parameters, direction, 2.0e-5
    )
    print(f"{representation} certificate: H1 directional complete", flush=True)
    h1_record = {
        "window_start": index,
        "cached_value": cached_value,
        "literal_value": literal_value,
        "value_absolute_difference": abs(cached_value - literal_value),
        "gradient_absolute_error": float(np.linalg.norm(difference)),
        "gradient_relative_error": float(
            np.linalg.norm(difference)
            / max(np.linalg.norm(np.asarray(literal_flat)), np.finfo(float).tiny)
        ),
        "state_tangent_adjoint": h1_duality,
        "parameter_directional": h1_directional,
    }
    h1.clear_parameter_tape()
    del fixed_y, cached_gradient, literal_gradient, cached_flat, literal_flat
    del difference, case_h1, h1
    collect()
    recursive = {}
    for horizon in (2, 5):
        print(f"{representation} certificate: H{horizon} active window", flush=True)
        case, objective = trajectory_objective(
            configuration, metadata, normalization, representation,
            parameters, horizon,
            windows=reset_windows(
                (index,), horizon, "accumulated", (1.0,) * horizon
            ),
        )
        recursive[f"H{horizon}"] = {
            "state_tangent_adjoint": _duality(
                case, objective, parameters, horizon, start=index
            ),
            "parameter_directional": _directional(
                objective, parameters, direction, 2.0e-5
            ),
            "objective_constructed": True,
            "window_start": index,
            "truth_targets": list(range(index + 1, index + horizon + 1)),
        }
        objective.clear_parameter_tape()
        del case, objective
        collect()
    result = {
        "status": "complete",
        "representation": representation,
        "evaluation_only": True,
        "optimizer_instantiated": False,
        "production_training_launched": False,
        "truth_generated": False,
        "training_truth_states": [0, 80],
        "heldout_accessed": False,
        "configuration_sha256": file_sha256(configuration_path),
        "fixed_preparation_sha256": file_sha256(preparation_path),
        "rain_output_preparation_sha256": file_sha256(rain_output_path),
        "rain_output_map_payload_sha256": rain_output["payload_sha256"],
        "architecture": RainMLPConfiguration(representation).to_record(),
        "B_architecture": RainMLPConfiguration("B").to_record(),
        "seed0_variant_parameter_pytree_sha256": parameter_pytree_sha256(parameters),
        "seed0_B_parameter_pytree_sha256": parameter_pytree_sha256(b_parameters),
        "local_output_map": local_certificate,
        "M1_active_state_directional": operator_directional,
        "M2_X_active_state_directional": fixed_x_directional,
        "H1_cache_literal": h1_record,
        "recursive": recursive,
        "objective_runner_compatibility": {
            "stages": ["M1", "M2-X", "H1", "H2", "H5"],
            "H1_fixed_cache": True,
            "H2_first_recursive": True,
            "fresh_optimizer_required_per_stage": True,
            "secant_history_transferred": False,
        },
        "wall_seconds": float(perf_counter() - started),
    }
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    write_json_record(destination, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--rain-output-preparation")
    parser.add_argument("--bplus-preparation")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="BPLUS")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rain_output_path = args.rain_output_preparation or args.bplus_preparation
    if rain_output_path is None:
        parser.error("a rain-output preparation is required")
    if args.rain_output_preparation and args.bplus_preparation:
        parser.error("use only one rain-output preparation argument")
    certify(
        args.configuration, args.preparation, rain_output_path,
        args.output, args.representation,
    )


if __name__ == "__main__":
    main()
