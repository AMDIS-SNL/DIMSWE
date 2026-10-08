"""Certification of the production dry RK4 and complete dry Lie HVP."""

from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest
from firedrake import COMM_SELF, Cofunction, Function, assemble, inner

import dimswe.meshes as dimswe_meshes
from dimswe.dry_lie_hvp import (
    DryLieHVPResult,
    DryLiePrimalCache,
    DryLieReducedGradientResult,
    DryLieReducedHVPResult,
    DryLieReverseResult,
    DryLieTangentCache,
    DryRK4HVPResult,
    DryRK4PrimalCache,
    DryRK4ReverseResult,
    DryRK4TangentCache,
)
from dimswe.logger import EmptyLogger
from dimswe.models import get_model
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.timestepping import get_timestepper


CFG = "tests/tswe_rol_small.cfg"
PHYSICAL_C0 = 0.14
PHYSICAL_C0_DIRECTION = 0.035
EPSILONS = (4.0e-2, 2.0e-2, 1.0e-2, 5.0e-3)
DIAGNOSTIC_EPSILONS = (0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001)
DIAGNOSTIC_DTS = (100.0, 50.0, 25.0, 12.5)
STAGE_LOCAL_EXACT_FLOOR = 1.0e-11
STAGE_LOCAL_FACTOR_RATIO_INTERVAL = (3.8, 4.2)
STAGE_LOCAL_FACTOR_RATIO_COUNT = 3


def _serial_solver_parameters():
    parameters = deepcopy(overall_solver_parameters)
    direct = {"ksp_type": "preonly", "pc_type": "lu"}
    for name in (
        "erkstage-f",
        "erkstage-aux",
        "erkstage-mu",
        "erkstage-muaux",
        "erk-dlambda",
        "erk-grad",
    ):
        parameters[name] = direct
    return parameters


def _function_values(value):
    with value.dat.vec_ro as vector:
        return vector.getArray(readonly=True).copy()


def _dual_values(value):
    with value.dat.vec_ro as vector:
        return vector.getArray(readonly=True).copy()


def _relative_error(computed, expected):
    denominator = max(np.linalg.norm(expected), np.finfo(float).tiny)
    return np.linalg.norm(computed - expected) / denominator


def _symmetric_relative_scalar_error(left, right):
    scale = max(abs(left), abs(right), np.finfo(float).tiny)
    return abs(left - right) / scale


def _l2_norm(case, value):
    return float(
        assemble(inner(value, value) * case["model"].spaces.dx)
    ) ** 0.5


def _json_normalize(value):
    """Recursively convert NumPy payload values to native JSON types."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return _json_normalize(value.tolist())
    if isinstance(value, dict):
        return {key: _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    return value


def _diagnostic_emit(record_property, name, payload):
    serialized = json.dumps(_json_normalize(payload), sort_keys=True)
    record_property(name, serialized)
    print(f"DRY_LIE_DIAGNOSTIC {name}={serialized}")


def _terminal_objective(case, state):
    residual = _terminal_residual(case, state, "diagnostic_objective_residual")
    return 0.5 * float(
        assemble(inner(residual, residual) * case["model"].spaces.dx)
    )


def _function_axpy(base, terms, name):
    result = base.copy(deepcopy=True)
    result.rename(name)
    for scale, value in terms:
        with result.dat.vec as result_vec, value.dat.vec_ro as value_vec:
            result_vec.axpy(float(scale), value_vec)
    return result


def _new_state(model, name):
    state, _, _ = model.get_x_var(name)
    return state


def _new_full_state(model, name):
    return model.get_full_var(name, split_x_and_aux=True)


@pytest.fixture(scope="module")
def production_case():
    parameters = get_parameters(CFG)
    logger = EmptyLogger()
    original_periodic_rectangle_mesh = dimswe_meshes.PeriodicRectangleMesh

    def comm_self_periodic_rectangle_mesh(*args, **kwargs):
        kwargs["comm"] = COMM_SELF
        return original_periodic_rectangle_mesh(*args, **kwargs)

    dimswe_meshes.PeriodicRectangleMesh = comm_self_periodic_rectangle_mesh
    try:
        model = get_model(parameters, logger, has_dynamics_statistics=False)
    finally:
        dimswe_meshes.PeriodicRectangleMesh = original_periodic_rectangle_mesh
    assert model.mesh.comm.size == 1

    coefficient, coefficient_sub, _ = model.get_coeff_var("coefficient")
    state_container, state_sub, _ = _new_full_state(
        model, "production_dry_lie_state"
    )
    time = model.get_t_var()
    model.initialize(state_sub, time)
    model.set_coeffs(parameters, coefficient_sub)
    split = get_timestepper(
        parameters, model, logger, _serial_solver_parameters()
    )
    split.set_coeff(coefficient)
    dry, hyperviscosity = split.time_integrators

    state = state_container[0]
    direction = _new_state(model, "production_dry_lie_direction")
    direction.assign(0.018 * state)
    direction.sub(0).assign(0.027 * state.sub(0))
    direction.sub(1).assign(-0.013 * state.sub(1))
    direction.sub(2).assign(0.021 * state.sub(2))
    probe = _new_state(model, "production_dry_lie_probe")
    probe.assign(-0.011 * state)
    probe.sub(0).assign(0.019 * state.sub(0))
    probe.sub(1).assign(0.014 * state.sub(1))
    probe_two = _new_state(model, "production_dry_lie_probe_two")
    probe_two.assign(0.009 * state)
    probe_two.sub(0).assign(-0.016 * state.sub(0))
    probe_two.sub(1).assign(0.023 * state.sub(1))
    probe_two.sub(2).assign(-0.012 * state.sub(2))
    target = _new_state(model, "production_dry_lie_target")
    target.assign(0.94 * state)

    return {
        "parameters": parameters,
        "model": model,
        "coefficient": coefficient,
        "coefficient_sub": coefficient_sub,
        "split": split,
        "dry": dry,
        "hyperviscosity": hyperviscosity,
        "state": state,
        "direction": direction,
        "probe": probe,
        "probe_two": probe_two,
        "target": target,
        "time": time,
        "dt": float(parameters["timestepping"]["dt"]),
    }


def _set_c0(case, c0):
    case["coefficient_sub"]["c0"].assign(float(c0))
    case["split"].set_coeff(case["coefficient"])


def _legacy_child_forward(case, child, state, time, name):
    output, output_sub, _ = _new_full_state(case["model"], name)
    child.reset_internal_vars()
    child.take_forward_step(
        output, output_sub, [state], float(time), case["dt"]
    )
    return output[0].copy(deepcopy=True)


def _legacy_lie_trajectory(case, nsteps, state, c0, name, dt=None):
    applied_dt = case["dt"] if dt is None else float(dt)
    _set_c0(case, c0)
    states = [state.copy(deepcopy=True)]
    for n in range(nsteps):
        output, output_sub, _ = _new_full_state(
            case["model"], f"{name}_state_{n + 1}"
        )
        case["split"].reset_internal_vars()
        case["split"].take_forward_step(
            output,
            output_sub,
            [states[-1]],
            float(case["time"]) + n * applied_dt,
            applied_dt,
        )
        states.append(output[0].copy(deepcopy=True))
    return tuple(states)


def _terminal_residual(case, state, name):
    result = _new_state(case["model"], name)
    result.assign(state - case["target"])
    return result


def _legacy_child_incoming_adjoint(case, child, state, lambda_plus, name):
    delta_gradient, _, _ = case["model"].get_coeff_var(
        f"{name}_delta_gradient"
    )
    delta_lambda = _new_state(case["model"], f"{name}_delta_lambda")
    child.reset_internal_vars()
    child.take_adjoint_step(
        delta_gradient,
        delta_lambda,
        lambda_plus,
        [state],
        float(case["time"]) + case["dt"],
        case["dt"],
    )
    result = _new_state(case["model"], f"{name}_lambda_in")
    result.assign(lambda_plus + delta_lambda)
    return result


def _legacy_reduced_gradient(case, nsteps, state, c0, name, dt=None):
    applied_dt = case["dt"] if dt is None else float(dt)
    states = _legacy_lie_trajectory(
        case, nsteps, state, c0, name, dt=applied_dt
    )
    current = _terminal_residual(case, states[-1], f"{name}_terminal")
    physical_c0_gradient = 0.0
    c0_index = case["model"].get_coeff_list().index("c0")

    for n in range(nsteps - 1, -1, -1):
        delta_gradient, _, _ = case["model"].get_coeff_var(
            f"{name}_delta_gradient_{n}"
        )
        delta_lambda = _new_state(
            case["model"], f"{name}_delta_lambda_{n}"
        )
        case["split"].reset_internal_vars()
        _, gradient = case["split"].take_adjoint_step(
            delta_gradient,
            delta_lambda,
            current,
            [states[n]],
            float(case["time"]) + (n + 1) * applied_dt,
            applied_dt,
        )
        physical_c0_gradient += float(gradient[c0_index])
        updated = _new_state(case["model"], f"{name}_lambda_{n}")
        updated.assign(current + delta_lambda)
        current = updated
    return physical_c0_gradient, current, states


def _perturbed_state(case, base, direction, sign, epsilon, name):
    result = _new_state(case["model"], name)
    result.assign(base + float(sign * epsilon) * direction)
    return result


def _centered_regime(errors, *, final_tolerance):
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert ratios[0] > 3.0
    assert ratios[1] > 3.0
    assert errors[-1] < errors[0]
    assert errors[-1] < final_tolerance
    return ratios


def _centered_exact_regime(errors, *, moderate_tolerance):
    # A centered difference of a map that is affine in the selected direction
    # has no truncation term.  Moderate epsilons expose the derivative, while
    # smaller epsilons can only add solve/subtraction roundoff.
    assert errors[0] < moderate_tolerance
    assert errors[1] < moderate_tolerance
    return [
        errors[i] / max(errors[i + 1], np.finfo(float).tiny)
        for i in range(len(errors) - 1)
    ]


def _stage_local_exact_regime(errors):
    """Certify centered convergence or an immediate roundoff-scale floor."""
    ratios = [
        errors[i] / max(errors[i + 1], np.finfo(float).tiny)
        for i in range(len(errors) - 1)
    ]
    if max(errors[:4]) < STAGE_LOCAL_EXACT_FLOOR:
        # The first four epsilons are the largest/moderate portion of the
        # diagnostic ladder.  When the exact derivative starts at roundoff,
        # smaller epsilons can only amplify solve/subtraction noise.
        factor_window_start = None
        factor_window_end = None
        factor_window_ratios = []
        factor_window_monotonic = None
        minimum_error_after_factor_window = min(errors)
        certified = True
        classification = "immediate_roundoff_floor"
    else:
        ratio_minimum, ratio_maximum = STAGE_LOCAL_FACTOR_RATIO_INTERVAL
        factor_window_start = next(
            (
                start
                for start in range(
                    len(ratios) - STAGE_LOCAL_FACTOR_RATIO_COUNT + 1
                )
                if all(
                    ratio_minimum <= ratio <= ratio_maximum
                    for ratio in ratios[
                        start : start + STAGE_LOCAL_FACTOR_RATIO_COUNT
                    ]
                )
            ),
            None,
        )
        if factor_window_start is None:
            factor_window_end = None
            factor_window_ratios = []
            factor_window_monotonic = False
            minimum_error_after_factor_window = min(errors)
            certified = False
            classification = "no_certified_stage_local_regime"
        else:
            factor_window_end = (
                factor_window_start + STAGE_LOCAL_FACTOR_RATIO_COUNT - 1
            )
            factor_window_ratios = ratios[
                factor_window_start : factor_window_end + 1
            ]
            factor_window_monotonic = all(
                errors[index + 1] < errors[index]
                for index in range(
                    factor_window_start, factor_window_end + 1
                )
            )
            minimum_error_after_factor_window = min(
                errors[factor_window_end + 1 :]
            )
            certified = (
                factor_window_monotonic
                and minimum_error_after_factor_window
                < STAGE_LOCAL_EXACT_FLOOR
            )
            classification = "factor_of_four_then_roundoff_floor"
    return {
        "certified": certified,
        "classification": classification,
        "factor_window_start": factor_window_start,
        "factor_window_end": factor_window_end,
        "factor_window_ratios": factor_window_ratios,
        "factor_window_monotonic": factor_window_monotonic,
        "minimum_error_after_factor_window": (
            minimum_error_after_factor_window
        ),
        "factor_ratio_interval": STAGE_LOCAL_FACTOR_RATIO_INTERVAL,
        "relative_error_floor_threshold": STAGE_LOCAL_EXACT_FLOOR,
        "ratios": ratios,
    }


def test_diagnostic_dry_rk4_independent_tangent_and_adjoint_oracles(
    production_case, record_property
):
    """Characterize new and legacy first derivatives without choosing an oracle."""
    case = production_case
    dry = case["dry"]
    primal = dry.take_dry_forward_step_cached(
        case["state"], case["time"], case["dt"]
    )
    residual = _terminal_residual(
        case, primal.state_out, "diagnostic_dry_terminal_residual"
    )
    lambda_plus_star = dry.dry_rk4_state_mass_map(residual)
    new_reverse = dry.take_dry_adjoint_step_cached(primal, lambda_plus_star)
    new_riesz = dry.dry_rk4_state_riesz_representative(
        new_reverse.state_adjoint_in
    )
    legacy_riesz = _legacy_child_incoming_adjoint(
        case,
        dry,
        case["state"],
        residual,
        "diagnostic_dry_legacy_adjoint",
    )
    legacy_dual = dry.dry_rk4_state_mass_map(legacy_riesz)
    new_roundtrip = dry.dry_rk4_state_mass_map(new_riesz)
    field_difference = _function_axpy(
        new_riesz,
        ((-1.0, legacy_riesz),),
        "diagnostic_new_legacy_adjoint_difference",
    )

    field_discrepancy = _relative_error(
        _function_values(new_riesz), _function_values(legacy_riesz)
    )
    field_discrepancy_norm = np.linalg.norm(
        _function_values(new_riesz) - _function_values(legacy_riesz)
    )
    dual_discrepancy = _relative_error(
        _dual_values(new_reverse.state_adjoint_in),
        _dual_values(legacy_dual),
    )
    dual_discrepancy_norm = np.linalg.norm(
        _dual_values(new_reverse.state_adjoint_in)
        - _dual_values(legacy_dual)
    )
    new_mass_roundtrip_error = _relative_error(
        _dual_values(new_roundtrip),
        _dual_values(new_reverse.state_adjoint_in),
    )

    probe_payload = []
    probes = (
        ("direction", case["direction"]),
        ("probe", case["probe"]),
        ("probe_two", case["probe_two"]),
    )
    for probe_name, probe in probes:
        tangent = dry.take_dry_tangent_step(primal, probe)
        terminal_pairing = dry.dry_rk4_dual_pairing(
            lambda_plus_star, tangent.state_direction_out
        )
        new_directional_gradient = dry.dry_rk4_dual_pairing(
            new_reverse.state_adjoint_in, probe
        )
        legacy_directional_gradient = float(
            assemble(inner(legacy_riesz, probe) * case["model"].spaces.dx)
        )

        tangent_fd_errors = []
        tangent_fd_absolute_errors = []
        centered_objective_derivatives = []
        new_scalar_errors = []
        legacy_scalar_errors = []
        new_scalar_absolute_errors = []
        legacy_scalar_absolute_errors = []
        for index, epsilon in enumerate(DIAGNOSTIC_EPSILONS):
            plus_state = _perturbed_state(
                case,
                case["state"],
                probe,
                1.0,
                epsilon,
                f"diagnostic_probe_plus_{probe_name}_{index}",
            )
            minus_state = _perturbed_state(
                case,
                case["state"],
                probe,
                -1.0,
                epsilon,
                f"diagnostic_probe_minus_{probe_name}_{index}",
            )
            plus = _legacy_child_forward(
                case,
                dry,
                plus_state,
                case["time"],
                f"diagnostic_probe_plus_forward_{probe_name}_{index}",
            )
            minus = _legacy_child_forward(
                case,
                dry,
                minus_state,
                case["time"],
                f"diagnostic_probe_minus_forward_{probe_name}_{index}",
            )
            centered_tangent = (
                _function_values(plus) - _function_values(minus)
            ) / (2.0 * epsilon)
            tangent_fd_errors.append(
                _relative_error(
                    _function_values(tangent.state_direction_out),
                    centered_tangent,
                )
            )
            tangent_fd_absolute_errors.append(
                np.linalg.norm(
                    _function_values(tangent.state_direction_out)
                    - centered_tangent
                )
            )
            centered_objective = (
                _terminal_objective(case, plus)
                - _terminal_objective(case, minus)
            ) / (2.0 * epsilon)
            centered_objective_derivatives.append(centered_objective)
            new_scalar_errors.append(
                _symmetric_relative_scalar_error(
                    new_directional_gradient, centered_objective
                )
            )
            legacy_scalar_errors.append(
                _symmetric_relative_scalar_error(
                    legacy_directional_gradient, centered_objective
                )
            )
            new_scalar_absolute_errors.append(
                abs(new_directional_gradient - centered_objective)
            )
            legacy_scalar_absolute_errors.append(
                abs(legacy_directional_gradient - centered_objective)
            )

        probe_payload.append(
            {
                "probe": probe_name,
                "tangent_fd_relative_errors": tangent_fd_errors,
                "tangent_fd_absolute_errors": tangent_fd_absolute_errors,
                "new_pairing_left": terminal_pairing,
                "new_pairing_right": new_directional_gradient,
                "new_pairing_absolute_error": abs(
                    terminal_pairing - new_directional_gradient
                ),
                "new_pairing_relative_error": (
                    _symmetric_relative_scalar_error(
                        terminal_pairing, new_directional_gradient
                    )
                ),
                "legacy_pairing_left": terminal_pairing,
                "legacy_pairing_right": legacy_directional_gradient,
                "legacy_pairing_absolute_error": abs(
                    terminal_pairing - legacy_directional_gradient
                ),
                "legacy_pairing_relative_error": (
                    _symmetric_relative_scalar_error(
                        terminal_pairing, legacy_directional_gradient
                    )
                ),
                "centered_objective_directional_derivatives": (
                    centered_objective_derivatives
                ),
                "new_scalar_directional_gradient": new_directional_gradient,
                "legacy_scalar_directional_gradient": (
                    legacy_directional_gradient
                ),
                "new_scalar_directional_gradient_relative_errors": (
                    new_scalar_errors
                ),
                "new_scalar_directional_gradient_absolute_errors": (
                    new_scalar_absolute_errors
                ),
                "legacy_scalar_directional_gradient_relative_errors": (
                    legacy_scalar_errors
                ),
                "legacy_scalar_directional_gradient_absolute_errors": (
                    legacy_scalar_absolute_errors
                ),
            }
        )

    payload = {
        "epsilons": DIAGNOSTIC_EPSILONS,
        "new_mass_roundtrip_relative_error": new_mass_roundtrip_error,
        "new_vs_legacy_riesz_field_difference_l2_norm": _l2_norm(
            case, field_difference
        ),
        "new_vs_legacy_riesz_field_difference_coefficient_norm": (
            field_discrepancy_norm
        ),
        "new_vs_legacy_riesz_field_relative_discrepancy": field_discrepancy,
        "new_vs_mass_mapped_legacy_dual_difference_coefficient_norm": (
            dual_discrepancy_norm
        ),
        "new_vs_mass_mapped_legacy_dual_relative_discrepancy": (
            dual_discrepancy
        ),
        "probes": probe_payload,
    }
    assert len(probe_payload) == 3
    assert all(
        len(item["tangent_fd_relative_errors"])
        == len(DIAGNOSTIC_EPSILONS)
        for item in probe_payload
    )
    _diagnostic_emit(record_property, "dry_rk4_independent_first_order", payload)


def test_diagnostic_dry_rk4_stage_graph_and_local_pairings(
    production_case, record_property
):
    """Localize any reverse discrepancy to stages, edges, or conventions."""
    case = production_case
    dry = case["dry"]
    primal = dry.take_dry_forward_step_cached(
        case["state"], case["time"], case["dt"]
    )
    graph = dry.dry_rk4_graph_diagnostics(primal)

    legacy_out = _legacy_child_forward(
        case,
        dry,
        case["state"],
        case["time"],
        "diagnostic_stage_legacy_forward",
    )
    legacy_tendencies = tuple(
        dry.Fi[i][0][0].copy(deepcopy=True) for i in range(4)
    )
    legacy_stage_states = tuple(
        _function_axpy(
            case["state"],
            (
                (
                    case["dt"] * float(dry.A[i, j]),
                    legacy_tendencies[j],
                )
                for j in range(i)
                if dry.A[i, j] != 0.0
            ),
            f"diagnostic_legacy_stage_state_{i}",
        )
        for i in range(4)
    )
    cached_legacy_stage_state_errors = tuple(
        _relative_error(
            _function_values(primal.stage_states[i]),
            _function_values(legacy_stage_states[i]),
        )
        for i in range(4)
    )
    cached_legacy_tendency_errors = tuple(
        _relative_error(
            _function_values(primal.stage_tendencies[i]),
            _function_values(legacy_tendencies[i]),
        )
        for i in range(4)
    )
    cached_legacy_stage_state_bitwise = tuple(
        np.array_equal(
            _function_values(primal.stage_states[i]),
            _function_values(legacy_stage_states[i]),
        )
        for i in range(4)
    )
    cached_legacy_tendency_bitwise = tuple(
        np.array_equal(
            _function_values(primal.stage_tendencies[i]),
            _function_values(legacy_tendencies[i]),
        )
        for i in range(4)
    )

    tangent = dry.take_dry_tangent_step(primal, case["direction"])
    residual = _terminal_residual(
        case, primal.state_out, "diagnostic_stage_terminal_residual"
    )
    lambda_plus_star = dry.dry_rk4_state_mass_map(residual)
    reverse = dry.take_dry_adjoint_step_cached(primal, lambda_plus_star)
    stage_pairings = dry.dry_rk4_stage_pairing_diagnostics(tangent, reverse)

    legacy_riesz = _legacy_child_incoming_adjoint(
        case,
        dry,
        case["state"],
        residual,
        "diagnostic_stage_legacy_reverse",
    )
    legacy_reverse_auxiliaries = tuple(
        dry.mui[i][0][0].copy(deepcopy=True) for i in range(4)
    )
    reverse_auxiliary_relative_differences = tuple(
        _relative_error(
            _function_values(reverse.stages[i].reverse_auxiliary),
            _function_values(legacy_reverse_auxiliaries[i]),
        )
        for i in range(4)
    )
    tendency_dual_relative_differences = []
    for i in range(4):
        legacy_tendency_dual = dry.dry_rk4_state_mass_map(
            legacy_reverse_auxiliaries[i]
        )
        tendency_dual_relative_differences.append(
            _relative_error(
                _dual_values(reverse.stages[i].tendency_adjoint),
                _dual_values(legacy_tendency_dual),
            )
        )
    new_riesz = dry.dry_rk4_state_riesz_representative(
        reverse.state_adjoint_in
    )

    factored_accumulation_errors = []
    exact_graph_accumulation_errors = []
    for i in range(4):
        factored_expected = case["dt"] * float(dry.b[i]) * _dual_values(
            lambda_plus_star
        )
        exact_graph_expected = factored_expected.copy()
        for j in range(i + 1, 4):
            factored_expected = factored_expected + (
                case["dt"]
                * float(dry.A[j, i])
                * _dual_values(reverse.stages[j].stage_state_adjoint)
            )
            for predecessor, contribution in (
                reverse.stages[
                    j
                ].predecessor_tendency_adjoint_contributions
            ):
                if predecessor == i:
                    exact_graph_expected += _dual_values(contribution)
        factored_accumulation_errors.append(
            _relative_error(
                _dual_values(reverse.stages[i].tendency_adjoint),
                factored_expected,
            )
        )
        exact_graph_accumulation_errors.append(
            _relative_error(
                _dual_values(reverse.stages[i].tendency_adjoint),
                exact_graph_expected,
            )
        )
    expected_incoming = _dual_values(lambda_plus_star).copy()
    for stage in reverse.stages:
        expected_incoming += _dual_values(stage.stage_state_adjoint)
    incoming_accumulation_error = _relative_error(
        _dual_values(reverse.state_adjoint_in), expected_incoming
    )

    payload = {
        "graph": graph,
        "cached_vs_independent_legacy_output_relative_error": _relative_error(
            _function_values(primal.state_out), _function_values(legacy_out)
        ),
        "cached_vs_independent_legacy_output_bitwise": np.array_equal(
            _function_values(primal.state_out), _function_values(legacy_out)
        ),
        "cached_vs_legacy_stage_state_relative_errors": (
            cached_legacy_stage_state_errors
        ),
        "cached_vs_legacy_stage_tendency_relative_errors": (
            cached_legacy_tendency_errors
        ),
        "cached_vs_legacy_stage_state_bitwise": (
            cached_legacy_stage_state_bitwise
        ),
        "cached_vs_legacy_stage_tendency_bitwise": (
            cached_legacy_tendency_bitwise
        ),
        "stage_pairings": tuple(asdict(item) for item in stage_pairings),
        "new_vs_legacy_reverse_auxiliary_relative_differences": (
            reverse_auxiliary_relative_differences
        ),
        "new_tendency_dual_vs_mass_mapped_legacy_mu_relative_differences": (
            tuple(tendency_dual_relative_differences)
        ),
        "new_vs_legacy_incoming_riesz_relative_difference": _relative_error(
            _function_values(new_riesz), _function_values(legacy_riesz)
        ),
        "first_compared_stage_in_reverse_order": 3,
        "stage_tendency_adjoint_accumulation_relative_errors": tuple(
            exact_graph_accumulation_errors
        ),
        "factored_stage_tendency_adjoint_accumulation_relative_errors": tuple(
            factored_accumulation_errors
        ),
        "incoming_identity_and_stage_accumulation_relative_error": (
            incoming_accumulation_error
        ),
        "new_vs_legacy_forward_metadata_equal": tuple(
            graph["new_stage_rhs_integral_metadata"] == metadata
            for metadata in graph["legacy_forward_stage_integral_metadata"]
        ),
        "new_mass_vs_legacy_forward_solver_parameters_equal": (
            graph["new_mass_solver_parameters"]
            == graph["legacy_forward_solver_parameters"]
        ),
        "new_mass_vs_legacy_reverse_solver_parameters_equal": (
            graph["new_mass_solver_parameters"]
            == graph["legacy_reverse_solver_parameters"]
        ),
    }
    assert graph["tableau_a"] == (
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0, 0.0),
        (0.0, 0.5, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    )
    assert graph["tableau_b"] == (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)
    assert graph["reverse_stage_order"] == (3, 2, 1, 0)
    assert not any(any(item) for item in graph["stage_aliases_scratch"])
    assert not any(graph["cached_state_tendency_aliases"])
    _diagnostic_emit(record_property, "dry_rk4_stage_graph", payload)


def test_diagnostic_dry_rk4_stage_local_exact_production_form_oracle(
    production_case, record_property
):
    """Compare exact stored and reconstructed B_i derivatives stage by stage."""
    case = production_case
    dry = case["dry"]
    primal = dry.take_dry_forward_step_cached(
        case["state"], case["time"], case["dt"]
    )
    direction_tangent = dry.take_dry_tangent_step(
        primal, case["direction"]
    )
    probe_tangent = dry.take_dry_tangent_step(primal, case["probe"])
    identities = dry.dry_rk4_stage_form_identity_diagnostics()

    stage_payload = []
    for i in range(4):
        stage_state = primal.stage_states[i]
        isolated_tendency = dry.dry_rk4_production_stage_tendency(
            primal, i, stage_state
        )
        direction_payload = []
        stage_directions = (
            ("direction", direction_tangent.stage_state_directions[i]),
            ("probe", probe_tangent.stage_state_directions[i]),
        )
        for direction_name, stage_direction in stage_directions:
            exact_tangent = dry.dry_rk4_production_stage_tangent(
                primal, i, stage_state, stage_direction
            )
            reconstructed_tangent = (
                dry.dry_rk4_reconstructed_stage_tangent(
                    primal, i, stage_state, stage_direction
                )
            )
            exact_values = _function_values(exact_tangent)
            reconstructed_values = _function_values(reconstructed_tangent)
            exact_fd_relative_errors = []
            exact_fd_absolute_errors = []
            reconstructed_fd_relative_errors = []
            reconstructed_fd_absolute_errors = []
            for epsilon in DIAGNOSTIC_EPSILONS:
                plus_tendency = (
                    dry.dry_rk4_perturbed_production_stage_tendency(
                        primal, i, stage_direction, epsilon
                    )
                )
                minus_tendency = (
                    dry.dry_rk4_perturbed_production_stage_tendency(
                        primal, i, stage_direction, -epsilon
                    )
                )
                centered = (
                    _function_values(plus_tendency)
                    - _function_values(minus_tendency)
                ) / (2.0 * epsilon)
                exact_fd_relative_errors.append(
                    _relative_error(exact_values, centered)
                )
                exact_fd_absolute_errors.append(
                    np.linalg.norm(exact_values - centered)
                )
                reconstructed_fd_relative_errors.append(
                    _relative_error(reconstructed_values, centered)
                )
                reconstructed_fd_absolute_errors.append(
                    np.linalg.norm(reconstructed_values - centered)
                )

            exact_fd_regime = _stage_local_exact_regime(
                exact_fd_relative_errors
            )
            direction_payload.append(
                {
                    "direction": direction_name,
                    "exact_production_tangent_vs_stage_fd_relative_errors": (
                        exact_fd_relative_errors
                    ),
                    "exact_production_tangent_vs_stage_fd_absolute_errors": (
                        exact_fd_absolute_errors
                    ),
                    "exact_production_tangent_vs_stage_fd_error_ratios": (
                        exact_fd_regime["ratios"]
                    ),
                    "exact_production_tangent_certification": exact_fd_regime,
                    "reconstructed_tangent_vs_stage_fd_relative_errors": (
                        reconstructed_fd_relative_errors
                    ),
                    "reconstructed_tangent_vs_stage_fd_absolute_errors": (
                        reconstructed_fd_absolute_errors
                    ),
                    "exact_production_vs_reconstructed_relative_error": (
                        _relative_error(exact_values, reconstructed_values)
                    ),
                    "exact_production_vs_reconstructed_absolute_error": (
                        np.linalg.norm(exact_values - reconstructed_values)
                    ),
                }
            )

        stage_payload.append(
            {
                "stage_index": i,
                "isolated_exact_solve_vs_cached_tendency_relative_error": (
                    _relative_error(
                        _function_values(isolated_tendency),
                        _function_values(primal.stage_tendencies[i]),
                    )
                ),
                "isolated_exact_solve_vs_cached_tendency_bitwise": (
                    np.array_equal(
                        _function_values(isolated_tendency),
                        _function_values(primal.stage_tendencies[i]),
                    )
                ),
                "directions": direction_payload,
            }
        )

    for i, identity in enumerate(identities):
        production_labels = {
            label
            for coefficient in identity["production_coefficients"]
            for label in coefficient["identity_labels"]
        }
        reconstructed_labels = {
            label
            for coefficient in identity["reconstructed_coefficients"]
            for label in coefficient["identity_labels"]
        }
        assert identity["stage_index"] == i
        assert identity["production_form_is_registered_generalrk_form"]
        assert not identity["forms_are_identical_objects"]
        assert identity["production_derivative_variable_is_live_coefficient"]
        assert identity[
            "reconstructed_derivative_variable_is_live_coefficient"
        ]
        assert "production_base_state_xk" in production_labels
        assert "production_base_state_xk" not in reconstructed_labels
        assert "reconstructed_state" in reconstructed_labels
        assert "reconstructed_state" not in production_labels
        for j in range(i):
            if dry.A[i, j] != 0.0:
                assert f"production_stage_tendency_F{j}" in production_labels

    all_stage_tangents_certified = all(
        direction["exact_production_tangent_certification"]["certified"]
        for stage in stage_payload
        for direction in stage["directions"]
    )
    _diagnostic_emit(
        record_property,
        "dry_rk4_stage_local_exact_production_form",
        {
            "epsilons": DIAGNOSTIC_EPSILONS,
            "form_identities": identities,
            "stages": stage_payload,
        },
    )
    assert len(stage_payload) == 4
    assert all(len(item["directions"]) == 2 for item in stage_payload)
    assert all(
        item["isolated_exact_solve_vs_cached_tendency_bitwise"]
        for item in stage_payload
    )
    assert all_stage_tangents_certified


def test_diagnostic_dry_rk4_incremental_independent_new_reverse_oracle(
    production_case, record_property
):
    """Compare incremental reverse with centered new and legacy reverses."""
    case = production_case
    dry = case["dry"]
    primal = dry.take_dry_forward_step_cached(
        case["state"], case["time"], case["dt"]
    )
    tangent = dry.take_dry_tangent_step(primal, case["direction"])
    residual = _terminal_residual(
        case, primal.state_out, "diagnostic_incremental_terminal_residual"
    )
    lambda_plus_star = dry.dry_rk4_state_mass_map(residual)
    mu_plus_star = dry.dry_rk4_state_mass_map(tangent.state_direction_out)
    incremental = dry.take_dry_incremental_adjoint_step(
        tangent, lambda_plus_star, mu_plus_star
    )
    exact_dual = _dual_values(incremental.incremental_state_adjoint_in)
    exact_riesz = dry.dry_rk4_state_riesz_representative(
        incremental.incremental_state_adjoint_in
    )
    exact_probe_pairing = dry.dry_rk4_dual_pairing(
        incremental.incremental_state_adjoint_in, case["probe"]
    )

    centered_new_errors = []
    centered_new_absolute_errors = []
    centered_legacy_errors = []
    centered_legacy_absolute_errors = []
    centered_new_reverse_pairing_errors = []
    centered_new_reverse_pairing_absolute_errors = []
    centered_tangent_pairing_errors = []
    centered_tangent_pairing_absolute_errors = []
    centered_new_reverse_pairings = []
    centered_tangent_pairings = []
    for index, epsilon in enumerate(DIAGNOSTIC_EPSILONS):
        plus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            1.0,
            epsilon,
            f"diagnostic_incremental_plus_state_{index}",
        )
        minus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            -1.0,
            epsilon,
            f"diagnostic_incremental_minus_state_{index}",
        )
        plus_primal = dry.take_dry_forward_step_cached(
            plus_state, case["time"], case["dt"]
        )
        minus_primal = dry.take_dry_forward_step_cached(
            minus_state, case["time"], case["dt"]
        )
        plus_residual = _terminal_residual(
            case,
            plus_primal.state_out,
            f"diagnostic_incremental_plus_residual_{index}",
        )
        minus_residual = _terminal_residual(
            case,
            minus_primal.state_out,
            f"diagnostic_incremental_minus_residual_{index}",
        )
        plus_star = dry.dry_rk4_state_mass_map(plus_residual)
        minus_star = dry.dry_rk4_state_mass_map(minus_residual)
        plus_new = dry.take_dry_adjoint_step_cached(plus_primal, plus_star)
        minus_new = dry.take_dry_adjoint_step_cached(minus_primal, minus_star)
        centered_new_dual = (
            _dual_values(plus_new.state_adjoint_in)
            - _dual_values(minus_new.state_adjoint_in)
        ) / (2.0 * epsilon)
        centered_new_errors.append(
            _relative_error(exact_dual, centered_new_dual)
        )
        centered_new_absolute_errors.append(
            np.linalg.norm(exact_dual - centered_new_dual)
        )

        plus_legacy = _legacy_child_incoming_adjoint(
            case,
            dry,
            plus_state,
            plus_residual,
            f"diagnostic_incremental_plus_legacy_{index}",
        )
        minus_legacy = _legacy_child_incoming_adjoint(
            case,
            dry,
            minus_state,
            minus_residual,
            f"diagnostic_incremental_minus_legacy_{index}",
        )
        centered_legacy_riesz = (
            _function_values(plus_legacy) - _function_values(minus_legacy)
        ) / (2.0 * epsilon)
        centered_legacy_errors.append(
            _relative_error(
                _function_values(exact_riesz), centered_legacy_riesz
            )
        )
        centered_legacy_absolute_errors.append(
            np.linalg.norm(
                _function_values(exact_riesz) - centered_legacy_riesz
            )
        )

        plus_new_probe_pairing = dry.dry_rk4_dual_pairing(
            plus_new.state_adjoint_in, case["probe"]
        )
        minus_new_probe_pairing = dry.dry_rk4_dual_pairing(
            minus_new.state_adjoint_in, case["probe"]
        )
        centered_new_pairing = (
            plus_new_probe_pairing - minus_new_probe_pairing
        ) / (2.0 * epsilon)
        centered_new_reverse_pairings.append(centered_new_pairing)
        centered_new_reverse_pairing_errors.append(
            _symmetric_relative_scalar_error(
                centered_new_pairing, exact_probe_pairing
            )
        )
        centered_new_reverse_pairing_absolute_errors.append(
            abs(centered_new_pairing - exact_probe_pairing)
        )

        plus_probe_tangent = dry.take_dry_tangent_step(
            plus_primal, case["probe"]
        )
        minus_probe_tangent = dry.take_dry_tangent_step(
            minus_primal, case["probe"]
        )
        plus_tangent_pairing = dry.dry_rk4_dual_pairing(
            plus_star, plus_probe_tangent.state_direction_out
        )
        minus_tangent_pairing = dry.dry_rk4_dual_pairing(
            minus_star, minus_probe_tangent.state_direction_out
        )
        centered_tangent_pairing = (
            plus_tangent_pairing - minus_tangent_pairing
        ) / (2.0 * epsilon)
        centered_tangent_pairings.append(centered_tangent_pairing)
        centered_tangent_pairing_errors.append(
            _symmetric_relative_scalar_error(
                centered_tangent_pairing, exact_probe_pairing
            )
        )
        centered_tangent_pairing_absolute_errors.append(
            abs(centered_tangent_pairing - exact_probe_pairing)
        )

    payload = {
        "epsilons": DIAGNOSTIC_EPSILONS,
        "new_incremental_vs_centered_new_dual_relative_errors": (
            centered_new_errors
        ),
        "new_incremental_vs_centered_new_dual_absolute_errors": (
            centered_new_absolute_errors
        ),
        "new_incremental_riesz_vs_centered_legacy_riesz_relative_errors": (
            centered_legacy_errors
        ),
        "new_incremental_riesz_vs_centered_legacy_riesz_absolute_errors": (
            centered_legacy_absolute_errors
        ),
        "exact_incremental_probe_pairing": exact_probe_pairing,
        "centered_new_reverse_probe_pairings": centered_new_reverse_pairings,
        "centered_new_reverse_probe_pairing_relative_errors": (
            centered_new_reverse_pairing_errors
        ),
        "centered_new_reverse_probe_pairing_absolute_errors": (
            centered_new_reverse_pairing_absolute_errors
        ),
        "centered_scalar_tangent_pairings": centered_tangent_pairings,
        "derivative_of_tangent_pairing_relative_errors": (
            centered_tangent_pairing_errors
        ),
        "derivative_of_tangent_pairing_absolute_errors": (
            centered_tangent_pairing_absolute_errors
        ),
    }
    assert len(centered_new_errors) == len(DIAGNOSTIC_EPSILONS)
    _diagnostic_emit(
        record_property, "dry_rk4_independent_incremental", payload
    )


def test_diagnostic_multistep_dt_amplification(
    production_case, record_property
):
    """Characterize one/three-step legacy discrepancies over a dt ladder."""
    case = production_case
    split = case["split"]
    measurements = []
    for dt in DIAGNOSTIC_DTS:
        for nsteps in (1, 3):
            _set_c0(case, PHYSICAL_C0)
            new = split.terminal_least_squares_gradient(
                nsteps,
                case["state"],
                case["time"],
                dt,
                case["target"],
            )
            legacy_c0, legacy_ic, legacy_states = _legacy_reduced_gradient(
                case,
                nsteps,
                case["state"],
                PHYSICAL_C0,
                f"diagnostic_amplification_{dt}_{nsteps}",
                dt=dt,
            )
            new_ic = split.dry_lie_state_riesz_representative(
                new.initial_condition_gradient
            )
            ic_difference = _function_axpy(
                new_ic,
                ((-1.0, legacy_ic),),
                f"diagnostic_ic_difference_{dt}_{nsteps}",
            )
            new_ic_values = _function_values(new_ic)
            legacy_ic_values = _function_values(legacy_ic)
            legacy_objective = _terminal_objective(case, legacy_states[-1])
            measurements.append(
                {
                    "dt": dt,
                    "nsteps": nsteps,
                    "terminal_state_l2_norm": _l2_norm(
                        case, new.states[-1]
                    ),
                    "legacy_terminal_state_l2_norm": _l2_norm(
                        case, legacy_states[-1]
                    ),
                    "objective": new.objective_value,
                    "legacy_objective": legacy_objective,
                    "objective_relative_difference": (
                        _symmetric_relative_scalar_error(
                            new.objective_value, legacy_objective
                        )
                    ),
                    "new_physical_c0_gradient": new.physical_c0_gradient,
                    "legacy_physical_c0_gradient": legacy_c0,
                    "physical_c0_gradient_relative_difference": (
                        _symmetric_relative_scalar_error(
                            new.physical_c0_gradient, legacy_c0
                        )
                    ),
                    "new_ic_gradient_riesz_l2_norm": _l2_norm(case, new_ic),
                    "legacy_ic_gradient_l2_norm": _l2_norm(case, legacy_ic),
                    "ic_gradient_riesz_difference_l2_norm": _l2_norm(
                        case, ic_difference
                    ),
                    "ic_gradient_riesz_relative_difference": _relative_error(
                        new_ic_values, legacy_ic_values
                    ),
                    "terminal_state_relative_difference": _relative_error(
                        _function_values(new.states[-1]),
                        _function_values(legacy_states[-1]),
                    ),
                }
            )
    _set_c0(case, PHYSICAL_C0)
    assert len(measurements) == 8
    _diagnostic_emit(
        record_property,
        "dry_lie_multistep_dt_amplification",
        {"measurements": measurements},
    )


def test_dry_rk4_cached_forward_and_tangent_match_independent_legacy_ladders(
    production_case, record_property
):
    case = production_case
    dry = case["dry"]
    primal = dry.take_dry_forward_step_cached(
        case["state"], case["time"], case["dt"]
    )
    legacy = _legacy_child_forward(
        case,
        dry,
        case["state"],
        case["time"],
        "dry_forward_legacy",
    )
    assert isinstance(primal, DryRK4PrimalCache)
    assert primal.t0 == float(case["time"])
    assert primal.dt == case["dt"]
    assert len(primal.stage_states) == len(primal.stage_tendencies) == 4
    np.testing.assert_allclose(
        _function_values(primal.state_out),
        _function_values(legacy),
        rtol=0.0,
        atol=3.0e-12,
    )

    tangent = dry.take_dry_tangent_step(primal, case["direction"])
    assert isinstance(tangent, DryRK4TangentCache)
    errors = []
    for index, epsilon in enumerate(EPSILONS):
        plus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            1.0,
            epsilon,
            f"dry_tangent_plus_state_{index}",
        )
        minus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            -1.0,
            epsilon,
            f"dry_tangent_minus_state_{index}",
        )
        plus = _legacy_child_forward(
            case,
            dry,
            plus_state,
            case["time"],
            f"dry_tangent_plus_{index}",
        )
        minus = _legacy_child_forward(
            case,
            dry,
            minus_state,
            case["time"],
            f"dry_tangent_minus_{index}",
        )
        centered = (
            _function_values(plus) - _function_values(minus)
        ) / (2.0 * epsilon)
        errors.append(
            _relative_error(
                _function_values(tangent.state_direction_out), centered
            )
        )
    ratios = _centered_regime(errors, final_tolerance=2.0e-5)
    record_property(
        "dry_rk4_tangent_ladder",
        json.dumps(
            {"epsilons": EPSILONS, "relative_errors": errors, "ratios": ratios}
        ),
    )


def test_dry_rk4_dual_reverse_matches_legacy_riesz_and_pairing(
    production_case,
):
    case = production_case
    dry = case["dry"]
    primal = dry.take_dry_forward_step_cached(
        case["state"], case["time"], case["dt"]
    )
    tangent = dry.take_dry_tangent_step(primal, case["direction"])
    residual = _terminal_residual(case, primal.state_out, "dry_reverse_residual")
    lambda_plus_star = dry.dry_rk4_state_mass_map(residual)
    reverse = dry.take_dry_adjoint_step_cached(primal, lambda_plus_star)
    assert isinstance(reverse, DryRK4ReverseResult)
    assert isinstance(reverse.state_adjoint_in, Cofunction)
    assert reverse.c0_gradient == 0.0
    assert reverse.reverse_stage_order == (3, 2, 1, 0)
    assert tuple(stage.stage_index for stage in reverse.stages) == (0, 1, 2, 3)
    assert all(isinstance(stage.tendency_adjoint, Cofunction) for stage in reverse.stages)
    for i, stage in enumerate(reverse.stages):
        expected_predecessors = tuple(
            j for j in range(i) if dry.A[i, j] != 0.0
        )
        assert tuple(
            index
            for index, _ in stage.predecessor_tendency_adjoint_contributions
        ) == expected_predecessors
        assert all(
            isinstance(contribution, Cofunction)
            for _, contribution in (
                stage.predecessor_tendency_adjoint_contributions
            )
        )

    legacy = _legacy_child_incoming_adjoint(
        case, dry, case["state"], residual, "dry_reverse_legacy"
    )
    representative = dry.dry_rk4_state_riesz_representative(
        reverse.state_adjoint_in
    )
    assert _relative_error(
        _function_values(representative), _function_values(legacy)
    ) < 2.0e-11

    left = dry.dry_rk4_dual_pairing(
        lambda_plus_star, tangent.state_direction_out
    )
    right = dry.dry_rk4_dual_pairing(
        reverse.state_adjoint_in, tangent.state_direction_in
    )
    np.testing.assert_allclose(left, right, rtol=3.0e-11, atol=1.0e-13)


def test_dry_rk4_incremental_reverse_matches_legacy_adjoint_ladder_and_pairing_derivative(
    production_case, record_property
):
    case = production_case
    dry = case["dry"]
    primal = dry.take_dry_forward_step_cached(
        case["state"], case["time"], case["dt"]
    )
    tangent = dry.take_dry_tangent_step(primal, case["direction"])
    residual = _terminal_residual(
        case, primal.state_out, "dry_incremental_residual"
    )
    lambda_plus_star = dry.dry_rk4_state_mass_map(residual)
    mu_plus_star = dry.dry_rk4_state_mass_map(tangent.state_direction_out)
    result = dry.take_dry_incremental_adjoint_step(
        tangent, lambda_plus_star, mu_plus_star
    )
    assert isinstance(result, DryRK4HVPResult)
    assert isinstance(result.incremental_state_adjoint_in, Cofunction)
    assert result.c0_hvp == 0.0
    assert result.reverse_stage_order == (3, 2, 1, 0)
    for i, stage in enumerate(result.incremental_stages):
        expected_predecessors = tuple(
            j for j in range(i) if dry.A[i, j] != 0.0
        )
        assert tuple(
            index
            for index, _ in (
                stage.incremental_predecessor_tendency_adjoint_contributions
            )
        ) == expected_predecessors
    exact = dry.dry_rk4_state_riesz_representative(
        result.incremental_state_adjoint_in
    )

    errors = []
    pairing_derivative_errors = []
    probe_tangent = dry.take_dry_tangent_step(primal, case["probe"])
    exact_pairing_derivative = dry.dry_rk4_dual_pairing(
        result.incremental_state_adjoint_in, case["probe"]
    )
    for index, epsilon in enumerate(EPSILONS):
        plus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            1.0,
            epsilon,
            f"dry_incremental_plus_state_{index}",
        )
        minus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            -1.0,
            epsilon,
            f"dry_incremental_minus_state_{index}",
        )
        plus_forward = _legacy_child_forward(
            case,
            dry,
            plus_state,
            case["time"],
            f"dry_incremental_plus_forward_{index}",
        )
        minus_forward = _legacy_child_forward(
            case,
            dry,
            minus_state,
            case["time"],
            f"dry_incremental_minus_forward_{index}",
        )
        lambda_plus = _terminal_residual(
            case, plus_forward, f"dry_incremental_plus_lambda_{index}"
        )
        lambda_minus = _terminal_residual(
            case, minus_forward, f"dry_incremental_minus_lambda_{index}"
        )
        adjoint_plus = _legacy_child_incoming_adjoint(
            case,
            dry,
            plus_state,
            lambda_plus,
            f"dry_incremental_plus_adjoint_{index}",
        )
        adjoint_minus = _legacy_child_incoming_adjoint(
            case,
            dry,
            minus_state,
            lambda_minus,
            f"dry_incremental_minus_adjoint_{index}",
        )
        centered = (
            _function_values(adjoint_plus) - _function_values(adjoint_minus)
        ) / (2.0 * epsilon)
        errors.append(_relative_error(centered, _function_values(exact)))

        plus_primal = dry.take_dry_forward_step_cached(
            plus_state, case["time"], case["dt"]
        )
        minus_primal = dry.take_dry_forward_step_cached(
            minus_state, case["time"], case["dt"]
        )
        plus_probe = dry.take_dry_tangent_step(plus_primal, case["probe"])
        minus_probe = dry.take_dry_tangent_step(minus_primal, case["probe"])
        plus_star = dry.dry_rk4_state_mass_map(lambda_plus)
        minus_star = dry.dry_rk4_state_mass_map(lambda_minus)
        plus_pairing = dry.dry_rk4_dual_pairing(
            plus_star, plus_probe.state_direction_out
        )
        minus_pairing = dry.dry_rk4_dual_pairing(
            minus_star, minus_probe.state_direction_out
        )
        centered_pairing = (plus_pairing - minus_pairing) / (2.0 * epsilon)
        pairing_derivative_errors.append(
            abs(centered_pairing - exact_pairing_derivative)
            / max(abs(exact_pairing_derivative), np.finfo(float).tiny)
        )

    ratios = _centered_regime(errors, final_tolerance=3.0e-5)
    pairing_ratios = _centered_regime(
        pairing_derivative_errors, final_tolerance=3.0e-5
    )
    record_property(
        "dry_rk4_incremental_ladders",
        json.dumps(
            {
                "epsilons": EPSILONS,
                "incoming_adjoint_relative_errors": errors,
                "incoming_adjoint_ratios": ratios,
                "pairing_derivative_relative_errors": pairing_derivative_errors,
                "pairing_derivative_ratios": pairing_ratios,
            }
        ),
    )


@pytest.mark.parametrize(
    ("use_state_direction", "delta_c0"),
    (
        (False, PHYSICAL_C0_DIRECTION),
        (True, 0.0),
        (True, PHYSICAL_C0_DIRECTION),
    ),
)
def test_two_child_lie_forward_and_all_tangent_direction_classes(
    production_case, use_state_direction, delta_c0, record_property
):
    case = production_case
    _set_c0(case, PHYSICAL_C0)
    split = case["split"]
    primal = split.take_forward_step_cached(
        case["state"], case["time"], case["dt"]
    )
    legacy = _legacy_lie_trajectory(
        case, 1, case["state"], PHYSICAL_C0, "lie_forward_legacy"
    )[-1]
    assert isinstance(primal, DryLiePrimalCache)
    assert primal.forward_child_order == ("dry_rk4", "hyperviscosity_euler")
    assert primal.dry.t0 == primal.hyperviscosity.t0 == float(case["time"])
    assert primal.dry.dt == primal.hyperviscosity.dt == case["dt"]
    np.testing.assert_allclose(
        _function_values(primal.state_out),
        _function_values(legacy),
        rtol=0.0,
        atol=4.0e-12,
    )

    if use_state_direction:
        direction = case["direction"]
    else:
        direction = _new_state(case["model"], "lie_zero_direction")
        direction.assign(0)
    tangent = split.take_tangent_step(primal, direction, delta_c0)
    assert isinstance(tangent, DryLieTangentCache)

    errors = []
    for index, epsilon in enumerate(EPSILONS):
        plus_state = _perturbed_state(
            case,
            case["state"],
            direction,
            1.0,
            epsilon,
            f"lie_tangent_plus_state_{use_state_direction}_{index}",
        )
        minus_state = _perturbed_state(
            case,
            case["state"],
            direction,
            -1.0,
            epsilon,
            f"lie_tangent_minus_state_{use_state_direction}_{index}",
        )
        plus = _legacy_lie_trajectory(
            case,
            1,
            plus_state,
            PHYSICAL_C0 + epsilon * delta_c0,
            f"lie_tangent_plus_{use_state_direction}_{index}",
        )[-1]
        minus = _legacy_lie_trajectory(
            case,
            1,
            minus_state,
            PHYSICAL_C0 - epsilon * delta_c0,
            f"lie_tangent_minus_{use_state_direction}_{index}",
        )[-1]
        centered = (
            _function_values(plus) - _function_values(minus)
        ) / (2.0 * epsilon)
        errors.append(
            _relative_error(
                _function_values(tangent.state_direction_out), centered
            )
        )
    if not use_state_direction:
        ratios = _centered_exact_regime(
            errors, moderate_tolerance=3.0e-7
        )
    else:
        ratios = _centered_regime(errors, final_tolerance=4.0e-5)
    record_property(
        f"lie_tangent_{use_state_direction}_{delta_c0}",
        json.dumps(
            {"epsilons": EPSILONS, "relative_errors": errors, "ratios": ratios}
        ),
    )


@pytest.mark.parametrize("nsteps", (1, 3))
def test_independent_reduced_gradient_api_matches_legacy(
    production_case, nsteps
):
    case = production_case
    _set_c0(case, PHYSICAL_C0)
    result = case["split"].terminal_least_squares_gradient(
        nsteps,
        case["state"],
        case["time"],
        case["dt"],
        case["target"],
    )
    assert isinstance(result, DryLieReducedGradientResult)
    assert isinstance(result.initial_condition_gradient, Cofunction)
    assert len(result.primal_caches) == nsteps
    legacy_c0, legacy_ic, _ = _legacy_reduced_gradient(
        case,
        nsteps,
        case["state"],
        PHYSICAL_C0,
        f"ordinary_reduced_legacy_{nsteps}",
    )
    representative = case["split"].dry_lie_state_riesz_representative(
        result.initial_condition_gradient
    )
    np.testing.assert_allclose(
        result.physical_c0_gradient,
        legacy_c0,
        rtol=3.0e-10,
        atol=1.0e-13,
    )
    assert _relative_error(
        _function_values(representative), _function_values(legacy_ic)
    ) < 3.0e-10


@pytest.mark.parametrize("nsteps", (1, 3))
@pytest.mark.parametrize(
    ("use_state_direction", "delta_c0", "direction_name"),
    (
        (False, PHYSICAL_C0_DIRECTION, "c0_only"),
        (True, 0.0, "initial_condition_only"),
        (True, PHYSICAL_C0_DIRECTION, "combined"),
    ),
)
def test_reduced_gradient_and_hvp_blocks_match_independent_legacy_ladders(
    production_case,
    nsteps,
    use_state_direction,
    delta_c0,
    direction_name,
    record_property,
):
    case = production_case
    _set_c0(case, PHYSICAL_C0)
    if use_state_direction:
        direction = case["direction"]
    else:
        direction = _new_state(case["model"], f"zero_{direction_name}")
        direction.assign(0)

    result = case["split"].terminal_least_squares_hvp(
        nsteps,
        case["state"],
        case["time"],
        case["dt"],
        case["target"],
        direction,
        delta_c0,
    )
    assert isinstance(result, DryLieReducedHVPResult)
    assert isinstance(result.initial_condition_gradient, Cofunction)
    assert isinstance(result.initial_condition_hvp, Cofunction)
    assert len(result.primal_caches) == len(result.tangent_caches) == nsteps
    assert len(result.states) == len(result.state_directions) == nsteps + 1
    assert all(
        reverse.reverse_child_order
        == ("hyperviscosity_euler", "dry_rk4")
        for reverse in result.reverse_results
    )
    assert all(
        reverse.dry.reverse_stage_order == (3, 2, 1, 0)
        for reverse in result.reverse_results
    )

    legacy_gradient, legacy_ic_gradient, legacy_states = _legacy_reduced_gradient(
        case,
        nsteps,
        case["state"],
        PHYSICAL_C0,
        f"legacy_base_{nsteps}_{direction_name}",
    )
    new_ic_gradient = case["split"].dry_lie_state_riesz_representative(
        result.initial_condition_gradient
    )
    legacy_residual = _terminal_residual(
        case,
        legacy_states[-1],
        f"legacy_objective_residual_{nsteps}_{direction_name}",
    )
    legacy_objective = 0.5 * float(
        assemble(inner(legacy_residual, legacy_residual) * case["model"].spaces.dx)
    )
    np.testing.assert_allclose(
        result.objective_value, legacy_objective, rtol=0.0, atol=2.0e-12
    )
    np.testing.assert_allclose(
        result.physical_c0_gradient,
        legacy_gradient,
        rtol=3.0e-10,
        atol=1.0e-13,
    )
    assert _relative_error(
        _function_values(new_ic_gradient), _function_values(legacy_ic_gradient)
    ) < 3.0e-10

    exact_ic_hvp = case["split"].dry_lie_state_riesz_representative(
        result.initial_condition_hvp
    )
    c0_errors = []
    ic_errors = []
    centered_c0 = []
    for index, epsilon in enumerate(EPSILONS):
        plus_state = _perturbed_state(
            case,
            case["state"],
            direction,
            1.0,
            epsilon,
            f"reduced_plus_state_{nsteps}_{direction_name}_{index}",
        )
        minus_state = _perturbed_state(
            case,
            case["state"],
            direction,
            -1.0,
            epsilon,
            f"reduced_minus_state_{nsteps}_{direction_name}_{index}",
        )
        plus_gradient, plus_ic, _ = _legacy_reduced_gradient(
            case,
            nsteps,
            plus_state,
            PHYSICAL_C0 + epsilon * delta_c0,
            f"reduced_plus_{nsteps}_{direction_name}_{index}",
        )
        minus_gradient, minus_ic, _ = _legacy_reduced_gradient(
            case,
            nsteps,
            minus_state,
            PHYSICAL_C0 - epsilon * delta_c0,
            f"reduced_minus_{nsteps}_{direction_name}_{index}",
        )
        centered_scalar = (plus_gradient - minus_gradient) / (2.0 * epsilon)
        centered_field = (
            _function_values(plus_ic) - _function_values(minus_ic)
        ) / (2.0 * epsilon)
        centered_c0.append(centered_scalar)
        c0_scale = max(abs(result.physical_c0_hvp), 1.0)
        c0_errors.append(
            abs(centered_scalar - result.physical_c0_hvp) / c0_scale
        )
        ic_errors.append(
            _relative_error(centered_field, _function_values(exact_ic_hvp))
        )

    if nsteps == 1 and direction_name == "c0_only":
        c0_ratios = _centered_exact_regime(
            c0_errors, moderate_tolerance=5.0e-7
        )
        ic_ratios = _centered_exact_regime(
            ic_errors, moderate_tolerance=5.0e-7
        )
    else:
        c0_ratios = _centered_regime(c0_errors, final_tolerance=6.0e-5)
        ic_ratios = _centered_regime(ic_errors, final_tolerance=6.0e-5)
    record_property(
        f"reduced_{nsteps}_{direction_name}",
        json.dumps(
            {
                "epsilons": EPSILONS,
                "exact_physical_c0_hvp": result.physical_c0_hvp,
                "centered_physical_c0_hvp": centered_c0,
                "physical_c0_relative_errors": c0_errors,
                "physical_c0_error_ratios": c0_ratios,
                "initial_condition_relative_errors": ic_errors,
                "initial_condition_error_ratios": ic_ratios,
            }
        ),
    )


def test_lie_pairing_reverse_order_and_mixed_block_symmetry(production_case):
    case = production_case
    _set_c0(case, PHYSICAL_C0)
    split = case["split"]
    primal = split.take_forward_step_cached(
        case["state"], case["time"], case["dt"]
    )
    tangent = split.take_tangent_step(
        primal, case["direction"], PHYSICAL_C0_DIRECTION
    )
    residual = _terminal_residual(case, primal.state_out, "lie_pairing_residual")
    lambda_plus_star = split.dry_lie_state_mass_map(residual)
    reverse = split.take_adjoint_step_cached(primal, lambda_plus_star)
    assert isinstance(reverse, DryLieReverseResult)
    assert reverse.reverse_child_order == (
        "hyperviscosity_euler",
        "dry_rk4",
    )
    left = split.dry_lie_dual_pairing(
        lambda_plus_star, tangent.state_direction_out
    )
    right = split.dry_lie_dual_pairing(
        reverse.state_adjoint_in, tangent.state_direction_in
    ) + reverse.physical_c0_gradient * PHYSICAL_C0_DIRECTION
    np.testing.assert_allclose(left, right, rtol=4.0e-10, atol=1.0e-12)

    mu_plus_star = split.dry_lie_state_mass_map(
        tangent.state_direction_out
    )
    incremental = split.take_incremental_adjoint_step(
        tangent, lambda_plus_star, mu_plus_star
    )
    assert isinstance(incremental, DryLieHVPResult)
    assert incremental.reverse_child_order == (
        "hyperviscosity_euler",
        "dry_rk4",
    )
    assert incremental.dry.reverse_stage_order == (3, 2, 1, 0)

    zero = _new_state(case["model"], "mixed_symmetry_zero")
    zero.assign(0)
    c0_only = split.terminal_least_squares_hvp(
        3,
        case["state"],
        case["time"],
        case["dt"],
        case["target"],
        zero,
        PHYSICAL_C0_DIRECTION,
    )
    ic_only = split.terminal_least_squares_hvp(
        3,
        case["state"],
        case["time"],
        case["dt"],
        case["target"],
        case["direction"],
        0.0,
    )
    mixed_xc = split.dry_lie_dual_pairing(
        c0_only.initial_condition_hvp, case["direction"]
    )
    mixed_cx = PHYSICAL_C0_DIRECTION * ic_only.physical_c0_hvp
    np.testing.assert_allclose(mixed_xc, mixed_cx, rtol=8.0e-9, atol=1.0e-10)


def test_inputs_caches_duals_are_owned_unmodified_and_bitwise_repeatable(
    production_case,
):
    case = production_case
    _set_c0(case, PHYSICAL_C0)
    state_before = _function_values(case["state"])
    direction_before = _function_values(case["direction"])
    target_before = _function_values(case["target"])

    first = case["split"].terminal_least_squares_hvp(
        3,
        case["state"],
        case["time"],
        case["dt"],
        case["target"],
        case["direction"],
        PHYSICAL_C0_DIRECTION,
    )
    assert isinstance(first, DryLieReducedHVPResult)
    first_cache = first.tangent_caches[0]
    assert first_cache.primal.state_in.dat is not case["state"].dat
    assert first_cache.state_direction_in.dat is not case["direction"].dat
    assert first_cache.primal.state_in.dat is not first_cache.primal.state_out.dat
    assert first_cache.primal.dry.stage_states[0].dat is not (
        first_cache.primal.dry.stage_tendencies[0].dat
    )
    assert first_cache.dry.stage_state_directions[0].dat is not (
        first_cache.dry.stage_tendency_directions[0].dat
    )
    cached_state = _function_values(first_cache.primal.state_in)
    cached_stage = _function_values(first_cache.primal.dry.stage_states[2])
    cached_direction = _function_values(first_cache.state_direction_in)

    case["split"].reset_internal_vars()
    assert np.array_equal(_function_values(first_cache.primal.state_in), cached_state)
    assert np.array_equal(
        _function_values(first_cache.primal.dry.stage_states[2]), cached_stage
    )
    assert np.array_equal(
        _function_values(first_cache.state_direction_in), cached_direction
    )
    assert np.array_equal(_function_values(case["state"]), state_before)
    assert np.array_equal(_function_values(case["direction"]), direction_before)
    assert np.array_equal(_function_values(case["target"]), target_before)

    second = case["split"].terminal_least_squares_hvp(
        3,
        case["state"],
        case["time"],
        case["dt"],
        case["target"],
        case["direction"],
        PHYSICAL_C0_DIRECTION,
    )
    assert first.objective_value == second.objective_value
    assert first.physical_c0_gradient == second.physical_c0_gradient
    assert first.physical_c0_hvp == second.physical_c0_hvp
    assert np.array_equal(
        _dual_values(first.initial_condition_gradient),
        _dual_values(second.initial_condition_gradient),
    )
    assert np.array_equal(
        _dual_values(first.initial_condition_hvp),
        _dual_values(second.initial_condition_hvp),
    )
    assert all(
        np.array_equal(_function_values(left), _function_values(right))
        for left, right in zip(first.states, second.states)
    )
