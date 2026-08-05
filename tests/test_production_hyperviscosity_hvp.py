"""Certification of the actual production hyperviscosity Euler child HVP."""

from copy import deepcopy
import json

import numpy as np
import pytest
from firedrake import (
    COMM_SELF,
    Cofunction,
    Function,
    TestFunction,
    TestFunctions,
    TrialFunction,
    TrialFunctions,
    assemble,
    grad,
    inner,
)
from petsc4py import PETSc
from ufl.algorithms import expand_derivatives

import dimswe.meshes as dimswe_meshes
from dimswe.hyperviscosity_hvp import (
    HyperviscosityHVPResult,
    HyperviscosityPrimalCache,
    HyperviscosityReverseResult,
    HyperviscosityTangentCache,
    _normalize_derivative_form,
)
from dimswe.logger import EmptyLogger
from dimswe.models import get_model
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.timestepping import get_timestepper


CFG = "tests/tswe_rol_small.cfg"
PHYSICAL_C0 = 0.14
PHYSICAL_C0_DIRECTION = 0.035


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


def _new_state(model, name):
    state, _, _ = model.get_x_var(name)
    return state


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
    state_container, state_sub, _ = model.get_full_var(
        "production_hvp_state", split_x_and_aux=True
    )
    time = model.get_t_var()
    model.initialize(state_sub, time)
    model.set_coeffs(parameters, coefficient_sub)

    split_timestepper = get_timestepper(
        parameters, model, logger, _serial_solver_parameters()
    )
    split_timestepper.set_coeff(coefficient)
    children = [
        child
        for child in split_timestepper.time_integrators
        if child.terms == ["hyperviscosity"]
    ]
    assert len(children) == 1, [
        child.terms for child in split_timestepper.time_integrators
    ]
    child = children[0]
    child.coeff_sub["c0"].assign(PHYSICAL_C0)

    # Use the initialized production field and deterministic proportional/non-
    # proportional directions.  The initialized double vortex is nonconstant,
    # which exercises both deployed stiffness applications.
    state = state_container[0]
    direction = _new_state(model, "production_hvp_state_direction")
    direction.assign(0.025 * state)
    direction.sub(0).assign(0.04 * state.sub(0))
    direction.sub(2).assign(-0.03 * state.sub(2))

    target = _new_state(model, "production_hvp_target")
    target.assign(0.92 * state)

    return {
        "parameters": parameters,
        "model": model,
        "split_timestepper": split_timestepper,
        "child": child,
        "state_container": state_container,
        "state": state,
        "direction": direction,
        "target": target,
        "time": time,
        "dt": float(parameters["timestepping"]["dt"]),
    }


def test_cached_api_rejects_uncertified_child_semantically(production_case):
    dry_children = [
        child
        for child in production_case["split_timestepper"].time_integrators
        if child.terms == ["model"]
    ]
    assert len(dry_children) == 1
    with pytest.raises(
        ValueError, match=r"terms=\['hyperviscosity'\] exactly"
    ):
        dry_children[0].take_forward_step_cached(
            production_case["state"],
            production_case["time"],
            production_case["dt"],
        )


def _cached_forward(case, state, c0):
    child = case["child"]
    child.coeff_sub["c0"].assign(float(c0))
    return child.take_forward_step_cached(
        state, case["time"], case["dt"]
    )


def _legacy_forward(case, state, c0, name):
    child = case["child"]
    child.coeff_sub["c0"].assign(float(c0))
    output, output_sub, _ = case["model"].get_full_var(
        name, split_x_and_aux=True
    )
    child.reset_internal_vars()
    child.take_forward_step(
        output,
        output_sub,
        [state],
        case["time"],
        case["dt"],
    )
    return output[0].copy(deepcopy=True)


def _legacy_gradient(case, state, c0, lambda_plus, name):
    child = case["child"]
    model = case["model"]
    child.coeff_sub["c0"].assign(float(c0))
    delta_gradient, _, _ = model.get_coeff_var(f"{name}_delta_gradient")
    delta_lambda, _, _ = model.get_x_var(f"{name}_delta_lambda")
    child.reset_internal_vars()
    _, gradient = child.take_adjoint_step(
        delta_gradient,
        delta_lambda,
        lambda_plus,
        [state],
        float(case["time"]) + case["dt"],
        case["dt"],
    )
    lambda_in = _new_state(model, f"{name}_lambda_in")
    lambda_in.assign(lambda_plus + delta_lambda)
    c0_index = model.get_coeff_list().index("c0")
    return float(gradient[c0_index]), lambda_in


def _perturbed_state(case, sign, epsilon):
    result = _new_state(case["model"], f"perturbed_state_{sign}_{epsilon}")
    result.assign(case["state"] + sign * epsilon * case["direction"])
    return result


def _terminal_residual(case, state_out, name):
    residual = _new_state(case["model"], name)
    residual.assign(state_out - case["target"])
    return residual


def _assembled_dense_operators(case):
    model = case["model"]
    space = model.dynamics.xspace
    test = TestFunction(space)
    trial = TrialFunction(space)
    mass_handle = assemble(
        inner(test, trial) * model.spaces.dx, mat_type="aij"
    ).M.handle
    tests = TestFunctions(space)
    trials = TrialFunctions(space)
    stiffness_form = sum(
        inner(grad(test_i), grad(trial_i)) * model.spaces.dx
        for test_i, trial_i in zip(tests, trials)
    )
    stiffness_handle = assemble(stiffness_form, mat_type="aij").M.handle
    size = mass_handle.getSize()[0]
    indices = np.arange(size, dtype=PETSc.IntType)
    return (
        np.asarray(mass_handle.getValues(indices, indices)),
        np.asarray(stiffness_handle.getValues(indices, indices)),
    )


def test_cached_forward_equals_legacy_and_has_actual_production_sign(
    production_case, record_property
):
    case = production_case
    cached = _cached_forward(case, case["state"], PHYSICAL_C0)
    legacy = _legacy_forward(
        case, case["state"], PHYSICAL_C0, "legacy_forward_equivalence"
    )
    assert isinstance(cached, HyperviscosityPrimalCache)
    np.testing.assert_allclose(
        _function_values(cached.state_out),
        _function_values(legacy),
        rtol=0.0,
        atol=2.0e-12,
    )
    assert cached.t0 == float(case["time"])
    assert cached.dt == case["dt"]
    assert cached.c0 == PHYSICAL_C0
    assert cached.s == case["parameters"]["hyperviscosity"]["s"]

    mass, stiffness = _assembled_dense_operators(case)
    state = _function_values(cached.state_in)
    factor = float(
        max(
            case["model"].spaces.mesh.dx / case["model"].spaces.order,
            case["model"].spaces.mesh.dy / case["model"].spaces.order,
        )
    )
    alpha = factor**cached.s
    dense_diagnostic = np.linalg.solve(mass, -stiffness @ state)
    dense_tendency = np.linalg.solve(
        mass, cached.c0 * alpha * stiffness @ dense_diagnostic
    )
    dense_state_out = state + cached.dt * dense_tendency
    diagnostic_error = _relative_error(
        _function_values(cached.diagnostic), dense_diagnostic
    )
    tendency_error = _relative_error(
        _function_values(cached.tendency), dense_tendency
    )
    forward_error = _relative_error(
        _function_values(cached.state_out), dense_state_out
    )
    assert diagnostic_error < 3.0e-13
    assert tendency_error < 3.0e-13
    assert forward_error < 3.0e-13
    record_property(
        "dense_forward_errors",
        json.dumps(
            {
                "diagnostic": diagnostic_error,
                "tendency": tendency_error,
                "state_out": forward_error,
            }
        ),
    )


@pytest.mark.parametrize(
    ("include_state_direction", "delta_c0"),
    [(False, PHYSICAL_C0_DIRECTION), (True, PHYSICAL_C0_DIRECTION)],
)
def test_parameter_only_and_combined_tangents_match_centered_forward_difference(
    production_case, include_state_direction, delta_c0, record_property
):
    case = production_case
    primal = _cached_forward(case, case["state"], PHYSICAL_C0)
    if include_state_direction:
        direction = case["direction"]
    else:
        direction = _new_state(case["model"], "zero_state_direction")
        direction.assign(0)
    tangent = case["child"].take_tangent_step(primal, direction, delta_c0)
    assert isinstance(tangent, HyperviscosityTangentCache)

    # The child is bilinear in (state, c0), so its centered directional
    # difference has no truncation term.  Moderate steps expose the exact
    # derivative while progressively smaller steps reveal subtraction/solve
    # roundoff; a factor-of-four regime is therefore neither expected nor
    # required.
    epsilons = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5)
    centered_errors = []
    for index, epsilon in enumerate(epsilons):
        plus_state = _new_state(
            case["model"], f"tangent_plus_state_{index}"
        )
        minus_state = _new_state(
            case["model"], f"tangent_minus_state_{index}"
        )
        plus_state.assign(case["state"] + epsilon * direction)
        minus_state.assign(case["state"] - epsilon * direction)
        plus = _legacy_forward(
            case,
            plus_state,
            PHYSICAL_C0 + epsilon * delta_c0,
            f"tangent_legacy_plus_{index}",
        )
        minus = _legacy_forward(
            case,
            minus_state,
            PHYSICAL_C0 - epsilon * delta_c0,
            f"tangent_legacy_minus_{index}",
        )
        centered = (_function_values(plus) - _function_values(minus)) / (
            2.0 * epsilon
        )
        centered_errors.append(
            _relative_error(
                _function_values(tangent.state_direction_out), centered
            )
        )
    mass, stiffness = _assembled_dense_operators(case)
    factor = float(
        max(
            case["model"].spaces.mesh.dx / case["model"].spaces.order,
            case["model"].spaces.mesh.dy / case["model"].spaces.order,
        )
    )
    alpha = factor**primal.s
    dense_diagnostic = np.linalg.solve(
        mass, -stiffness @ _function_values(primal.state_in)
    )
    dense_diagnostic_direction = np.linalg.solve(
        mass, -stiffness @ _function_values(direction)
    )
    dense_tendency_direction = np.linalg.solve(
        mass,
        delta_c0 * alpha * stiffness @ dense_diagnostic
        + primal.c0 * alpha * stiffness @ dense_diagnostic_direction,
    )
    dense_direction_out = (
        _function_values(direction) + primal.dt * dense_tendency_direction
    )
    dense_error = _relative_error(
        _function_values(tangent.state_direction_out), dense_direction_out
    )
    assert centered_errors[0] < 2.0e-8
    assert centered_errors[1] < 2.0e-8
    assert dense_error < 5.0e-13
    record_property(
        f"tangent_relative_error_{include_state_direction}",
        json.dumps(
            {
                "epsilons": epsilons,
                "centered_errors": centered_errors,
                "dense": float(dense_error),
                "interpretation": "smaller-step error is solve/subtraction roundoff",
            }
        ),
    )


def test_dual_reverse_matches_legacy_gradient_and_riesz_state_pullback(
    production_case, record_property
):
    case = production_case
    primal = _cached_forward(case, case["state"], PHYSICAL_C0)
    residual = _terminal_residual(case, primal.state_out, "reverse_residual")
    lambda_plus_star = case["child"].hyperviscosity_state_mass_map(residual)
    reverse = case["child"].take_adjoint_step_cached(
        primal, lambda_plus_star
    )
    assert isinstance(reverse, HyperviscosityReverseResult)
    assert isinstance(reverse.state_adjoint_in, Cofunction)
    assert isinstance(reverse.tendency_adjoint, Cofunction)
    assert isinstance(reverse.diagnostic_adjoint, Cofunction)

    helper = case["child"]._get_hyperviscosity_hvp_helper()
    zero_debug = helper._last_main_state_zero_debug
    assert zero_debug.dependency_absent
    assert zero_debug.structural_zero
    assert zero_debug.assembly_bypassed
    assert zero_debug.normalized_metadata["integral_count"] == 0
    assert type(expand_derivatives(zero_debug.raw_derivative)) is type(
        zero_debug.expanded_derivative
    )
    assert _normalize_derivative_form(
        zero_debug.raw_derivative
    ).structural_zero
    with pytest.raises(
        TypeError,
        match=r"nonzero dual form.*arguments=0.*expected_arguments=1",
    ):
        helper._assemble_dual(
            zero_debug.contracted,
            helper.state_dual_space,
            "rank_zero_nonzero_must_not_assemble",
        )
    assert np.count_nonzero(_dual_values(reverse.main_state_adjoint)) == 0
    assert np.linalg.norm(_dual_values(reverse.diagnostic_state_adjoint)) > 0.0
    assert np.linalg.norm(_dual_values(reverse.state_adjoint_in)) > 0.0
    diagnostic_pullback_identity_error = _relative_error(
        _dual_values(reverse.state_adjoint_in)
        - _dual_values(lambda_plus_star),
        _dual_values(reverse.diagnostic_state_adjoint),
    )
    assert diagnostic_pullback_identity_error < 5.0e-13

    legacy_gradient, legacy_lambda_in = _legacy_gradient(
        case,
        case["state"],
        PHYSICAL_C0,
        residual,
        "ordinary_reverse_legacy",
    )
    gradient_error = abs(reverse.c0_gradient - legacy_gradient) / max(
        abs(legacy_gradient), np.finfo(float).tiny
    )
    new_lambda_in = (
        case["child"].hyperviscosity_state_riesz_representative(
            reverse.state_adjoint_in
        )
    )
    state_error = _relative_error(
        _function_values(new_lambda_in), _function_values(legacy_lambda_in)
    )
    mass, stiffness = _assembled_dense_operators(case)
    factor = float(
        max(
            case["model"].spaces.mesh.dx / case["model"].spaces.order,
            case["model"].spaces.mesh.dy / case["model"].spaces.order,
        )
    )
    alpha = factor**primal.s
    state = _function_values(primal.state_in)
    diagnostic = np.linalg.solve(mass, -stiffness @ state)
    lambda_plus = _function_values(residual)
    psi = primal.dt * lambda_plus
    dense_gradient = alpha * psi @ stiffness @ diagnostic
    dense_diagnostic_adjoint = primal.c0 * alpha * stiffness @ psi
    dense_diagnostic_reverse = np.linalg.solve(
        mass, dense_diagnostic_adjoint
    )
    dense_lambda_in = mass @ lambda_plus - stiffness @ dense_diagnostic_reverse
    dense_gradient_error = abs(reverse.c0_gradient - dense_gradient) / max(
        abs(dense_gradient), np.finfo(float).tiny
    )
    dense_state_error = _relative_error(
        _dual_values(reverse.state_adjoint_in), dense_lambda_in
    )
    assert gradient_error < 2.0e-13
    assert state_error < 5.0e-13
    assert dense_gradient_error < 6.0e-13
    assert dense_state_error < 6.0e-13
    record_property(
        "legacy_reverse_errors",
        json.dumps(
            {
                "gradient": gradient_error,
                "state_riesz": state_error,
                "dense_gradient": dense_gradient_error,
                "dense_state_dual": dense_state_error,
                "diagnostic_pullback_identity": (
                    diagnostic_pullback_identity_error
                ),
                "main_state_zero_raw": zero_debug.raw_metadata,
                "main_state_zero_expanded": zero_debug.expanded_metadata,
            }
        ),
    )


def test_pairing_identity_for_combined_state_and_physical_c0_direction(
    production_case,
):
    case = production_case
    primal = _cached_forward(case, case["state"], PHYSICAL_C0)
    tangent = case["child"].take_tangent_step(
        primal, case["direction"], PHYSICAL_C0_DIRECTION
    )
    residual = _terminal_residual(case, primal.state_out, "pairing_residual")
    lambda_plus_star = case["child"].hyperviscosity_state_mass_map(residual)
    reverse = case["child"].take_adjoint_step_cached(
        primal, lambda_plus_star
    )
    left = case["child"].hyperviscosity_dual_pairing(
        lambda_plus_star, tangent.state_direction_out
    )
    right = case["child"].hyperviscosity_dual_pairing(
        reverse.state_adjoint_in, tangent.state_direction_in
    ) + reverse.c0_gradient * PHYSICAL_C0_DIRECTION
    np.testing.assert_allclose(left, right, rtol=3.0e-13, atol=0.0)


def test_exact_hvp_has_both_mixed_terms_and_matches_dense_oracle(
    production_case, record_property
):
    case = production_case
    primal = _cached_forward(case, case["state"], PHYSICAL_C0)
    tangent = case["child"].take_tangent_step(
        primal, case["direction"], PHYSICAL_C0_DIRECTION
    )
    residual = _terminal_residual(case, primal.state_out, "hvp_residual")
    lambda_plus_star = case["child"].hyperviscosity_state_mass_map(residual)
    mu_plus_star = case["child"].hyperviscosity_state_mass_map(
        tangent.state_direction_out
    )
    result = case["child"].take_incremental_adjoint_step(
        tangent, lambda_plus_star, mu_plus_star
    )
    assert isinstance(result, HyperviscosityHVPResult)
    assert isinstance(result.incremental_state_adjoint_in, Cofunction)
    assert isinstance(result.incremental_tendency_adjoint, Cofunction)
    assert isinstance(result.incremental_diagnostic_adjoint, Cofunction)

    helper = case["child"]._get_hyperviscosity_hvp_helper()
    pure_control_debug = helper._last_pure_control_zero_debug
    if pure_control_debug.structural_zero:
        assert pure_control_debug.assembly_bypassed
    else:
        assert not pure_control_debug.assembly_bypassed
        assert pure_control_debug.normalized_metadata["argument_count"] == 0
        assert pure_control_debug.normalized_metadata["domain_count"] >= 1
    assert type(
        expand_derivatives(pure_control_debug.raw_derivative)
    ) is type(pure_control_debug.expanded_derivative)
    pure_control_normalized = _normalize_derivative_form(
        pure_control_debug.raw_derivative
    )
    assert (
        pure_control_normalized.structural_zero
        == pure_control_debug.structural_zero
    )
    assert pure_control_debug.assembled_scalar_value == (
        result.c0_hvp_from_pure_control_curvature
    )
    assert result.ordinary.c0_gradient != 0.0
    assert result.c0_hvp_from_incremental_adjoint != 0.0
    assert result.c0_hvp_from_state_direction != 0.0
    mixed_scale = max(
        abs(result.c0_hvp_from_incremental_adjoint),
        abs(result.c0_hvp_from_state_direction),
        abs(
            result.c0_hvp_from_incremental_adjoint
            + result.c0_hvp_from_state_direction
        ),
        np.finfo(float).tiny,
    )
    pure_control_tolerance = 256.0 * np.finfo(float).eps * mixed_scale
    dense_pure_control = 0.0
    pure_control_dense_error = abs(
        result.c0_hvp_from_pure_control_curvature - dense_pure_control
    )
    record_property(
        "pure_control_curvature",
        json.dumps(
            {
                "assembled_value": (
                    result.c0_hvp_from_pure_control_curvature
                ),
                "dense_oracle": dense_pure_control,
                "absolute_error": pure_control_dense_error,
                "scale": mixed_scale,
                "tolerance": pure_control_tolerance,
                "assembly_bypassed": pure_control_debug.assembly_bypassed,
                "raw_metadata": pure_control_debug.raw_metadata,
                "expanded_metadata": pure_control_debug.expanded_metadata,
                "normalized_metadata": (
                    pure_control_debug.normalized_metadata
                ),
            }
        ),
    )
    assert pure_control_dense_error <= pure_control_tolerance
    assert result.c0_hvp == (
        result.c0_hvp_from_incremental_adjoint
        + result.c0_hvp_from_state_direction
        + result.c0_hvp_from_pure_control_curvature
    )
    assert result.c0_hvp != pytest.approx(
        result.c0_hvp_from_incremental_adjoint
    )
    assert result.c0_hvp != pytest.approx(result.c0_hvp_from_state_direction)

    mass, stiffness = _assembled_dense_operators(case)
    factor = float(
        max(
            case["model"].spaces.mesh.dx / case["model"].spaces.order,
            case["model"].spaces.mesh.dy / case["model"].spaces.order,
        )
    )
    alpha = factor**primal.s
    state = _function_values(primal.state_in)
    direction = _function_values(tangent.state_direction_in)
    diagnostic = np.linalg.solve(mass, -stiffness @ state)
    diagnostic_direction = np.linalg.solve(mass, -stiffness @ direction)
    lambda_plus = _function_values(residual)
    mu_plus = _function_values(tangent.state_direction_out)
    psi = primal.dt * lambda_plus
    delta_psi = primal.dt * mu_plus
    dense_incremental_term = alpha * delta_psi @ stiffness @ diagnostic
    dense_state_term = alpha * psi @ stiffness @ diagnostic_direction
    dense_hvp = dense_incremental_term + dense_state_term
    hyperviscosity_operator = alpha * np.linalg.solve(
        mass, stiffness @ np.linalg.solve(mass, stiffness)
    )
    child_operator = (
        np.eye(mass.shape[0])
        - primal.dt * primal.c0 * hyperviscosity_operator
    )
    dense_lambda_plus_star = mass @ lambda_plus
    dense_mu_plus_star = mass @ mu_plus
    dense_incremental_state_adjoint = (
        child_operator.T @ dense_mu_plus_star
        - primal.dt
        * tangent.delta_c0
        * hyperviscosity_operator.T
        @ dense_lambda_plus_star
    )
    dense_error = abs(result.c0_hvp - dense_hvp) / max(
        abs(dense_hvp), np.finfo(float).tiny
    )
    dense_incremental_state_error = _relative_error(
        _dual_values(result.incremental_state_adjoint_in),
        dense_incremental_state_adjoint,
    )
    assert dense_error < 6.0e-13
    assert dense_incremental_state_error < 8.0e-13
    np.testing.assert_allclose(
        result.c0_hvp_from_incremental_adjoint,
        dense_incremental_term,
        rtol=6.0e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.c0_hvp_from_state_direction,
        dense_state_term,
        rtol=6.0e-13,
        atol=0.0,
    )
    record_property(
        "hvp_terms",
        json.dumps(
            {
                "incremental_adjoint": result.c0_hvp_from_incremental_adjoint,
                "state_direction": result.c0_hvp_from_state_direction,
                "pure_control": result.c0_hvp_from_pure_control_curvature,
                "pure_control_tolerance": pure_control_tolerance,
                "pure_control_dense_error": pure_control_dense_error,
                "pure_control_assembly_bypassed": (
                    pure_control_debug.assembly_bypassed
                ),
                "total": result.c0_hvp,
                "dense_relative_error": dense_error,
                "dense_incremental_state_relative_error": (
                    dense_incremental_state_error
                ),
                "pure_control_zero_raw": pure_control_debug.raw_metadata,
                "pure_control_zero_expanded": (
                    pure_control_debug.expanded_metadata
                ),
            }
        ),
    )


def test_hvp_and_incremental_state_pullback_match_centered_legacy_quantities(
    production_case, record_property
):
    case = production_case
    primal = _cached_forward(case, case["state"], PHYSICAL_C0)
    tangent = case["child"].take_tangent_step(
        primal, case["direction"], PHYSICAL_C0_DIRECTION
    )
    residual = _terminal_residual(case, primal.state_out, "fd_hvp_residual")
    lambda_plus_star = case["child"].hyperviscosity_state_mass_map(residual)
    mu_plus_star = case["child"].hyperviscosity_state_mass_map(
        tangent.state_direction_out
    )
    result = case["child"].take_incremental_adjoint_step(
        tangent, lambda_plus_star, mu_plus_star
    )
    exact = result.c0_hvp
    exact_incremental_state = (
        case["child"].hyperviscosity_state_riesz_representative(
            result.incremental_state_adjoint_in
        )
    )

    epsilons = (4.0e-2, 2.0e-2, 1.0e-2, 5.0e-3)
    errors = []
    state_pullback_errors = []
    centered_values = []
    for index, epsilon in enumerate(epsilons):
        state_plus = _perturbed_state(case, 1.0, epsilon)
        state_minus = _perturbed_state(case, -1.0, epsilon)
        c0_plus = PHYSICAL_C0 + epsilon * PHYSICAL_C0_DIRECTION
        c0_minus = PHYSICAL_C0 - epsilon * PHYSICAL_C0_DIRECTION
        forward_plus = _legacy_forward(
            case, state_plus, c0_plus, f"fd_hvp_forward_plus_{index}"
        )
        forward_minus = _legacy_forward(
            case, state_minus, c0_minus, f"fd_hvp_forward_minus_{index}"
        )
        lambda_plus = _terminal_residual(
            case, forward_plus, f"fd_hvp_lambda_plus_{index}"
        )
        lambda_minus = _terminal_residual(
            case, forward_minus, f"fd_hvp_lambda_minus_{index}"
        )
        gradient_plus, state_adjoint_plus = _legacy_gradient(
            case,
            state_plus,
            c0_plus,
            lambda_plus,
            f"fd_hvp_gradient_plus_{index}",
        )
        gradient_minus, state_adjoint_minus = _legacy_gradient(
            case,
            state_minus,
            c0_minus,
            lambda_minus,
            f"fd_hvp_gradient_minus_{index}",
        )
        centered = (gradient_plus - gradient_minus) / (2.0 * epsilon)
        centered_state_pullback = (
            _function_values(state_adjoint_plus)
            - _function_values(state_adjoint_minus)
        ) / (2.0 * epsilon)
        centered_values.append(centered)
        errors.append(abs(centered - exact))
        state_pullback_errors.append(
            _relative_error(
                centered_state_pullback,
                _function_values(exact_incremental_state),
            )
        )

    rates = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    state_pullback_rates = [
        state_pullback_errors[i] / state_pullback_errors[i + 1]
        for i in range(len(state_pullback_errors) - 1)
    ]
    assert all(rate > 3.8 for rate in rates)
    assert all(rate > 3.8 for rate in state_pullback_rates)
    assert errors[-1] / abs(exact) < 2.0e-5
    assert state_pullback_errors[-1] < 2.0e-5
    record_property(
        "hvp_centered_difference",
        json.dumps(
            {
                "exact": exact,
                "epsilons": epsilons,
                "centered": centered_values,
                "errors": errors,
                "error_ratios": rates,
                "incremental_state_pullback_relative_errors": (
                    state_pullback_errors
                ),
                "incremental_state_pullback_error_ratios": (
                    state_pullback_rates
                ),
            }
        ),
    )


def test_inputs_caches_and_duals_are_owned_unmodified_and_repeatable(
    production_case,
):
    case = production_case
    child = case["child"]
    state_before = _function_values(case["state"])
    direction_before = _function_values(case["direction"])
    primal = _cached_forward(case, case["state"], PHYSICAL_C0)
    tangent = child.take_tangent_step(
        primal, case["direction"], PHYSICAL_C0_DIRECTION
    )
    assert primal.state_in.dat is not primal.state_out.dat
    assert primal.state_in.dat is not primal.tendency.dat
    assert tangent.state_direction_in.dat is not tangent.state_direction_out.dat
    assert tangent.state_direction_in.dat is not primal.state_in.dat
    primal_state_cache = _function_values(primal.state_in)
    primal_diagnostic_cache = _function_values(primal.diagnostic)
    tangent_direction_cache = _function_values(tangent.state_direction_in)

    residual = _terminal_residual(case, primal.state_out, "ownership_residual")
    lambda_plus_star = child.hyperviscosity_state_mass_map(residual)
    mu_plus_star = child.hyperviscosity_state_mass_map(
        tangent.state_direction_out
    )
    lambda_before = _dual_values(lambda_plus_star)
    mu_before = _dual_values(mu_plus_star)
    first = child.take_incremental_adjoint_step(
        tangent, lambda_plus_star, mu_plus_star
    )
    assert np.array_equal(_dual_values(lambda_plus_star), lambda_before)
    assert np.array_equal(_dual_values(mu_plus_star), mu_before)

    caller_state_copy = case["state"].copy(deepcopy=True)
    caller_direction_copy = case["direction"].copy(deepcopy=True)
    case["state"].assign(0)
    case["direction"].assign(0)
    child.reset_internal_vars()
    assert np.array_equal(_function_values(primal.state_in), primal_state_cache)
    assert np.array_equal(
        _function_values(primal.diagnostic), primal_diagnostic_cache
    )
    assert np.array_equal(
        _function_values(tangent.state_direction_in), tangent_direction_cache
    )
    case["state"].assign(caller_state_copy)
    case["direction"].assign(caller_direction_copy)
    assert np.array_equal(_function_values(case["state"]), state_before)
    assert np.array_equal(_function_values(case["direction"]), direction_before)

    second_primal = _cached_forward(case, case["state"], PHYSICAL_C0)
    second_tangent = child.take_tangent_step(
        second_primal, case["direction"], PHYSICAL_C0_DIRECTION
    )
    assert second_primal.state_in.dat is not primal.state_in.dat
    assert second_primal.diagnostic.dat is not primal.diagnostic.dat
    assert second_tangent.state_direction_in.dat is not tangent.state_direction_in.dat
    second_residual = _terminal_residual(
        case, second_primal.state_out, "ownership_second_residual"
    )
    second_lambda = child.hyperviscosity_state_mass_map(second_residual)
    second_mu = child.hyperviscosity_state_mass_map(
        second_tangent.state_direction_out
    )
    second = child.take_incremental_adjoint_step(
        second_tangent, second_lambda, second_mu
    )
    assert np.array_equal(
        _function_values(primal.state_out), _function_values(second_primal.state_out)
    )
    assert np.array_equal(
        _function_values(tangent.state_direction_out),
        _function_values(second_tangent.state_direction_out),
    )
    assert first.c0_hvp == second.c0_hvp
    assert np.array_equal(
        _dual_values(first.incremental_state_adjoint_in),
        _dual_values(second.incremental_state_adjoint_in),
    )
