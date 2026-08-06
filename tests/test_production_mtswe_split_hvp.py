"""Certification of exact HVPs for the complete production MTSWE split."""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import json

import numpy as np
import pytest
import ufl
from firedrake import (
    COMM_SELF,
    Cofunction,
    Function,
    SpatialCoordinate,
    as_vector,
    assemble,
    cos,
    inner,
    pi,
    sin,
)

import dimswe.meshes as dimswe_meshes
from dimswe.logger import EmptyLogger
from dimswe.models import get_model
from dimswe.mtswe_split_hvp import (
    MTSWEReducedHVPResult,
    MTSWESplitHVPResult,
    MTSWESplitPrimalCache,
    MTSWESplitReverseResult,
    MTSWESplitTangentCache,
    MoistEulerHVPResult,
    MoistEulerPrimalCache,
)
from dimswe.parameters import get_parameters, overall_solver_parameters
from dimswe.physics import qsat
from dimswe.timestepping import get_timestepper


CFG = "tests/mtswe_small.cfg"
PHYSICAL_C0 = 0.07
DELTA_C0 = 0.012
EPSILONS = (0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001)
REDUCED_EPSILONS = (0.1, 0.05, 0.025, 0.0125)
STRICT_FLOOR = 1.0e-9
FACTOR_INTERVAL = (3.5, 4.5)
GRADIENT_DIAGNOSTIC_EPSILONS = (
    0.8,
    0.4,
    0.2,
    0.1,
    0.05,
    0.025,
    0.0125,
    0.00625,
    0.003125,
    0.0015625,
    0.00078125,
)
REDUCED_DIAGNOSTIC_EPSILONS = (
    0.4,
    0.2,
    0.1,
    0.05,
    0.025,
    0.0125,
    0.00625,
    0.003125,
    0.0015625,
    0.00078125,
)
OBJECTIVE_DIRECTIONAL_STRICT_FLOOR = STRICT_FLOOR
SCALAR_GRADIENT_STRICT_FLOOR = STRICT_FLOOR
FIELD_HVP_STRICT_FLOOR = STRICT_FLOOR
ROUNDOFF_SAFETY_FACTOR = 64.0
ACTIVE_BRANCH_NAMES = (
    "condensation",
    "evaporation",
    "evaporation_cap",
    "rain",
)


@dataclass(frozen=True)
class DualFDDiagnosticResult:
    """Stable named return contract for a field-valued FD diagnostic."""

    record: dict
    centered_dual: Cofunction
    numerator_dual: Cofunction
    error_dual: Cofunction


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


def _values(value):
    with value.dat.vec_ro as vector:
        return vector.getArray(readonly=True).copy()


def _relative_error(computed, expected):
    scale = max(np.linalg.norm(expected), np.finfo(float).tiny)
    return float(np.linalg.norm(computed - expected) / scale)


def _scalar_error(computed, expected):
    scale = max(abs(computed), abs(expected), np.finfo(float).tiny)
    return abs(computed - expected) / scale


def _ratios(errors):
    return [
        float(errors[i]) / max(float(errors[i + 1]), np.finfo(float).tiny)
        for i in range(len(errors) - 1)
    ]


def _factor_window(errors):
    ratios = _ratios(errors)
    for start in range(max(0, len(ratios) - 2)):
        window = ratios[start : start + 3]
        if len(window) != 3:
            continue
        if not all(
            FACTOR_INTERVAL[0] <= ratio <= FACTOR_INTERVAL[1]
            for ratio in window
        ):
            continue
        if all(errors[i] > errors[i + 1] for i in range(start, start + 3)):
            return start, start + 2, window
    return None, None, ()


def _scalar_fd_diagnostic(
    epsilon, exact, plus, minus, repeated_absolute_error
):
    numerator = float(plus) - float(minus)
    centered = numerator / (2.0 * float(epsilon))
    absolute_error = abs(float(exact) - centered)
    scale = max(abs(float(exact)), abs(centered), np.finfo(float).tiny)
    raw_floor = (
        float(repeated_absolute_error)
        + np.finfo(float).eps * (abs(float(plus)) + abs(float(minus)))
    ) / (2.0 * float(epsilon))
    record = {
        "epsilon": float(epsilon),
        "exact": float(exact),
        "centered": centered,
        "absolute_error": absolute_error,
        "relative_error": absolute_error / scale,
        "scale": scale,
        "plus_magnitude": abs(float(plus)),
        "minus_magnitude": abs(float(minus)),
        "subtraction_numerator_magnitude": abs(numerator),
        "repeated_unperturbed_absolute_error": float(repeated_absolute_error),
        "raw_roundoff_absolute_floor": raw_floor,
        "raw_roundoff_relative_floor": raw_floor / scale,
    }
    return record


def _scale_aware_classifier(
    records,
    *,
    strict_floor,
    relative_error_key="relative_error",
    relative_floor_key="raw_roundoff_relative_floor",
    allow_immediate_floor=False,
    independent_checks=False,
    active_set_safe=False,
    active_set_truncated=False,
    strictly_interior_safe=False,
    secondary_metric=False,
    primary_metric_certified=False,
):
    """Certify convergence against a measured subtraction-floor model."""
    errors = [float(record[relative_error_key]) for record in records]
    epsilons = [float(record["epsilon"]) for record in records]
    attainable_floors = [
        max(
            float(strict_floor),
            ROUNDOFF_SAFETY_FACTOR * float(record[relative_floor_key]),
        )
        for record in records
    ]
    start, end, factor_ratios = _factor_window(errors)
    minimum_index = int(np.argmin(errors))
    minimum_record = records[minimum_index]
    measured_floor = ROUNDOFF_SAFETY_FACTOR * float(
        minimum_record[relative_floor_key]
    )
    attainable_floor = attainable_floors[minimum_index]
    floor_reached = errors[minimum_index] <= attainable_floor
    floor_reaching_indices = tuple(
        index
        for index, (error, floor) in enumerate(
            zip(errors, attainable_floors)
        )
        if error <= floor
    )
    first_floor_reaching_index = (
        floor_reaching_indices[0] if floor_reaching_indices else None
    )
    truncation_prefix_indices = (
        tuple(range(first_floor_reaching_index + 1))
        if first_floor_reaching_index is not None
        else ()
    )
    truncation_prefix_epsilons = tuple(
        epsilons[index] for index in truncation_prefix_indices
    )
    truncation_prefix_errors = tuple(
        errors[index] for index in truncation_prefix_indices
    )
    truncation_prefix_monotone = all(
        truncation_prefix_errors[index]
        > truncation_prefix_errors[index + 1]
        for index in range(len(truncation_prefix_errors) - 1)
    )
    upturn_after_minimum = any(
        errors[index] > errors[minimum_index]
        for index in range(minimum_index + 1, len(errors))
    )
    post_floor_upturn = (
        first_floor_reaching_index is not None
        and any(
            errors[index] > errors[index - 1]
            for index in range(
                first_floor_reaching_index + 1, len(errors)
            )
        )
    )
    enough_immediate_floor_records = len(errors) >= 3
    all_errors_below_strict_floor = (
        enough_immediate_floor_records
        and all(error <= float(strict_floor) for error in errors)
    )
    all_errors_within_per_record_attainable_floor = (
        enough_immediate_floor_records
        and all(
            error <= floor
            for error, floor in zip(errors, attainable_floors)
        )
    )
    initial_floor_prefix_length = 0
    for error, floor in zip(errors, attainable_floors):
        if error > floor:
            break
        initial_floor_prefix_length += 1
    initial_floor_prefix_indices = tuple(range(initial_floor_prefix_length))
    initial_floor_prefix_epsilons = tuple(
        epsilons[index] for index in initial_floor_prefix_indices
    )
    initial_floor_prefix_errors = tuple(
        errors[index] for index in initial_floor_prefix_indices
    )
    initial_floor_prefix_floors = tuple(
        attainable_floors[index] for index in initial_floor_prefix_indices
    )
    first_floor_escape_index = (
        initial_floor_prefix_length
        if initial_floor_prefix_length < len(errors)
        else None
    )
    post_floor_prefix_errors = (
        tuple(errors[first_floor_escape_index:])
        if first_floor_escape_index is not None
        else ()
    )
    post_floor_prefix_floors = (
        tuple(attainable_floors[first_floor_escape_index:])
        if first_floor_escape_index is not None
        else ()
    )
    initial_floor_prefix_kind = (
        "strict_floor"
        if initial_floor_prefix_errors
        and all(
            error <= float(strict_floor)
            for error in initial_floor_prefix_errors
        )
        else "per_record_attainable_floor"
    )
    maximum_initial_floor_prefix_error = (
        max(initial_floor_prefix_errors)
        if initial_floor_prefix_errors
        else None
    )
    maximum_post_floor_prefix_error = (
        max(post_floor_prefix_errors) if post_floor_prefix_errors else None
    )
    post_prefix_outside_floor = any(
        error > floor
        for error, floor in zip(
            post_floor_prefix_errors, post_floor_prefix_floors
        )
    )
    post_prefix_cancellation_escape = (
        maximum_initial_floor_prefix_error is not None
        and maximum_post_floor_prefix_error is not None
        and maximum_post_floor_prefix_error
        > maximum_initial_floor_prefix_error
    )
    immediate_strict_floor = all_errors_below_strict_floor
    fitted_order = None
    fitted_indices = ()
    fitted_epsilon_interval = None
    order_estimate_kind = None
    monotone_prefix_length = len(truncation_prefix_indices)
    monotone_prefix = truncation_prefix_monotone
    truncated_fallback_eligible = (
        active_set_truncated
        and active_set_safe
        and strictly_interior_safe
        and monotone_prefix_length >= 2
        and monotone_prefix
    )
    ratios = _ratios(errors)
    secondary_factor_window_start = None
    secondary_factor_window_end = None
    secondary_factor_window_ratios = ()
    for factor_start in range(max(0, len(ratios) - 1)):
        factor_pair = ratios[factor_start : factor_start + 2]
        if len(factor_pair) != 2:
            continue
        if not all(
            FACTOR_INTERVAL[0] <= ratio <= FACTOR_INTERVAL[1]
            for ratio in factor_pair
        ):
            continue
        if not all(
            errors[index] > errors[index + 1]
            for index in range(factor_start, factor_start + 2)
        ):
            continue
        secondary_factor_window_start = factor_start
        secondary_factor_window_end = factor_start + 1
        secondary_factor_window_ratios = factor_pair
        break
    later_floor_contact_indices = tuple(
        index
        for index in range(minimum_index + 1, len(errors))
        if errors[index] <= attainable_floors[index]
    )
    first_later_floor_contact_index = (
        later_floor_contact_indices[0]
        if later_floor_contact_indices
        else None
    )
    minimum_to_later_floor_contact_distance = (
        first_later_floor_contact_index - minimum_index
        if first_later_floor_contact_index is not None
        else None
    )
    if start is not None:
        assert minimum_index >= end + 1
        assert floor_reached, {
            "errors": errors,
            "attainable_floor": attainable_floor,
            "minimum_index": minimum_index,
        }
        assert upturn_after_minimum or errors[minimum_index] <= float(
            strict_floor
        )
        classification = "factor_of_four_then_measured_subtraction_floor"
    elif allow_immediate_floor and (
        all_errors_within_per_record_attainable_floor
    ):
        assert independent_checks
        assert active_set_safe
        assert enough_immediate_floor_records
        assert floor_reached
        if all_errors_below_strict_floor:
            classification = "immediate_strict_floor"
        else:
            classification = "immediate_per_record_attainable_floor"
    elif allow_immediate_floor and initial_floor_prefix_length >= 3:
        assert independent_checks
        assert active_set_safe
        assert len(errors) >= 4
        assert first_floor_escape_index is not None
        assert post_prefix_outside_floor
        assert post_prefix_cancellation_escape, {
            "initial_floor_prefix_errors": initial_floor_prefix_errors,
            "post_floor_prefix_errors": post_floor_prefix_errors,
        }
        classification = "immediate_floor_prefix_then_roundoff_escape"
    elif truncated_fallback_eligible:
        fitted_indices = truncation_prefix_indices
        fitted_epsilons = np.asarray(
            [epsilons[index] for index in fitted_indices]
        )
        fitted_errors = np.asarray(
            [errors[index] for index in fitted_indices]
        )
        assert np.all(fitted_epsilons > 0.0)
        assert np.all(fitted_errors > 0.0)
        if monotone_prefix_length >= 3:
            design = np.column_stack(
                (
                    np.log(fitted_epsilons),
                    np.ones(monotone_prefix_length),
                )
            )
            fitted_order, _ = np.linalg.lstsq(
                design, np.log(fitted_errors), rcond=None
            )[0]
            fitted_order = float(fitted_order)
            order_estimate_kind = "least_squares_log_log"
        else:
            assert monotone_prefix_length == 2
            assert independent_checks
            assert errors[0] > attainable_floors[0]
            assert errors[1] <= attainable_floors[1]
            fitted_order = float(
                np.log(errors[0] / errors[1])
                / np.log(epsilons[0] / epsilons[1])
            )
            order_estimate_kind = "two_point_actual_epsilon_ratio"
        fitted_epsilon_interval = (
            float(fitted_epsilons[0]),
            float(fitted_epsilons[-1]),
        )
        assert 1.7 <= fitted_order <= 2.3, {
            "fitted_order": fitted_order,
            "fitted_indices": fitted_indices,
            "fitted_epsilons": fitted_epsilons,
            "fitted_errors": fitted_errors,
        }
        assert floor_reached, {
            "minimum_error": errors[minimum_index],
            "strict_floor": float(strict_floor),
            "attainable_floor": attainable_floor,
        }
        if monotone_prefix_length == 2:
            assert post_floor_upturn
            classification = (
                "active_set_truncated_single_quadratic_step_into_floor"
            )
        else:
            if minimum_index + 1 < len(errors):
                assert upturn_after_minimum
            classification = (
                "active_set_truncated_near_quadratic_then_subtraction_floor"
            )
    elif secondary_metric:
        assert independent_checks
        assert active_set_safe
        assert primary_metric_certified
        assert secondary_factor_window_start is not None
        factor_transition_index = secondary_factor_window_end + 1
        assert minimum_index in (
            factor_transition_index,
            factor_transition_index + 1,
        )
        assert upturn_after_minimum
        assert first_later_floor_contact_index is not None
        assert minimum_to_later_floor_contact_distance <= 2
        classification = (
            "secondary_metric_two_factor_steps_then_bracketed_floor"
        )
    else:
        assert allow_immediate_floor and independent_checks, errors
        assert active_set_safe
        assert first_floor_reaching_index != 0, {
            "errors": errors,
            "attainable_floors": attainable_floors,
        }
        assert floor_reached, {
            "errors": errors,
            "attainable_floor": attainable_floor,
            "minimum_index": minimum_index,
        }
        assert upturn_after_minimum
        classification = "immediate_measured_subtraction_floor"
    record = {
        "classification": classification,
        "ratios": _ratios(errors),
        "factor_window_start": start,
        "factor_window_end": end,
        "factor_window_ratios": factor_ratios,
        "minimum_error_index": minimum_index,
        "minimum_error_epsilon": float(minimum_record["epsilon"]),
        "minimum_error": errors[minimum_index],
        "raw_measured_relative_floor": float(
            minimum_record[relative_floor_key]
        ),
        "roundoff_safety_factor": ROUNDOFF_SAFETY_FACTOR,
        "attainable_relative_floor": attainable_floor,
        "attainable_relative_floors": tuple(attainable_floors),
        "strict_relative_floor": float(strict_floor),
        "selected_errors": tuple(errors),
        "selected_record_count": len(errors),
        "all_errors_below_strict_floor": (
            all_errors_below_strict_floor
        ),
        "all_errors_within_per_record_attainable_floor": (
            all_errors_within_per_record_attainable_floor
        ),
        "initial_floor_prefix_indices": initial_floor_prefix_indices,
        "initial_floor_prefix_epsilons": initial_floor_prefix_epsilons,
        "initial_floor_prefix_errors": initial_floor_prefix_errors,
        "initial_floor_prefix_attainable_floors": (
            initial_floor_prefix_floors
        ),
        "initial_floor_prefix_kind": initial_floor_prefix_kind,
        "first_floor_escape_index": first_floor_escape_index,
        "post_floor_prefix_errors": post_floor_prefix_errors,
        "post_floor_prefix_attainable_floors": post_floor_prefix_floors,
        "maximum_initial_floor_prefix_error": (
            maximum_initial_floor_prefix_error
        ),
        "maximum_post_floor_prefix_error": maximum_post_floor_prefix_error,
        "post_prefix_cancellation_escape": post_prefix_cancellation_escape,
        "first_floor_reaching_index": first_floor_reaching_index,
        "truncation_prefix_indices": truncation_prefix_indices,
        "truncation_prefix_epsilons": truncation_prefix_epsilons,
        "active_set_safe": bool(active_set_safe),
        "active_set_truncated": bool(active_set_truncated),
        "strictly_interior_safe_ladder": bool(strictly_interior_safe),
        "independent_checks": bool(independent_checks),
        "fitted_order": fitted_order,
        "fitted_indices": fitted_indices,
        "fitted_epsilon_interval": fitted_epsilon_interval,
        "order_estimate_kind": order_estimate_kind,
        "monotone_prefix_length": monotone_prefix_length,
        "monotone_prefix": monotone_prefix,
        "truncated_fallback_eligible": truncated_fallback_eligible,
        "immediate_strict_floor": immediate_strict_floor,
        "upturn_after_minimum": upturn_after_minimum,
        "post_floor_upturn": post_floor_upturn,
        "secondary_metric": bool(secondary_metric),
        "primary_metric_certified": bool(primary_metric_certified),
        "secondary_factor_window_start": secondary_factor_window_start,
        "secondary_factor_window_end": secondary_factor_window_end,
        "secondary_factor_window_ratios": secondary_factor_window_ratios,
        "secondary_factor_ratio_indices": (
            tuple(
                range(
                    secondary_factor_window_start,
                    secondary_factor_window_end + 1,
                )
            )
            if secondary_factor_window_start is not None
            else ()
        ),
        "minimum_local_attainable_floor": attainable_floor,
        "minimum_to_local_floor_ratio": (
            errors[minimum_index] / attainable_floor
        ),
        "first_later_floor_contact_index": (
            first_later_floor_contact_index
        ),
        "minimum_to_later_floor_contact_distance": (
            minimum_to_later_floor_contact_distance
        ),
        "certified": True,
    }
    return record


def _new_state(case, name):
    state, _, _ = case["model"].get_x_var(name)
    return state


def _new_full_state(case, name):
    return case["model"].get_full_var(name, split_x_and_aux=True)


def _copy_state(value, name):
    result = value.copy(deepcopy=True)
    result.rename(name)
    return result


def _dual_linear_combination(space, terms, name):
    result = Cofunction(space, name=name)
    result.zero()
    for scale, value in terms:
        with result.dat.vec as target, value.dat.vec_ro as increment:
            target.axpy(float(scale), increment)
    return result


def _dual_natural_norm(case, value, name):
    representative = case["helper"].state_riesz_representative(value, name)
    squared = case["helper"].dual_pairing(value, representative)
    scale = max(np.linalg.norm(_values(value)) ** 2, 1.0)
    assert squared >= -100.0 * np.finfo(float).eps * scale
    return float(np.sqrt(max(squared, 0.0)))


def _dual_reproducibility(case, first, second, name):
    difference = _dual_linear_combination(
        first.function_space(),
        ((1.0, first), (-1.0, second)),
        f"{name}_difference",
    )
    return {
        "coefficient_vector_absolute_error": float(
            np.linalg.norm(_values(difference))
        ),
        "natural_absolute_error": _dual_natural_norm(
            case, difference, f"{name}_difference_riesz"
        ),
        "first_coefficient_vector_norm": float(np.linalg.norm(_values(first))),
        "second_coefficient_vector_norm": float(np.linalg.norm(_values(second))),
        "first_natural_norm": _dual_natural_norm(
            case, first, f"{name}_first_riesz"
        ),
        "second_natural_norm": _dual_natural_norm(
            case, second, f"{name}_second_riesz"
        ),
    }


def _dual_fd_diagnostic(
    case,
    epsilon,
    exact,
    plus,
    minus,
    reproducibility,
    name,
):
    numerator = _dual_linear_combination(
        exact.function_space(),
        ((1.0, plus), (-1.0, minus)),
        f"{name}_numerator",
    )
    centered = _dual_linear_combination(
        exact.function_space(),
        (
            (1.0 / (2.0 * float(epsilon)), plus),
            (-1.0 / (2.0 * float(epsilon)), minus),
        ),
        f"{name}_centered",
    )
    error = _dual_linear_combination(
        exact.function_space(),
        ((1.0, exact), (-1.0, centered)),
        f"{name}_error",
    )
    coefficient_norms = {
        "exact": float(np.linalg.norm(_values(exact))),
        "centered": float(np.linalg.norm(_values(centered))),
        "plus": float(np.linalg.norm(_values(plus))),
        "minus": float(np.linalg.norm(_values(minus))),
        "numerator": float(np.linalg.norm(_values(numerator))),
        "absolute_error": float(np.linalg.norm(_values(error))),
    }
    natural_norms = {
        "exact": _dual_natural_norm(case, exact, f"{name}_exact_riesz"),
        "centered": _dual_natural_norm(
            case, centered, f"{name}_centered_riesz"
        ),
        "plus": _dual_natural_norm(case, plus, f"{name}_plus_riesz"),
        "minus": _dual_natural_norm(case, minus, f"{name}_minus_riesz"),
        "numerator": _dual_natural_norm(
            case, numerator, f"{name}_numerator_riesz"
        ),
        "absolute_error": _dual_natural_norm(
            case, error, f"{name}_error_riesz"
        ),
    }
    coefficient_scale = max(
        coefficient_norms["exact"],
        coefficient_norms["centered"],
        np.finfo(float).tiny,
    )
    natural_scale = max(
        natural_norms["exact"],
        natural_norms["centered"],
        np.finfo(float).tiny,
    )
    coefficient_floor = (
        reproducibility["coefficient_vector_absolute_error"]
        + np.finfo(float).eps
        * (coefficient_norms["plus"] + coefficient_norms["minus"])
    ) / (2.0 * float(epsilon))
    natural_floor = (
        reproducibility["natural_absolute_error"]
        + np.finfo(float).eps
        * (natural_norms["plus"] + natural_norms["minus"])
    ) / (2.0 * float(epsilon))
    record = {
        "epsilon": float(epsilon),
        "coefficient_vector": {
            **coefficient_norms,
            "relative_error": coefficient_norms["absolute_error"]
            / coefficient_scale,
            "scale": coefficient_scale,
            "raw_roundoff_absolute_floor": coefficient_floor,
            "raw_roundoff_relative_floor": coefficient_floor
            / coefficient_scale,
        },
        "natural_mass_riesz": {
            **natural_norms,
            "relative_error": natural_norms["absolute_error"] / natural_scale,
            "scale": natural_scale,
            "raw_roundoff_absolute_floor": natural_floor,
            "raw_roundoff_relative_floor": natural_floor / natural_scale,
        },
    }
    return DualFDDiagnosticResult(
        record=record,
        centered_dual=centered,
        numerator_dual=numerator,
        error_dual=error,
    )


def test_dual_fd_diagnostic_result_contract():
    assert tuple(field.name for field in fields(DualFDDiagnosticResult)) == (
        "record",
        "centered_dual",
        "numerator_dual",
        "error_dual",
    )


def test_active_set_truncated_scale_classifier_contract():
    epsilons = (0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125)
    gradient_errors = (
        1.1752523129319e-8,
        2.9379549891086098e-9,
        5.744368650193705e-10,
        1.974488495805344e-10,
        1.4970337730376229e-9,
    )
    ic_probe_errors = (
        1.5740918751133455e-9,
        3.951326930992529e-10,
        1.7302564355017118e-10,
        6.866683477406852e-11,
        2.0571161006801823e-10,
    )
    combined_probe_errors = (
        1.3853932072132853e-9,
        3.6785589214053265e-10,
        3.214568728623676e-10,
        1.9950858649456812e-10,
        1.053114074097447e-9,
    )
    ic_natural_errors = (
        7.799172758518771e-10,
        2.2756663810021927e-10,
        1.8309531331481302e-10,
        2.878037488206652e-10,
        7.724697794749904e-10,
    )
    combined_natural_errors = (
        8.763633521462349e-10,
        2.4954256423046307e-10,
        1.8831030672507957e-10,
        3.2823104880127766e-10,
        9.941986000958176e-10,
    )

    def records(errors):
        return tuple(
            {
                "epsilon": epsilon,
                "relative_error": error,
                "raw_roundoff_relative_floor": 0.0,
            }
            for epsilon, error in zip(epsilons, errors)
        )

    result = _scale_aware_classifier(
        records(gradient_errors),
        strict_floor=STRICT_FLOOR,
        active_set_safe=True,
        active_set_truncated=True,
        strictly_interior_safe=True,
    )
    assert result["classification"] == (
        "active_set_truncated_near_quadratic_then_subtraction_floor"
    )
    assert 1.7 <= result["fitted_order"] <= 2.3
    assert result["first_floor_reaching_index"] == 2
    assert result["truncation_prefix_indices"] == (0, 1, 2)
    assert result["fitted_indices"] == (0, 1, 2)
    assert result["fitted_epsilon_interval"] == (0.0125, 0.003125)
    assert result["monotone_prefix_length"] == 3
    assert result["minimum_error_index"] == 3
    assert result["minimum_error"] < STRICT_FLOOR
    assert result["upturn_after_minimum"]

    for errors, expected_order in (
        (ic_probe_errors, 1.9941106248714737),
        (combined_probe_errors, 1.9130829001594707),
    ):
        result = _scale_aware_classifier(
            records(errors),
            strict_floor=STRICT_FLOOR,
            independent_checks=True,
            active_set_safe=True,
            active_set_truncated=True,
            strictly_interior_safe=True,
        )
        assert result["classification"] == (
            "active_set_truncated_single_quadratic_step_into_floor"
        )
        assert result["first_floor_reaching_index"] == 1
        assert result["truncation_prefix_indices"] == (0, 1)
        assert result["truncation_prefix_epsilons"] == (0.0125, 0.00625)
        assert result["fitted_indices"] == (0, 1)
        assert result["fitted_order"] == pytest.approx(expected_order)
        assert result["order_estimate_kind"] == (
            "two_point_actual_epsilon_ratio"
        )
        assert result["strict_relative_floor"] == STRICT_FLOOR
        assert result["attainable_relative_floors"] == (
            STRICT_FLOOR,
        ) * len(epsilons)
        assert result["independent_checks"]
        assert result["post_floor_upturn"]

        with pytest.raises(AssertionError):
            _scale_aware_classifier(
                records(errors),
                strict_floor=STRICT_FLOOR,
                independent_checks=False,
                active_set_safe=True,
                active_set_truncated=True,
                strictly_interior_safe=True,
            )

    for errors in (ic_natural_errors, combined_natural_errors):
        result = _scale_aware_classifier(
            records(errors),
            strict_floor=STRICT_FLOOR,
            allow_immediate_floor=True,
            independent_checks=True,
            active_set_safe=True,
            active_set_truncated=True,
            strictly_interior_safe=True,
        )
        assert result["classification"] == "immediate_strict_floor"
        assert result["strict_relative_floor"] == 1.0e-9
        assert result["selected_errors"] == errors
        assert result["selected_record_count"] == 5
        assert result["attainable_relative_floors"] == (
            STRICT_FLOOR,
        ) * len(epsilons)
        assert result["all_errors_below_strict_floor"]
        assert result[
            "all_errors_within_per_record_attainable_floor"
        ]
        assert result["fitted_order"] is None
        assert result["fitted_indices"] == ()
        assert result["order_estimate_kind"] is None

        for independent_checks, active_set_safe in (
            (False, True),
            (True, False),
        ):
            with pytest.raises(AssertionError):
                _scale_aware_classifier(
                    records(errors),
                    strict_floor=STRICT_FLOOR,
                    allow_immediate_floor=True,
                    independent_checks=independent_checks,
                    active_set_safe=active_set_safe,
                    active_set_truncated=True,
                    strictly_interior_safe=True,
                )

    above_every_floor = list(ic_natural_errors)
    above_every_floor[2] = 1.01e-9
    with pytest.raises(AssertionError):
        _scale_aware_classifier(
            records(above_every_floor),
            strict_floor=STRICT_FLOOR,
            allow_immediate_floor=True,
            independent_checks=True,
            active_set_safe=True,
            active_set_truncated=True,
            strictly_interior_safe=True,
        )


def test_immediate_floor_prefix_roundoff_escape_classifier_contract():
    epsilons = REDUCED_DIAGNOSTIC_EPSILONS[1:]
    c0_scalar_errors = (
        2.3659715736786227e-11,
        5.7248518327684635e-12,
        6.794392392598897e-11,
        4.0248262778803686e-11,
        4.949186997509626e-10,
        2.728665730007977e-10,
        1.4960916071338244e-9,
        3.39478304376295e-9,
        4.231807469635109e-9,
    )

    def records(errors, raw_floors=None):
        if raw_floors is None:
            raw_floors = (0.0,) * len(errors)
        return tuple(
            {
                "epsilon": epsilon,
                "relative_error": error,
                "raw_roundoff_relative_floor": raw_floor,
            }
            for epsilon, error, raw_floor in zip(
                epsilons, errors, raw_floors
            )
        )

    result = _scale_aware_classifier(
        records(c0_scalar_errors),
        strict_floor=STRICT_FLOOR,
        allow_immediate_floor=True,
        independent_checks=True,
        active_set_safe=True,
    )
    assert result["classification"] == (
        "immediate_floor_prefix_then_roundoff_escape"
    )
    assert result["initial_floor_prefix_indices"] == tuple(range(6))
    assert result["initial_floor_prefix_epsilons"] == epsilons[:6]
    assert result["initial_floor_prefix_errors"] == c0_scalar_errors[:6]
    assert result["initial_floor_prefix_attainable_floors"] == (
        STRICT_FLOOR,
    ) * 6
    assert result["initial_floor_prefix_kind"] == "strict_floor"
    assert result["first_floor_escape_index"] == 6
    assert result["post_floor_prefix_errors"] == c0_scalar_errors[6:]
    assert result["maximum_initial_floor_prefix_error"] == max(
        c0_scalar_errors[:6]
    )
    assert result["maximum_post_floor_prefix_error"] == max(
        c0_scalar_errors[6:]
    )
    assert result["post_prefix_cancellation_escape"]
    assert result["fitted_order"] is None
    assert result["order_estimate_kind"] is None

    irregular_probe_errors = (
        4.0e-10,
        2.0e-10,
        3.0e-10,
        2.0e-9,
        5.0e-10,
        3.0e-9,
    )
    irregular = _scale_aware_classifier(
        records(irregular_probe_errors),
        strict_floor=STRICT_FLOOR,
        allow_immediate_floor=True,
        independent_checks=True,
        active_set_safe=True,
    )
    assert irregular["classification"] == (
        "immediate_floor_prefix_then_roundoff_escape"
    )
    assert irregular["initial_floor_prefix_indices"] == (0, 1, 2)
    assert irregular["first_floor_escape_index"] == 3
    assert irregular["post_floor_prefix_errors"] == (
        2.0e-9,
        5.0e-10,
        3.0e-9,
    )

    for independent_checks, active_set_safe in (
        (False, True),
        (True, False),
    ):
        with pytest.raises(AssertionError):
            _scale_aware_classifier(
                records(c0_scalar_errors),
                strict_floor=STRICT_FLOOR,
                allow_immediate_floor=True,
                independent_checks=independent_checks,
                active_set_safe=active_set_safe,
            )

    for invalid_errors in (
        (1.0e-10, 2.0e-10, 2.0e-9, 3.0e-9),
        (1.0e-10, 2.0e-10, 1.01e-9, 2.0e-9, 3.0e-9),
    ):
        with pytest.raises(AssertionError):
            _scale_aware_classifier(
                records(invalid_errors),
                strict_floor=STRICT_FLOOR,
                allow_immediate_floor=True,
                independent_checks=True,
                active_set_safe=True,
            )

    no_escape_errors = (8.0e-9, 7.0e-9, 6.0e-9, 1.1e-9)
    no_escape_raw_floors = (
        1.0e-8 / ROUNDOFF_SAFETY_FACTOR,
        1.0e-8 / ROUNDOFF_SAFETY_FACTOR,
        1.0e-8 / ROUNDOFF_SAFETY_FACTOR,
        0.0,
    )
    with pytest.raises(AssertionError):
        _scale_aware_classifier(
            records(no_escape_errors, no_escape_raw_floors),
            strict_floor=STRICT_FLOOR,
            allow_immediate_floor=True,
            independent_checks=True,
            active_set_safe=True,
        )


def test_secondary_metric_bracketed_floor_classifier_contract():
    epsilons = REDUCED_DIAGNOSTIC_EPSILONS[:6]
    errors = (
        8.717998581810343e-8,
        2.1792455305071106e-8,
        5.4588100254692644e-9,
        1.5940985005644644e-9,
        3.1297725209364616e-9,
        2.089126084432338e-9,
    )
    minimum_floor = 1.50729765042339e-9
    contact_floor = 2.2e-9

    def records(values, floors=None):
        if floors is None:
            floors = (STRICT_FLOOR,) * len(values)
        return tuple(
            {
                "epsilon": epsilon,
                "relative_error": error,
                "raw_roundoff_relative_floor": (
                    floor / ROUNDOFF_SAFETY_FACTOR
                    if floor > STRICT_FLOOR
                    else 0.0
                ),
            }
            for epsilon, error, floor in zip(epsilons, values, floors)
        )

    floors = (
        STRICT_FLOOR,
        STRICT_FLOOR,
        STRICT_FLOOR,
        minimum_floor,
        STRICT_FLOOR,
        contact_floor,
    )
    result = _scale_aware_classifier(
        records(errors, floors),
        strict_floor=STRICT_FLOOR,
        allow_immediate_floor=True,
        independent_checks=True,
        active_set_safe=True,
        secondary_metric=True,
        primary_metric_certified=True,
    )
    assert result["classification"] == (
        "secondary_metric_two_factor_steps_then_bracketed_floor"
    )
    assert result["secondary_metric"]
    assert result["primary_metric_certified"]
    assert result["secondary_factor_ratio_indices"] == (0, 1)
    assert result["secondary_factor_window_ratios"] == pytest.approx(
        (4.000466427379417, 3.992162248437603)
    )
    assert result["minimum_error_index"] == 3
    assert result["minimum_error"] == errors[3]
    assert result["minimum_local_attainable_floor"] == pytest.approx(
        minimum_floor
    )
    assert result["minimum_to_local_floor_ratio"] == pytest.approx(
        errors[3] / minimum_floor
    )
    assert result["first_later_floor_contact_index"] == 5
    assert result["minimum_to_later_floor_contact_distance"] == 2
    assert result["upturn_after_minimum"]
    assert result["fitted_order"] is None

    for independent_checks, active_set_safe, primary_certified in (
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ):
        with pytest.raises(AssertionError):
            _scale_aware_classifier(
                records(errors, floors),
                strict_floor=STRICT_FLOOR,
                allow_immediate_floor=True,
                independent_checks=independent_checks,
                active_set_safe=active_set_safe,
                secondary_metric=True,
                primary_metric_certified=primary_certified,
            )

    fewer_factor_steps = (
        8.0e-8,
        3.0e-8,
        1.0e-8,
        errors[3],
        errors[4],
        errors[5],
    )
    no_upturn = errors[:4] + (errors[3], errors[3])
    no_later_contact_floors = floors[:5] + (STRICT_FLOOR,)
    invalid_cases = (
        (fewer_factor_steps, floors),
        (no_upturn, floors),
        (errors, no_later_contact_floors),
    )
    for invalid_errors, invalid_floors in invalid_cases:
        with pytest.raises(AssertionError):
            _scale_aware_classifier(
                records(invalid_errors, invalid_floors),
                strict_floor=STRICT_FLOOR,
                allow_immediate_floor=True,
                independent_checks=True,
                active_set_safe=True,
                secondary_metric=True,
                primary_metric_certified=True,
            )

    far_contact_errors = errors + (2.0e-9,)
    far_contact_floors = floors[:-1] + (STRICT_FLOOR, contact_floor)
    far_contact_epsilons = REDUCED_DIAGNOSTIC_EPSILONS[:7]
    far_contact_records = tuple(
        {
            "epsilon": epsilon,
            "relative_error": error,
            "raw_roundoff_relative_floor": (
                floor / ROUNDOFF_SAFETY_FACTOR
                if floor > STRICT_FLOOR
                else 0.0
            ),
        }
        for epsilon, error, floor in zip(
            far_contact_epsilons,
            far_contact_errors,
            far_contact_floors,
        )
    )
    with pytest.raises(AssertionError):
        _scale_aware_classifier(
            far_contact_records,
            strict_floor=STRICT_FLOOR,
            allow_immediate_floor=True,
            independent_checks=True,
            active_set_safe=True,
            secondary_metric=True,
            primary_metric_certified=True,
        )


def _state_axpy(base, terms, name):
    result = _copy_state(base, name)
    for scale, value in terms:
        with result.dat.vec as target, value.dat.vec_ro as increment:
            target.axpy(float(scale), increment)
    return result


def _perturbed_state(case, base, direction, scale, name):
    result = _new_state(case, name)
    result.assign(base + float(scale) * direction)
    return result


def _zero_state(case, name):
    result = _new_state(case, name)
    result.assign(0)
    return result


def _json_normalize(value):
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
    if isinstance(value, (tuple, list)):
        return [_json_normalize(item) for item in value]
    return value


def _emit(record_property, name, payload):
    text = json.dumps(_json_normalize(payload), sort_keys=True)
    record_property(name, text)
    print(f"MTSWE_SPLIT_DIAGNOSTIC {name}={text}")


def _floor_aware_centered_classifier(errors, floor=STRICT_FLOOR):
    """Require centered convergence followed by a strict floor, or a floor."""
    errors = [float(value) for value in errors]
    ratios = [
        errors[i] / max(errors[i + 1], np.finfo(float).tiny)
        for i in range(len(errors) - 1)
    ]
    moderate_count = min(3, len(errors))
    if max(errors[:moderate_count]) < floor:
        return {
            "classification": "immediate_roundoff_floor",
            "ratios": ratios,
            "factor_window_start": None,
            "factor_window_end": None,
            "minimum_after_window": min(errors),
            "floor": floor,
            "certified": True,
        }
    window_start = None
    for start in range(max(0, len(ratios) - 2)):
        window = ratios[start : start + 3]
        if len(window) == 3 and all(
            FACTOR_INTERVAL[0] <= ratio <= FACTOR_INTERVAL[1]
            for ratio in window
        ):
            if all(errors[i] > errors[i + 1] for i in range(start, start + 3)):
                window_start = start
                break
    assert window_start is not None, errors
    window_end = window_start + 2
    minimum_after = min(errors[window_end + 1 :])
    assert minimum_after < floor, errors
    return {
        "classification": "factor_of_four_then_roundoff_floor",
        "ratios": ratios,
        "factor_window_start": window_start,
        "factor_window_end": window_end,
        "minimum_after_window": minimum_after,
        "floor": floor,
        "certified": True,
    }


def _absolute_floor_assessment(records):
    absolute_errors = [float(record["absolute_error"]) for record in records]
    minimum_index = int(np.argmin(absolute_errors))
    minimum = records[minimum_index]
    attainable = ROUNDOFF_SAFETY_FACTOR * float(
        minimum["raw_roundoff_absolute_floor"]
    )
    return {
        "minimum_error_index": minimum_index,
        "minimum_error_epsilon": float(minimum["epsilon"]),
        "minimum_absolute_error": absolute_errors[minimum_index],
        "raw_measured_absolute_floor": float(
            minimum["raw_roundoff_absolute_floor"]
        ),
        "roundoff_safety_factor": ROUNDOFF_SAFETY_FACTOR,
        "attainable_absolute_floor": attainable,
        "certified": absolute_errors[minimum_index] <= attainable,
    }


@pytest.fixture(scope="module")
def production_case():
    parameters = get_parameters(CFG)
    parameters["timestepping"]["subcycle_list"] = [2, 1, 2, 1]
    parameters["hyperviscosity"]["treat_as_coeffs"] = True
    parameters["threewayphysics"]["treat_as_coeffs"] = False
    logger = EmptyLogger()
    original_mesh = dimswe_meshes.PeriodicRectangleMesh

    def comm_self_mesh(*args, **kwargs):
        kwargs["comm"] = COMM_SELF
        return original_mesh(*args, **kwargs)

    dimswe_meshes.PeriodicRectangleMesh = comm_self_mesh
    try:
        model = get_model(parameters, logger, has_dynamics_statistics=False)
    finally:
        dimswe_meshes.PeriodicRectangleMesh = original_mesh
    assert model.mesh.comm.size == 1

    coefficient, coefficient_sub, _ = model.get_coeff_var("coefficient")
    state_container, state_sub, _ = model.get_full_var(
        "production_mtswe_state", split_x_and_aux=True
    )
    time = model.get_t_var()
    model.initialize(state_sub, time)
    model.set_coeffs(parameters, coefficient_sub)

    x = SpatialCoordinate(model.mesh)
    lx = model.initcond.Lx
    ly = model.initcond.Ly
    mode_x = sin(2.0 * pi * x[0] / lx)
    mode_y = cos(2.0 * pi * x[1] / ly)
    height = 750.0 + 4.0 * mode_x + 3.0 * mode_y
    entropy_density = height * model.initcond.g * (
        1.02 + 0.0015 * mode_x - 0.0010 * mode_y
    )
    state_sub["v"].project(
        as_vector([25.0 + 1.5 * mode_y, 17.0 + 1.0 * mode_x])
    )
    state_sub["h"].project(height)
    state_sub["S"].project(entropy_density)
    state_sub["Qv"].project(0.0030 * height)
    state_sub["Qc"].project(0.0010 * height)
    state_sub["Qr"].project(0.0002 * height)

    split = get_timestepper(
        parameters, model, logger, _serial_solver_parameters()
    )
    split.set_coeff(coefficient)
    helper = split._get_mtswe_split_hvp_helper()

    direction = model.get_x_var("production_mtswe_direction")[0]
    direction.sub(0).project(
        as_vector([0.22 * mode_x, -0.17 * mode_y])
    )
    direction.sub(1).project(0.18 * mode_y)
    direction.sub(2).project(1.7 * mode_x - 1.1 * mode_y)
    direction.sub(3).project(1.1e-5 * height * (1.0 + 0.2 * mode_x))
    direction.sub(4).project(-8.0e-6 * height * (1.0 - 0.2 * mode_y))
    direction.sub(5).project(6.0e-6 * height * (1.0 + 0.1 * mode_x))

    probe = model.get_x_var("production_mtswe_probe")[0]
    probe.sub(0).project(as_vector([-0.15 * mode_y, 0.19 * mode_x]))
    probe.sub(1).project(-0.12 * mode_x)
    probe.sub(2).project(1.3 * mode_y)
    probe.sub(3).project(-7.0e-6 * height * mode_y)
    probe.sub(4).project(5.0e-6 * height * mode_x)
    probe.sub(5).project(9.0e-6 * height * mode_y)

    target = model.get_x_var("production_mtswe_target")[0]
    target.assign(0.985 * state_container[0])

    return {
        "parameters": parameters,
        "model": model,
        "coefficient": coefficient,
        "coefficient_sub": coefficient_sub,
        "split": split,
        "helper": helper,
        "dry": split.time_integrators[0],
        "hyper": split.time_integrators[1],
        "dg": split.time_integrators[2],
        "moist": split.time_integrators[3],
        "state": state_container[0],
        "direction": direction,
        "probe": probe,
        "target": target,
        "t0": float(time),
        "dt": float(parameters["timestepping"]["dt"]),
    }


def _set_c0(case, value):
    case["coefficient_sub"]["c0"].assign(float(value))
    case["split"].set_coeff(case["coefficient"])


def _legacy_complete_step(case, state, c0, time, name):
    _set_c0(case, c0)
    output, output_sub, _ = _new_full_state(case, name)
    case["split"].reset_internal_vars()
    case["split"].take_forward_step(
        output, output_sub, [state], float(time), case["dt"]
    )
    return _copy_state(output[0], f"{name}_copy")


def _independent_child_boundaries(case, state, c0, name):
    _set_c0(case, c0)
    boundaries = [_copy_state(state, f"{name}_boundary_0")]
    current = boundaries[0]
    for integrator_index, child in enumerate(case["split"].time_integrators):
        count = case["split"].subcycle_list[integrator_index]
        child_dt = case["dt"] / count
        for subcycle_index in range(count):
            output, output_sub, _ = _new_full_state(
                case, f"{name}_{integrator_index}_{subcycle_index}"
            )
            child.reset_internal_vars()
            child.take_forward_step(
                output,
                output_sub,
                [current],
                case["t0"] + subcycle_index * child_dt,
                child_dt,
            )
            current = _copy_state(
                output[0], f"{name}_boundary_{len(boundaries)}"
            )
            boundaries.append(current)
    return tuple(boundaries)


def _legacy_trajectory(case, nsteps, state, c0, name):
    states = [_copy_state(state, f"{name}_state_0")]
    for n in range(nsteps):
        states.append(
            _legacy_complete_step(
                case,
                states[-1],
                c0,
                case["t0"] + n * case["dt"],
                f"{name}_step_{n}",
            )
        )
    return tuple(states)


def _terminal_residual(case, state, name):
    residual = _new_state(case, name)
    residual.assign(state - case["target"])
    return residual


def _objective(case, state):
    residual = _terminal_residual(case, state, "mtswe_objective_residual")
    return 0.5 * float(
        assemble(inner(residual, residual) * case["model"].spaces.dx)
    )


def _legacy_reduced_gradient(case, nsteps, state, c0, name):
    states = _legacy_trajectory(case, nsteps, state, c0, name)
    current = _terminal_residual(case, states[-1], f"{name}_terminal")
    c0_gradient = 0.0
    c0_index = case["model"].get_coeff_list().index("c0")
    for n in range(nsteps - 1, -1, -1):
        delta_gradient = case["model"].get_coeff_var(
            f"{name}_delta_gradient_{n}"
        )[0]
        delta_lambda = _new_state(case, f"{name}_delta_lambda_{n}")
        case["split"].reset_internal_vars()
        _, gradient = case["split"].take_adjoint_step(
            delta_gradient,
            delta_lambda,
            current,
            [states[n]],
            case["t0"] + (n + 1) * case["dt"],
            case["dt"],
        )
        c0_gradient += float(gradient[c0_index])
        updated = _new_state(case, f"{name}_legacy_adjoint_{n}")
        updated.assign(current + delta_lambda)
        current = updated
    return c0_gradient, current, states


def _moist_cache(parent_cache):
    assert parent_cache.children[-1].name == "moist_euler"
    cache = parent_cache.children[-1].cache
    assert isinstance(cache, MoistEulerPrimalCache)
    return cache


def _moist_signed_switch_summary(case, parent_cache, name):
    """Independently sample signed distances to every deployed moist switch."""
    cache = _moist_cache(parent_cache)
    state = cache.stage_state
    field_indices = {
        field: index
        for index, field in enumerate(case["model"].get_x_var_list())
    }
    fields = {
        field: state.sub(index) for field, index in field_indices.items()
    }
    term = case["helper"].moist_helper.term
    h = fields["h"]
    qv = fields["Qv"] / h
    qc = fields["Qc"] / h
    entropy = fields["S"] / h
    beta2 = term.g * term.L
    q_sat = qsat(h, entropy, term.B, term.q0, term.H0, term.g)
    gamma_v = 1.0 / (1.0 + q_sat * 20.0 * beta2 / term.g)
    condensation = gamma_v * (qv - q_sat) / term.tau_v
    evaporation = gamma_v * (q_sat - qv) / term.tau_v
    evaporation_positive = ufl.max_value(0.0, evaporation)
    cap_difference = qc / term.dt - evaporation_positive
    rain = term.gamma_r * (qc - term.qprecip) / term.tau_r
    expressions = {
        "condensation": condensation,
        "evaporation": evaporation,
        "evaporation_cap": cap_difference,
        "rain": rain,
        "depth_denominator": h + term.B,
    }
    space = fields["Qv"].function_space()
    result = {}
    for switch_name, expression in expressions.items():
        sampled = Function(space, name=f"{name}_{switch_name}")
        sampled.interpolate(expression)
        values = np.array(sampled.dat.data_ro, dtype=float, copy=True).reshape(-1)
        assert values.size > 0
        result[switch_name] = {
            "minimum_signed_value": float(np.min(values)),
            "maximum_signed_value": float(np.max(values)),
            "minimum_absolute_margin": float(np.min(np.abs(values))),
            "positive_dof_count": int(np.count_nonzero(values > 0.0)),
            "negative_dof_count": int(np.count_nonzero(values < 0.0)),
            "zero_dof_count": int(np.count_nonzero(values == 0.0)),
        }
    result["stored_signature"] = cache.active_set.signature
    result["stored_absolute_margins"] = {
        "condensation": cache.active_set.condensation_margin,
        "evaporation": cache.active_set.evaporation_margin,
        "evaporation_cap": cache.active_set.evaporation_cap_margin,
        "rain": cache.active_set.rain_margin,
        "depth_denominator": cache.active_set.depth_denominator_margin,
    }
    return result


def _mask_changes(base_signature, candidate_signature):
    result = {}
    for index, branch in enumerate(ACTIVE_BRANCH_NAMES):
        base = np.asarray(base_signature[index], dtype=bool)
        candidate = np.asarray(candidate_signature[index], dtype=bool)
        assert base.shape == candidate.shape
        result[branch] = int(np.count_nonzero(base != candidate))
    return result


def _switch_summary_well_separated(summary):
    for switch_name in (*ACTIVE_BRANCH_NAMES, "depth_denominator"):
        switch = summary[switch_name]
        scale = max(
            abs(switch["minimum_signed_value"]),
            abs(switch["maximum_signed_value"]),
            np.finfo(float).tiny,
        )
        if switch["minimum_absolute_margin"] <= (
            128.0 * np.finfo(float).eps * scale
        ):
            return False
    return True


def _trajectory_active_set_record(
    case, base_caches, plus_caches, minus_caches, epsilon, name
):
    assert len(base_caches) == len(plus_caches) == len(minus_caches)
    timesteps = []
    safe = True
    for timestep, (base, plus, minus) in enumerate(
        zip(base_caches, plus_caches, minus_caches)
    ):
        base_active = _moist_cache(base).active_set
        plus_active = _moist_cache(plus).active_set
        minus_active = _moist_cache(minus).active_set
        plus_changes = _mask_changes(
            base_active.signature, plus_active.signature
        )
        minus_changes = _mask_changes(
            base_active.signature, minus_active.signature
        )
        base_summary = _moist_signed_switch_summary(
            case, base, f"{name}_base_{timestep}"
        )
        plus_summary = _moist_signed_switch_summary(
            case, plus, f"{name}_plus_{timestep}"
        )
        minus_summary = _moist_signed_switch_summary(
            case, minus, f"{name}_minus_{timestep}"
        )
        well_separated = all(
            _switch_summary_well_separated(summary)
            for summary in (base_summary, plus_summary, minus_summary)
        )
        timestep_safe = (
            not any(plus_changes.values())
            and not any(minus_changes.values())
            and well_separated
        )
        safe = safe and timestep_safe
        timesteps.append(
            {
                "timestep": timestep,
                "safe": timestep_safe,
                "well_separated_from_switches": well_separated,
                "base": base_summary,
                "plus": plus_summary,
                "minus": minus_summary,
                "plus_changed_dofs": plus_changes,
                "minus_changed_dofs": minus_changes,
            }
        )
    return {
        "epsilon": float(epsilon),
        "safe": safe,
        "timesteps": timesteps,
    }


def _safe_records_strictly_inside(records):
    """Return the largest sampled safe interval and its strict interior."""
    interval_start = None
    for index in range(len(records)):
        if all(record["active_set"]["safe"] for record in records[index:]):
            interval_start = index
            break
    if interval_start is None:
        return None, []
    largest_safe = float(records[interval_start]["epsilon"])
    return largest_safe, list(records[interval_start + 1 :])


def _assert_active_sets_equal(base_caches, plus_caches, minus_caches):
    assert len(base_caches) == len(plus_caches) == len(minus_caches)
    for base, plus, minus in zip(base_caches, plus_caches, minus_caches):
        base_active = _moist_cache(base).active_set
        plus_active = _moist_cache(plus).active_set
        minus_active = _moist_cache(minus).active_set
        assert plus_active.signature == base_active.signature
        assert minus_active.signature == base_active.signature
        assert min(
            base_active.condensation_margin,
            base_active.evaporation_margin,
            base_active.evaporation_cap_margin,
            base_active.rain_margin,
            base_active.depth_denominator_margin,
            plus_active.condensation_margin,
            plus_active.evaporation_margin,
            plus_active.evaporation_cap_margin,
            plus_active.rain_margin,
            plus_active.depth_denominator_margin,
            minus_active.condensation_margin,
            minus_active.evaporation_margin,
            minus_active.evaporation_cap_margin,
            minus_active.rain_margin,
            minus_active.depth_denominator_margin,
        ) > 0.0


def _direction(case, kind):
    if kind == "c0":
        return _zero_state(case, "mtswe_c0_only_state_direction"), DELTA_C0
    if kind == "ic":
        return case["direction"], 0.0
    if kind == "combined":
        return case["direction"], DELTA_C0
    raise ValueError(kind)


def test_exact_complete_graph_and_every_child_boundary(
    production_case, record_property
):
    case = production_case
    _set_c0(case, PHYSICAL_C0)
    primal = case["split"].take_mtswe_forward_step_cached(
        case["state"], case["t0"], case["dt"]
    )
    legacy = _legacy_complete_step(
        case,
        case["state"],
        PHYSICAL_C0,
        case["t0"],
        "mtswe_complete_legacy",
    )
    independent = _independent_child_boundaries(
        case, case["state"], PHYSICAL_C0, "mtswe_independent_children"
    )
    diagnostics = case["helper"].production_graph_diagnostics()
    assert isinstance(primal, MTSWESplitPrimalCache)
    assert primal.forward_child_order == (
        "dry_rk4_0",
        "dry_rk4_1",
        "hyperviscosity_euler",
        "dg_ssprk43_0",
        "dg_ssprk43_1",
        "moist_euler",
    )
    assert diagnostics["reverse_child_order"] == tuple(
        reversed(primal.forward_child_order)
    )
    assert len(primal.boundary_states) == len(independent) == 7
    assert np.array_equal(_values(primal.state_out), _values(legacy))
    assert all(
        np.array_equal(_values(cached), _values(production))
        for cached, production in zip(primal.boundary_states, independent)
    )
    for before_index, after_index in ((0, 1), (1, 2), (2, 3)):
        for field_index in (3, 4, 5):
            np.testing.assert_array_equal(
                primal.boundary_states[after_index].sub(field_index).dat.data_ro,
                primal.boundary_states[before_index].sub(field_index).dat.data_ro,
            )
    for before_index, after_index in ((3, 4), (4, 5)):
        for field_index in (0, 1, 2):
            np.testing.assert_array_equal(
                primal.boundary_states[after_index].sub(field_index).dat.data_ro,
                primal.boundary_states[before_index].sub(field_index).dat.data_ro,
            )
    for field_index in (0, 1):
        np.testing.assert_array_equal(
            primal.boundary_states[6].sub(field_index).dat.data_ro,
            primal.boundary_states[5].sub(field_index).dat.data_ro,
        )
    assert all(
        item["production_form_is_registered_generalrk_form"]
        for item in diagnostics["dry_form_identities"]
    )
    assert all(
        item["production_form_is_registered_generalrk_form"]
        for item in diagnostics["dg_form_identities"]
    )
    assert diagnostics["moist_form_identity"][
        "form_is_registered_generalrk_form"
    ]
    assert diagnostics["moist_form_identity"]["derivative_variable_is_live"]
    assert not diagnostics["limiter_post_step_invoked"]
    _emit(record_property, "production_graph", diagnostics)


def test_dg_cached_stages_tangent_reverse_and_incremental(
    production_case, record_property
):
    case = production_case
    dg = case["helper"].dg_helper
    child = case["dg"]
    child_dt = case["dt"] / 2.0
    _set_c0(case, PHYSICAL_C0)
    primal = child.take_dg_forward_step_cached(
        case["state"], case["t0"], child_dt
    )
    independent_out, independent_sub, _ = _new_full_state(
        case, "dg_independent_out"
    )
    child.reset_internal_vars()
    child.take_forward_step(
        independent_out,
        independent_sub,
        [case["state"]],
        case["t0"],
        child_dt,
    )
    independent_tendencies = tuple(
        _copy_state(child.Fi[i][0][0], f"dg_independent_F{i}")
        for i in range(4)
    )
    independent_stages = tuple(
        _state_axpy(
            case["state"],
            (
                (
                    child_dt * float(child.A[i, j]),
                    independent_tendencies[j],
                )
                for j in range(i)
                if child.A[i, j] != 0.0
            ),
            f"dg_independent_stage_{i}",
        )
        for i in range(4)
    )
    assert np.array_equal(_values(primal.state_out), _values(independent_out[0]))
    assert all(
        np.array_equal(_values(left), _values(right))
        for left, right in zip(primal.stage_tendencies, independent_tendencies)
    )
    assert all(
        np.array_equal(_values(left), _values(right))
        for left, right in zip(primal.stage_states, independent_stages)
    )
    for field_index in (0, 1, 2):
        np.testing.assert_array_equal(
            primal.state_out.sub(field_index).dat.data_ro,
            primal.state_in.sub(field_index).dat.data_ro,
        )

    tangent = child.take_dg_tangent_step(primal, case["direction"])
    lambda_plus = dg.state_mass_map(case["probe"], "dg_lambda_plus")
    reverse = child.take_dg_adjoint_step_cached(primal, lambda_plus)
    pair_out = dg.dual_pairing(lambda_plus, tangent.state_direction_out)
    pair_in = dg.dual_pairing(reverse.state_adjoint_in, case["direction"])
    assert _scalar_error(pair_in, pair_out) < 2.0e-12
    stage_pairings = dg.stage_pairing_diagnostics(tangent, reverse)
    assert max(item.relative_error for item in stage_pairings) < 2.0e-12
    for field_index in (0, 1, 2):
        np.testing.assert_array_equal(
            tangent.state_direction_out.sub(field_index).dat.data_ro,
            tangent.state_direction_in.sub(field_index).dat.data_ro,
        )

    tangent_errors = []
    incremental_errors = []
    mu_plus = dg.state_mass_map(case["probe"], "dg_mu_plus")
    exact_incremental = child.take_dg_incremental_adjoint_step(
        tangent, lambda_plus, mu_plus
    )
    for index, epsilon in enumerate(EPSILONS):
        plus_state = _perturbed_state(
            case, case["state"], case["direction"], epsilon, f"dg_plus_{index}"
        )
        minus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            -epsilon,
            f"dg_minus_{index}",
        )
        plus = child.take_dg_forward_step_cached(
            plus_state, case["t0"], child_dt
        )
        minus = child.take_dg_forward_step_cached(
            minus_state, case["t0"], child_dt
        )
        centered = (_values(plus.state_out) - _values(minus.state_out)) / (
            2.0 * epsilon
        )
        tangent_errors.append(
            _relative_error(_values(tangent.state_direction_out), centered)
        )
        plus_lambda = Cofunction(lambda_plus.function_space())
        plus_lambda.assign(lambda_plus)
        with plus_lambda.dat.vec as target, mu_plus.dat.vec_ro as increment:
            target.axpy(epsilon, increment)
        minus_lambda = Cofunction(lambda_plus.function_space())
        minus_lambda.assign(lambda_plus)
        with minus_lambda.dat.vec as target, mu_plus.dat.vec_ro as increment:
            target.axpy(-epsilon, increment)
        plus_reverse = child.take_dg_adjoint_step_cached(plus, plus_lambda)
        minus_reverse = child.take_dg_adjoint_step_cached(minus, minus_lambda)
        centered_reverse = (
            _values(plus_reverse.state_adjoint_in)
            - _values(minus_reverse.state_adjoint_in)
        ) / (2.0 * epsilon)
        incremental_errors.append(
            _relative_error(
                _values(exact_incremental.incremental_state_adjoint_in),
                centered_reverse,
            )
        )
    tangent_regime = _floor_aware_centered_classifier(tangent_errors)
    incremental_regime = _floor_aware_centered_classifier(incremental_errors)
    identity = dg.stage_form_identity_diagnostics()
    assert all(
        item["production_form_is_registered_generalrk_form"]
        and item["production_derivative_variable_is_live_coefficient"]
        for item in identity
    )
    assert reverse.reverse_stage_order == (3, 2, 1, 0)
    assert exact_incremental.reverse_stage_order == (3, 2, 1, 0)
    _emit(
        record_property,
        "dg_child",
        {
            "tangent_errors": tangent_errors,
            "tangent_regime": tangent_regime,
            "incremental_errors": incremental_errors,
            "incremental_regime": incremental_regime,
            "stage_pairings": [asdict(item) for item in stage_pairings],
            "form_identities": identity,
        },
    )


def test_dg_stage_local_exact_production_forms(production_case, record_property):
    case = production_case
    child = case["dg"]
    dg = case["helper"].dg_helper
    child_dt = case["dt"] / 2.0
    primal = child.take_dg_forward_step_cached(
        case["state"], case["t0"], child_dt
    )
    tangent = child.take_dg_tangent_step(primal, case["direction"])
    stage_payload = []
    for stage in range(4):
        exact = dg.production_stage_tangent(
            primal,
            stage,
            primal.stage_states[stage],
            tangent.stage_state_directions[stage],
        )
        reconstructed = dg.reconstructed_stage_tangent(
            primal,
            stage,
            primal.stage_states[stage],
            tangent.stage_state_directions[stage],
        )
        errors = []
        for epsilon in EPSILONS:
            plus = dg.perturbed_production_stage_tendency(
                primal,
                stage,
                tangent.stage_state_directions[stage],
                epsilon,
            )
            minus = dg.perturbed_production_stage_tendency(
                primal,
                stage,
                tangent.stage_state_directions[stage],
                -epsilon,
            )
            centered = (_values(plus) - _values(minus)) / (2.0 * epsilon)
            errors.append(_relative_error(_values(exact), centered))
        regime = _floor_aware_centered_classifier(errors, floor=1.0e-10)
        stage_payload.append(
            {
                "stage": stage,
                "exact_errors": errors,
                "exact_regime": regime,
                "exact_vs_reconstructed": _relative_error(
                    _values(exact), _values(reconstructed)
                ),
            }
        )
    assert len(stage_payload) == 4
    _emit(record_property, "dg_stage_local", stage_payload)


def test_moist_exact_form_active_set_invariants_and_incremental(
    production_case, record_property
):
    case = production_case
    moist = case["moist"]
    helper = case["helper"].moist_helper
    primal = moist.take_moist_forward_step_cached(
        case["state"], case["t0"], case["dt"]
    )
    tangent = moist.take_moist_tangent_step(primal, case["direction"])
    identity = helper.form_identity_diagnostics()
    assert identity["form_is_registered_generalrk_form"]
    assert identity["derivative_variable_is_live"]
    assert not identity["direct_c0_dependency"]
    for field_index in (0, 1):
        np.testing.assert_array_equal(
            primal.state_out.sub(field_index).dat.data_ro,
            primal.state_in.sub(field_index).dat.data_ro,
        )
        np.testing.assert_array_equal(
            tangent.state_direction_out.sub(field_index).dat.data_ro,
            tangent.state_direction_in.sub(field_index).dat.data_ro,
        )

    dx = case["model"].spaces.dx
    beta2 = helper.term.g * helper.term.L
    total_before = float(
        assemble(
            (primal.state_in.sub(3) + primal.state_in.sub(4) + primal.state_in.sub(5))
            * dx
        )
    )
    total_after = float(
        assemble(
            (primal.state_out.sub(3) + primal.state_out.sub(4) + primal.state_out.sub(5))
            * dx
        )
    )
    thermal_before = float(
        assemble((primal.state_in.sub(2) - beta2 * primal.state_in.sub(3)) * dx)
    )
    thermal_after = float(
        assemble((primal.state_out.sub(2) - beta2 * primal.state_out.sub(3)) * dx)
    )
    tangent_total_before = float(
        assemble(
            (
                tangent.state_direction_in.sub(3)
                + tangent.state_direction_in.sub(4)
                + tangent.state_direction_in.sub(5)
            )
            * dx
        )
    )
    tangent_total_after = float(
        assemble(
            (
                tangent.state_direction_out.sub(3)
                + tangent.state_direction_out.sub(4)
                + tangent.state_direction_out.sub(5)
            )
            * dx
        )
    )
    tangent_thermal_before = float(
        assemble(
            (
                tangent.state_direction_in.sub(2)
                - beta2 * tangent.state_direction_in.sub(3)
            )
            * dx
        )
    )
    tangent_thermal_after = float(
        assemble(
            (
                tangent.state_direction_out.sub(2)
                - beta2 * tangent.state_direction_out.sub(3)
            )
            * dx
        )
    )
    assert _scalar_error(total_after, total_before) < 2.0e-13
    assert _scalar_error(thermal_after, thermal_before) < 2.0e-13
    assert _scalar_error(tangent_total_after, tangent_total_before) < 2.0e-12
    assert _scalar_error(tangent_thermal_after, tangent_thermal_before) < 2.0e-12

    water_representative = _zero_state(case, "moist_water_representative")
    water_representative.sub(3).assign(1.0)
    water_representative.sub(4).assign(1.0)
    water_representative.sub(5).assign(1.0)
    water_dual = helper.state_mass_map(
        water_representative, "moist_water_invariant_dual"
    )
    water_reverse = moist.take_moist_adjoint_step_cached(primal, water_dual)
    assert _relative_error(
        _values(water_reverse.state_adjoint_in), _values(water_dual)
    ) < 2.0e-12
    thermal_representative = _zero_state(
        case, "moist_thermal_representative"
    )
    thermal_representative.sub(2).assign(1.0)
    thermal_representative.sub(3).assign(-beta2)
    thermal_dual = helper.state_mass_map(
        thermal_representative, "moist_thermal_invariant_dual"
    )
    thermal_reverse = moist.take_moist_adjoint_step_cached(primal, thermal_dual)
    assert _relative_error(
        _values(thermal_reverse.state_adjoint_in), _values(thermal_dual)
    ) < 2.0e-12
    zero_dual = helper.state_mass_map(
        _zero_state(case, "moist_zero_incremental_representative"),
        "moist_zero_incremental_dual",
    )
    water_incremental = moist.take_moist_incremental_adjoint_step(
        tangent, water_dual, zero_dual
    )
    assert (
        np.linalg.norm(_values(water_incremental.incremental_state_adjoint_in))
        / max(np.linalg.norm(_values(water_dual)), np.finfo(float).tiny)
        < 2.0e-12
    )

    lambda_plus = helper.state_mass_map(case["probe"], "moist_lambda_plus")
    reverse = moist.take_moist_adjoint_step_cached(primal, lambda_plus)
    pair_out = helper.dual_pairing(lambda_plus, tangent.state_direction_out)
    pair_in = helper.dual_pairing(reverse.state_adjoint_in, case["direction"])
    assert _scalar_error(pair_out, pair_in) < 2.0e-12
    mu_plus = helper.state_mass_map(case["probe"], "moist_mu_plus")
    incremental = moist.take_moist_incremental_adjoint_step(
        tangent, lambda_plus, mu_plus
    )
    assert isinstance(incremental, MoistEulerHVPResult)

    tangent_errors = []
    incremental_errors = []
    active_payload = []
    for index, epsilon in enumerate(EPSILONS):
        plus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            epsilon,
            f"moist_plus_{index}",
        )
        minus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            -epsilon,
            f"moist_minus_{index}",
        )
        plus = moist.take_moist_forward_step_cached(
            plus_state, case["t0"], case["dt"]
        )
        minus = moist.take_moist_forward_step_cached(
            minus_state, case["t0"], case["dt"]
        )
        assert plus.active_set.signature == primal.active_set.signature
        assert minus.active_set.signature == primal.active_set.signature
        centered = (_values(plus.state_out) - _values(minus.state_out)) / (
            2.0 * epsilon
        )
        tangent_errors.append(
            _relative_error(_values(tangent.state_direction_out), centered)
        )
        plus_lambda = Cofunction(lambda_plus.function_space())
        plus_lambda.assign(lambda_plus)
        with plus_lambda.dat.vec as target, mu_plus.dat.vec_ro as increment:
            target.axpy(epsilon, increment)
        minus_lambda = Cofunction(lambda_plus.function_space())
        minus_lambda.assign(lambda_plus)
        with minus_lambda.dat.vec as target, mu_plus.dat.vec_ro as increment:
            target.axpy(-epsilon, increment)
        plus_reverse = moist.take_moist_adjoint_step_cached(plus, plus_lambda)
        minus_reverse = moist.take_moist_adjoint_step_cached(minus, minus_lambda)
        centered_reverse = (
            _values(plus_reverse.state_adjoint_in)
            - _values(minus_reverse.state_adjoint_in)
        ) / (2.0 * epsilon)
        incremental_errors.append(
            _relative_error(
                _values(incremental.incremental_state_adjoint_in),
                centered_reverse,
            )
        )
        active_payload.append(
            {
                "epsilon": epsilon,
                "base": asdict(primal.active_set),
                "plus": asdict(plus.active_set),
                "minus": asdict(minus.active_set),
            }
        )
    tangent_regime = _floor_aware_centered_classifier(tangent_errors)
    incremental_regime = _floor_aware_centered_classifier(incremental_errors)
    assert reverse.reverse_stage_order == (0,)
    assert incremental.reverse_stage_order == (0,)
    _emit(
        record_property,
        "moist_child",
        {
            "identity": identity,
            "active_sets": active_payload,
            "tangent_errors": tangent_errors,
            "tangent_regime": tangent_regime,
            "incremental_errors": incremental_errors,
            "incremental_regime": incremental_regime,
            "invariants": {
                "total_before": total_before,
                "total_after": total_after,
                "thermal_before": thermal_before,
                "thermal_after": thermal_after,
                "tangent_total_before": tangent_total_before,
                "tangent_total_after": tangent_total_after,
                "tangent_thermal_before": tangent_thermal_before,
                "tangent_thermal_after": tangent_thermal_after,
            },
        },
    )


@pytest.mark.parametrize("kind", ("c0", "ic", "combined"))
def test_complete_tangent_deployed_forward_and_pairing(
    production_case, kind, record_property
):
    case = production_case
    direction, delta_c0 = _direction(case, kind)
    _set_c0(case, PHYSICAL_C0)
    primal = case["split"].take_mtswe_forward_step_cached(
        case["state"], case["t0"], case["dt"]
    )
    tangent = case["split"].take_mtswe_tangent_step(
        primal, direction, delta_c0
    )
    lambda_plus = case["helper"].state_mass_map(
        case["probe"], f"mtswe_{kind}_lambda_plus"
    )
    reverse = case["split"].take_mtswe_adjoint_step_cached(
        primal, lambda_plus
    )
    assert isinstance(tangent, MTSWESplitTangentCache)
    assert isinstance(reverse, MTSWESplitReverseResult)
    assert tuple(item.name for item in reverse.children) == tuple(
        reversed(primal.forward_child_order)
    )
    for child_reverse in reverse.children:
        if child_reverse.name.startswith(("dry_rk4", "dg_ssprk43")):
            assert child_reverse.result.reverse_stage_order == (3, 2, 1, 0)
        elif child_reverse.name == "moist_euler":
            assert child_reverse.result.reverse_stage_order == (0,)
    tangent_by_name = {
        item.primal.name: item.cache for item in tangent.children
    }
    reverse_by_name = {item.name: item.result for item in reverse.children}
    local_pairing_errors = {}
    for child_name in (
        "dry_rk4_0",
        "dry_rk4_1",
        "dg_ssprk43_0",
        "dg_ssprk43_1",
    ):
        helper = (
            case["helper"].dry_helper
            if child_name.startswith("dry_rk4")
            else case["helper"].dg_helper
        )
        pairings = helper.stage_pairing_diagnostics(
            tangent_by_name[child_name], reverse_by_name[child_name]
        )
        local_pairing_errors[child_name] = tuple(
            item.relative_error for item in pairings
        )
        assert max(local_pairing_errors[child_name]) < 5.0e-12
    pair_out = case["helper"].dual_pairing(
        lambda_plus, tangent.state_direction_out
    )
    pair_in = case["helper"].dual_pairing(
        reverse.state_adjoint_in, direction
    ) + reverse.physical_c0_gradient * delta_c0
    assert _scalar_error(pair_out, pair_in) < 5.0e-12

    errors = []
    active_records = []
    for index, epsilon in enumerate(EPSILONS):
        plus_state = _perturbed_state(
            case, case["state"], direction, epsilon, f"full_{kind}_plus_{index}"
        )
        minus_state = _perturbed_state(
            case,
            case["state"],
            direction,
            -epsilon,
            f"full_{kind}_minus_{index}",
        )
        plus_c0 = PHYSICAL_C0 + epsilon * delta_c0
        minus_c0 = PHYSICAL_C0 - epsilon * delta_c0
        plus = _legacy_complete_step(
            case,
            plus_state,
            plus_c0,
            case["t0"],
            f"full_{kind}_legacy_plus_{index}",
        )
        minus = _legacy_complete_step(
            case,
            minus_state,
            minus_c0,
            case["t0"],
            f"full_{kind}_legacy_minus_{index}",
        )
        centered = (_values(plus) - _values(minus)) / (2.0 * epsilon)
        errors.append(
            _relative_error(_values(tangent.state_direction_out), centered)
        )
        _set_c0(case, plus_c0)
        plus_cache = case["split"].take_mtswe_forward_step_cached(
            plus_state, case["t0"], case["dt"]
        )
        _set_c0(case, minus_c0)
        minus_cache = case["split"].take_mtswe_forward_step_cached(
            minus_state, case["t0"], case["dt"]
        )
        _assert_active_sets_equal((primal,), (plus_cache,), (minus_cache,))
        active_records.append(
            {
                "epsilon": epsilon,
                "plus": asdict(_moist_cache(plus_cache).active_set),
                "minus": asdict(_moist_cache(minus_cache).active_set),
            }
        )
    _set_c0(case, PHYSICAL_C0)
    regime = _floor_aware_centered_classifier(errors)
    _emit(
        record_property,
        f"complete_tangent_{kind}",
        {
            "errors": errors,
            "regime": regime,
            "pair_out": pair_out,
            "pair_in": pair_in,
            "local_stage_pairing_errors": local_pairing_errors,
            "active_sets": active_records,
        },
    )


@pytest.mark.parametrize("nsteps", (1, 3))
def test_reduced_gradient_against_legacy_and_scalar_objective(
    production_case, nsteps, record_property
):
    case = production_case
    _set_c0(case, PHYSICAL_C0)
    new = case["split"].mtswe_terminal_least_squares_gradient(
        nsteps,
        case["state"],
        case["t0"],
        case["dt"],
        case["target"],
    )
    legacy_c0, legacy_ic, _ = _legacy_reduced_gradient(
        case,
        nsteps,
        case["state"],
        PHYSICAL_C0,
        f"mtswe_legacy_gradient_{nsteps}",
    )
    new_ic_riesz = case["helper"].state_riesz_representative(
        new.initial_condition_gradient,
        f"mtswe_new_ic_riesz_{nsteps}",
    )
    assert _relative_error(_values(new_ic_riesz), _values(legacy_ic)) < 2.0e-10
    assert _scalar_error(new.physical_c0_gradient, legacy_c0) < 2.0e-10

    repeated = case["split"].mtswe_terminal_least_squares_gradient(
        nsteps,
        case["state"],
        case["t0"],
        case["dt"],
        case["target"],
    )
    objective_reproducibility = abs(
        new.objective_value - repeated.objective_value
    )
    c0_reproducibility = abs(
        new.physical_c0_gradient - repeated.physical_c0_gradient
    )
    ic_reproducibility = _dual_reproducibility(
        case,
        new.initial_condition_gradient,
        repeated.initial_condition_gradient,
        f"reduced_gradient_reproducibility_{nsteps}",
    )

    state_directional = case["helper"].dual_pairing(
        new.initial_condition_gradient, case["direction"]
    )
    combined_directional = state_directional + DELTA_C0 * new.physical_c0_gradient
    state_records = []
    combined_original_records = []
    for index, epsilon in enumerate(EPSILONS):
        plus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            epsilon,
            f"gradient_plus_{nsteps}_{index}",
        )
        minus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            -epsilon,
            f"gradient_minus_{nsteps}_{index}",
        )
        plus_state_only = _legacy_trajectory(
            case,
            nsteps,
            plus_state,
            PHYSICAL_C0,
            f"objective_ic_plus_{nsteps}_{index}",
        )[-1]
        minus_state_only = _legacy_trajectory(
            case,
            nsteps,
            minus_state,
            PHYSICAL_C0,
            f"objective_ic_minus_{nsteps}_{index}",
        )[-1]
        plus_state_objective = _objective(case, plus_state_only)
        minus_state_objective = _objective(case, minus_state_only)
        state_records.append(
            _scalar_fd_diagnostic(
                epsilon,
                state_directional,
                plus_state_objective,
                minus_state_objective,
                objective_reproducibility,
            )
        )
        plus_combined = _legacy_trajectory(
            case,
            nsteps,
            plus_state,
            PHYSICAL_C0 + epsilon * DELTA_C0,
            f"objective_combined_plus_{nsteps}_{index}",
        )[-1]
        minus_combined = _legacy_trajectory(
            case,
            nsteps,
            minus_state,
            PHYSICAL_C0 - epsilon * DELTA_C0,
            f"objective_combined_minus_{nsteps}_{index}",
        )[-1]
        combined_original_records.append(
            _scalar_fd_diagnostic(
                epsilon,
                combined_directional,
                _objective(case, plus_combined),
                _objective(case, minus_combined),
                objective_reproducibility,
            )
        )

    combined_diagnostic_records = []
    for index, epsilon in enumerate(GRADIENT_DIAGNOSTIC_EPSILONS):
        plus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            epsilon,
            f"gradient_diagnostic_plus_{nsteps}_{index}",
        )
        minus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            -epsilon,
            f"gradient_diagnostic_minus_{nsteps}_{index}",
        )
        _set_c0(case, PHYSICAL_C0 + epsilon * DELTA_C0)
        plus = case["split"].mtswe_terminal_least_squares_gradient(
            nsteps,
            plus_state,
            case["t0"],
            case["dt"],
            case["target"],
        )
        _set_c0(case, PHYSICAL_C0 - epsilon * DELTA_C0)
        minus = case["split"].mtswe_terminal_least_squares_gradient(
            nsteps,
            minus_state,
            case["t0"],
            case["dt"],
            case["target"],
        )
        record = _scalar_fd_diagnostic(
            epsilon,
            combined_directional,
            plus.objective_value,
            minus.objective_value,
            objective_reproducibility,
        )
        record["active_set"] = _trajectory_active_set_record(
            case,
            new.primal_caches,
            plus.primal_caches,
            minus.primal_caches,
            epsilon,
            f"gradient_active_{nsteps}_{index}",
        )
        record["in_original_ladder"] = epsilon in EPSILONS
        combined_diagnostic_records.append(record)

    _set_c0(case, PHYSICAL_C0)
    state_errors = [record["relative_error"] for record in state_records]
    state_regime = _floor_aware_centered_classifier(state_errors)
    largest_safe, safe_combined_records = _safe_records_strictly_inside(
        combined_diagnostic_records
    )
    _emit(
        record_property,
        f"reduced_gradient_diagnostic_{nsteps}",
        {
            "new_c0": new.physical_c0_gradient,
            "legacy_c0": legacy_c0,
            "ic_riesz_error": _relative_error(
                _values(new_ic_riesz), _values(legacy_ic)
            ),
            "new_vs_legacy_c0_relative_error": _scalar_error(
                new.physical_c0_gradient, legacy_c0
            ),
            "reproducibility": {
                "base_objective_magnitude": abs(new.objective_value),
                "repeated_objective_magnitude": abs(
                    repeated.objective_value
                ),
                "objective_absolute_error": objective_reproducibility,
                "base_c0_gradient_magnitude": abs(
                    new.physical_c0_gradient
                ),
                "repeated_c0_gradient_magnitude": abs(
                    repeated.physical_c0_gradient
                ),
                "c0_gradient_absolute_error": c0_reproducibility,
                "initial_condition_gradient": ic_reproducibility,
            },
            "original_epsilons": EPSILONS,
            "diagnostic_epsilons": GRADIENT_DIAGNOSTIC_EPSILONS,
            "state_objective_records": state_records,
            "state_objective_ratios": _ratios(
                [record["relative_error"] for record in state_records]
            ),
            "state_regime": state_regime,
            "combined_original_records": combined_original_records,
            "combined_original_ratios": _ratios(
                [
                    record["relative_error"]
                    for record in combined_original_records
                ]
            ),
            "combined_diagnostic_records": combined_diagnostic_records,
            "combined_diagnostic_ratios": _ratios(
                [
                    record["relative_error"]
                    for record in combined_diagnostic_records
                ]
            ),
            "largest_symmetric_active_set_safe_epsilon": largest_safe,
            "strictly_inside_safe_epsilons": tuple(
                record["epsilon"] for record in safe_combined_records
            ),
        },
    )
    assert largest_safe is not None
    assert len(safe_combined_records) >= 4
    combined_regime = _scale_aware_classifier(
        safe_combined_records,
        strict_floor=OBJECTIVE_DIRECTIONAL_STRICT_FLOOR,
        active_set_safe=all(
            record["active_set"]["safe"]
            for record in safe_combined_records
        ),
        active_set_truncated=(
            largest_safe < max(GRADIENT_DIAGNOSTIC_EPSILONS)
        ),
        strictly_interior_safe=True,
    )
    _emit(
        record_property,
        f"reduced_gradient_certification_{nsteps}",
        {
            "combined_regime": combined_regime,
            "strict_objective_directional_floor": (
                OBJECTIVE_DIRECTIONAL_STRICT_FLOOR
            ),
        },
    )


def test_complete_incremental_reverse_centered_new_reverse(
    production_case, record_property
):
    case = production_case
    _set_c0(case, PHYSICAL_C0)
    primal = case["split"].take_mtswe_forward_step_cached(
        case["state"], case["t0"], case["dt"]
    )
    tangent = case["split"].take_mtswe_tangent_step(
        primal, case["direction"], DELTA_C0
    )
    lambda_plus = case["helper"].state_mass_map(
        case["probe"], "complete_incremental_lambda_plus"
    )
    mu_plus = case["helper"].state_mass_map(
        case["direction"], "complete_incremental_mu_plus"
    )
    exact = case["split"].take_mtswe_incremental_adjoint_step(
        tangent, lambda_plus, mu_plus
    )
    assert isinstance(exact, MTSWESplitHVPResult)
    state_errors = []
    scalar_errors = []
    for index, epsilon in enumerate(EPSILONS):
        plus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            epsilon,
            f"incremental_complete_plus_{index}",
        )
        minus_state = _perturbed_state(
            case,
            case["state"],
            case["direction"],
            -epsilon,
            f"incremental_complete_minus_{index}",
        )
        _set_c0(case, PHYSICAL_C0 + epsilon * DELTA_C0)
        plus_primal = case["split"].take_mtswe_forward_step_cached(
            plus_state, case["t0"], case["dt"]
        )
        _set_c0(case, PHYSICAL_C0 - epsilon * DELTA_C0)
        minus_primal = case["split"].take_mtswe_forward_step_cached(
            minus_state, case["t0"], case["dt"]
        )
        _assert_active_sets_equal((primal,), (plus_primal,), (minus_primal,))
        plus_lambda = Cofunction(lambda_plus.function_space())
        plus_lambda.assign(lambda_plus)
        with plus_lambda.dat.vec as target, mu_plus.dat.vec_ro as increment:
            target.axpy(epsilon, increment)
        minus_lambda = Cofunction(lambda_plus.function_space())
        minus_lambda.assign(lambda_plus)
        with minus_lambda.dat.vec as target, mu_plus.dat.vec_ro as increment:
            target.axpy(-epsilon, increment)
        plus_reverse = case["split"].take_mtswe_adjoint_step_cached(
            plus_primal, plus_lambda
        )
        minus_reverse = case["split"].take_mtswe_adjoint_step_cached(
            minus_primal, minus_lambda
        )
        centered_state = (
            _values(plus_reverse.state_adjoint_in)
            - _values(minus_reverse.state_adjoint_in)
        ) / (2.0 * epsilon)
        centered_scalar = (
            plus_reverse.physical_c0_gradient
            - minus_reverse.physical_c0_gradient
        ) / (2.0 * epsilon)
        state_errors.append(
            _relative_error(_values(exact.incremental_state_adjoint_in), centered_state)
        )
        scalar_errors.append(_scalar_error(exact.physical_c0_hvp, centered_scalar))
    _set_c0(case, PHYSICAL_C0)
    state_regime = _floor_aware_centered_classifier(state_errors)
    scalar_regime = _floor_aware_centered_classifier(scalar_errors)
    assert exact.reverse_child_order == tuple(reversed(primal.forward_child_order))
    _emit(
        record_property,
        "complete_incremental_reverse",
        {
            "state_errors": state_errors,
            "state_regime": state_regime,
            "scalar_errors": scalar_errors,
            "scalar_regime": scalar_regime,
        },
    )


@pytest.mark.parametrize("nsteps", (1, 3))
@pytest.mark.parametrize("kind", ("c0", "ic", "combined"))
def test_reduced_hvp_centered_gradients(
    production_case, nsteps, kind, record_property
):
    case = production_case
    direction, delta_c0 = _direction(case, kind)
    _set_c0(case, PHYSICAL_C0)
    input_snapshots = {
        "state": _values(case["state"]),
        "direction": _values(case["direction"]),
        "target": _values(case["target"]),
        "physical_c0": _values(case["coefficient_sub"]["c0"]),
    }
    exact = case["split"].mtswe_terminal_least_squares_hvp(
        nsteps,
        case["state"],
        case["t0"],
        case["dt"],
        case["target"],
        direction,
        delta_c0,
    )
    assert isinstance(exact, MTSWEReducedHVPResult)
    assert isinstance(exact.initial_condition_gradient, Cofunction)
    assert isinstance(exact.initial_condition_hvp, Cofunction)

    repeated = case["split"].mtswe_terminal_least_squares_hvp(
        nsteps,
        case["state"],
        case["t0"],
        case["dt"],
        case["target"],
        direction,
        delta_c0,
    )
    gradient_reproducibility = _dual_reproducibility(
        case,
        exact.initial_condition_gradient,
        repeated.initial_condition_gradient,
        f"reduced_hvp_gradient_reproducibility_{kind}_{nsteps}",
    )
    hvp_reproducibility = _dual_reproducibility(
        case,
        exact.initial_condition_hvp,
        repeated.initial_condition_hvp,
        f"reduced_hvp_block_reproducibility_{kind}_{nsteps}",
    )
    scalar_gradient_reproducibility = abs(
        exact.physical_c0_gradient - repeated.physical_c0_gradient
    )
    scalar_hvp_reproducibility = abs(
        exact.physical_c0_hvp - repeated.physical_c0_hvp
    )
    probes = (
        ("probe", case["probe"]),
        ("direction", case["direction"]),
        ("state", case["state"]),
    )
    probe_base_reproducibility = {
        name: abs(
            case["helper"].dual_pairing(
                exact.initial_condition_gradient, probe
            )
            - case["helper"].dual_pairing(
                repeated.initial_condition_gradient, probe
            )
        )
        for name, probe in probes
    }

    diagnostic_records = []
    dual_diagnostic_results = []
    base_caches = exact.primal_caches
    for index, epsilon in enumerate(REDUCED_DIAGNOSTIC_EPSILONS):
        plus_state = _perturbed_state(
            case,
            case["state"],
            direction,
            epsilon,
            f"reduced_{kind}_plus_{nsteps}_{index}",
        )
        minus_state = _perturbed_state(
            case,
            case["state"],
            direction,
            -epsilon,
            f"reduced_{kind}_minus_{nsteps}_{index}",
        )
        _set_c0(case, PHYSICAL_C0 + epsilon * delta_c0)
        plus = case["split"].mtswe_terminal_least_squares_gradient(
            nsteps,
            plus_state,
            case["t0"],
            case["dt"],
            case["target"],
        )
        _set_c0(case, PHYSICAL_C0 - epsilon * delta_c0)
        minus = case["split"].mtswe_terminal_least_squares_gradient(
            nsteps,
            minus_state,
            case["t0"],
            case["dt"],
            case["target"],
        )
        active_set = _trajectory_active_set_record(
            case,
            base_caches,
            plus.primal_caches,
            minus.primal_caches,
            epsilon,
            f"reduced_hvp_active_{kind}_{nsteps}_{index}",
        )
        state_diagnostic = _dual_fd_diagnostic(
            case,
            epsilon,
            exact.initial_condition_hvp,
            plus.initial_condition_gradient,
            minus.initial_condition_gradient,
            gradient_reproducibility,
            f"reduced_hvp_state_{kind}_{nsteps}_{index}",
        )
        dual_diagnostic_results.append(state_diagnostic)
        assert isinstance(state_diagnostic.centered_dual, Cofunction)
        assert isinstance(state_diagnostic.numerator_dual, Cofunction)
        assert isinstance(state_diagnostic.error_dual, Cofunction)
        state_record = state_diagnostic.record
        scalar_record = _scalar_fd_diagnostic(
            epsilon,
            exact.physical_c0_hvp,
            plus.physical_c0_gradient,
            minus.physical_c0_gradient,
            scalar_gradient_reproducibility,
        )
        probe_records = {}
        for probe_name, probe in probes:
            exact_pairing = case["helper"].dual_pairing(
                exact.initial_condition_hvp, probe
            )
            plus_pairing = case["helper"].dual_pairing(
                plus.initial_condition_gradient, probe
            )
            minus_pairing = case["helper"].dual_pairing(
                minus.initial_condition_gradient, probe
            )
            probe_records[probe_name] = _scalar_fd_diagnostic(
                epsilon,
                exact_pairing,
                plus_pairing,
                minus_pairing,
                probe_base_reproducibility[probe_name],
            )
        diagnostic_records.append(
            {
                "epsilon": float(epsilon),
                "in_original_ladder": epsilon in REDUCED_EPSILONS,
                "active_set": active_set,
                "state_block": state_record,
                "scalar_block": scalar_record,
                "probe_pairings": probe_records,
            }
        )

    _set_c0(case, PHYSICAL_C0)
    input_nonmutation_evidence = {
        "state": np.array_equal(
            _values(case["state"]), input_snapshots["state"]
        ),
        "direction": np.array_equal(
            _values(case["direction"]), input_snapshots["direction"]
        ),
        "target": np.array_equal(
            _values(case["target"]), input_snapshots["target"]
        ),
        "physical_c0_restored": np.array_equal(
            _values(case["coefficient_sub"]["c0"]),
            input_snapshots["physical_c0"],
        ),
    }
    assert all(input_nonmutation_evidence.values())
    repeatability_evidence = {
        "gradient_coefficient_vector": (
            gradient_reproducibility["coefficient_vector_absolute_error"]
            == 0.0
        ),
        "gradient_natural": (
            gradient_reproducibility["natural_absolute_error"] == 0.0
        ),
        "hvp_coefficient_vector": (
            hvp_reproducibility["coefficient_vector_absolute_error"] == 0.0
        ),
        "hvp_natural": (
            hvp_reproducibility["natural_absolute_error"] == 0.0
        ),
        "scalar_gradient": scalar_gradient_reproducibility == 0.0,
        "scalar_hvp": scalar_hvp_reproducibility == 0.0,
        "probe_gradient_pairings": all(
            error == 0.0 for error in probe_base_reproducibility.values()
        ),
    }
    assert all(repeatability_evidence.values())
    assert len(dual_diagnostic_results) == len(REDUCED_DIAGNOSTIC_EPSILONS)
    largest_safe, safe_records = _safe_records_strictly_inside(
        diagnostic_records
    )
    _emit(
        record_property,
        f"reduced_hvp_diagnostic_{kind}_{nsteps}",
        {
            "original_epsilons": REDUCED_EPSILONS,
            "diagnostic_epsilons": REDUCED_DIAGNOSTIC_EPSILONS,
            "reproducibility": {
                "gradient_state_block": gradient_reproducibility,
                "hvp_state_block": hvp_reproducibility,
                "scalar_gradient_absolute_error": (
                    scalar_gradient_reproducibility
                ),
                "base_scalar_gradient_magnitude": abs(
                    exact.physical_c0_gradient
                ),
                "repeated_scalar_gradient_magnitude": abs(
                    repeated.physical_c0_gradient
                ),
                "scalar_hvp_absolute_error": scalar_hvp_reproducibility,
                "base_scalar_hvp_magnitude": abs(exact.physical_c0_hvp),
                "repeated_scalar_hvp_magnitude": abs(
                    repeated.physical_c0_hvp
                ),
                "probe_gradient_pairings": probe_base_reproducibility,
            },
            "exact_state_hvp_coefficient_vector_norm": float(
                np.linalg.norm(_values(exact.initial_condition_hvp))
            ),
            "exact_state_hvp_natural_norm": _dual_natural_norm(
                case,
                exact.initial_condition_hvp,
                f"reduced_hvp_exact_natural_{kind}_{nsteps}",
            ),
            "exact_scalar_hvp_magnitude": abs(exact.physical_c0_hvp),
            "records": diagnostic_records,
            "diagnostic_ratios": {
                "state_coefficient_vector": _ratios(
                    [
                        record["state_block"]["coefficient_vector"][
                            "relative_error"
                        ]
                        for record in diagnostic_records
                    ]
                ),
                "state_natural_mass_riesz": _ratios(
                    [
                        record["state_block"]["natural_mass_riesz"][
                            "relative_error"
                        ]
                        for record in diagnostic_records
                    ]
                ),
                "scalar": _ratios(
                    [
                        record["scalar_block"]["relative_error"]
                        for record in diagnostic_records
                    ]
                ),
                "probe_pairings": {
                    probe_name: _ratios(
                        [
                            record["probe_pairings"][probe_name][
                                "relative_error"
                            ]
                            for record in diagnostic_records
                        ]
                    )
                    for probe_name, _ in probes
                },
            },
            "original_ladder_records": tuple(
                record
                for record in diagnostic_records
                if record["in_original_ladder"]
            ),
            "largest_symmetric_active_set_safe_epsilon": largest_safe,
            "strictly_inside_safe_epsilons": tuple(
                record["epsilon"] for record in safe_records
            ),
        },
    )
    assert largest_safe is not None
    assert len(safe_records) >= 4
    selected_active_set_safe = all(
        record["active_set"]["safe"] for record in safe_records
    )
    assert selected_active_set_safe
    active_set_truncated = largest_safe < max(REDUCED_DIAGNOSTIC_EPSILONS)

    natural_records = [
        {
            "epsilon": record["epsilon"],
            **record["state_block"]["natural_mass_riesz"],
        }
        for record in safe_records
    ]
    coefficient_records = [
        {
            "epsilon": record["epsilon"],
            **record["state_block"]["coefficient_vector"],
        }
        for record in safe_records
    ]
    scalar_records = [record["scalar_block"] for record in safe_records]
    common_independent_evidence = {
        "selected_active_set_safe": selected_active_set_safe,
        "exact_gradient_repeatability": all(
            repeatability_evidence[key]
            for key in (
                "gradient_coefficient_vector",
                "gradient_natural",
                "scalar_gradient",
                "probe_gradient_pairings",
            )
        ),
        "exact_hvp_repeatability": all(
            repeatability_evidence[key]
            for key in (
                "hvp_coefficient_vector",
                "hvp_natural",
                "scalar_hvp",
            )
        ),
        "input_nonmutation": all(input_nonmutation_evidence.values()),
    }
    scalar_independent_checks = all(common_independent_evidence.values())
    scalar_regime = _scale_aware_classifier(
        scalar_records,
        strict_floor=SCALAR_GRADIENT_STRICT_FLOOR,
        allow_immediate_floor=True,
        independent_checks=scalar_independent_checks,
        active_set_safe=selected_active_set_safe,
        active_set_truncated=active_set_truncated,
        strictly_interior_safe=True,
    )
    probe_regimes = {}
    for probe_name, _ in probes:
        probe_series = [
            record["probe_pairings"][probe_name] for record in safe_records
        ]
        probe_regimes[probe_name] = _scale_aware_classifier(
            probe_series,
            strict_floor=SCALAR_GRADIENT_STRICT_FLOOR,
            allow_immediate_floor=True,
            independent_checks=(
                scalar_independent_checks and scalar_regime["certified"]
            ),
            active_set_safe=selected_active_set_safe,
            active_set_truncated=active_set_truncated,
            strictly_interior_safe=True,
        )
    absolute_natural_floor = _absolute_floor_assessment(natural_records)
    field_independent_evidence = {
        **common_independent_evidence,
        "probe_pairing_regimes_certified": all(
            regime["certified"] for regime in probe_regimes.values()
        ),
        "scalar_hvp_regime_certified": scalar_regime["certified"],
        "absolute_natural_floor_certified": absolute_natural_floor[
            "certified"
        ],
    }
    field_independent_checks = all(field_independent_evidence.values())
    state_natural_regime = _scale_aware_classifier(
        natural_records,
        strict_floor=FIELD_HVP_STRICT_FLOOR,
        allow_immediate_floor=True,
        independent_checks=field_independent_checks,
        active_set_safe=selected_active_set_safe,
        active_set_truncated=active_set_truncated,
        strictly_interior_safe=True,
    )
    state_coefficient_regime = _scale_aware_classifier(
        coefficient_records,
        strict_floor=FIELD_HVP_STRICT_FLOOR,
        allow_immediate_floor=True,
        independent_checks=field_independent_checks,
        active_set_safe=selected_active_set_safe,
        active_set_truncated=active_set_truncated,
        strictly_interior_safe=True,
        secondary_metric=True,
        primary_metric_certified=(
            state_natural_regime["certified"]
            and absolute_natural_floor["certified"]
        ),
    )
    _emit(
        record_property,
        f"reduced_hvp_certification_{kind}_{nsteps}",
        {
            "state_natural_regime": state_natural_regime,
            "state_coefficient_vector_regime": state_coefficient_regime,
            "state_coefficient_vector_ratios": _ratios(
                [record["relative_error"] for record in coefficient_records]
            ),
            "state_natural_ratios": _ratios(
                [record["relative_error"] for record in natural_records]
            ),
            "state_absolute_natural_floor": absolute_natural_floor,
            "scalar_regime": scalar_regime,
            "probe_pairing_regimes": probe_regimes,
            "field_independent_evidence": field_independent_evidence,
            "field_independent_checks": field_independent_checks,
            "input_nonmutation_evidence": input_nonmutation_evidence,
            "repeatability_evidence": repeatability_evidence,
            "strict_limits": {
                "field_hvp": FIELD_HVP_STRICT_FLOOR,
                "scalar_gradient": SCALAR_GRADIENT_STRICT_FLOOR,
            },
        },
    )


def test_mixed_block_symmetry(production_case, record_property):
    case = production_case
    zero = _zero_state(case, "mixed_symmetry_zero")
    _set_c0(case, PHYSICAL_C0)
    c0_block = case["split"].mtswe_terminal_least_squares_hvp(
        1,
        case["state"],
        case["t0"],
        case["dt"],
        case["target"],
        zero,
        DELTA_C0,
    )
    ic_block = case["split"].mtswe_terminal_least_squares_hvp(
        1,
        case["state"],
        case["t0"],
        case["dt"],
        case["target"],
        case["direction"],
        0.0,
    )
    left = case["helper"].dual_pairing(
        c0_block.initial_condition_hvp, case["direction"]
    )
    right = DELTA_C0 * ic_block.physical_c0_hvp
    absolute_discrepancy = abs(left - right)
    relative_discrepancy = _scalar_error(left, right)
    assert relative_discrepancy < 5.0e-10
    _emit(
        record_property,
        "mixed_block_symmetry",
        {
            "state_pairing": left,
            "scalar_pairing": right,
            "absolute_discrepancy": absolute_discrepancy,
            "relative_discrepancy": relative_discrepancy,
            "c0_to_state_coefficient_vector_norm": float(
                np.linalg.norm(_values(c0_block.initial_condition_hvp))
            ),
            "c0_to_state_natural_norm": _dual_natural_norm(
                case,
                c0_block.initial_condition_hvp,
                "mixed_symmetry_c0_to_state_riesz",
            ),
            "state_to_c0_scalar_magnitude": abs(ic_block.physical_c0_hvp),
        },
    )


def test_cofunction_mass_roundtrip_ownership_nonmutation_and_repeatability(
    production_case,
):
    case = production_case
    _set_c0(case, PHYSICAL_C0)
    input_before = _values(case["state"])
    direction_before = _values(case["direction"])
    first = case["split"].take_mtswe_forward_step_cached(
        case["state"], case["t0"], case["dt"]
    )
    first_tangent = case["split"].take_mtswe_tangent_step(
        first, case["direction"], DELTA_C0
    )
    terminal = case["helper"].state_mass_map(
        case["probe"], "ownership_terminal_dual"
    )
    reverse = case["split"].take_mtswe_adjoint_step_cached(first, terminal)
    incremental = case["split"].take_mtswe_incremental_adjoint_step(
        first_tangent, terminal, terminal
    )
    assert isinstance(reverse.state_adjoint_in, Cofunction)
    assert isinstance(incremental.incremental_state_adjoint_in, Cofunction)
    representative = case["helper"].state_riesz_representative(
        terminal, "ownership_riesz"
    )
    remapped = case["helper"].state_mass_map(
        representative, "ownership_remapped"
    )
    assert _relative_error(_values(remapped), _values(terminal)) < 2.0e-13
    assert np.array_equal(_values(case["state"]), input_before)
    assert np.array_equal(_values(case["direction"]), direction_before)
    assert first.state_in.dat is not case["state"].dat
    assert first.state_out.dat is not first.state_in.dat
    assert all(
        boundary.dat is not case["state"].dat for boundary in first.boundary_states
    )
    assert all(
        first.boundary_states[i].dat is not first.boundary_states[j].dat
        for i in range(len(first.boundary_states))
        for j in range(i + 1, len(first.boundary_states))
    )
    case["split"].reset_internal_vars()
    np.testing.assert_array_equal(_values(first.state_in), input_before)
    second = case["split"].take_mtswe_forward_step_cached(
        case["state"], case["t0"], case["dt"]
    )
    second_tangent = case["split"].take_mtswe_tangent_step(
        second, case["direction"], DELTA_C0
    )
    assert np.array_equal(_values(first.state_out), _values(second.state_out))
    assert np.array_equal(
        _values(first_tangent.state_direction_out),
        _values(second_tangent.state_direction_out),
    )
    assert first.state_out.dat is not second.state_out.dat
