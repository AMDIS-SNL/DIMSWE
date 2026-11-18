"""Unit tests for the pure helpers of workflows/steady_state.py (spec validation, cfg rewriting)."""
import math

import pytest

from workflows.steady_state import (check_loss_fields, coefficient_name, from_parameter_value, is_time_invariant_cfg, model_parameters,
                                    steady_loss_label, to_parameter_value, truth_coefficient_value,
                                    validate_steady_spec)


def spec(**changes):
    """A valid steady-state spec with overrides."""
    base = {'target': 'coeffs', 'truth': {'mode': 'per-run'},
            'parameters': {'elliptic.coefficients.kappa': {'initial': 1.0, 'bounds': [0.1, 10.0], 'log': True}},
            'losses': [{'type': 'steady', 'fields': ['u']}]}
    base.update(changes)
    return base


def test_helpers():
    """Names, labels, log transform and the time-invariance check."""
    assert coefficient_name('elliptic.coefficients.mu') == 'mu'
    for bad in ('hyperviscosity.c0', 'elliptic.coefficients.kappa.value'):
        with pytest.raises(ValueError):
            coefficient_name(bad)
    assert steady_loss_label({'fields': ['sigma', 'u']}) == 'steady-sigma-u'
    assert steady_loss_label({'fields': ['u'], 'label': 'my loss'}) != 'steady-u'
    assert math.isclose(from_parameter_value(to_parameter_value(2.0, True), True), 2.0)
    assert to_parameter_value(2.0, False) == 2.0
    assert is_time_invariant_cfg({'model': {'type': 'elliptic'}})
    assert not is_time_invariant_cfg({'model': {'type': 'metriplectic'}})


def test_validate_steady_spec():
    """A valid spec passes and defaults the loss fields."""
    s = spec(losses=[{'type': 'steady'}])
    validate_steady_spec(s)
    assert s['losses'][0]['fields'] == ['u']


@pytest.mark.parametrize("changes", [
    {'target': 'ic-field'},
    {'truth': {'mode': 'joint'}},
    {'losses': []},
    {'losses': [{'type': 'final'}]},
    {'losses': [{'type': 'steady', 'fields': ['p']}]},
    {'losses': [{'type': 'steady', 'colour': 1}]},
    {'losses': [{'type': 'steady'}, {'type': 'steady'}]},          # duplicate labels
    {'losses': [{'type': 'steady', 'regularization': -1.0}]},
    {'parameters': {}},
    {'parameters': {'hyperviscosity.c0': 1.0}},
    {'parameters': {'elliptic.coefficients.kappa': {'bounds': [2.0, 1.0]}}},
    {'parameters': {'elliptic.coefficients.kappa': {'log': True, 'bounds': [0.0, 1.0]}}},
    {'parameters': {'elliptic.coefficients.kappa': {'scale': 0.0}}},
    {'parameters': {'elliptic.coefficients.kappa': {'fixed': True}}},   # nothing free
    {'parameters': {'elliptic.coefficients.kappa': {'shape': 'round'}}},
])
def test_validate_steady_spec_errors(changes):
    """Invalid specs raise ValueError."""
    with pytest.raises(ValueError):
        validate_steady_spec(spec(**changes))


def test_model_parameters_marks_coefficients():
    """Listed coefficients become parameters; the truth's values (problem data) are kept."""
    truth = {'model': {'type': 'elliptic'}, 'elliptic': {'form_degree': 2, 'coefficients': {
        'kappa': {'value': 2.0}, 'mu': {'kind': 'problem'}}}}
    s = spec(parameters={'elliptic.coefficients.kappa': {'initial': 1.0, 'log': True, 'bounds': [0.1, 10.0],
                                                         'scale': 3.0},
                         'elliptic.coefficients.mu': {'field': 'DG0'}})
    parameters = model_parameters(truth, s)
    kappa = parameters['elliptic']['coefficients']['kappa']
    assert kappa == {'value': 2.0, 'treat_as_coeff': True, 'log_parametrization': True, 'bounds': [0.1, 10.0],
                     'scaling': 3.0}
    mu = parameters['elliptic']['coefficients']['mu']
    assert mu['kind'] == 'field' and mu['space'] == 'DG0' and mu['init'] == 'problem' and mu['treat_as_coeff']
    assert truth['elliptic']['coefficients']['kappa'] == {'value': 2.0}   # truth cfg not modified
    assert truth_coefficient_value(truth, 'kappa') == 2.0
    assert truth_coefficient_value(truth, 'mu') is None
    assert truth_coefficient_value(truth, 'c') == 0.0


def test_model_parameters_rejects_tensors_and_problem_constants():
    """Tensor coefficients cannot be optimized; problem expressions only as fields."""
    truth = {'elliptic': {'coefficients': {'kappa': {'rank': 'tensor', 'value': [[1, 0], [0, 1]]},
                                           'mu': {'kind': 'problem'}}}}
    with pytest.raises(ValueError):
        model_parameters(truth, spec(parameters={'elliptic.coefficients.kappa': 1.0}))
    with pytest.raises(ValueError):
        model_parameters(truth, spec(parameters={'elliptic.coefficients.mu': 1.0}))


def test_check_loss_fields():
    """sigma is rejected for k = 0 truth runs, accepted otherwise."""
    with pytest.raises(ValueError):
        check_loss_fields({'elliptic': {'form_degree': 0}}, ['sigma', 'u'])
    check_loss_fields({'elliptic': {'form_degree': 0}}, ['u'])
    check_loss_fields({'elliptic': {'form_degree': 1}}, ['sigma', 'u'])
