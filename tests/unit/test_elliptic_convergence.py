"""Unit tests for the pure helpers of workflows/elliptic_convergence.py."""
import math
from pathlib import Path

import pytest

from workflows.elliptic_convergence import (compute_observed_rates, observed_rate, parse_args,
                                            parse_overrides, refinement_parameters)


def test_parse_overrides():
    """Values are parsed as YAML; malformed items are rejected."""
    assert parse_overrides(['elliptic.form_degree=2', 'mesh.periodic_directions=[x, y]',
                            'elliptic.complex=curl-div', 'elliptic.quadrature_degree=null']) == {
        'elliptic.form_degree': 2, 'mesh.periodic_directions': ['x', 'y'],
        'elliptic.complex': 'curl-div', 'elliptic.quadrature_degree': None}
    for bad in (['no-equals'], ['=3']):
        with pytest.raises(ValueError):
            parse_overrides(bad)


def test_refinement_parameters_does_not_modify_template():
    """Overrides, mesh size and output path are applied to a copy."""
    template = {'mesh': {'type': 'rectangle', 'nx': 2}, 'elliptic': {'form_degree': 0}}
    parameters = refinement_parameters(template, 16, {'elliptic.form_degree': 1}, Path('/tmp/sweep/nx16'))
    assert parameters['mesh']['nx'] == parameters['mesh']['ny'] == parameters['mesh']['nz'] == 16
    assert parameters['elliptic']['form_degree'] == 1
    assert parameters['output']['outfile_name'] == '/tmp/sweep/nx16/sim'
    assert parameters['output']['output_aux_vars'] is False
    assert template == {'mesh': {'type': 'rectangle', 'nx': 2}, 'elliptic': {'form_degree': 0}}


def test_observed_rate():
    """Second-order data give rate 2; undefined cases give None."""
    assert math.isclose(observed_rate(4e-2, 1e-2, 8, 16), 2.0)
    assert observed_rate(0.0, 1e-2, 8, 16) is None
    assert observed_rate(1e-2, float('nan'), 8, 16) is None
    assert observed_rate(1e-2, 1e-3, 8, 8) is None


def test_compute_observed_rates():
    """Rows are sorted by nx; the first row has no rates; missing errors give None."""
    rows = compute_observed_rates([16, 8, 32], [{'u_error': 0.25, 'du_error': 1.0},
                                                {'u_error': 2.0, 'du_error': 2.0},
                                                {'u_error': 0.03125}])
    assert [r['nx'] for r in rows] == [8, 16, 32]
    assert rows[0]['rate_u_error'] is None
    assert math.isclose(rows[1]['rate_u_error'], 3.0) and math.isclose(rows[1]['rate_du_error'], 1.0)
    assert math.isclose(rows[2]['rate_u_error'], 3.0) and rows[2]['rate_du_error'] is None
    with pytest.raises(ValueError):
        compute_observed_rates([8], [])


def test_parse_args():
    """--nx is required; --set may be repeated."""
    args = parse_args(['elliptic.cfg', '--nx', '4', '8', '--set', 'a=1', '--set', 'b=2'])
    assert args.nx == [4, 8] and args.overrides == ['a=1', 'b=2'] and not args.skip_existing
    with pytest.raises(SystemExit):
        parse_args(['elliptic.cfg'])
