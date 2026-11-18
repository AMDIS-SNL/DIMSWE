"""Unit tests for dimswe/plotting/plot_elliptic.py (pure helpers and the convergence plot)."""
import matplotlib

matplotlib.use("Agg")

from dimswe.plotting.plot_elliptic import (ERROR_SERIES, convergence_series, elliptic_field_degrees,  # noqa: E402
                                           elliptic_fields, expected_rates, is_vector_field,
                                           plot_convergence)


def test_field_degrees_and_shapes():
    """sigma is a (k-1)-form, u/p k-forms, du a (k+1)-form; only interior degrees are vectors."""
    degrees = elliptic_field_degrees(1)
    assert degrees == {'sigma': 0, 'u': 1, 'p': 1, 'u_exact': 1, 'u_error': 1, 'sigma_error': 0, 'du': 2}
    assert [is_vector_field(n, 1, 2) for n in ('sigma', 'u', 'du')] == [False, True, False]
    assert [is_vector_field(n, 2, 3) for n in ('sigma', 'u', 'du')] == [True, True, False]
    assert not is_vector_field('u', 0, 1) and not is_vector_field('sigma', 1, 1)


def test_elliptic_fields():
    """Field lists match what run_elliptic writes."""
    assert elliptic_fields(0, 2, True, True) == ['u', 'p', 'u_exact', 'u_error', 'du']
    assert elliptic_fields(2, 2, False, False) == ['sigma', 'u']
    assert elliptic_fields(1, 3, True, True) == ['sigma', 'u', 'p', 'u_exact', 'u_error', 'sigma_error', 'du']


def test_expected_rates():
    """u gains an order for k = 0; everything else converges at the spatial order."""
    assert expected_rates(2, 0) == {'u_error': 3, 'sigma_error': 2, 'du_error': 2, 'dsigma_error': 2}
    assert expected_rates(1, 2)['u_error'] == 1


def test_convergence_series_filters_and_orders():
    """Zero, None and non-finite errors are dropped; series follow the fixed ERROR_SERIES order."""
    rows = [{'nx': 4, 'u_error': 1e-2, 'du_error': 0.0, 'sigma_error': None, 'p_error': float('nan')},
            {'nx': 8, 'u_error': 2e-3, 'du_error': 1e-3, 'sigma_error': 5e-3}]
    series = convergence_series(rows)
    assert list(series) == ['u_error', 'sigma_error', 'du_error']
    assert series['u_error'] == ([4, 8], [1e-2, 2e-3])
    assert series['du_error'] == ([8], [1e-3])
    assert list(series) == [name for name in ERROR_SERIES if name in series]


def test_plot_convergence(tmp_path):
    """A synthetic summary gives a PNG; an empty one gives nothing."""
    summary = {'elliptic': {'form_degree': 1, 'complex': 'curl-div'}, 'spatial-discretization': {'order': 2},
               'rows': [{'nx': n, 'u_error': n ** -2.0, 'sigma_error': 3 * n ** -2.0, 'du_error': n ** -2.0,
                         'dsigma_error': 2 * n ** -2.0, 'p_error': 0.0} for n in (4, 8, 16)]}
    path = plot_convergence(summary, tmp_path / 'convergence.png')
    assert path.exists() and path.stat().st_size > 0
    assert plot_convergence({'rows': []}, tmp_path / 'none.png') is None
