"""Tests of the positivity diagnostics and statistics: cell means and whole-cell lower bounds.

For the CF, LP and H1 models (DG densities of degree 1, and continuous degree-3 h and S in H1), the
diagnostics <dens>_cellmean and <dens>_cellmin must give the exact cell mean and a lower bound of the
density over the whole cell, which is the exact cell minimum for degree <= 1. The cloud water is set
to a field with negative regions, so negative minima are also checked.
"""

import numpy as np
import pytest
from firedrake import Function, FunctionSpace, SpatialCoordinate, TestFunction, assemble, cos, dx, pi, sin

from dimswe.bernstein import field_degree
from dimswe.logger import EmptyLogger
from dimswe.models import get_model
from dimswe.parameters import get_parameters

CFGS = ['tests/mtswe-cf.cfg', 'tests/mtswe-lp.cfg', 'tests/mtswe-h1.cfg']


def model_with_diagnostics(cfg):
    """Return (model, state subfunctions, density spaces) after creating and computing the diagnostics."""
    model = get_model(get_parameters(cfg), EmptyLogger(), has_dynamics_statistics=True)
    xn, xn_sub, _ = model.get_full_var('xn', split_x_and_aux=True)
    t = model.get_t_var()
    model.initialize(xn_sub, t)
    # cloud water with negative regions (positive and negative cell minima)
    X, Y = SpatialCoordinate(model.mesh)
    L = model.initcond.Lx
    xn_sub['Qc'].interpolate(1e-4 * xn_sub['h'] * (0.3 + sin(2 * pi * X / L + 0.4) * cos(4 * pi * Y / L + 0.9)))
    _, coeff_sub, _ = model.get_coeff_var('coeff')
    model.create_diagnostics(xn_sub, t, coeff_sub)
    model.compute_diagnostics()
    variables = model.diagnostics.variableset
    spaces = {dens: variables.spacelist[variables.varlist.index(dens)] for dens in variables.density_names}
    return model, xn_sub, spaces


def cell_samples(field, degree):
    """Return an (ncells, nsamples) array of field values at the equispaced nodes of DG(degree + 2).

    The nodes include the cell vertices and edges, and the interpolation is exact (the field has
    degree <= degree per direction), so these are exact point values over each closed cell.
    """
    mesh = field.function_space().mesh()
    fine = FunctionSpace(mesh, 'DG', degree + 2, variant='equispaced')
    values = Function(fine).interpolate(field).dat.data_ro
    return values[fine.cell_node_map().values]


def cell_values(dg0_field):
    """Return the DG0 field value of every cell, in cell order."""
    space = dg0_field.function_space()
    return dg0_field.dat.data_ro[space.cell_node_map().values[:, 0]]


@pytest.mark.parametrize('cfg', CFGS)
def test_cell_means_and_whole_cell_minima(cfg):
    """cellmean is the exact cell mean, cellmin <= the density on the whole cell (= the minimum for p <= 1)."""
    model, xn_sub, spaces = model_with_diagnostics(cfg)
    diagnostics = model.get_diagnostics()
    V0 = FunctionSpace(model.mesh, 'DG', 0)
    volume = assemble(TestFunction(V0) * dx).dat.data_ro
    for dens, space in spaces.items():
        density = xn_sub[dens]
        degree = field_degree(space)
        scale = float(np.max(np.abs(density.dat.data_ro)))

        exact_mean = Function(V0)
        exact_mean.dat.data[:] = assemble(TestFunction(V0) * density * dx(degree=2 * degree + 2)).dat.data_ro / volume
        assert np.allclose(cell_values(diagnostics[dens + '_cellmean']), cell_values(exact_mean), rtol=1e-10, atol=1e-12 * scale), dens

        lower_bound = cell_values(diagnostics[dens + '_cellmin'])
        sampled_minimum = np.min(cell_samples(density, degree), axis=1)
        assert np.all(lower_bound <= sampled_minimum + 1e-10 * scale), dens
        if degree <= 1:
            assert np.allclose(lower_bound, sampled_minimum, rtol=1e-10, atol=1e-12 * scale), dens

    qc_minimum = cell_values(diagnostics['Qc_cellmin'])
    assert np.any(qc_minimum < 0) and np.any(qc_minimum > 0)
    assert np.all(cell_values(diagnostics['h_cellmin']) > 0)


@pytest.mark.parametrize('cfg', CFGS)
def test_positivity_statistics_are_domain_extrema_of_the_diagnostics(cfg):
    """<dens>_<measure>_min/max equal the min/max over the domain of the positivity diagnostics."""
    model, xn_sub, spaces = model_with_diagnostics(cfg)
    t = model.get_t_var()
    _, coeff_sub, _ = model.get_coeff_var('coeff_stats')
    model.create_statistics(xn_sub, t, coeff_sub)
    model.compute_statistics(0, 0)
    statistics = model.get_statistics()
    diagnostics = model.get_diagnostics()
    for dens in spaces:
        for measure in ('cellmean', 'cellmin'):
            values = diagnostics[f'{dens}_{measure}'].dat.data_ro
            assert statistics[f'{dens}_{measure}_min'][0] == pytest.approx(float(np.min(values)), rel=1e-12, abs=1e-300)
            assert statistics[f'{dens}_{measure}_max'][0] == pytest.approx(float(np.max(values)), rel=1e-12, abs=1e-300)
    assert statistics['Qc_cellmin_min'][0] < 0 < statistics['Qc_cellmin_max'][0]


def test_positivity_plots(tmp_path, monkeypatch):
    """The positivity figure is written, and statistics starting at zero get an absolute-change plot."""
    from dimswe.plotting.plotting import plot_positivity_statistics, plot_statistic
    monkeypatch.chdir(tmp_path)
    steps = np.arange(5, dtype=float)
    stats = {'Qc_cellmean_min': 1e-4 - 1e-5 * steps, 'Qc_cellmean_max': 2e-4 + 0 * steps,
             'Qc_cellmin_min': -1e-6 * steps, 'Qc_cellmin_max': 3e-4 + 0 * steps}
    plot_positivity_statistics(stats, 'Qc')
    plot_statistic(stats['Qc_cellmin_min'], 'Qc_cellmin_min')
    plot_statistic(stats['Qc_cellmean_max'], 'Qc_cellmean_max')
    assert (tmp_path / 'Qc-positivity.png').exists()
    assert (tmp_path / 'Qc_cellmin_min-change.png').exists()
    assert (tmp_path / 'Qc_cellmean_max-fractional-change.png').exists()
