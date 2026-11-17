"""Integration tests for the Hodge Laplacian eigenproblem (TODO-elliptic §4.11, §6).

Named test_elliptic_eigensolver.py so it does not clash with tests/unit/test_elliptic_eigen.py
(the test directories are not packages, so test module basenames must be unique).
"""
import math
from pathlib import Path

import h5py
import matplotlib
import numpy
import pytest
from firedrake import CheckpointFile

from dimswe.elliptic_eigen import HodgeLaplacianEigenproblem
from dimswe.parameters import get_parameters
from dimswe.plotting.plot_elliptic import plot_elliptic_eigen
from run_model import run_model
from tests.conftest import make_elliptic_parameters

matplotlib.use('Agg')

COMPLEX_FOR_DIM = {1: [None], 2: ['grad-curl', 'curl-div'], 3: [None]}
ALL_CASES = [(dim, cx, k) for dim in (1, 2, 3) for cx in COMPLEX_FOR_DIM[dim] for k in range(dim + 1)]
BC_SETS = {1: ['periodic', 'essential', 'natural', 'mixed'],
           2: ['periodic', 'essential', 'natural', 'mixed', 'partial-essential', 'partial-natural'],
           3: ['periodic', 'essential', 'mixed']}
# Resolution, order, number of eigenvalues and tolerance on |lambda_h - lambda| / max(1, lambda).
SETUP = {1: (16, 2, 8, 1e-2), 2: (12, 2, 8, 5e-2), 3: (6, 1, 6, 0.15)}
TEST_CFG = Path(__file__).resolve().parent.parent / 'elliptic-eigen.cfg'


def eigen_parameters(dim, cx, k, bcs, n, order, nev, **eigen):
    """Parameters of an elliptic-eigen model."""
    parameters = make_elliptic_parameters(dim, cx, k, bcs, n=n, order=order)
    parameters['model']['type'] = 'elliptic-eigen'
    parameters['elliptic']['eigen'] = dict({'nev': nev, 'target': -1.0, 'tolerance': 1e-10}, **eigen)
    return parameters


def solve(parameters):
    """Build, solve and compute statistics."""
    model = HodgeLaplacianEigenproblem(parameters)
    model.solve()
    model.create_statistics()
    model.compute_statistics()
    return model


def cases():
    """Every (dim, complex, k) with every BC set of its dimension; 3D marked slow."""
    params = []
    for dim, cx, k in ALL_CASES:
        for bcs in BC_SETS[dim]:
            marks = [pytest.mark.slow] if dim == 3 else []
            params.append(pytest.param(dim, cx, k, bcs, marks=marks,
                                       id=str(dim) + "D-" + (cx + "-" if cx else "") + "k" + str(k) + "-" + bcs))
    return params


@pytest.mark.parametrize("dim, cx, k, bcs", cases())
def test_eigenvalues_match_exact(dim, cx, k, bcs):
    """The first nev eigenvalues match the exact box spectrum with multiplicities (so no spurious
    modes in the window), and the zero multiplicity equals the harmonic dimension."""
    n, order, nev, tol = SETUP[dim]
    model = solve(eigen_parameters(dim, cx, k, bcs, n, order, nev))
    stats = model.get_statistics()
    assert len(stats['eigenvalues']) == nev
    assert numpy.max(numpy.abs(stats['eigenvalue_imag'])) < 1e-6
    errors = stats['eigenvalue_errors']
    assert numpy.max(errors) < tol, (stats['eigenvalues'], stats['exact_eigenvalues'])
    assert stats['num_zero_eigenvalues'][0] == min(model.num_harmonic, nev)
    assert stats['num_harmonic_forms'][0] == model.num_harmonic


@pytest.mark.parametrize("cx, k, bcs, order", [
    ('grad-curl', 0, 'natural', 1), ('grad-curl', 1, 'essential', 1), ('curl-div', 1, 'natural', 1),
    ('curl-div', 2, 'periodic', 1), ('grad-curl', 1, 'partial-essential', 2),
])
def test_eigenvalue_convergence(cx, k, bcs, order):
    """Eigenvalue errors of the first non-zero modes converge at ~ 2 * order."""
    nev = 6
    errors = []
    for n in (4, 8):
        stats = solve(eigen_parameters(2, cx, k, bcs, n, order, nev)).get_statistics()
        exact = stats['exact_eigenvalues']
        nonzero = exact > 1e-8
        errors.append(numpy.max(numpy.abs(stats['eigenvalues'][nonzero] - exact[nonzero]) / exact[nonzero]))
    rate = math.log(errors[0] / errors[1]) / math.log(2.0)
    assert rate >= 2 * order - 0.5, (rate, errors)


def test_harmonic_deflation():
    """Deflating the harmonic forms of the torus removes the zero eigenvalues only."""
    model = solve(eigen_parameters(2, 'grad-curl', 1, 'periodic', 8, 2, 6, deflate_harmonic=True))
    stats = model.get_statistics()
    assert model.num_harmonic == 2 and stats['deflated'][0] == 1
    assert stats['num_zero_eigenvalues'][0] == 0
    assert numpy.min(stats['exact_eigenvalues']) > 1.0
    assert numpy.max(stats['eigenvalue_errors']) < 5e-2


def test_run_model_eigen_and_plots(tmp_path):
    """run_model writes one checkpoint index per mode plus the eigenvalue statistics; plots work."""
    parameters = get_parameters(TEST_CFG)
    parameters['output']['outfile_name'] = str(tmp_path / 'sim')
    model = run_model(parameters)
    assert isinstance(model, HodgeLaplacianEigenproblem)
    nev = parameters['elliptic']['eigen']['nev']
    with h5py.File(tmp_path / 'sim.h5', 'r') as handle:
        assert handle['eigenvalues'].shape == (nev,)
        assert handle['exact_eigenvalues'].shape == (nev,)
        assert handle['num_zero_eigenvalues'][0] == handle['num_harmonic_forms'][0] == 1
    with CheckpointFile(str(tmp_path / 'sim.h5'), 'r') as checkpoint:
        mesh = checkpoint.load_mesh()
        for i in range(nev):
            for name in ('sigma', 'u'):
                checkpoint.load_function(mesh, name, idx=i)
    assert (tmp_path / 'sim.pvd').exists()
    written = plot_elliptic_eigen(parameters, outdir=tmp_path / 'plots', max_modes=2)
    names = {Path(p).name for p in written}
    assert {'eigenvalues.png', 'mode0-sigma.png', 'mode0-u.png', 'mode0-u-mag.png', 'mode1-u.png'} <= names
