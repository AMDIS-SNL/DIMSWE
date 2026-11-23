"""Iterative solvers for the elliptic problems (TODO-elliptic §4.12, §6).

- riesz-amg / riesz-gmg / riesz-pmg / schur-mass reproduce the direct solve;
- iteration counts stay bounded under mesh refinement (and coefficient contrast for the scalar
  Riesz maps);
- the nullspace (and lagrange) harmonic treatments work with iterative solvers;
- the mesh hierarchy (mesh.refinement_levels) keeps the face registry and periodicity;
- the eigenproblem's iterative inner solve matches the direct one;
- the scaling workflow runs end to end;
- an MPI smoke test (marked mpi, run with `mpiexec -n 2 pytest -m mpi ...`).
"""
import copy
from pathlib import Path

import numpy
import pytest
import yaml
from firedrake import COMM_WORLD, assemble, dx, inner

from dimswe.boundary_conditions import get_boundary_registry
from dimswe.elliptic import EllipticProblem
from dimswe.elliptic_eigen import HodgeLaplacianEigenproblem
from dimswe.ufl_helpers import ufl_attribute
from tests.conftest import make_elliptic_parameters
from workflows.elliptic_solver_scaling import main as scaling_main


def run(parameters, mesh=None):
    """Solve with statistics."""
    model = EllipticProblem(parameters, mesh=mesh)
    model.solve()
    model.create_statistics()
    model.compute_statistics()
    return model


def iterative(dim, cx, k, bcs, solver, n, order=2, levels=0, constraint='nullspace', **kwargs):
    """Parameters for an iterative solve (n = cells of the coarsest level when levels > 0)."""
    parameters = make_elliptic_parameters(dim, cx, k, bcs, n=n, order=order, solver=solver,
                                          harmonic_constraint=constraint, **kwargs)
    parameters['mesh']['refinement_levels'] = levels
    parameters['elliptic']['ksp_rtol'] = 1e-11
    return parameters


def l2(f):
    """L2 norm."""
    return assemble(inner(f, f) * dx) ** 0.5


def direct_reference(model):
    """The direct solve of the same problem on the same mesh (only the solver changes)."""
    parameters = copy.deepcopy(model.parameters)
    parameters['elliptic']['solver'] = 'elliptic-direct'
    return run(parameters, mesh=model.mesh)


def assert_matches_direct(model, *args, **kwargs):
    """The iterative solution equals the direct one of the same problem on the same mesh."""
    direct = direct_reference(model)
    assert model.ksp_reason > 0, model.ksp_reason
    for name in direct.get_x_var_list():
        diff = l2(model.sub[name] - direct.sub[name])
        assert diff < 1e-7 * max(1.0, l2(direct.sub[name])), (name, diff)


CASES_2D = [('grad-curl', 0), ('grad-curl', 1), ('curl-div', 1), ('curl-div', 2), ('grad-curl', 2)]


@pytest.mark.parametrize("cx, k", CASES_2D)
@pytest.mark.parametrize("bcs", ['essential', 'natural', 'periodic'])
def test_riesz_amg_matches_direct(cx, k, bcs):
    """riesz-amg (BoomerAMG / LU fallback / bjacobi) reproduces the direct solution."""
    model = run(iterative(2, cx, k, bcs, 'riesz-amg', n=8))
    assert_matches_direct(model, 2, cx, k, bcs, 2)


@pytest.mark.parametrize("cx, k", CASES_2D)
def test_riesz_gmg_matches_direct(cx, k):
    """riesz-gmg on a two-level hierarchy (coarse 4x4, fine 16x16)."""
    model = run(iterative(2, cx, k, 'essential', 'riesz-gmg', n=4, levels=2))
    assert model.mesh.refinement_levels == 2 and model.parameters['mesh']['nx'] == 4
    assert_matches_direct(model)


@pytest.mark.parametrize("cx, k", [('grad-curl', 1), ('curl-div', 1), ('grad-curl', 0)])
def test_riesz_pmg_matches_direct(cx, k):
    """riesz-pmg (order 3 -> 1) reproduces the direct solution."""
    model = run(iterative(2, cx, k, 'natural', 'riesz-pmg', n=6, order=3))
    assert_matches_direct(model, 2, cx, k, 'natural', 3)


@pytest.mark.parametrize("cx, bcs", [('curl-div', 'essential'), ('grad-curl', 'natural')])
def test_schur_mass_matches_direct(cx, bcs):
    """k = n mixed Poisson/Darcy with the Schur-complement solver."""
    model = run(iterative(2, cx, 2, bcs, 'schur-mass', n=8))
    assert_matches_direct(model, 2, cx, 2, bcs, 2)


@pytest.mark.parametrize("solver, levels, cx, k", [
    ('riesz-gmg', None, 'grad-curl', 1), ('riesz-gmg', None, 'curl-div', 1), ('riesz-gmg', None, 'grad-curl', 0),
    ('riesz-amg', 0, 'grad-curl', 0),
])
def test_iterations_bounded_under_refinement(solver, levels, cx, k):
    """Over three refinements the iteration count stays below a bound and grows by at most 1.5x."""
    counts = []
    for refine in (1, 2, 3):
        if solver == 'riesz-gmg':
            parameters = iterative(2, cx, k, 'essential', solver, n=4, levels=refine)
        else:
            parameters = iterative(2, cx, k, 'essential', solver, n=4 * 2 ** refine)
        parameters['elliptic']['ksp_rtol'] = 1e-8
        model = run(parameters)
        assert model.ksp_reason > 0
        counts.append(model.ksp_iterations)
    assert max(counts) < 200, counts
    assert counts[-1] <= 1.5 * counts[0] + 5, counts


def test_coefficient_contrast_robustness():
    """kappa jumping by 1 -> 10^3 (two-material problem): riesz-gmg iterations stay bounded."""
    counts = {}
    for contrast in (1.0, 1e3):
        parameters = iterative(2, 'grad-curl', 0, 'essential', 'riesz-gmg', n=4, levels=2,
                               problem='two-material-sine', contrast=contrast)
        parameters['elliptic']['coefficients'] = {'kappa': {'kind': 'problem'}}
        parameters['elliptic']['ksp_rtol'] = 1e-8
        model = run(parameters)
        assert model.ksp_reason > 0
        counts[contrast] = model.ksp_iterations
    assert counts[1e3] <= 3 * counts[1.0] + 10, counts


@pytest.mark.parametrize("constraint", ['nullspace', 'lagrange'])
def test_harmonic_forms_with_iterative_solver(constraint):
    """Fully periodic k = 1 (two harmonic forms): both treatments converge to the direct solution."""
    model = run(iterative(2, 'curl-div', 1, 'periodic', 'riesz-amg', n=6, constraint=constraint,
                          problem='incompatible-source', amplitude=2.0))
    assert model.num_harmonic == 2
    assert model.get_statistics()['harmonic_orthogonality'][0] < 1e-7
    assert_matches_direct(model, 2, 'curl-div', 1, 'periodic', 2, constraint=constraint)


def test_hierarchy_keeps_boundaries_and_periodicity():
    """The finest mesh of a hierarchy keeps the face registry, bounds and periodic directions."""
    model = run(iterative(2, 'grad-curl', 1, 'partial-essential', 'riesz-gmg', n=4, levels=1))
    mesh = model.mesh
    assert mesh.periodic_directions == ('x',)
    registry = get_boundary_registry(mesh, 'rectangle')
    assert set(registry) == {'bottom', 'top'}
    assert model.get_statistics()['u_error'][0] < 0.1
    assert mesh.dx == pytest.approx(1.0 / 8)


@pytest.mark.slow
def test_3d_extruded_hierarchy_gmg():
    """3D Q-family box: ExtrudedMeshHierarchy + riesz-gmg (extruded star patches) matches direct."""
    model = run(iterative(3, None, 1, 'essential', 'riesz-gmg', n=2, order=1, levels=1))
    assert isinstance(ufl_attribute(model.mesh, 'cell_dimension'), tuple)
    assert_matches_direct(model)


@pytest.mark.parametrize("cx, k, bcs", [('grad-curl', 0, 'essential'), ('curl-div', 1, 'natural')])
def test_eigen_iterative_inner_solve(cx, k, bcs):
    """Shift-and-invert with the Riesz-preconditioned iterative inner solve matches the direct one."""
    eigenvalues = {}
    for inner_solver in ('direct', 'iterative'):
        parameters = make_elliptic_parameters(2, cx, k, bcs, n=8, order=2)
        parameters['model']['type'] = 'elliptic-eigen'
        parameters['elliptic']['eigen'] = {'nev': 5, 'target': -1.0, 'tolerance': 1e-10,
                                           'inner_solver': inner_solver}
        model = HodgeLaplacianEigenproblem(parameters)
        model.solve()
        eigenvalues[inner_solver] = numpy.array(model.eigenvalues)
    assert numpy.allclose(eigenvalues['iterative'], eigenvalues['direct'], rtol=1e-6, atol=1e-8), eigenvalues


def test_scaling_workflow(tmp_path):
    """The scaling sweep writes runs, summary.yaml/csv and the iterations plot."""
    cfg = Path(__file__).resolve().parent.parent / 'elliptic.cfg'
    scaling_main([str(cfg), '--nx', '4', '8', '--solver', 'riesz-amg', 'elliptic-direct', '--order', '1',
                  '--runs-root', str(tmp_path), '--name', 'scaling'])
    sweep = tmp_path / 'scaling'
    summary = yaml.safe_load((sweep / 'summary.yaml').read_text())
    rows = summary['rows']
    assert len(rows) == 4 and {row['solver'] for row in rows} == {'riesz-amg', 'elliptic-direct'}
    assert all(row['num_dofs'] > 0 and row['ksp_converged_reason'] > 0 for row in rows)
    assert (sweep / 'summary.csv').exists() and (sweep / 'iterations.png').exists()


@pytest.mark.mpi
def test_parallel_smoke():
    """Under mpiexec -n 2: riesz-amg converges and matches the direct solve (skipped in serial)."""
    if COMM_WORLD.size < 2:
        pytest.skip("run with mpiexec -n 2 pytest -m mpi")
    model = run(iterative(2, 'curl-div', 1, 'essential', 'riesz-amg', n=8))
    assert_matches_direct(model, 2, 'curl-div', 1, 'essential', 2)
