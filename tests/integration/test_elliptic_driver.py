"""Driver and workflow integration of the elliptic model (TODO-elliptic §4.7)."""
from pathlib import Path

import h5py
import matplotlib
import pytest
import yaml
from firedrake import CheckpointFile, assemble, dx, inner

from dimswe.elliptic import EllipticProblem
from dimswe.logger import EmptyLogger
from dimswe.models import get_model
from dimswe.parameters import get_parameters
from run_model import run_model
from workflows.elliptic_convergence import main as convergence_main
from dimswe.plotting.plot_elliptic import plot_elliptic

matplotlib.use('Agg')
pytestmark = pytest.mark.integration

TEST_CFG = Path(__file__).resolve().parent.parent / 'elliptic.cfg'


def test_get_model_dispatches_elliptic():
    """model.type: elliptic builds an EllipticProblem."""
    model = get_model(get_parameters(TEST_CFG), EmptyLogger())
    assert isinstance(model, EllipticProblem) and model.is_time_invariant


def test_run_model_writes_checkpoint(tmp_path):
    """run_model solves once and writes sim.h5 (fields + statistics) and sim.pvd."""
    parameters = get_parameters(TEST_CFG)
    parameters['output']['outfile_name'] = str(tmp_path / 'sim')
    model = run_model(parameters)
    assert isinstance(model, EllipticProblem)
    assert model.get_x_var_list() == ['sigma', 'u', 'p']
    assert (tmp_path / 'sim.pvd').exists()

    with h5py.File(tmp_path / 'sim.h5', 'r') as handle:
        for name in model.get_statistics_list():
            assert name in handle and handle[name].shape == (1,)
        assert handle['u_error'][0] > 0.0 and handle['num_harmonic_forms'][0] == 1

    with CheckpointFile(str(tmp_path / 'sim.h5'), 'r') as checkpoint:
        mesh = checkpoint.load_mesh()
        for name in model.get_x_var_list() + model.get_diagnostics_list():
            loaded = checkpoint.load_function(mesh, name, idx=0)
            reference = model.sub[name] if name in model.sub else model.get_diagnostics()[name]
            norm_loaded = assemble(inner(loaded, loaded) * dx) ** 0.5
            norm_reference = assemble(inner(reference, reference) * dx) ** 0.5
            assert abs(norm_loaded - norm_reference) < 1e-10 * max(1.0, norm_reference), name


def test_run_model_rejects_initial_state(tmp_path):
    """initial_state does not apply to time-invariant models."""
    parameters = get_parameters(TEST_CFG)
    parameters['output']['outfile_name'] = str(tmp_path / 'sim')
    with pytest.raises(ValueError):
        run_model(parameters, initial_state=lambda model, xn, t: None)


def test_convergence_workflow(tmp_path):
    """The CLI runs an nx sweep (1D Dirichlet/Neumann, k = 0, order 2) and records rates (u: 3, du: 2)."""
    convergence_main([str(TEST_CFG), '--nx', '4', '8', '--runs-root', str(tmp_path), '--name', 'sweep',
                      '--set', 'mesh.type=line', '--set', 'mesh.periodic_directions=[]',
                      '--set', 'elliptic.form_degree=0', '--set', 'elliptic.complex=null',
                      '--set', 'elliptic.boundary_conditions={left: essential, right: natural}',
                      '--set', 'spatial-discretization.order=2'])
    sweep = tmp_path / 'sweep'
    for nx in (4, 8):
        assert (sweep / f'nx{nx}' / 'sim.h5').exists() and (sweep / f'nx{nx}' / 'run.cfg').exists()
    assert (sweep / 'summary.csv').exists() and (sweep / 'convergence.png').exists()
    summary = yaml.safe_load((sweep / 'summary.yaml').read_text())
    assert summary['nx'] == [4, 8] and summary['elliptic']['form_degree'] == 0
    rows = summary['rows']
    assert rows[0]['rate_u_error'] is None
    assert rows[1]['rate_u_error'] > 2.5 and rows[1]['rate_du_error'] > 1.5  # expected 3 and 2
    # Re-running with --skip-existing reuses the runs and reproduces the summary.
    convergence_main([str(TEST_CFG), '--nx', '4', '8', '--runs-root', str(tmp_path), '--name', 'sweep',
                      '--skip-existing', '--set', 'mesh.type=line', '--set', 'mesh.periodic_directions=[]',
                      '--set', 'elliptic.form_degree=0', '--set', 'elliptic.complex=null',
                      '--set', 'elliptic.boundary_conditions={left: essential, right: natural}',
                      '--set', 'spatial-discretization.order=2'])
    again = yaml.safe_load((sweep / 'summary.yaml').read_text())
    assert again['rows'] == rows


def test_plot_elliptic_2d(tmp_path):
    """2D grad-curl k = 1: scalar plots for 0-/2-forms, quiver + magnitude for 1-forms."""
    parameters = get_parameters(TEST_CFG)
    parameters['output']['outfile_name'] = str(tmp_path / 'sim')
    run_model(parameters)
    written = plot_elliptic(parameters, outdir=tmp_path / 'plots')
    names = {Path(p).name for p in written}
    assert names == {'sigma.png', 'u.png', 'u-mag.png', 'p.png', 'p-mag.png', 'u_exact.png', 'u_exact-mag.png',
                     'u_error.png', 'u_error-mag.png', 'sigma_error.png', 'du.png'}
    assert all(Path(p).exists() for p in written)


def test_plot_elliptic_1d_and_3d(tmp_path):
    """1D: line plots with u_h overlaid on u_exact; 3D: VTK only (no static plots)."""
    parameters = get_parameters(TEST_CFG)
    parameters['mesh'].update({'type': 'line', 'periodic_directions': []})
    parameters['elliptic'].update({'form_degree': 0, 'complex': None,
                                   'boundary_conditions': {'left': 'essential', 'right': 'natural'}})
    parameters['output']['outfile_name'] = str(tmp_path / 'sim')
    run_model(parameters)
    written = plot_elliptic(parameters, outdir=tmp_path / 'plots')
    assert {Path(p).name for p in written} == {'u.png', 'u_exact.png', 'u_error.png', 'du.png'}
    three_d = get_parameters(TEST_CFG)
    three_d['mesh']['type'] = 'box'
    assert plot_elliptic(three_d, outdir=tmp_path / 'plots3d') == []
