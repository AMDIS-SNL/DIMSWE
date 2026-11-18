"""optimize_params.py routes elliptic truth runs to the steady-state optimization (TODO-elliptic §4.10)."""
import math
import sys

import pytest
import yaml

from dimswe.parameters import get_parameters
from run_model import run_model
from tests.conftest import make_elliptic_parameters
from workflows.common import dump_yaml

pytestmark = pytest.mark.integration


def make_truth_run(run_dir, kappa, k=1):
    """An elliptic truth run directory (run.cfg + sim.h5) with constant kappa.

    k = 1 (grad-curl) has both sigma and u, so sigma-based losses are meaningful.
    """
    parameters = make_elliptic_parameters(2, 'grad-curl', k, 'essential', n=6, order=2, problem='sine')
    parameters['elliptic']['coefficients'] = {'kappa': {'value': kappa}}
    parameters['output'].update({'outfile_name': str(run_dir / 'sim'), 'output_aux_vars': False,
                                 'output_freq': 1, 'stat_freq': 1})
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(parameters, run_dir / 'run.cfg')
    run_model(get_parameters(run_dir / 'run.cfg'))
    return parameters


def run_optimize(monkeypatch, spec_path, *extra):
    """Run workflows.optimize_params.main with command-line arguments."""
    from workflows import optimize_params
    monkeypatch.setattr(sys, 'argv', ['optimize_params.py', str(spec_path), *extra])
    optimize_params.main()


def test_kappa_recovery_from_elliptic_truth(tmp_path, monkeypatch):
    """One truth run with kappa = 2; both losses recover it from kappa = 1 (log, bounds)."""
    make_truth_run(tmp_path / 'truth', 2.0)
    spec = {'optimization_name': 'kappa', 'truth': {'runs': [str(tmp_path / 'truth')]}, 'target': 'coeffs',
            'parameters': {'elliptic.coefficients.kappa': {'initial': 1.0, 'bounds': [0.1, 10.0], 'log': True}},
            'losses': [{'type': 'steady', 'fields': ['u']}, {'type': 'steady', 'fields': ['sigma', 'u'],
                                                            'label': 'both'}],
            'optimizer': {'gtol': 1e-10, 'ftol': 1e-14, 'maxiter': 100, 'check_gradient': True},
            'output_root': str(tmp_path / 'opt')}
    spec_path = tmp_path / 'optimize.cfg'
    dump_yaml(spec, spec_path)
    run_optimize(monkeypatch, spec_path)
    for label in ('steady-u', 'both'):
        job_dir = tmp_path / 'opt' / 'kappa' / f'truth__{label}'
        result = yaml.safe_load((job_dir / 'result.yaml').read_text())
        assert result['steady_state'] and result['parameters']['kappa']['true'] == 2.0
        assert result['parameters']['kappa']['rel_error'] < 1e-3
        assert result['final_loss'] < 1e-3 * result['initial_loss']
        taylor = yaml.safe_load((job_dir / 'taylor-test.yaml').read_text())
        assert all(abs(r - 2.0) < 0.2 for r in taylor['rates'][:3]), taylor['rates']
        optimal = yaml.safe_load((job_dir / 'optimal.cfg').read_text())
        assert math.isclose(optimal['elliptic']['coefficients']['kappa']['value'], 2.0, rel_tol=1e-3)
        assert (job_dir / 'history.csv').exists() and (job_dir / 'optimize.cfg').exists()


def test_dry_run_plans_steady_jobs(tmp_path, monkeypatch, capsys):
    """--dry-run plans steady jobs without solving."""
    make_truth_run(tmp_path / 'truth', 1.5)
    spec = {'optimization_name': 'dry', 'truth': {'runs': [str(tmp_path / 'truth')]}, 'target': 'coeffs',
            'parameters': {'elliptic.coefficients.kappa': 1.0}, 'losses': [{'type': 'steady'}],
            'output_root': str(tmp_path / 'opt')}
    spec_path = tmp_path / 'optimize.cfg'
    dump_yaml(spec, spec_path)
    run_optimize(monkeypatch, spec_path, '--dry-run')
    out = capsys.readouterr().out
    assert 'steady state' in out and 'truth__steady-u' in out
    assert not (tmp_path / 'opt' / 'dry').exists()


def test_sigma_loss_rejected_for_primal_truth(tmp_path, monkeypatch):
    """A sigma loss on a k = 0 truth run fails with an explicit message, before loading data."""
    make_truth_run(tmp_path / 'truth', 1.5, k=0)
    spec = {'optimization_name': 'bad', 'truth': {'runs': [str(tmp_path / 'truth')]}, 'target': 'coeffs',
            'parameters': {'elliptic.coefficients.kappa': 1.0},
            'losses': [{'type': 'steady', 'fields': ['sigma', 'u']}], 'output_root': str(tmp_path / 'opt')}
    spec_path = tmp_path / 'optimize.cfg'
    dump_yaml(spec, spec_path)
    with pytest.raises(RuntimeError, match="does not exist for elliptic.form_degree 0"):
        run_optimize(monkeypatch, spec_path)
