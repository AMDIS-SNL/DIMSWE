"""Unit tests for the pure parts of dimswe/elliptic_eigen.py (exact spectra, eigen config)."""
from itertools import product
from math import pi

import numpy
import pytest

from dimswe.elliptic_eigen import EigenConfig, direction_modes, exact_box_eigenvalues
from dimswe.harmonic_forms import harmonic_index_sets


def close(a, b):
    """Elementwise comparison of eigenvalue lists."""
    return len(a) == len(b) and numpy.allclose(a, b, rtol=1e-12, atol=1e-12)


def test_direction_modes():
    """cos/sin factors, periodic pairs and quarter waves."""
    assert close(direction_modes(1.0, 'essential', True, 3), [0, pi ** 2, 4 * pi ** 2])      # cos
    assert close(direction_modes(1.0, 'essential', False, 3), [pi ** 2, 4 * pi ** 2, 9 * pi ** 2])  # sin
    assert close(direction_modes(1.0, 'natural', True, 2), [pi ** 2, 4 * pi ** 2])           # sin
    assert close(direction_modes(1.0, 'natural', False, 2), [0, pi ** 2])                    # cos
    assert close(direction_modes(1.0, 'periodic', False, 5), [0] + [(2 * pi) ** 2] * 2 + [(4 * pi) ** 2] * 2)
    assert close(direction_modes(2.0, 'mixed', True, 2), [(pi / 4) ** 2, (3 * pi / 4) ** 2])


@pytest.mark.parametrize("k, dim, lengths, periodic, bcs, expected", [
    # 1D Dirichlet / Neumann / periodic / mixed scalar Laplacian
    (0, 1, [1.0], (), {'left': 'essential', 'right': 'essential'}, [pi ** 2, 4 * pi ** 2, 9 * pi ** 2]),
    (0, 1, [2.0], (), {'left': 'essential', 'right': 'essential'}, [(pi / 2) ** 2, pi ** 2]),
    (0, 1, [1.0], (), {'left': 'natural', 'right': 'natural'}, [0, pi ** 2, 4 * pi ** 2]),
    (0, 1, [1.0], ('x',), {}, [0, 4 * pi ** 2, 4 * pi ** 2, 16 * pi ** 2]),
    (0, 1, [1.0], (), {'left': 'essential', 'right': 'natural'}, [(pi / 2) ** 2, (3 * pi / 2) ** 2]),
    # 1D 1-forms: essential -> cos (constants harmonic), natural -> sin
    (1, 1, [1.0], (), {'left': 'essential', 'right': 'essential'}, [0, pi ** 2, 4 * pi ** 2]),
    (1, 1, [1.0], (), {'left': 'natural', 'right': 'natural'}, [pi ** 2, 4 * pi ** 2]),
    # unit square, Neumann k = 0: 0, pi^2 (x2), 2 pi^2, 4 pi^2 (x2)
    (0, 2, [1.0, 1.0], (), {'left': 'natural', 'right': 'natural', 'bottom': 'natural', 'top': 'natural'},
     [0, pi ** 2, pi ** 2, 2 * pi ** 2, 4 * pi ** 2, 4 * pi ** 2]),
    # unit torus, 1-forms: 0 (x2, the harmonic dx, dy), then (2 pi)^2 with multiplicity 8
    (1, 2, [1.0, 1.0], ('x', 'y'), {}, [0, 0] + [(2 * pi) ** 2] * 8),
])
def test_exact_box_eigenvalues(k, dim, lengths, periodic, bcs, expected):
    """Known spectra with multiplicities."""
    assert close(exact_box_eigenvalues(k, dim, lengths, periodic, bcs, len(expected)), expected)


def all_configs():
    """Every (k, dim, periodic set, per-direction BC kind) combination."""
    configs = []
    kinds = {'periodic': None, 'essential': ('essential', 'essential'), 'natural': ('natural', 'natural'),
             'mixed': ('essential', 'natural')}
    for dim in (1, 2, 3):
        for choice in product(kinds, repeat=dim):
            periodic = tuple(d for d, c in zip('xyz', choice) if c == 'periodic')
            bcs = {d: kinds[c] for d, c in zip('xyz', choice) if c != 'periodic'}
            for k in range(dim + 1):
                configs.append((k, dim, periodic, bcs))
    return configs


@pytest.mark.parametrize("k, dim, periodic, bcs", all_configs())
def test_zero_multiplicity_matches_harmonic_rule(k, dim, periodic, bcs):
    """The multiplicity of eigenvalue 0 is the harmonic dimension of the §1.4 rule."""
    expected = len(harmonic_index_sets(k, dim, periodic, bcs))
    spectrum = exact_box_eigenvalues(k, dim, [1.0, 1.3, 0.7][:dim], periodic, bcs, expected + 4)
    assert sum(1 for value in spectrum if abs(value) < 1e-12) == expected
    assert spectrum == sorted(spectrum)


def test_exact_box_eigenvalues_rejects_inconsistent_bcs():
    """BCs on periodic directions and missing faces are rejected."""
    with pytest.raises(ValueError):
        exact_box_eigenvalues(0, 1, [1.0], ('x',), {'left': 'natural', 'right': 'natural'}, 3)
    with pytest.raises(ValueError):
        exact_box_eigenvalues(0, 2, [1.0, 1.0], (), {'left': 'natural', 'right': 'natural'}, 3)
    with pytest.raises(ValueError):
        exact_box_eigenvalues(3, 2, [1.0, 1.0], ('x', 'y'), {}, 3)


def test_eigen_config():
    """Defaults, overrides, validation and the SLEPc options."""
    default = EigenConfig.from_parameters({'elliptic': {}})
    assert (default.nev, default.target, default.deflate_harmonic) == (20, -1.0, False)
    config = EigenConfig.from_parameters({'elliptic': {'eigen': {'nev': 5, 'ncv': 12, 'target': 2.0, 'max_it': 50}}})
    params = config.solver_parameters()
    assert params['st_type'] == 'sinvert' and params['eps_target'] == 2.0 and params['eps_max_it'] == 50
    assert params['st_pc_factor_mat_solver_type'] == 'mumps' and 'eps_gen_non_hermitian' in params
    for bad in ({'nev': 0}, {'nev': 5, 'ncv': 3}, {'tolerance': 0.0}):
        with pytest.raises(ValueError):
            EigenConfig.from_parameters({'elliptic': {'eigen': bad}})


def test_eigen_request_margin():
    """SLEPc is asked for a margin beyond nev (repeated eigenvalues) with a large enough subspace."""
    config = EigenConfig.from_parameters({'elliptic': {'eigen': {'nev': 8}}})
    assert config.num_requested() == 12
    assert config.subspace_size() == 28
    assert EigenConfig.from_parameters({'elliptic': {'eigen': {'nev': 8, 'ncv': 40}}}).subspace_size() == 40
    # A configured ncv below the request is replaced by the default rule.
    assert EigenConfig.from_parameters({'elliptic': {'eigen': {'nev': 8, 'ncv': 10}}}).subspace_size() == 28
