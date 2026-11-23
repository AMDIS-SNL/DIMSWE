"""Elliptic problems on disks, annuli and balls (TODO-elliptic §4.14 stage 2, and the §4.13 annulus).

- RadialProblem MMS convergence on the disk, annulus (quads and the natural harmonic 1-form) and
  ball, with inhomogeneous exact boundary data;
- the annulus has one harmonic 1-form for natural BCs (and for essential BCs in curl-div), found
  automatically by the numerical path (harmonic_forms: auto on a non-box mesh);
- box problems are rejected on radial meshes.
"""
import math

import pytest
from firedrake import COMM_WORLD

from dimswe.elliptic import EllipticProblem

pytestmark = pytest.mark.skipif(COMM_WORLD.size > 1, reason="numerical harmonic forms use serial Betti numbers")


def radial_parameters(mesh_type, k, complex, bcs, order=2, simplicial=True, problem='radial', **mesh):
    """Elliptic parameters on a radial mesh."""
    family = 'P-' if simplicial else 'Q'
    mesh_cfg = {'type': mesh_type, 'simplicial_cells': simplicial}
    mesh_cfg.update(mesh)
    return {
        'mesh': mesh_cfg,
        'spatial-discretization': {'family': family, 'order': order, 'lump_mass': False},
        'model': {'type': 'elliptic'},
        'elliptic': {'form_degree': k, 'complex': complex, 'boundary_conditions': bcs, 'boundary_data': 'exact',
                     'harmonic_forms': 'auto', 'harmonic_constraint': 'nullspace', 'solver': 'elliptic-direct'},
        'elliptic-problem': {'name': problem},
        'output': {'loglevel': 0},
    }


def solve(parameters):
    """Solve with statistics."""
    model = EllipticProblem(parameters)
    model.solve()
    model.create_statistics()
    model.compute_statistics()
    return model


def rate(errors, hs):
    """Observed order between two resolutions."""
    return math.log(errors[0] / errors[1]) / math.log(hs[0] / hs[1])


def expected_u_rate(order, k):
    """P-/Q rates (as in the box tests): order + 1 for k = 0, order otherwise."""
    return order + 1 if k == 0 else order


@pytest.mark.parametrize("k, complex, bcs", [
    (0, 'grad-curl', {'outer': 'essential'}), (0, 'grad-curl', {'outer': 'natural'}),
    (1, 'curl-div', {'outer': 'natural'}), (2, 'curl-div', {'outer': 'essential'}),
])
def test_disk_mms(k, complex, bcs):
    """MMS on the unit disk (polygonal boundary, so order 1 and coarse refinements)."""
    errors = []
    for refinement in (2, 3):
        errors.append(solve(radial_parameters('unit-disk', k, complex, bcs, order=1,
                                              refinement=refinement)).get_statistics()['u_error'][0])
    # Halving h per refinement level; the polygonal boundary limits the rate to <= 2.
    assert rate(errors, (2.0, 1.0)) >= min(expected_u_rate(1, k), 2) - 0.4, errors


@pytest.mark.parametrize("k, complex, bcs", [
    (0, 'grad-curl', {'inner': 'essential', 'outer': 'natural'}),
    (1, 'grad-curl', {'inner': 'natural', 'outer': 'natural'}),
    (1, 'curl-div', {'inner': 'essential', 'outer': 'essential'}),
    (2, 'curl-div', {'inner': 'natural', 'outer': 'essential'}),
])
def test_annulus_mms(k, complex, bcs):
    """MMS on the quadrilateral annulus (curved cells approximated by straight edges)."""
    errors = []
    for nr in (4, 8):
        model = solve(radial_parameters('annulus', k, complex, bcs, order=1, simplicial=False, nr=nr, nt=4 * nr))
        errors.append(model.get_statistics()['u_error'][0])
    assert rate(errors, (2.0, 1.0)) >= min(expected_u_rate(1, k), 2) - 0.4, errors


@pytest.mark.parametrize("complex, bcs, count", [
    ('grad-curl', {'inner': 'natural', 'outer': 'natural'}, 1),
    ('curl-div', {'inner': 'natural', 'outer': 'natural'}, 1),
    ('curl-div', {'inner': 'essential', 'outer': 'essential'}, 1),
    ('grad-curl', {'inner': 'essential', 'outer': 'natural'}, 0),
])
def test_annulus_harmonic_one_form(complex, bcs, count):
    """The annulus has one harmonic 1-form for natural (or for both-essential) BCs, found numerically."""
    model = solve(radial_parameters('annulus', 1, complex, bcs, order=2, simplicial=False, nr=4, nt=16))
    stats = model.get_statistics()
    assert model.num_harmonic == count
    if count:
        assert stats['harmonic_gap_ratio'][0] > 1e6 and stats['harmonic_orthogonality'][0] < 1e-8


@pytest.mark.slow
def test_ball_mms():
    """k = 0 Dirichlet on the unit ball converges."""
    errors = [solve(radial_parameters('unit-ball', 0, None, {'outer': 'essential'}, order=1,
                                      refinement=r)).get_statistics()['u_error'][0] for r in (1, 2)]
    assert errors[1] < 0.5 * errors[0], errors


def test_box_problem_rejected_on_radial_mesh():
    """Normalized-coordinate problems need a box; the error names the radial problem."""
    with pytest.raises(ValueError, match="radial"):
        EllipticProblem(radial_parameters('unit-disk', 0, 'grad-curl', {'outer': 'essential'}, problem='sine'))
