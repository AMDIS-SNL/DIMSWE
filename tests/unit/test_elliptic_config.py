"""Unit tests for EllipticConfig validation and solver selection (TODO-elliptic §3, §4.6)."""
import copy

import pytest

from dimswe.elliptic import EllipticConfig, elliptic_solver_parameters
from dimswe.elliptic_problems import get_elliptic_problem
from dimswe.meshes import get_mesh_and_spaces


def base(elliptic_parameters, **elliptic):
    """2D curl-div k = 1 parameters with essential BCs, with elliptic-section overrides."""
    parameters = elliptic_parameters(2, 'curl-div', 1, 'essential', n=2, order=1)
    parameters['elliptic'].update(elliptic)
    return parameters


def test_valid_config(elliptic_parameters):
    """A valid section parses, with defaults for the optional entries."""
    config = EllipticConfig.from_parameters(base(elliptic_parameters))
    assert (config.dim, config.k, config.complex) == (2, 1, 'curl-div')
    assert config.harmonic_constraint == 'lagrange' and config.solver == 'elliptic-direct'
    assert config.error_quadrature_degree == 2 * 1 + 4
    assert config.quadrature_degree == 2 * 1 + 2  # always explicit (lagrange/nullspace consistency)
    assert config.registry is None and config.face_bcs is None
    # 1D/3D ignore the complex.
    config3 = EllipticConfig.from_parameters(elliptic_parameters(3, 'whatever', 2, 'natural', n=2, order=1))
    assert config3.complex is None


@pytest.mark.parametrize("elliptic", [
    {'form_degree': 3},                      # k > dim
    {'form_degree': -1},                     # k < 0
    {'form_degree': 1.0},                    # not an integer
    {'complex': None},                       # 2D needs a complex
    {'complex': 'div-curl'},                 # unknown complex
    {'harmonic_constraint': 'penalty'},
    {'harmonic_forms': 'sometimes'},
    {'boundary_data': 'random'},
    {'solver': 'cg'},
    {'quadrature_degree': 0},
    {'error_quadrature_degree': 2.5},
    {'lagrange_regularization': 0.0},
])
def test_invalid_elliptic_options(elliptic_parameters, elliptic):
    """Every invalid elliptic option raises ValueError."""
    with pytest.raises(ValueError):
        EllipticConfig.from_parameters(base(elliptic_parameters, **elliptic))


def test_missing_sections_and_degree(elliptic_parameters):
    """The elliptic section and form_degree are required."""
    parameters = base(elliptic_parameters)
    del parameters['elliptic']['form_degree']
    with pytest.raises(ValueError):
        EllipticConfig.from_parameters(parameters)
    parameters = base(elliptic_parameters)
    del parameters['elliptic']
    with pytest.raises(ValueError):
        EllipticConfig.from_parameters(parameters)


@pytest.mark.parametrize("change", [
    {'lump_mass': True},       # always rejected for elliptic problems
    {'family': 'P-'},          # not available yet (§4.9)
    {'order': 0},
])
def test_invalid_discretization(elliptic_parameters, change):
    """lump_mass, unsupported families and invalid orders are rejected."""
    parameters = base(elliptic_parameters)
    parameters['spatial-discretization'].update(change)
    with pytest.raises(ValueError):
        EllipticConfig.from_parameters(parameters)


def test_q_family_rejects_simplicial_cells(elliptic_parameters):
    """Family/cell mismatch: Q needs quadrilateral/hexahedral cells."""
    parameters = base(elliptic_parameters)
    parameters['mesh']['simplicial_cells'] = True
    with pytest.raises(ValueError):
        EllipticConfig.from_parameters(parameters)


def build_spaces(parameters):
    """Mesh and spaces for the parameters (problem used as the domain)."""
    return get_mesh_and_spaces(parameters, get_elliptic_problem(parameters))[1]


def test_boundary_conditions_parsed_with_spaces(elliptic_parameters):
    """With spaces, the registry and per-face BCs are attached."""
    parameters = elliptic_parameters(2, 'grad-curl', 1, 'partial-essential', n=3, order=1)
    config = EllipticConfig.from_parameters(parameters, build_spaces(parameters))
    assert list(config.registry) == ['bottom', 'top']
    assert config.face_bcs == {'bottom': 'essential', 'top': 'essential'}


@pytest.mark.parametrize("faces", [
    {'bottom': 'essential', 'top': 'essential', 'left': 'natural'},   # BC on a periodic face
    {'bottom': 'essential'},                                           # missing BC
])
def test_boundary_condition_errors(elliptic_parameters, faces):
    """BCs on periodic faces and missing BCs on bounded faces are rejected."""
    parameters = elliptic_parameters(2, 'grad-curl', 1, 'partial-essential', n=3, order=1)
    parameters['elliptic']['boundary_conditions'] = faces
    spaces = build_spaces(parameters)
    with pytest.raises(ValueError):
        EllipticConfig.from_parameters(parameters, spaces)


# ---------------------------------------------------------------------------------------------
# Solver selection
# ---------------------------------------------------------------------------------------------

def config_with(elliptic_parameters, **elliptic):
    """Config for solver-selection tests."""
    return EllipticConfig.from_parameters(base(elliptic_parameters, **elliptic))


def test_solver_lagrange_with_harmonic_forms(elliptic_parameters):
    """lagrange with harmonic forms: matrix-free GMRES with the scalar R blocks as one split."""
    params = elliptic_solver_parameters(config_with(elliptic_parameters), 2, 4, [2, 3])
    assert params['mat_type'] == 'matfree' and params['ksp_type'] == 'gmres'
    assert params['ksp_pc_side'] == 'right'
    assert params['pc_fieldsplit_0_fields'] == '0,1' and params['pc_fieldsplit_1_fields'] == '2,3'
    with pytest.raises(ValueError):
        elliptic_solver_parameters(config_with(elliptic_parameters, solver='elliptic-minres'), 2, 4, [2, 3])


def test_solver_nullspace_and_plain(elliptic_parameters):
    """nullspace upgrades direct to the singular variant; without harmonic forms plain LU."""
    nullspace = config_with(elliptic_parameters, harmonic_constraint='nullspace')
    params = elliptic_solver_parameters(nullspace, 1, 2)
    assert params['ksp_type'] == 'gmres' and params['ksp_pc_side'] == 'right' and params['pc_type'] == 'lu'
    plain = elliptic_solver_parameters(config_with(elliptic_parameters), 0, 2)
    assert plain['ksp_type'] == 'preonly' and plain['pc_type'] == 'lu'


def test_solver_minres_and_overrides(elliptic_parameters):
    """minres: per-block LU fieldsplit (or plain LU for one block); solver_options win."""
    config = config_with(elliptic_parameters, solver='elliptic-minres', harmonic_constraint='nullspace',
                         solver_options={'ksp_rtol': 1e-6})
    params = elliptic_solver_parameters(config, 1, 2)
    assert params['ksp_type'] == 'minres' and params['pc_type'] == 'fieldsplit'
    assert params['fieldsplit_0_pc_type'] == 'lu' and params['fieldsplit_1_pc_type'] == 'lu'
    assert params['ksp_rtol'] == 1e-6
    single = elliptic_solver_parameters(config, 1, 1)
    assert single['pc_type'] == 'lu' and 'pc_fieldsplit_type' not in single
    # The shared defaults are not modified by the overrides.
    from dimswe.parameters import elliptic_minres
    assert elliptic_minres['ksp_rtol'] == 1e-12 and 'pc_type' not in copy.deepcopy(elliptic_minres)
