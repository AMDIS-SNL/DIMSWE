"""Unit tests for dimswe/elliptic_solvers.py: the solver-parameter builder (pure Python)."""
import pytest

from dimswe.elliptic_solvers import (ITERATIVE_SOLVERS, SolverContext, block_space_types, eigen_inner_solver_options,
                                     get_elliptic_solver_parameters, hypre_auxiliary_space, space_type,
                                     validate_block_solvers, validate_solver_choice)

COMPLEX_FOR_DIM = {1: [None], 2: ['grad-curl', 'curl-div'], 3: [None]}
ALL_CASES = [(dim, cx, k) for dim in (1, 2, 3) for cx in COMPLEX_FOR_DIM[dim] for k in range(dim + 1)]


def ctx(name, k, dim, complex=None, **kwargs):
    """Solver context with defaults suited to name."""
    defaults = {'order': 2, 'refinement_levels': 2}
    defaults.update(kwargs)
    return SolverContext(name=name, k=k, dim=dim, complex=complex, **defaults)


def test_space_types():
    """H1 / Hcurl / Hdiv / L2 per form degree and complex."""
    assert space_type(0, 2, 'grad-curl') == 'H1' and space_type(2, 2, 'curl-div') == 'L2'
    assert space_type(1, 2, 'grad-curl') == 'Hcurl' and space_type(1, 2, 'curl-div') == 'Hdiv'
    assert [space_type(j, 3, None) for j in range(4)] == ['H1', 'Hcurl', 'Hdiv', 'L2']
    assert [space_type(j, 1, None) for j in range(2)] == ['H1', 'L2']
    assert block_space_types(0, 2, 'grad-curl') == ['H1']
    assert block_space_types(2, 3, None) == ['Hcurl', 'Hdiv']
    with pytest.raises(ValueError):
        space_type(3, 2, 'grad-curl')


def test_hypre_auxiliary_space():
    """AMS for lowest-order N1curl on simplices, ADS for 3D RT_1, None otherwise."""
    assert hypre_auxiliary_space('Hcurl', 'P-', 1, 2, 'triangle') == 'ams'
    assert hypre_auxiliary_space('Hcurl', 'P-', 1, 3, 'tetrahedron') == 'ams'
    assert hypre_auxiliary_space('Hdiv', 'P-', 1, 3, 'tetrahedron') == 'ads'
    assert hypre_auxiliary_space('Hdiv', 'P-', 1, 2, 'triangle') is None       # 2D H(div)
    assert hypre_auxiliary_space('Hcurl', 'P-', 2, 3, 'tetrahedron') is None   # high order
    assert hypre_auxiliary_space('Hcurl', 'Q', 1, 2, 'quadrilateral') is None  # tensor-product


@pytest.mark.parametrize("name", ['riesz-amg', 'riesz-gmg', 'riesz-pmg'])
@pytest.mark.parametrize("dim, cx, k", ALL_CASES)
def test_riesz_fieldsplit_layout(name, dim, cx, k):
    """MINRES with one fieldsplit per present block, in block order; single blocks are unsplit."""
    setup = get_elliptic_solver_parameters(ctx(name, k, dim, cx))
    params = setup.parameters
    assert params['ksp_type'] == 'minres' and setup.preconditioner == 'riesz'
    types = block_space_types(k, dim, cx)
    assert setup.block_types == types
    if len(types) == 1:
        assert 'pc_fieldsplit_type' not in params
        return
    assert params['pc_type'] == 'fieldsplit' and params['pc_fieldsplit_type'] == 'additive'
    for i, stype in enumerate(types):
        assert params['pc_fieldsplit_' + str(i) + '_fields'] == str(i)
        prefix = 'fieldsplit_' + str(i) + '_'
        assert params[prefix + 'ksp_type'] == 'preonly'
        if stype == 'L2':
            assert params[prefix + 'pc_type'] == 'bjacobi' and params[prefix + 'sub_pc_type'] == 'lu'
    assert 'fieldsplit_' + str(len(types)) + '_ksp_type' not in params


def test_riesz_amg_block_solvers():
    """BoomerAMG on H1, AMS/ADS where supported, LU fallback otherwise, bjacobi on L2."""
    params = get_elliptic_solver_parameters(ctx('riesz-amg', 1, 2, 'grad-curl', family='P-', order=1,
                                                cell='triangle')).parameters
    assert params['fieldsplit_0_pc_hypre_type'] == 'boomeramg'
    assert params['fieldsplit_1_pc_python_type'] == 'firedrake.HypreAMS'
    params = get_elliptic_solver_parameters(ctx('riesz-amg', 2, 3, family='P-', order=1, cell='tetrahedron')).parameters
    assert params['fieldsplit_0_pc_python_type'] == 'firedrake.HypreAMS'
    assert params['fieldsplit_1_pc_python_type'] == 'firedrake.HypreADS'
    params = get_elliptic_solver_parameters(ctx('riesz-amg', 1, 2, 'curl-div', cell='quadrilateral')).parameters
    assert params['fieldsplit_1_pc_type'] == 'lu'          # RTCF: no supported auxiliary-space solver
    params = get_elliptic_solver_parameters(ctx('riesz-amg', 0, 2, 'grad-curl', block_solvers={'H1': 'gamg'})).parameters
    assert params['pc_type'] == 'gamg'


def test_riesz_gmg_and_pmg_smoothers():
    """Multigrid runs on the Riesz block (AssembledPC); Jacobi on H1 levels, vertex-star patches
    (extruded variant on extruded meshes) otherwise."""
    params = get_elliptic_solver_parameters(ctx('riesz-gmg', 1, 2, 'curl-div', cell='quadrilateral')).parameters
    for i in (0, 1):
        assert params['fieldsplit_' + str(i) + '_pc_python_type'] == 'firedrake.AssembledPC'
    a0, a1 = 'fieldsplit_0_assembled_', 'fieldsplit_1_assembled_'
    assert params[a0 + 'pc_type'] == 'mg' and params[a0 + 'mg_levels_pc_type'] == 'jacobi'
    assert params[a1 + 'mg_levels_pc_python_type'] == 'firedrake.ASMStarPC'
    assert params[a1 + 'mg_levels_pc_star_construct_dim'] == 0
    assert params[a1 + 'mg_levels_pc_star_sub_sub_pc_type'] == 'lu'   # exact patch solves
    assert params[a1 + 'mg_coarse_pc_type'] == 'lu'
    params = get_elliptic_solver_parameters(ctx('riesz-gmg', 1, 3, cell='extruded')).parameters
    assert params[a1 + 'mg_levels_pc_python_type'] == 'firedrake.ASMExtrudedStarPC'
    assert params[a1 + 'mg_levels_pc_estar_sub_sub_pc_type'] == 'lu'
    params = get_elliptic_solver_parameters(ctx('riesz-pmg', 1, 2, 'grad-curl')).parameters
    assert params['fieldsplit_1_pc_python_type'] == 'firedrake.AssembledPC'
    assert params[a1 + 'pc_python_type'] == 'firedrake.PMGPC'
    assert params[a1 + 'pmg_mg_coarse_degree'] == 1
    assert params[a1 + 'pmg_mg_levels_pc_python_type'] == 'firedrake.ASMStarPC'
    # Single-block (k = 0) problems are wrapped too; the lagrange path keeps its own AssembledPC.
    assert get_elliptic_solver_parameters(ctx('riesz-gmg', 0, 2, 'grad-curl')).parameters['pc_python_type'] == \
        'firedrake.AssembledPC'
    params = get_elliptic_solver_parameters(ctx('riesz-gmg', 1, 2, 'curl-div', harmonic_constraint='lagrange',
                                                num_harmonic=2)).parameters
    assert params['fieldsplit_1_assembled_pc_type'] == 'mg'
    assert 'fieldsplit_1_assembled_assembled_pc_type' not in params


def test_lu_override_and_rtol():
    """block_solvers: lu forces exact block solves; rtol and max_it are passed through."""
    params = get_elliptic_solver_parameters(ctx('riesz-gmg', 1, 2, 'grad-curl', block_solvers={'Hcurl': 'lu'},
                                                rtol=1e-8, max_it=77)).parameters
    assert params['fieldsplit_1_pc_type'] == 'lu' and params['ksp_rtol'] == 1e-8 and params['ksp_max_it'] == 77


def test_riesz_with_lagrange_multipliers():
    """lagrange with m harmonic forms: matrix-free MINRES, AssembledPC per block, R blocks together."""
    setup = get_elliptic_solver_parameters(ctx('riesz-amg', 1, 2, 'curl-div', harmonic_constraint='lagrange',
                                               num_harmonic=2))
    params = setup.parameters
    assert setup.preconditioner == 'riesz+R' and params['mat_type'] == 'matfree'
    assert params['pc_fieldsplit_0_fields'] == '0' and params['pc_fieldsplit_1_fields'] == '1'
    assert params['pc_fieldsplit_2_fields'] == '2,3'
    assert params['fieldsplit_0_pc_python_type'] == 'firedrake.AssembledPC'
    assert params['fieldsplit_0_assembled_pc_hypre_type'] == 'boomeramg'
    assert params['fieldsplit_2_pc_type'] == 'none'
    # nullspace (or no harmonic forms) keeps the assembled path
    assert get_elliptic_solver_parameters(ctx('riesz-amg', 1, 2, 'curl-div', num_harmonic=2)).preconditioner == 'riesz'


def test_schur_mass():
    """k = n only: full Schur fieldsplit with selfp; rejects R blocks."""
    setup = get_elliptic_solver_parameters(ctx('schur-mass', 2, 2, 'curl-div'))
    params = setup.parameters
    assert setup.preconditioner is None and params['pc_fieldsplit_type'] == 'schur'
    assert params['pc_fieldsplit_schur_precondition'] == 'selfp' and params['ksp_type'] == 'fgmres'
    with pytest.raises(ValueError):
        get_elliptic_solver_parameters(ctx('schur-mass', 2, 2, 'curl-div', harmonic_constraint='lagrange',
                                           num_harmonic=1))


@pytest.mark.parametrize("name, k, dim, order, levels", [
    ('riesz-gmg', 1, 2, 2, 0),     # no hierarchy
    ('riesz-pmg', 1, 2, 1, 2),     # nothing to coarsen
    ('schur-mass', 1, 2, 2, 2),    # not k = n
])
def test_invalid_choices(name, k, dim, order, levels):
    """Choices that cannot work are rejected."""
    with pytest.raises(ValueError):
        validate_solver_choice(name, k, dim, order, levels)


def test_validate_block_solvers():
    """Unknown space types and variants are rejected."""
    assert validate_block_solvers({'H1': 'gamg'}) == {'H1': 'gamg'}
    for bad in ({'H2': 'lu'}, {'Hcurl': 'boomeramg'}):
        with pytest.raises(ValueError):
            validate_block_solvers(bad)
    with pytest.raises(ValueError):
        get_elliptic_solver_parameters(ctx('cg', 0, 2, 'grad-curl'))
    assert set(ITERATIVE_SOLVERS) == {'riesz-amg', 'riesz-gmg', 'riesz-pmg', 'schur-mass'}


def test_eigen_inner_solver_options():
    """st_-prefixed GMRES with an algebraic fieldsplit for mixed problems."""
    options, types = eigen_inner_solver_options(1, 2, 'curl-div', rtol=1e-9)
    assert types == ['H1', 'Hdiv'] and options['st_ksp_type'] == 'gmres' and options['st_ksp_rtol'] == 1e-9
    assert options['st_pc_type'] == 'fieldsplit' and options['st_fieldsplit_0_pc_type'] == 'hypre'
    assert options['st_fieldsplit_1_pc_type'] == 'lu'
    options, types = eigen_inner_solver_options(0, 2, 'grad-curl')
    assert types == ['H1'] and options['st_pc_type'] == 'hypre'
