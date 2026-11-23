"""Solver-parameter builders for the elliptic Hodge Laplacian problems (TODO-elliptic §4.12).

Pure Python (no Firedrake objects), so the option dictionaries are unit-testable. The block
structure of the system is [sigma (k > 0), u, R multipliers (lagrange with harmonic forms)]:

    -(mu sigma, tau) + (u, d tau) + (d sigma, v) + (kappa du, dv) + (c u, v)   (symmetric, indefinite)

Iterative solvers (names in ITERATIVE_SOLVERS):
- riesz-amg / riesz-gmg / riesz-pmg: MINRES with the block-diagonal Riesz-map preconditioner of
  Arnold-Falk-Winther, P = diag(R_{k-1}, R_k), R_{k-1} = (mu sigma, tau) + (d sigma, d tau) and
  R_k = ((1 + c) u, v) + (kappa du, dv) (dimswe.elliptic.riesz_form), split with an additive
  PETSc fieldsplit. The blocks are solved approximately:
  - riesz-amg: algebraic. H1: BoomerAMG (or GAMG); H(curl)/H(div): Hypre AMS/ADS where the
    installed Firedrake supports them (lowest-order N1curl/RT on simplices, see
    hypre_auxiliary_space), LU otherwise; L2: block Jacobi with exact LU of the (block-diagonal)
    DG mass.
  - riesz-gmg: geometric multigrid on the MeshHierarchy (mesh.refinement_levels >= 1), Chebyshev
    smoothing with Jacobi (H1) or vertex-star patches (ASMStarPC; ASMExtrudedStarPC on extruded
    meshes) for H(curl)/H(div), the AFW smoother; LU on the coarsest level.
  - riesz-pmg: p-multigrid (firedrake.PMGPC) down to degree 1 with the same smoothers and LU on
    the degree-1 level (order >= 2).
  With 'lagrange' and harmonic forms the system has R blocks; it is then solved matrix-free with
  MINRES, each non-R block preconditioned by firedrake.AssembledPC + its block solver and the R
  blocks by the identity (their Riesz block is sum_i p_i q_i). This works but the R rows are dense,
  so the 'nullspace' treatment is recommended for large problems.
- schur-mass (k = n only, mixed Poisson/Darcy): full Schur-complement fieldsplit on [sigma, u]
  with the 'selfp' approximation S ~ A11 - A10 diag(A00)^{-1} A01 (a diagonal-mass approximation),
  GMRES on the mass block and BoomerAMG on S, inside FGMRES.

block_solvers (elliptic.block_solvers) selects per-space variants: {'H1': 'hypre'|'gamg'|'lu',
'Hcurl': 'auto'|'lu', 'Hdiv': 'auto'|'lu', 'L2': 'bjacobi'|'lu'} ('auto' = the solver's default).
'lu' always forces an exact block solve; the other values only matter for riesz-amg.
"""
from dataclasses import dataclass, field
from typing import Optional

ITERATIVE_SOLVERS = ('riesz-amg', 'riesz-gmg', 'riesz-pmg', 'schur-mass')
SPACE_TYPES = ('H1', 'Hcurl', 'Hdiv', 'L2')
BLOCK_VARIANTS = {
    'H1': ('auto', 'hypre', 'gamg', 'lu'),
    'Hcurl': ('auto', 'lu'),
    'Hdiv': ('auto', 'lu'),
    'L2': ('auto', 'bjacobi', 'lu'),
}
DEFAULT_RTOL = 1e-10
DEFAULT_MAX_IT = 2000
LU = {'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}


def space_type(j, dim, complex):
    """Sobolev type of the j-form space: 'H1' (j = 0), 'L2' (j = dim), else 'Hcurl'/'Hdiv'.

    In 2D the 1-form space is Hcurl for the grad-curl complex and Hdiv for curl-div; in 3D
    1-forms are Hcurl and 2-forms Hdiv.
    """
    if j < 0 or j > dim:
        raise ValueError("form degree " + str(j) + " is out of range for dimension " + str(dim))
    if j == 0:
        return 'H1'
    if j == dim:
        return 'L2'
    if dim == 2:
        return 'Hcurl' if complex == 'grad-curl' else 'Hdiv'
    return 'Hcurl' if j == 1 else 'Hdiv'


def block_space_types(k, dim, complex):
    """Space types of the [sigma, u] blocks (sigma only for k > 0)."""
    types = [space_type(k - 1, dim, complex)] if k > 0 else []
    return types + [space_type(k, dim, complex)]


def hypre_auxiliary_space(stype, family, order, dim, cell):
    """'ams', 'ads' or None: where Hypre's auxiliary-space solvers are expected to work.

    Firedrake's HypreAMS/HypreADS build the discrete gradient/curl by interpolation and need
    lowest-order Nedelec/Raviart-Thomas elements on simplices: AMS for N1curl_1 (2D grad-curl and
    3D 1-forms), ADS for 3D RT_1. Everything else (higher order, Q/S families, 2D H(div)) returns
    None and gets the riesz-amg fallback (LU), or should use riesz-gmg/riesz-pmg.
    """
    if family != 'P-' or int(order) != 1 or cell not in ('triangle', 'tetrahedron'):
        return None
    if stype == 'Hcurl':
        return 'ams'
    if stype == 'Hdiv' and dim == 3:
        return 'ads'
    return None


@dataclass
class SolverContext:
    """Everything the builder needs to know about the problem (pure data)."""
    name: str
    k: int
    dim: int
    complex: Optional[str] = None
    family: str = 'Q'
    order: int = 1
    cell: str = 'quadrilateral'
    harmonic_constraint: str = 'nullspace'
    num_harmonic: int = 0
    refinement_levels: int = 0
    block_solvers: dict = field(default_factory=dict)
    rtol: float = DEFAULT_RTOL
    max_it: int = DEFAULT_MAX_IT

    @property
    def use_multiplier(self):
        """True if the system has R multiplier blocks (lagrange with harmonic forms)."""
        return self.harmonic_constraint == 'lagrange' and self.num_harmonic > 0


@dataclass
class SolverSetup:
    """Result of get_elliptic_solver_parameters.

    parameters: PETSc options for the solve.
    preconditioner: which preconditioner form the model must build: 'riesz', 'riesz+R' (Riesz
        blocks plus sum_i p_i q_i for the R blocks) or None (the operator itself).
    block_types: space types of the non-R blocks, in block order.
    """
    parameters: dict
    preconditioner: Optional[str]
    block_types: list


def validate_block_solvers(block_solvers):
    """Check elliptic.block_solvers: {space type: variant}; returns a clean dict."""
    block_solvers = dict(block_solvers or {})
    for stype, variant in block_solvers.items():
        if stype not in BLOCK_VARIANTS:
            raise ValueError("block_solvers key must be one of " + str(list(BLOCK_VARIANTS)) + ", got " + repr(stype))
        if variant not in BLOCK_VARIANTS[stype]:
            raise ValueError("block_solvers." + stype + " must be one of " + str(list(BLOCK_VARIANTS[stype]))
                             + ", got " + repr(variant))
    return block_solvers


def validate_solver_choice(name, k, dim, order, refinement_levels, block_solvers=None):
    """Raise ValueError for iterative-solver choices that cannot work for the problem.

    riesz-gmg needs mesh.refinement_levels >= 1, riesz-pmg needs order >= 2, schur-mass needs
    k = dim (the mixed Poisson/Darcy structure [sigma, u] with no stiffness block).
    """
    validate_block_solvers(block_solvers)
    if name not in ITERATIVE_SOLVERS:
        return
    if name == 'riesz-gmg' and int(refinement_levels) < 1:
        raise ValueError("solver riesz-gmg needs a mesh hierarchy: set mesh.refinement_levels >= 1")
    if name == 'riesz-pmg' and int(order) < 2:
        raise ValueError("solver riesz-pmg needs spatial-discretization.order >= 2 (it coarsens to degree 1)")
    if name == 'schur-mass' and k != dim:
        raise ValueError("solver schur-mass is the mixed Poisson/Darcy solver: it needs form_degree = dim")


def _prefixed(prefix, options):
    """{prefix + key: value} for every option."""
    return {prefix + key: value for key, value in options.items()}


def _star_pc(cell):
    """Vertex-star patch smoother (ASMExtrudedStarPC on extruded meshes)."""
    return 'firedrake.ASMExtrudedStarPC' if cell in ('extruded', 'extruded2d') else 'firedrake.ASMStarPC'


def _multigrid_options(stype, cell, prefix):
    """Chebyshev-smoothed multigrid (mg or pmg_mg) with an LU coarse solve, as PC options."""
    options = {'levels_ksp_type': 'chebyshev', 'levels_ksp_max_it': 2,
               'coarse_ksp_type': 'preonly', 'coarse_pc_type': 'lu',
               'coarse_pc_factor_mat_solver_type': 'mumps'}
    if stype == 'H1':
        options['levels_pc_type'] = 'jacobi'
    else:
        options['levels_pc_type'] = 'python'
        options['levels_pc_python_type'] = _star_pc(cell)
        # Vertex stars with exact (LU) patch solves, set explicitly rather than relying on the
        # patch preconditioner's defaults.
        star = 'levels_pc_estar_' if cell in ('extruded', 'extruded2d') else 'levels_pc_star_'
        options[star + 'construct_dim'] = 0
        options[star + 'sub_sub_pc_type'] = 'lu'
    return _prefixed(prefix, options)


def block_pc_options(stype, ctx):
    """PC options (no ksp_type) for one Riesz block of space type stype under solver ctx.name."""
    variant = ctx.block_solvers.get(stype, 'auto')
    if variant == 'lu':
        return dict(LU)
    if stype == 'L2':
        # The DG/DPC Riesz block is a mass matrix (d of an n-form is 0): block diagonal per cell.
        return {'pc_type': 'bjacobi', 'sub_pc_type': 'lu'}
    if ctx.name == 'riesz-amg':
        if stype == 'H1':
            return {'pc_type': 'gamg'} if variant == 'gamg' else {'pc_type': 'hypre', 'pc_hypre_type': 'boomeramg'}
        aux = hypre_auxiliary_space(stype, ctx.family, ctx.order, ctx.dim, ctx.cell)
        if aux == 'ams':
            return {'pc_type': 'python', 'pc_python_type': 'firedrake.HypreAMS'}
        if aux == 'ads':
            return {'pc_type': 'python', 'pc_python_type': 'firedrake.HypreADS'}
        return dict(LU)  # no supported auxiliary-space solver: exact block solve
    if ctx.name == 'riesz-gmg':
        return _on_riesz_block({'pc_type': 'mg', **_multigrid_options(stype, ctx.cell, 'mg_')}, ctx)
    if ctx.name == 'riesz-pmg':
        options = {'pc_type': 'python', 'pc_python_type': 'firedrake.PMGPC', 'pmg_mg_coarse_degree': 1}
        options.update(_multigrid_options(stype, ctx.cell, 'pmg_mg_'))
        return _on_riesz_block(options, ctx)
    raise ValueError("no block solver for " + repr(ctx.name))


def _on_riesz_block(options, ctx):
    """Run a rediscretizing multigrid (PCMG on a hierarchy, PMGPC) on the Riesz block itself.

    Firedrake's multigrid rebuilds the coarse operators by rediscretizing the *system* form of the
    split block, not the preconditioner form aP. Inside a fieldsplit the system form is the block of
    the indefinite Hodge operator (-(mu sigma, tau) for sigma, the singular kappa (du, dv) for u),
    so the coarse levels approximate the wrong operator and the iteration count grows with every
    level (measured: 20, 64, 138 over three refinements, against 6 with exact block solves).
    firedrake.AssembledPC makes the block's Pmat (the Riesz block) the operator of an inner PC, so
    the multigrid is built on the Riesz map at every level. The 'lagrange' (matrix-free) path
    already wraps every block in AssembledPC, so there the options are used unchanged.
    """
    if ctx.use_multiplier:
        return options
    wrapped = {'pc_type': 'python', 'pc_python_type': 'firedrake.AssembledPC'}
    wrapped.update(_prefixed('assembled_', options))
    return wrapped


def _schur_mass_parameters(ctx):
    """Full Schur complement on [sigma, u] with the selfp (diagonal-mass) approximation."""
    return {
        'ksp_type': 'fgmres', 'ksp_rtol': ctx.rtol, 'ksp_atol': 1e-14, 'ksp_max_it': ctx.max_it,
        'pc_type': 'fieldsplit', 'pc_fieldsplit_type': 'schur',
        'pc_fieldsplit_schur_fact_type': 'full', 'pc_fieldsplit_schur_precondition': 'selfp',
        'pc_fieldsplit_0_fields': '0', 'pc_fieldsplit_1_fields': '1',
        # A00 = -(mu sigma, tau) is negative definite: GMRES (CG/MINRES would reject it) + Jacobi.
        'fieldsplit_0_ksp_type': 'gmres', 'fieldsplit_0_ksp_rtol': 1e-12, 'fieldsplit_0_pc_type': 'jacobi',
        'fieldsplit_1_ksp_type': 'preonly', 'fieldsplit_1_pc_type': 'hypre', 'fieldsplit_1_pc_hypre_type': 'boomeramg',
    }


def get_elliptic_solver_parameters(ctx):
    """Solver parameters and preconditioner kind for an iterative solver (see the module docstring).

    Returns a SolverSetup. Raises ValueError for invalid choices (validate_solver_choice, and
    schur-mass with R multiplier blocks).
    """
    if ctx.name not in ITERATIVE_SOLVERS:
        raise ValueError("unknown iterative solver " + repr(ctx.name) + " (known: " + str(list(ITERATIVE_SOLVERS)) + ")")
    ctx.block_solvers = validate_block_solvers(ctx.block_solvers)
    validate_solver_choice(ctx.name, ctx.k, ctx.dim, ctx.order, ctx.refinement_levels, ctx.block_solvers)
    types = block_space_types(ctx.k, ctx.dim, ctx.complex)
    if ctx.name == 'schur-mass':
        if ctx.use_multiplier:
            raise ValueError("schur-mass does not support R multiplier blocks; use harmonic_constraint: nullspace")
        return SolverSetup(_schur_mass_parameters(ctx), None, types)

    params = {'ksp_type': 'minres', 'ksp_rtol': ctx.rtol, 'ksp_atol': 1e-14, 'ksp_max_it': ctx.max_it}
    if ctx.use_multiplier:
        # Matrix-free MINRES; non-R blocks preconditioned by AssembledPC + their block solver,
        # all R blocks together by the identity (their aP block is sum_i p_i q_i).
        nblocks = len(types)
        r_fields = ','.join(str(i) for i in range(nblocks, nblocks + ctx.num_harmonic))
        params.update({'mat_type': 'matfree', 'pmat_type': 'matfree',
                       'pc_type': 'fieldsplit', 'pc_fieldsplit_type': 'additive'})
        for i, stype in enumerate(types):
            prefix = 'fieldsplit_' + str(i) + '_'
            params['pc_fieldsplit_' + str(i) + '_fields'] = str(i)
            params.update({prefix + 'ksp_type': 'preonly', prefix + 'pc_type': 'python',
                           prefix + 'pc_python_type': 'firedrake.AssembledPC'})
            params.update(_prefixed(prefix + 'assembled_', block_pc_options(stype, ctx)))
        last = 'fieldsplit_' + str(nblocks) + '_'
        params['pc_fieldsplit_' + str(nblocks) + '_fields'] = r_fields
        params.update({last + 'ksp_type': 'preonly', last + 'pc_type': 'none'})
        return SolverSetup(params, 'riesz+R', types)
    if len(types) == 1:
        params.update(block_pc_options(types[0], ctx))
        return SolverSetup(params, 'riesz', types)
    params.update({'pc_type': 'fieldsplit', 'pc_fieldsplit_type': 'additive'})
    for i, stype in enumerate(types):
        prefix = 'fieldsplit_' + str(i) + '_'
        params['pc_fieldsplit_' + str(i) + '_fields'] = str(i)
        params[prefix + 'ksp_type'] = 'preonly'
        params.update(_prefixed(prefix, block_pc_options(stype, ctx)))
    return SolverSetup(params, 'riesz', types)


def algebraic_block_pc_options(stype):
    """Algebraic block solvers usable without Firedrake context (e.g. inside SLEPc's ST)."""
    if stype == 'H1':
        return {'pc_type': 'hypre', 'pc_hypre_type': 'boomeramg'}
    if stype == 'L2':
        return {'pc_type': 'bjacobi', 'sub_pc_type': 'lu'}
    return dict(LU)


def eigen_inner_solver_options(k, dim, complex, rtol=1e-12, max_it=1000):
    """SLEPc ST options for an iterative shift-and-invert inner solve (prefix 'st_').

    GMRES on A - target M, preconditioned by the assembled Riesz matrix (set as the ST
    preconditioner matrix) with an additive fieldsplit of algebraic block solvers (the fields are
    passed as index sets, since SLEPc's KSP has no Firedrake DM). Returns (options, block_types).
    """
    types = block_space_types(k, dim, complex)
    options = {'st_ksp_type': 'gmres', 'st_ksp_rtol': rtol, 'st_ksp_atol': 1e-14, 'st_ksp_max_it': max_it}
    if len(types) == 1:
        options.update(_prefixed('st_', algebraic_block_pc_options(types[0])))
        return options, types
    options.update({'st_pc_type': 'fieldsplit', 'st_pc_fieldsplit_type': 'additive'})
    for i, stype in enumerate(types):
        prefix = 'st_fieldsplit_' + str(i) + '_'
        options[prefix + 'ksp_type'] = 'preonly'
        options.update(_prefixed(prefix, algebraic_block_pc_options(stype)))
    return options, types
