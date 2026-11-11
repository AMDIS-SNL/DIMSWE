"""Repeated assembly of the same form into a preallocated tensor (HVP performance plan, item B0).

A bare ``firedrake.assemble(form)`` redoes the per-call setup work every time: it processes the form
symbolically, builds the parallel loops and allocates a new output. For the large adjoint and
second-order adjoint increment forms of the RK steps this cost 0.1-0.5 s per call, against a few
milliseconds for the actual kernels (docs/hvp_performance_plan.md, section 2.3). The stage
solvers do not pay it, because a LinearVariationalSolver keeps a persistent assembler for its
right-hand side.

PersistentAssembler does the same for forms that are assembled outside a solver. The persistent
assembly API differs between Firedrake versions, so it tries, in order:

1. ``firedrake.assemble.get_assembler(form).assemble(tensor=...)``, recent Firedrake;
2. ``firedrake.assemble.create_assembly_callable(form, tensor=...)``, older Firedrake;
3. ``firedrake.assemble(form, tensor=...)``, which at least reuses the output.

Each candidate is checked against a plain ``assemble(form)`` on its first ``validate_calls``
uses. Values change between calls, so this also checks that the persistent assembler sees
updated coefficients and does not accumulate into the old tensor. A candidate that disagrees or
raises is dropped for the next one. The decision is made collectively over the mesh
communicator, so all ranks always use the same strategy.

The second half of the module removes the per-solver setup of the explicit RK stage solves
(plan item C1, docs/hvp_performance_plan.md section 2.3, first-call setup): AffineSolver
solves an affine residual with one assembly and a MassSolver whose factorization is shared by
every solve of the same space (shared_mass_solver). PersistentProjector uses the same pieces for
repeated projections of fixed expressions, as in the initial-condition map.
"""

import itertools

import numpy as np

from .numpy_helpers import create_flattened_numpy_arr_from_mixed_function

# relative tolerance of the validation against a plain assemble (both run the same kernels, so
# the results normally agree to round-off)
VALIDATION_RTOL = 1e-10
# factor of the probe in AffineSolver's check: a power of two, so scaling the solution by it and
# back is exact, and large, so the check is not swamped by the assembly round-off of r0
PROBE_SCALE = 2.0 ** 13


class FiredrakeAssemblyBackend():
    """The Firedrake operations PersistentAssembler needs, kept separate so tests can replace them."""

    def assemble(self, form):
        """Assemble form with a fresh output, the slow but reference way."""
        from firedrake import assemble
        return assemble(form)

    def candidates(self, form, tensor):
        """Return (name, factory) pairs; each factory returns a callable that assembles into tensor.

        Factories may raise if their API is missing; PersistentAssembler then moves on.
        """
        from firedrake import assemble

        def from_get_assembler():
            from firedrake.assemble import get_assembler
            assembler = get_assembler(form)
            return lambda: assembler.assemble(tensor=tensor)

        def from_assembly_callable():
            from firedrake.assemble import create_assembly_callable
            return create_assembly_callable(form, tensor=tensor)

        def from_assemble_with_tensor():
            return lambda: assemble(form, tensor=tensor)

        return [("get_assembler", from_get_assembler), ("create_assembly_callable", from_assembly_callable),
                ("assemble(tensor=)", from_assemble_with_tensor)]

    def zero(self, tensor):
        """Zero the output tensor in place."""
        tensor.dat.zero()

    def flatten(self, tensor):
        """Return a rank-local flattened copy of the tensor values."""
        return create_flattened_numpy_arr_from_mixed_function(tensor)

    def scale(self, func, factor):
        """Multiply the values of a Function in place (all components, owned dofs; halos follow)."""
        with func.dat.vec as vec:
            vec.scale(factor)

    def _comm(self, tensor):
        """Return the mesh communicator of a Function or Cofunction (COMM_WORLD if it has none)."""
        try:
            return tensor.function_space().mesh().comm
        except AttributeError:
            from mpi4py import MPI
            return MPI.COMM_WORLD

    def any_rank(self, tensor, flag):
        """Return True if flag is True on any rank of the tensor's mesh communicator."""
        return bool(self._comm(tensor).allreduce(bool(flag), op=_mpi_lor()))

    def global_max(self, tensor, value):
        """Return the maximum of value over the ranks of the tensor's mesh communicator."""
        from mpi4py import MPI
        return float(self._comm(tensor).allreduce(float(value), op=MPI.MAX))


def _mpi_lor():
    """Return MPI.LOR (imported lazily, so the module imports without mpi4py)."""
    from mpi4py import MPI
    return MPI.LOR


class PersistentAssembler():
    """Assemble one fixed form repeatedly into a reused output tensor (see the module doc).

    Call assemble_flat() whenever the form's coefficients have changed. It returns a new rank-local
    flattened numpy array of the assembled values, as
    create_flattened_numpy_arr_from_mixed_function(assemble(form)) did, so callers may keep or
    accumulate the result while the tensor is reused.
    """

    def __init__(self, form, validate_calls=2, backend=None):
        self.form = form
        self.validate_calls = validate_calls
        self.backend = FiredrakeAssemblyBackend() if backend is None else backend
        self.tensor = None
        # name of the strategy in use, or None before the first call; 'assemble' means the
        # plain-assemble fallback after every candidate failed
        self.strategy = None
        self.calls = 0
        self._candidates = []
        self._run = None
        self._validated = 0

    def _next_candidate(self):
        """Switch to the next candidate whose factory works; fall back to plain assemble at the end."""
        while self._candidates:
            name, factory = self._candidates.pop(0)
            try:
                run = factory()
            except Exception:
                continue
            self.strategy, self._run, self._validated = name, run, 0
            return
        self.strategy, self._run = 'assemble', None

    def _run_candidate(self):
        """Assemble with the current candidate into the zeroed tensor; return the flat values or None on error."""
        try:
            self.backend.zero(self.tensor)
            self._run()
            return self.backend.flatten(self.tensor)
        except Exception:
            return None

    def assemble_tensor(self):
        """Assemble into the reused tensor and return it (no copy), or None if that is not possible yet.

        Only available once the tensor exists and the current candidate needs no further
        validation (validate_calls reached, or 0 as AffineSolver uses it); otherwise the caller
        uses assemble_flat. The tensor is overwritten by the next assembly.
        """
        if self.tensor is None or self._run is None or self._validated < self.validate_calls:
            return None
        self.calls += 1
        self.backend.zero(self.tensor)
        self._run()
        return self.tensor

    def assemble_flat(self):
        """Assemble the form at the current coefficient values and return a flattened copy."""
        self.calls += 1
        if self.tensor is None:
            # first call: the reference assembly also allocates the reusable tensor
            self.tensor = self.backend.assemble(self.form)
            self._candidates = list(self.backend.candidates(self.form, self.tensor))
            self._next_candidate()
            return self.backend.flatten(self.tensor)

        while self.strategy != 'assemble' and self._validated < self.validate_calls:
            reference = self.backend.flatten(self.backend.assemble(self.form))
            values = self._run_candidate()
            bad = values is None or values.shape != reference.shape or not np.allclose(
                values, reference, rtol=VALIDATION_RTOL, atol=VALIDATION_RTOL * (np.max(np.abs(reference), initial=0.0) + 1e-300))
            if self.backend.any_rank(self.tensor, bad):
                self._next_candidate()
                continue
            self._validated += 1
            return reference

        if self.strategy == 'assemble':
            return self.backend.flatten(self.backend.assemble(self.form))
        values = self._run_candidate()
        if values is None:
            raise RuntimeError('persistent assembly with ' + self.strategy + ' failed after it was validated')
        return values


def _set_flat_values(func, arr):
    """Write a rank-local flattened array (as returned by assemble_flat) into a (mixed) Function or Cofunction."""
    offset = 0
    for dat in func.dat.split:
        target = dat.data_wo
        target[...] = np.reshape(arr[offset:offset + target.size], target.shape)
        offset += target.size


def is_direct_solver(solver_parameters):
    """Return True if solver_parameters ask for a direct solve (preonly KSP with an LU or Cholesky PC).

    Only direct solves are exact enough for AffineSolver's residual check, so only they use the
    shared factored mass matrix; iterative solver parameters keep the LinearVariationalSolver.
    """
    return solver_parameters.get('ksp_type') == 'preonly' and solver_parameters.get('pc_type') in ('lu', 'cholesky')


def mass_solver_key(form, solver_parameters):
    """Return a hashable key under which equal mass matrices share one MassSolver.

    Two operator forms get the same matrix when they have the same signature (structure and
    elements), the same coefficients and constants (the signature renumbers them, so they are
    compared by count), the same argument spaces (compared by identity) and the same solver
    parameters.
    """
    coefficients = tuple(c.count() for c in form.coefficients())
    constants = tuple(c.count() for c in getattr(form, 'constants', lambda: ())())
    spaces = tuple(id(argument.function_space()) for argument in form.arguments())
    parameters = tuple(sorted((key, repr(value)) for key, value in solver_parameters.items()))
    return (form.signature(), coefficients, constants, spaces, parameters)


def shared_mass_solver(cache, form, solver_parameters, options_prefix='erk-mass'):
    """Return the MassSolver for the operator form from cache (a dict), creating it on first use.

    The cache is shared by every sub-integrator of a model (GeneralRK keeps it on the model), so
    all the stage, adjoint, tangent, second-order adjoint and increment solves of one space use
    a single assembled and factored matrix (plan item C1).
    """
    key = mass_solver_key(form, solver_parameters)
    solver = cache.get(key)
    if solver is None:
        solver = MassSolver(form, solver_parameters, options_prefix=options_prefix)
        cache[key] = solver
    return solver


# distinct options prefixes for the MassSolvers of one process
_MASS_SOLVER_IDS = itertools.count()


def flatten_solver_parameters(parameters, prefix=''):
    """Flatten nested solver parameters ({'mat_mumps': {'icntl_14': 50}}) to PETSc option names.

    Returns {'mat_mumps_icntl_14': 50, ...}; booleans become 'true' / 'false' and None an empty
    value (a flag option), as in Firedrake's own options handling.
    """
    flat = {}
    for key, value in parameters.items():
        name = prefix + str(key)
        if isinstance(value, dict):
            flat.update(flatten_solver_parameters(value, name + '_'))
        elif isinstance(value, bool):
            flat[name] = 'true' if value else 'false'
        elif value is None:
            flat[name] = ''
        else:
            flat[name] = value
    return flat


# mass_inverse modes of MassSolver (solver parameter 'mass_inverse', default 'auto'; kept in sync
# with workflows/optimize_params.py MASS_INVERSE_MODES by tests/unit/test_workflow_optimizer_tables.py)
MASS_INVERSE_MODES = ('auto', 'diagonal', 'cellwise', 'factor')
# a matrix is treated as diagonal if its off-diagonal part changes A v by at most this, relative
# to max|diag(A)| max|v|, for a random v (lumped masses have off-diagonal round-off only)
DIAGONAL_RTOL = 1e-12
# a cheap inverse (diagonal, cellwise, per field) must reproduce v from A v to this relative accuracy
INVERSE_RTOL = 1e-10


def petsc_solver_parameters(solver_parameters):
    """Return solver_parameters without the keys that only MassSolver reads (mass_inverse)."""
    return {key: value for key, value in solver_parameters.items() if key != 'mass_inverse'}


def is_l2_space(space):
    """Return True if a (non-mixed) function space is discontinuous (UFL Sobolev space L2), as DG is.

    Only then is its mass matrix block diagonal with one block per cell, so that the cellwise
    (Slate) inverse is its exact inverse.
    """
    sobolev = getattr(space.ufl_element(), 'sobolev_space', None)
    if callable(sobolev):
        # older UFL versions had a method
        sobolev = sobolev()
    return sobolev is not None and str(sobolev) == 'L2'


class MassSolver():
    """The inverse of an operator matrix M (in practice a mass matrix), shared by many solves (plan item C1).

    Every explicit RK stage equation, its adjoint, tangent-linear and second-order adjoint
    counterparts and the lambda / lambda_dot increments read M var = b with the x-space (or the
    aux-space) mass matrix. The inverse is set up once, on the first solve, and reused for every
    right-hand side after that. Before, each of the up to 8 x nstages + 2 LinearVariationalSolvers
    of a sub-integrator assembled and factored its own copy, which was about 16 s of LU
    factorizations in the first calls of the baseline job.

    How M is inverted is chosen on the first solve by the solver parameter mass_inverse:

    - 'auto' (the default) uses the cheapest exact inverse that applies, per field of a mixed
      space: pointwise division where the matrix is diagonal (a lumped mass, e.g. continuous
      spectral elements with GLL quadrature, lump_mass: true), the cellwise (Slate) inverse for
      discontinuous (L2, e.g. DG) fields, and a factorization for the remaining fields. Fields
      are only treated separately when the form does not couple them.
    - 'diagonal' only uses pointwise division (per field), 'cellwise' prefers the cellwise
      inverse for discontinuous fields (a diagonal field is still divided), and 'factor' always
      LU-factors the whole matrix, as before.

    A triangular solve with MUMPS costs about 4.5 ms per right-hand side on 4 ranks even for a
    diagonal matrix (about half of every steady evaluation in the profile of
    docs/hvp_performance_plan.md section 2.3); a division or a cellwise product is local and
    almost free. Every cheap inverse is checked once on its setup: for a random v it must give
    back v from A v to INVERSE_RTOL. If that fails (or raises), the whole matrix is factored
    instead and inverse_notes says why. method describes the inverse in use ('diagonal',
    'cellwise', 'fieldsplit(diagonal, cellwise, ksp)', 'ksp' or 'linear_solver').
    """

    def __init__(self, form, solver_parameters, options_prefix='erk-mass', petscmat=None):
        """form: the operator form (for one field of a mixed operator possibly None);
        solver_parameters: direct solver parameters, optionally with mass_inverse;
        petscmat: an already assembled PETSc matrix to invert instead of assembling form (the
        diagonal block of one field, see _try_fieldsplit).
        """
        self.form = form
        self._given_petscmat = petscmat
        self.solver_parameters = dict(solver_parameters)
        self.options_prefix = options_prefix
        self.mass_inverse = self.solver_parameters.get('mass_inverse', 'auto')
        if self.mass_inverse not in MASS_INVERSE_MODES:
            raise ValueError('mass_inverse must be one of ' + ', '.join(MASS_INVERSE_MODES) + ', not ' + repr(self.mass_inverse))
        # 'diagonal', 'cellwise', 'fieldsplit', 'ksp' (a bare PETSc KSP), 'linear_solver'
        # (Firedrake's LinearSolver, if the bare KSP failed; ksp_error says why) or None before
        # the first solve
        self.kind = None
        self.ksp_error = None
        # why cheaper inverses were not used (or were rejected by the check)
        self.inverse_notes = []
        self.ksp = None
        self.solver = None
        self.matrix = None
        self.inverse_diagonal = None
        self.inverse_matrix = None
        # for 'fieldsplit': [(PETSc IS of the field, MassSolver of its diagonal block)]
        self.components = None
        self.solves = 0

    @property
    def method(self):
        """A description of the inverse in use (None before the first solve)."""
        if self.kind == 'fieldsplit':
            return 'fieldsplit(' + ', '.join(component.method for _, component in self.components) + ')'
        return self.kind

    def _parameters(self):
        """Return (mat_type, PETSc solver parameters) for the matrix and its KSP."""
        parameters = petsc_solver_parameters(self.solver_parameters)
        mat_type = parameters.pop('mat_type', 'aij')
        # as Firedrake's solver defaults: a parallel direct solver for LU / Cholesky
        if parameters.get('pc_type') in ('lu', 'cholesky'):
            parameters.setdefault('pc_factor_mat_solver_type', 'mumps')
        return mat_type, parameters

    def _assemble_matrix(self):
        """Assemble the operator once, as a monolithic matrix (also for mixed spaces), which LU needs."""
        if self.matrix is None:
            from firedrake import assemble
            self.matrix = assemble(self.form, mat_type=self._parameters()[0])
        return self.matrix

    def _petscmat(self):
        """Return the PETSc matrix to invert: the given block, or the assembled form."""
        if self._given_petscmat is not None:
            return self._given_petscmat
        return self._assemble_matrix().petscmat

    def _row_space(self):
        """Return the function space of the test function (the rows of M), or None without a form."""
        return None if self.form is None else self.form.arguments()[0].function_space()

    def _mpi_comm(self):
        """Return the mpi4py communicator of the operator (of its mesh, or of the given matrix)."""
        if self._given_petscmat is not None:
            return self._given_petscmat.getComm().tompi4py()
        return self._row_space().mesh().comm

    def _all_ranks(self, flag):
        """Return True only if flag is True on every rank, so that all ranks take the same path.

        An error caught on some ranks only would otherwise send those ranks into a fallback while
        the others continue, and the next collective call (a MUMPS solve, a norm) would mix them up.
        """
        return not self._mpi_comm().allreduce(not bool(flag), op=_mpi_lor())

    def _probe(self, petscmat):
        """Return a seeded random vector compatible with the columns of petscmat (per-rank seed)."""
        probe = petscmat.createVecRight()
        rng = np.random.default_rng(20240 + petscmat.getComm().getRank())
        probe.setArray(rng.standard_normal(probe.getLocalSize()))
        return probe

    def _reset(self):
        """Drop a rejected cheap inverse (the assembled matrix is kept)."""
        self.kind = None
        self.inverse_diagonal = None
        self.inverse_matrix = None
        self.components = None

    def _try_diagonal(self):
        """Use pointwise division if the assembled matrix is diagonal; return whether it is.

        The test is collective: it compares A v with diag(A) v for a random v, with global norms.
        """
        from firedrake.petsc import PETSc

        petscmat = self._petscmat()
        diagonal = petscmat.createVecLeft()
        petscmat.getDiagonal(diagonal)
        probe = self._probe(petscmat)
        image = petscmat.createVecLeft()
        petscmat.mult(probe, image)
        difference = diagonal.duplicate()
        difference.pointwiseMult(diagonal, probe)
        difference.axpy(-1.0, image)
        magnitude = diagonal.copy()
        magnitude.abs()
        smallest = magnitude.min()[1]
        scale = diagonal.norm(PETSc.NormType.INFINITY) * probe.norm(PETSc.NormType.INFINITY)
        error = difference.norm(PETSc.NormType.INFINITY)
        if smallest == 0.0 or error > DIAGONAL_RTOL * scale:
            self.inverse_notes.append('not diagonal (relative off-diagonal action {:.1e})'.format(error / scale if scale > 0 else float('inf')))
            return False
        diagonal.reciprocal()
        self.inverse_diagonal, self.kind = diagonal, 'diagonal'
        return True

    def _try_cellwise(self):
        """Use the cellwise inverse, assembled from the Slate tensor inverse of the form."""
        from firedrake import Tensor, assemble

        inverse = assemble(Tensor(self.form).inv, mat_type='aij')
        # keep the Firedrake matrix alive with its PETSc Mat
        self._inverse = inverse
        self.inverse_matrix, self.kind = inverse.petscmat, 'cellwise'
        return True

    def _field_forms(self, nfields):
        """Return {i: form of the diagonal block of field i}, from split_form, or {} if unavailable.

        The forms are only needed for the cellwise inverse of a field. The index format of
        split_form differs between Firedrake versions ((i, j) or ((i,), (j,))), so the indices are
        flattened before they are compared.
        """
        def flat(indices):
            result = []
            for item in indices:
                if isinstance(item, (tuple, list)):
                    result.extend(flat(item))
                else:
                    result.append(int(item))
            return tuple(result)

        try:
            from firedrake.formmanipulation import split_form
            forms = {}
            for block in split_form(self.form):
                indices = flat(block.indices)
                if len(indices) == 2 and indices[0] == indices[1] and indices[0] < nfields:
                    forms[indices[0]] = block.form
            return forms
        except Exception as error:
            self.inverse_notes.append('split_form failed: ' + repr(error))
            return {}

    def _try_fieldsplit(self):
        """Invert a mixed operator that does not couple its fields field by field; return whether possible.

        Coupling is checked on the assembled matrix: every off-diagonal field block (a PETSc
        sub-matrix on the field index sets) must vanish. Each field then gets its own MassSolver
        of its diagonal sub-matrix, which uses a cheap inverse if one applies and a factorization
        otherwise. This is only worth it if at least one field has a cheap inverse; otherwise the
        whole matrix is factored at once.
        """
        from firedrake.petsc import PETSc

        space = self._row_space()
        nfields = len(space)
        petscmat = self._petscmat()
        field_ises = list(space.dof_dset.field_ises)
        scale = petscmat.norm(PETSc.NormType.INFINITY)
        for i in range(nfields):
            for j in range(nfields):
                if i != j:
                    block = petscmat.createSubMatrix(field_ises[i], field_ises[j])
                    if block.norm(PETSc.NormType.INFINITY) > DIAGONAL_RTOL * scale:
                        self.inverse_notes.append('the form couples fields {} and {}'.format(i, j))
                        return False
        forms = self._field_forms(nfields)
        components = []
        for index in range(nfields):
            block = petscmat.createSubMatrix(field_ises[index], field_ises[index])
            component = MassSolver(forms.get(index), self.solver_parameters,
                                   options_prefix=self.options_prefix + '_f' + str(index), petscmat=block)
            component._setup_cheap(allow_split=False)
            components.append(component)
        if all(component.kind is None for component in components):
            self.inverse_notes.append('no field has a cheap inverse: ' + '; '.join(
                note for component in components for note in component.inverse_notes))
            return False
        for component in components:
            if component.kind is None:
                component._setup_ksp()
        self.components = list(zip(field_ises, components))
        self.kind = 'fieldsplit'
        return True

    def _setup_cheap(self, allow_split=True):
        """Set up the cheapest inverse that mass_inverse allows, other than a factorization.

        Returns True if one was set up (kind is then set), False otherwise (with a note). Errors
        are turned into notes, so the caller can fall back to a factorization.
        """
        try:
            if not self._try_diagonal():
                space = self._row_space()
                if space is not None and len(space) > 1:
                    if allow_split:
                        self._try_fieldsplit()
                elif self.mass_inverse in ('auto', 'cellwise'):
                    if space is None:
                        self.inverse_notes.append('no form for a cellwise inverse')
                    elif is_l2_space(space):
                        self._try_cellwise()
                    else:
                        self.inverse_notes.append('not discontinuous (no cellwise inverse)')
        except Exception as error:
            self.inverse_notes.append('cheap inverse failed: ' + repr(error))
            self._reset()
        ok = self.kind is not None
        # collective: a cheap inverse is only used if it was set up on every rank
        if not self._all_ranks(ok):
            if ok:
                self.inverse_notes.append('cheap inverse failed on another rank')
            self._reset()
            return False
        return True

    def _check_inverse(self):
        """Return True if the inverse in use gives back a random v from A v to INVERSE_RTOL (collective)."""
        from firedrake.petsc import PETSc

        petscmat = self._petscmat()
        probe = self._probe(petscmat)
        image = petscmat.createVecLeft()
        petscmat.mult(probe, image)
        recovered = probe.duplicate()
        self._apply(image, recovered)
        recovered.axpy(-1.0, probe)
        error = recovered.norm(PETSc.NormType.INFINITY) / probe.norm(PETSc.NormType.INFINITY)
        if error > INVERSE_RTOL:
            self.inverse_notes.append('{} rejected by the check (relative error {:.1e})'.format(self.method, error))
            return False
        return True

    def _setup(self):
        """Choose and set up the inverse on the first solve (see the class doc)."""
        if self.mass_inverse != 'factor':
            if self._setup_cheap():
                try:
                    accepted = self._check_inverse()
                except Exception as error:
                    self.inverse_notes.append('check of ' + str(self.method) + ' failed: ' + repr(error))
                    accepted = False
                if self._all_ranks(accepted):
                    return
                self._reset()
        error = None
        try:
            self._setup_ksp()
        except Exception as caught:
            error = caught
        # collective: if the KSP failed on any rank, every rank uses the LinearSolver
        if not self._all_ranks(error is None):
            self._setup_linear_solver(error or RuntimeError('KSP setup failed on another rank'))

    def _setup_ksp(self):
        """Create a bare PETSc KSP on the matrix and factor it now.

        Firedrake's LinearSolver is avoided where possible: in recent Firedrake it is a
        LinearVariationalSolver underneath, so every solve copies b and x and assembles another
        residual A x - b, which made each stage solve about 2 ms slower than the old per-stage
        LinearVariationalSolver.

        The solver parameters are put in the global PETSc options database under a prefix unique
        to this solver only while the KSP reads them and factors the matrix (KSPSetUp), and are
        removed again afterwards. This needs no Firedrake / petsctools OptionsManager, whose
        location differs between Firedrake versions.
        """
        from firedrake.petsc import PETSc

        petscmat = self._petscmat()
        ksp = PETSc.KSP().create(comm=petscmat.getComm())
        ksp.setOperators(petscmat)
        prefix = self.options_prefix + '_' + str(next(_MASS_SOLVER_IDS)) + '_'
        ksp.setOptionsPrefix(prefix)
        database = PETSc.Options()
        flat = flatten_solver_parameters(self._parameters()[1])
        for key, value in flat.items():
            database[prefix + key] = value
        try:
            ksp.setFromOptions()
            # factor now, while the options (e.g. pc_factor_mat_solver_type) are present
            ksp.setUp()
        finally:
            for key in flat:
                database.delValue(prefix + key)
        self.ksp, self.kind = ksp, 'ksp'

    def _setup_linear_solver(self, error):
        """Fall back to Firedrake's LinearSolver after the bare KSP raised error."""
        from firedrake import LinearSolver
        if self._given_petscmat is not None:
            # a field block has no Firedrake matrix to fall back on
            raise error
        self.ksp_error = repr(error)
        self.solver = LinearSolver(self._assemble_matrix(), solver_parameters=petsc_solver_parameters(self.solver_parameters),
                                   options_prefix=self.options_prefix)
        self.ksp, self.kind = None, 'linear_solver'

    def _apply(self, rhs, solution):
        """Set the PETSc Vec solution to M^{-1} rhs with the inverse in use (not 'linear_solver')."""
        if self.kind == 'diagonal':
            solution.pointwiseMult(rhs, self.inverse_diagonal)
        elif self.kind == 'cellwise':
            self.inverse_matrix.mult(rhs, solution)
        elif self.kind == 'fieldsplit':
            # the fields are not coupled: solve each block on its part of the vectors
            for field_is, component in self.components:
                rhs_part = rhs.getSubVector(field_is)
                solution_part = solution.getSubVector(field_is)
                try:
                    component._apply(rhs_part, solution_part)
                finally:
                    rhs.restoreSubVector(field_is, rhs_part)
                    solution.restoreSubVector(field_is, solution_part)
        elif self.kind == 'ksp':
            self.ksp.solve(rhs, solution)
            if self.ksp.getConvergedReason() < 0:
                raise RuntimeError('mass-matrix solve did not converge: reason ' + str(self.ksp.getConvergedReason()))
        else:
            raise RuntimeError('no PETSc inverse set up (' + str(self.kind) + ')')

    def solve(self, x, b, scale=1.0):
        """Solve M x = b and then multiply x by scale (-1 solves M x = -b without negating b).

        b is an assembled right-hand side (a Cofunction, read only) and x a Function. The inverse
        is chosen on the first call (see the class doc). If the bare KSP fails on its setup or
        first solve, Firedrake's LinearSolver is used from then on.
        """
        if self.kind is None:
            self._setup()
        if self.kind != 'linear_solver':
            # sub-vectors (fieldsplit) need a right-hand side that is not locked read-only
            rhs_vec = b.dat.vec if self.kind == 'fieldsplit' else b.dat.vec_ro
            error = None
            try:
                with rhs_vec as rhs, x.dat.vec_wo as solution:
                    self._apply(rhs, solution)
            except Exception as caught:
                if self.solves > 0:
                    raise
                error = caught
            # the first solve decides collectively whether the KSP works (later errors just raise)
            if self.solves == 0 and not self._all_ranks(error is None):
                error = error or RuntimeError('mass solve failed on another rank')
                if self.kind != 'ksp':
                    raise error
                self._setup_linear_solver(error)
        if self.kind == 'linear_solver':
            self.solver.solve(x, b)
        self.solves += 1
        if scale != 1.0:
            with x.dat.vec as solution:
                solution.scale(scale)


class AffineSolver():
    """Solve residual(var) = 0 for a residual that is affine in var, with a shared factored operator.

    Every explicit RK stage residual (forward, aux, adjoint, tangent-linear, second-order adjoint)
    and the lambda / lambda_dot increment residuals read residual(var) = M var + r0, where M is
    the mass matrix of var's space and r0 does not depend on var. So the solve is:

    1. zero var, so that assembling the residual gives r0 exactly;
    2. assemble the residual once, with a PersistentAssembler;
    3. solve M var = -r0 with the shared MassSolver.

    -r0 is also the dual increment the adjoint steps need for lambda and lambda_dot, so solve()
    returns it (flattened). Before, each solve was a LinearVariationalSolver with its own matrix
    and LU factorization, whose right-hand side L = action(a, var) - residual and SNES residual
    action(a, var) - L each needed a UFL expansion of the whole (for the second-order solves,
    very large) form at construction, and the increments assembled their right-hand side twice.

    The first validate_calls solves are checked cheaply with one more assembly: at the probe
    var = c x (c = PROBE_SCALE, x the solution) the residual must equal (c - 1)(-r0), to
    VALIDATION_RTOL relative to c max|r0| over all ranks. A wrong operator, a residual that is
    not affine in var, a wrong solve, or an assembler that misses coefficient updates or
    accumulates into its old output all fail this check. A solve with r0 = 0 proves nothing and
    is not counted. fallback_reason records why a check failed. On a
    failure or an error, every rank switches for good to the reference solver, the old
    LinearVariationalSolver, built by make_reference only then ('reference' strategy). Without a
    mass solver (iterative solver parameters, see is_direct_solver) the reference solver is used
    from the start, exactly as before.
    """

    def __init__(self, residual, var, mass_solver, make_reference, validate_calls=2, returns_rhs=False,
                 backend=None, make_assembler=None, make_rhs=None):
        """residual: the UFL residual form (affine in var); var: the unknown Function;
        mass_solver: a MassSolver for the operator, or None to use the reference solver only;
        make_reference: returns the fallback solver (an object with solve());
        returns_rhs: whether solve() must return the flattened -r0 (the increments) also on the
        reference path. make_assembler(form, validate_calls) and make_rhs() (the right-hand side
        buffer passed to the mass solve) default to PersistentAssembler and a copy of the
        assembled residual; tests replace them, and backend, which provides any_rank and global_max.
        """
        self.residual = residual
        self.var = var
        self.mass_solver = mass_solver
        self.make_reference = make_reference
        self.validate_calls = validate_calls
        self.returns_rhs = returns_rhs
        self.backend = FiredrakeAssemblyBackend() if backend is None else backend
        self.make_assembler = make_assembler or (lambda form, validate_calls: PersistentAssembler(form, validate_calls=validate_calls))
        self.make_rhs = make_rhs
        # 'mass' until a check fails, then 'reference'
        self.strategy = 'mass' if mass_solver is not None else 'reference'
        self.calls = 0
        self.reference_calls = 0
        self.assembler = None
        self.reference = None
        self._validated = 0
        self._rhs = None
        # why the mass path was dropped (None while it is in use, or with iterative parameters)
        self.fallback_reason = None if mass_solver is not None else 'iterative solver parameters'
        self._error = None

    def _get_assembler(self):
        """Return the residual's PersistentAssembler, created on first use.

        On the mass path it is not validated against plain assembles: the residual check covers
        it. On the reference path (where only the returned increment uses it) it is.
        """
        if self.assembler is None:
            self.assembler = self.make_assembler(self.residual, 0 if self.strategy == 'mass' else 2)
        return self.assembler

    def _solve_with_mass(self, rhs_flat):
        """Solve M var = rhs for the flattened rhs; return True on success, False if it raised."""
        try:
            if self._rhs is None:
                if self.make_rhs is not None:
                    self._rhs = self.make_rhs()
                else:
                    # a buffer of the type this Firedrake version's assemble returns (a Cofunction)
                    self._rhs = self.assembler.tensor.copy(deepcopy=True)
            _set_flat_values(self._rhs, rhs_flat)
            self.mass_solver.solve(self.var, self._rhs)
            return True
        except Exception as error:
            self._error = repr(error)
            return False

    def _solve_reference(self):
        """Solve with the reference solver (built on first use); return -r0 if returns_rhs, else None."""
        rhs_flat = None
        if self.returns_rhs:
            self.var.dat.zero()
            rhs_flat = -self._get_assembler().assemble_flat()
        if self.reference is None:
            self.reference = self.make_reference()
        self.reference_calls += 1
        self.reference.solve()
        return rhs_flat

    def solve(self):
        """Solve for var at the current coefficient values; return the flattened -r0 (or None, see returns_rhs)."""
        self.calls += 1
        if self.strategy == 'reference':
            return self._solve_reference()
        # var enters the residual only through M var: zeroing it leaves exactly r0
        self.var.dat.zero()
        if self._validated >= self.validate_calls:
            # steady path: solve M var = r0 straight from the assembler's tensor and flip the
            # sign of the solution, without numpy copies; only the increments flatten r0
            assemble_tensor = getattr(self._get_assembler(), 'assemble_tensor', None)
            tensor = assemble_tensor() if assemble_tensor is not None else None
            if tensor is not None:
                self.mass_solver.solve(self.var, tensor, scale=-1.0)
                return -self.backend.flatten(tensor) if self.returns_rhs else None
        rhs_flat = -self._get_assembler().assemble_flat()
        solved = self._solve_with_mass(rhs_flat)
        if self._validated >= self.validate_calls:
            if not solved:
                raise RuntimeError('the mass-matrix solve failed after it was validated')
            return rhs_flat
        # validation on every rank: the solve worked, and the residual is affine in var with the
        # operator M. With x the solution, M x = -r0, so the residual at the probe c x must be
        # c M x + r0 = (c - 1) (-r0). Checking a probe instead of the residual at x itself, which
        # should vanish, keeps the tolerance relative to c |r0|: the assembly round-off of r0,
        # which can be large against |r0| when r0 is a sum of cancelling terms, does not matter.
        failed = self.backend.any_rank(self.var, not solved)
        scale = 0.0
        if failed:
            reason = 'mass solve raised: ' + (self._error or 'on another rank')
        else:
            scale = self.backend.global_max(self.var, np.max(np.abs(rhs_flat), initial=0.0))
            if scale > 0.0:
                self.backend.scale(self.var, PROBE_SCALE)
                probe = self._get_assembler().assemble_flat()
                self.backend.scale(self.var, 1.0 / PROBE_SCALE)
                error = self.backend.global_max(self.var, np.max(np.abs(probe - (PROBE_SCALE - 1.0) * rhs_flat), initial=0.0))
                failed = error > VALIDATION_RTOL * PROBE_SCALE * scale
                reason = 'residual check: relative error {:.3e} (tolerance {:.0e})'.format(error / (PROBE_SCALE * scale), VALIDATION_RTOL)
        if failed:
            # drop the mass path; the validated assembler of the reference path is created afresh
            self.fallback_reason = reason
            self.strategy = 'reference'
            self.assembler = None
            return self._solve_reference()
        # a zero right-hand side tells nothing about the operator: not counted
        if scale > 0.0:
            self._validated += 1
        return rhs_flat


class PersistentProjector():
    """Project one fixed UFL expression onto one target Function, repeatedly and cheaply.

    target.project(expr) builds a new LinearSolver every call: it assembles and LU-factors the
    mass matrix of the target space and processes both forms again. For the initial-condition
    map (ParametricInitialCondition) that was 0.4-0.6 s per objective, 0.8-1.0 s per gradient
    and about 2 s per HVP at every new parameter point, almost all of it the factorization
    (docs/hvp_performance_plan.md, section 2.3, first-call setup). The projection solves
    M target = b with b = assemble(inner(expr, v)*dx), the same forms Firedrake's project uses.
    Here the mass matrix comes from shared_mass_solver (factored once and shared by every
    projector onto the same space), and b from a PersistentAssembler, which sees the current
    values of the expression's coefficients (for example the R-space parameter Functions) on
    every call.

    Iterative solver parameters (see is_direct_solver) keep target.project, as does any later
    call after the mass solve raised (fallback_reason says why).
    """

    def __init__(self, target, expr, mass_cache, solver_parameters, options_prefix='projection-mass'):
        """target: the Function (possibly a subfunction of a mixed Function) to project onto;
        expr: the UFL expression (or number) to project; mass_cache: the dict of shared
        MassSolvers (see shared_mass_solver); solver_parameters: those of the projection.
        """
        import ufl
        from firedrake import TestFunction, TrialFunction, dx, inner

        self.target = target
        self.expr = expr
        self.solver_parameters = dict(solver_parameters)
        self.fallback_reason = None
        self.mass_solver = None
        self.assembler = None
        self._rhs = None
        if not is_direct_solver(self.solver_parameters):
            self.fallback_reason = 'iterative solver parameters'
            return
        space = target.function_space()
        test, trial = TestFunction(space), TrialFunction(space)
        # the forms of firedrake.projection: a = inner(u, v)*dx, L = inner(expr, v)*dx
        self.mass_solver = shared_mass_solver(mass_cache, inner(trial, test)*dx, self.solver_parameters,
                                              options_prefix=options_prefix)
        self.assembler = PersistentAssembler(inner(ufl.as_ufl(expr), test)*dx)

    @property
    def strategy(self):
        """'mass' (shared factored mass matrix) or 'project' (Firedrake's project, the fallback)."""
        return 'project' if self.fallback_reason is not None else 'mass'

    def _rhs_tensor(self):
        """Assemble b at the current coefficient values and return it as a Cofunction.

        After the PersistentAssembler has validated itself, its reused output tensor is returned
        directly; before that (and for the plain-assemble fallback) the flattened values are
        copied into a buffer of the same type.
        """
        tensor = self.assembler.assemble_tensor()
        if tensor is not None:
            return tensor
        values = self.assembler.assemble_flat()
        if self._rhs is None:
            self._rhs = self.assembler.tensor.copy(deepcopy=True)
        _set_flat_values(self._rhs, values)
        return self._rhs

    def project(self):
        """Project the expression at its current coefficient values into target; return target."""
        if self.fallback_reason is None:
            error = None
            try:
                self.mass_solver.solve(self.target, self._rhs_tensor())
            except Exception as caught:
                error = caught
            comm = self.target.function_space().mesh().comm
            # collective: every rank falls back together, or none does
            if not comm.allreduce(error is not None, op=_mpi_lor()):
                return self.target
            self.fallback_reason = 'mass solve raised: ' + (repr(error) if error is not None else 'on another rank')
            if comm.rank == 0:
                import warnings
                warnings.warn('PersistentProjector falls back to Function.project: ' + self.fallback_reason)
        try:
            self.target.project(self.expr, solver_parameters=petsc_solver_parameters(self.solver_parameters))
        except Exception as error:
            raise RuntimeError('Function.project failed after the persistent projection did ('
                               + str(self.fallback_reason) + ')') from error
        return self.target
