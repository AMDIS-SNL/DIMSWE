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
every solve of the same space (shared_mass_solver).
"""

import numpy as np

from .numpy_helpers import create_flattened_numpy_arr_from_mixed_function

# relative tolerance of the validation against a plain assemble (both run the same kernels, so
# the results normally agree to round-off)
VALIDATION_RTOL = 1e-10


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


class MassSolver():
    """A factored operator matrix M (in practice a mass matrix), shared by many solves (plan item C1).

    Every explicit RK stage equation, its adjoint, tangent-linear and second-order adjoint
    counterparts and the lambda / lambda_dot increments read M var = b with the x-space (or the
    aux-space) mass matrix. The matrix is assembled and factored once, on the first solve, and
    reused for every right-hand side after that. Before, each of the up to 8 x nstages + 2
    LinearVariationalSolvers of a sub-integrator assembled and factored its own copy, which was
    about 16 s of LU factorizations in the first calls of the baseline job.
    """

    def __init__(self, form, solver_parameters, options_prefix='erk-mass'):
        self.form = form
        self.solver_parameters = dict(solver_parameters)
        self.options_prefix = options_prefix
        self.solver = None
        self.solves = 0

    def solve(self, x, b):
        """Solve M x = b, where b is an assembled right-hand side (Cofunction) and x a Function."""
        if self.solver is None:
            from firedrake import LinearSolver, assemble
            # a monolithic matrix (also for mixed spaces), which LU needs; as LinearVariationalSolver
            matrix = assemble(self.form, mat_type=self.solver_parameters.get('mat_type', 'aij'))
            self.solver = LinearSolver(matrix, solver_parameters=self.solver_parameters,
                                       options_prefix=self.options_prefix)
        self.solves += 1
        self.solver.solve(x, b)


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

    The first validate_calls solves are checked cheaply: the residual is assembled again at the
    solution and must vanish, max|residual(var)| <= VALIDATION_RTOL max|r0| over all ranks. A
    wrong operator, an assembler that misses coefficient updates or accumulates into its old
    output all fail this check. A solve with r0 = 0 proves nothing and is not counted. On a
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
        except Exception:
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
        rhs_flat = -self._get_assembler().assemble_flat()
        solved = self._solve_with_mass(rhs_flat)
        if self._validated >= self.validate_calls:
            if not solved:
                raise RuntimeError('the mass-matrix solve failed after it was validated')
            return rhs_flat
        # validation: the residual at the solution must vanish (to round-off) on every rank
        failed = self.backend.any_rank(self.var, not solved)
        if not failed:
            scale = self.backend.global_max(self.var, np.max(np.abs(rhs_flat), initial=0.0))
            error = self.backend.global_max(self.var, np.max(np.abs(self._get_assembler().assemble_flat()), initial=0.0))
            failed = error > VALIDATION_RTOL * scale
        if failed:
            # drop the mass path; the validated assembler of the reference path is created afresh
            self.strategy = 'reference'
            self.assembler = None
            return self._solve_reference()
        if scale > 0.0:
            self._validated += 1
        return rhs_flat
