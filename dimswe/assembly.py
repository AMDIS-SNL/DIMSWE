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

    def any_rank(self, tensor, flag):
        """Return True if flag is True on any rank of the tensor's mesh communicator."""
        try:
            comm = tensor.function_space().mesh().comm
        except AttributeError:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        return bool(comm.allreduce(bool(flag), op=_mpi_lor()))


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
