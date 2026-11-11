"""Tests for the shared-factorization stage and increment solves (dimswe/assembly.py AffineSolver, plan item C1).

Every explicit RK stage equation and the lambda / lambda_dot increments read
residual(var) = M var + r0. AffineSolver zeroes var, assembles r0 once, and solves M var = -r0
with a MassSolver whose factorization is shared by every solve of the same space, across all
sub-integrators of a model. Its first solves are checked by assembling the residual again at the
solution, which must vanish; a failure switches to the old LinearVariationalSolver for good.

The strategy logic is tested with fakes: a correct solve, a wrong operator, a failing solve, an
assembler that misses coefficient updates, a zero right-hand side (which validates nothing),
and iterative solver parameters (old solver from the start). The real RK steps of
tests/tswe_small.cfg are then checked: every solve uses the shared mass path, the x-space
factorization is shared by all sub-integrators, and gradients and HVPs equal those of the old
solvers to round-off.
"""

import numpy as np
import pytest

pytest.importorskip("firedrake")

from dimswe.assembly import AffineSolver, is_direct_solver  # noqa: E402

RTOL = 1e-10
SIZE = 5


class FakeDat:
    """Minimal stand-in for a PyOP2 Dat with one component: data_ro/data_wo, split and zero."""

    def __init__(self, size):
        self.values = np.zeros(size)

    @property
    def data_ro(self):
        return self.values

    @property
    def data_wo(self):
        return self.values

    @property
    def split(self):
        return (self,)

    def zero(self):
        self.values[:] = 0.0


class FakeFunction:
    """A Function (or Cofunction) with a FakeDat."""

    def __init__(self, size):
        self.dat = FakeDat(size)


class FakeAssembler:
    """Assembles the problem's affine residual diag(mass) var + r0 at the current var and r0; counts calls.

    stale=True returns the values of its first call forever, like a persistent assembler that
    does not see coefficient updates.
    """

    def __init__(self, problem, stale=False):
        self.problem, self.stale = problem, stale
        self.calls = 0
        self._first = None

    def assemble_flat(self):
        self.calls += 1
        values = self.problem.mass * self.problem.var.dat.values + self.problem.r0
        if self.stale:
            if self._first is None:
                self._first = values.copy()
            return self._first.copy()
        return values


class FakeMassSolver:
    """Solves diag(mass) x = b, optionally wrong by a factor, or raising."""

    def __init__(self, mass, factor=1.0, fail=False):
        self.mass, self.factor, self.fail = mass, factor, fail
        self.calls = 0

    def solve(self, x, b):
        self.calls += 1
        if self.fail:
            raise RuntimeError('simulated solver failure')
        x.dat.values[:] = self.factor * b.dat.values / self.mass


class FakeReferenceSolver:
    """The old LinearVariationalSolver: solves the residual for var exactly from the current r0."""

    def __init__(self, problem):
        self.problem = problem
        self.calls = 0

    def solve(self):
        self.calls += 1
        self.problem.var.dat.values[:] = -self.problem.r0 / self.problem.mass


class FakeBackend:
    """Single-rank reductions."""

    def any_rank(self, tensor, flag):
        return bool(flag)

    def global_max(self, tensor, value):
        return float(value)


class FakeProblem:
    """An affine residual on SIZE dofs, with the solver under test and its fakes (seeded)."""

    def __init__(self, seed, mass_kwargs=None, stale=False, direct=True, returns_rhs=True):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.mass = 1.0 + rng.random(SIZE)
        self.r0 = rng.standard_normal(SIZE)
        self.var = FakeFunction(SIZE)
        self.assemblers = []
        self.references = []
        self.stale = stale
        self.mass_solver = FakeMassSolver(self.mass, **(mass_kwargs or {})) if direct else None
        self.solver = AffineSolver(None, self.var, self.mass_solver, self._make_reference, returns_rhs=returns_rhs,
                                   backend=FakeBackend(), make_assembler=self._make_assembler,
                                   make_rhs=lambda: FakeFunction(SIZE))

    def _make_assembler(self, form, validate_calls):
        # only the first assembler (of the mass path) can be stale; the reference path gets a new one
        assembler = FakeAssembler(self, stale=self.stale and not self.assemblers)
        self.assemblers.append(assembler)
        return assembler

    def _make_reference(self):
        reference = FakeReferenceSolver(self)
        self.references.append(reference)
        return reference

    def run(self, ncalls, zero_first=False):
        """Solve ncalls times with new r0 (the first one zero if zero_first); check every result."""
        for call in range(ncalls):
            self.r0 = np.zeros(SIZE) if (zero_first and call == 0) else self.rng.standard_normal(SIZE)
            self.var.dat.values[:] = 99.0  # a stale value must not leak into the result
            returned = self.solver.solve()
            np.testing.assert_allclose(self.var.dat.values, -self.r0 / self.mass, rtol=1e-14, atol=1e-300)
            if self.solver.returns_rhs:
                np.testing.assert_allclose(returned, -self.r0, rtol=1e-14, atol=1e-300)


def test_a_correct_mass_solve_is_validated_by_its_residual_then_used_alone():
    problem = FakeProblem(seed=1)
    problem.run(6)
    assert problem.solver.strategy == 'mass'
    assert problem.references == []
    # one assembly per solve, plus the residual check of the first two
    assert problem.assemblers[0].calls == 6 + 2
    assert problem.mass_solver.calls == 6


@pytest.mark.parametrize("case", ["wrong operator", "solver error", "stale assembler"])
def test_a_failed_residual_check_falls_back_to_the_old_solver(case):
    kwargs = {"wrong operator": {"factor": 1.5}, "solver error": {"fail": True}}.get(case)
    # the stale assembler gives the right r0 on its first call, but the residual it "assembles"
    # at the solution is r0 again, so the check fails
    problem = FakeProblem(seed=3, mass_kwargs=kwargs, stale=(case == "stale assembler"))
    problem.run(4)
    assert problem.solver.strategy == 'reference'
    assert len(problem.references) == 1
    # the returned increments come from a new, validated assembler
    assert len(problem.assemblers) == 2


def test_a_zero_right_hand_side_does_not_count_as_validation():
    problem = FakeProblem(seed=5)
    problem.run(3, zero_first=True)
    assert problem.solver.strategy == 'mass'
    # the zero call and two informative calls were all checked
    assert problem.assemblers[0].calls == 3 + 3


def test_iterative_solver_parameters_keep_the_old_solver_without_zeroing():
    assert is_direct_solver({'ksp_type': 'preonly', 'pc_type': 'lu'})
    assert not is_direct_solver({'ksp_type': 'cg', 'pc_type': 'jacobi'})
    problem = FakeProblem(seed=7, direct=False, returns_rhs=False)
    problem.run(3)
    assert problem.solver.strategy == 'reference'
    assert problem.references[0].calls == 3
    # a stage solve (no returned increment) does not assemble at all, as before
    assert problem.assemblers == []


def test_rk_solves_share_one_x_space_factorization_and_match_the_old_solvers():
    from dimswe.assembly import shared_mass_solver
    from tests.gradient_helpers import flat, setup_problem

    setup = setup_problem(3, obs_steps=[1, 3])
    optimizer = setup.optimizer
    integrators = setup.timestepper.time_integrators
    rng = np.random.default_rng(60)
    coeffs = flat(setup.coeff) * 1.02
    ics = (flat(setup.x0[0]) * 1.01)[None, :]
    coeffs_dir = 0.1 * flat(setup.coeff) * rng.standard_normal(coeffs.size)
    ics_dir = 0.01 * (np.abs(ics) + 1.0) * rng.standard_normal(ics.shape)

    def evaluate():
        """Return (gradient, HVP), with the checkpoint dropped so every call solves every equation."""
        optimizer.clear_checkpoint()
        gradient = np.asarray(optimizer.jac(coeffs, ics))
        optimizer.clear_checkpoint()
        hvp = np.asarray(optimizer.hessp(coeffs, ics, coeffs_dir_arr=coeffs_dir, ics_dir_arr=ics_dir))
        return gradient, hvp

    def all_solvers(integrator):
        return list(integrator._stage_solvers.values()) + list(integrator._increment_solvers.values())

    # the first calls validate the mass path; the later ones use it alone
    for _ in range(2):
        mass_gradient, mass_hvp = evaluate()
    parameters = integrators[0].solver_parameters['erkstage-f']
    x_mass = shared_mass_solver(integrators[0]._mass_solvers, integrators[0].mass_x, parameters)
    for integrator in integrators:
        assert set(integrator._increment_solvers) == {'L_delta_lambda', 'L_delta_lambda_dot'}, integrator
        nstages = integrator.nstages
        # forward, adjoint, tangent and second-order adjoint solves of every stage (x part)
        for kind in ('F', 'mu', 'tlm_F', 'soa_mu'):
            assert all(kind + str(i) in integrator._stage_solvers for i in range(nstages)), (integrator, kind)
        assert all(solver.strategy == 'mass' for solver in all_solvers(integrator)), integrator
        # one cache for the whole model, and one x-space factorization in it
        assert integrator._mass_solvers is integrators[0]._mass_solvers
        assert integrator._stage_solvers['F0'].mass_solver is x_mass
        assert integrator._increment_solvers['L_delta_lambda_dot'].mass_solver is x_mass
    assert x_mass.solver is not None and x_mass.solves > 0
    for integrator in integrators:
        for solver in all_solvers(integrator):
            solver.strategy = 'reference'
    try:
        reference_gradient, reference_hvp = evaluate()
    finally:
        for integrator in integrators:
            for solver in all_solvers(integrator):
                solver.strategy = 'mass'
    scale = lambda arr: np.max(np.abs(arr))
    print("gradient diff", scale(mass_gradient - reference_gradient) / scale(reference_gradient))
    print("hvp diff", scale(mass_hvp - reference_hvp) / scale(reference_hvp))
    assert scale(mass_gradient - reference_gradient) <= RTOL * scale(reference_gradient)
    assert scale(mass_hvp - reference_hvp) <= RTOL * scale(reference_hvp)
