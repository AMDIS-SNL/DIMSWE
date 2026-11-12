"""Tests for dimswe/assembly.py, the persistent increment assembly of plan item B0.

The strategy selection and validation logic is tested with a fake backend whose "forms" are
callables returning the current values. That covers a strategy that is correct, one that keeps
stale values (as a persistent assembler that does not see coefficient updates would), and ones
that raise. Real Firedrake assembly is then checked against a plain assemble() for a form whose
coefficient changes between calls, and in the RK adjoint steps through repeated gradients and HVPs.
"""

import numpy as np
import pytest

pytest.importorskip("firedrake")

from dimswe.assembly import PersistentAssembler  # noqa: E402


class FakeTensor:
    """Output tensor of the fake backend: a numpy array of values."""

    def __init__(self, values):
        self.values = np.array(values, dtype=float)


class FakeForm:
    """A 'form' whose assembled values are set by the test, standing in for changing coefficients."""

    def __init__(self, values):
        self.values = np.array(values, dtype=float)

    def __call__(self):
        return self.values.copy()


class FakeBackend:
    """Backend with configurable candidate strategies; counts the plain (slow) assemblies."""

    def __init__(self, kinds):
        self.kinds = kinds
        self.plain_assemblies = 0

    def assemble(self, form):
        self.plain_assemblies += 1
        return FakeTensor(form())

    def candidates(self, form, tensor):
        def make(kind):
            def factory():
                if kind == "missing":
                    raise ImportError("API not available in this version")
                if kind == "good":
                    # accumulates, so it is only right because the tensor is zeroed first
                    def run():
                        tensor.values += form()
                    return run
                if kind == "stale":
                    frozen = form()

                    def run():
                        tensor.values[:] = frozen
                    return run
                if kind == "raises":
                    def run():
                        raise RuntimeError("assembly failed")
                    return run
                raise ValueError(kind)
            return factory

        return [(kind, make(kind)) for kind in self.kinds]

    def zero(self, tensor):
        tensor.values[:] = 0.0

    def flatten(self, tensor):
        return tensor.values.copy()

    def any_rank(self, tensor, flag):
        return bool(flag)


def run_calls(assembler, form, ncalls, seed=0):
    """Assemble ncalls times with new form values each time; return (results, expected)."""
    rng = np.random.default_rng(seed)
    results, expected = [], []
    for _ in range(ncalls):
        form.values = rng.standard_normal(form.values.size)
        expected.append(form.values.copy())
        results.append(assembler.assemble_flat())
    return results, expected


def test_a_correct_strategy_is_validated_then_used_without_plain_assembly():
    form = FakeForm(np.zeros(5))
    backend = FakeBackend(["good"])
    assembler = PersistentAssembler(form, validate_calls=2, backend=backend)
    results, expected = run_calls(assembler, form, 6, seed=1)
    for result, values in zip(results, expected):
        np.testing.assert_allclose(result, values)
    assert assembler.strategy == "good"
    # one reference assembly on the first call, one per validated call, none afterwards
    assert backend.plain_assemblies == 3


def test_a_stale_strategy_is_dropped_for_the_next_one():
    form = FakeForm(np.zeros(4))
    backend = FakeBackend(["stale", "good"])
    assembler = PersistentAssembler(form, backend=backend)
    results, expected = run_calls(assembler, form, 5, seed=2)
    for result, values in zip(results, expected):
        np.testing.assert_allclose(result, values)
    assert assembler.strategy == "good"


def test_missing_apis_and_failing_strategies_are_skipped():
    form = FakeForm(np.zeros(3))
    assembler = PersistentAssembler(form, backend=FakeBackend(["missing", "raises", "good"]))
    results, expected = run_calls(assembler, form, 4, seed=3)
    for result, values in zip(results, expected):
        np.testing.assert_allclose(result, values)
    assert assembler.strategy == "good"


def test_plain_assemble_is_the_last_resort():
    form = FakeForm(np.zeros(3))
    backend = FakeBackend(["missing", "stale", "raises"])
    assembler = PersistentAssembler(form, backend=backend)
    results, expected = run_calls(assembler, form, 4, seed=4)
    for result, values in zip(results, expected):
        np.testing.assert_allclose(result, values)
    assert assembler.strategy == "assemble"


def test_results_are_copies_of_the_reused_tensor():
    form = FakeForm(np.ones(3))
    assembler = PersistentAssembler(form, backend=FakeBackend(["good"]))
    results, expected = run_calls(assembler, form, 5, seed=5)
    # accumulating into a returned array, as the timesteppers do, must not touch later results
    results[-2] += 100.0
    np.testing.assert_allclose(results[-1], expected[-1])
    assert results[-1] is not assembler.tensor.values


def test_firedrake_persistent_assembly_matches_assemble_as_coefficients_change():
    from firedrake import Constant, Function, FunctionSpace, TestFunction, UnitSquareMesh, assemble, dx, inner

    from dimswe.numpy_helpers import create_flattened_numpy_arr_from_mixed_function

    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    c = Constant(1.0)
    v = TestFunction(V)
    form = inner(c * f * f + f, v) * dx
    assembler = PersistentAssembler(form)
    rng = np.random.default_rng(6)
    for k in range(5):
        f.dat.data[:] = rng.standard_normal(f.dat.data.shape)
        c.assign(1.0 + k)
        expected = create_flattened_numpy_arr_from_mixed_function(assemble(form))
        np.testing.assert_allclose(assembler.assemble_flat(), expected, rtol=1e-12, atol=1e-14)
    print("strategy", assembler.strategy)
    assert assembler.strategy != "assemble"


def test_rk_adjoint_steps_use_persistent_assemblers_and_stay_repeatable():
    from dimswe.optimize import ParametricInitialCondition
    from tests.gradient_helpers import setup_problem

    setup = setup_problem(3, obs_steps=[1, 3])
    ic_map = ParametricInitialCondition(setup.model, setup.parameters, ["ox", "oy"])
    params = ic_map.get_initial_values(setup.parameters) + np.array([0.01, -0.005])
    problem = setup.optimizer.reduced_problem(params, opt_type="ic-params", coeffs0=setup.coeff, ic_map=ic_map,
                                              scaling_factors=np.array([0.1, 0.1]))
    x = problem.to_full(problem.z0)
    direction = np.array([1.0, 0.5])
    # each increment form is validated on its first few uses, within the first substeps of the
    # first call; every later substep and call uses the persistent results
    grads = [problem.full_jac(x) for _ in range(4)]
    hvps = [problem.full_hessp(x, direction) for _ in range(4)]
    for grad in grads[1:]:
        np.testing.assert_allclose(grad, grads[0], rtol=1e-12, atol=0.0)
    for hvp in hvps[1:]:
        np.testing.assert_allclose(hvp, hvps[0], rtol=1e-12, atol=0.0)
    for integrator in setup.timestepper.time_integrators:
        # the lambda increments assemble inside their AffineSolvers (plan item C1), the coefficient
        # increments (if any) in _increment_assemblers
        assemblers = dict(integrator._increment_assemblers)
        for name in ("L_delta_lambda", "L_delta_lambda_dot"):
            assemblers[name] = integrator._increment_solvers[name].assembler
        print(integrator, {name: a.strategy for name, a in assemblers.items()})
        assert all(a.strategy != "assemble" for a in assemblers.values())
