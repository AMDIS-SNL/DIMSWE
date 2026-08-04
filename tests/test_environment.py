"""Smoke checks for the validated serial development environment."""

from importlib.metadata import version

import numpy as np
from packaging.version import Version
import pytest


def test_core_firedrake_petsc_mpi_smoke():
    import firedrake
    import mpi4py
    import petsc4py
    from mpi4py import MPI
    from petsc4py import PETSc

    mesh = firedrake.UnitSquareMesh(1, 1, comm=firedrake.COMM_SELF)
    petsc_runtime_version = PETSc.Sys.getVersion()
    petsc4py_distribution_version = Version(version("petsc4py"))
    mpi4py_distribution_version = Version(version("mpi4py"))

    assert mesh.comm.size == 1
    assert MPI.COMM_SELF.Get_size() == 1
    assert len(petsc4py_distribution_version.release) >= 2
    assert (
        petsc4py_distribution_version.release[:2]
        == petsc_runtime_version[:2]
    )
    assert mpi4py_distribution_version.release


@pytest.mark.jax
def test_optional_firedrake_jax_bridge_float64_smoke():
    jax = pytest.importorskip(
        "jax",
        reason="JAX is optional; install the Firedrake JAX extra for this smoke test",
    )

    # This must precede the Firedrake bridge import: Firedrake/PETSc uses
    # double-precision scalars in the validated local environment.
    jax.config.update("jax_enable_x64", True)
    assert jax.config.x64_enabled
    assert jax.numpy.asarray(1.0).dtype == jax.numpy.float64

    bridge = pytest.importorskip(
        "firedrake.ml.jax.ml_operator",
        reason=(
            "the Firedrake JAX bridge is optional; install Firedrake with its "
            "JAX extra for this smoke test"
        ),
    )
    assert bridge.JaxOperator is not None
    assert callable(bridge.ml_operator)


@pytest.mark.rol
def test_optional_pyrol_import_and_construction_smoke():
    pyrol = pytest.importorskip(
        "pyrol",
        reason="PyROL is optional; install the rol-python distribution for this smoke test",
    )
    from pyrol import Bounds, Objective, ParameterList, Problem, Solver
    from pyrol.vectors import NumPyVector

    class SmokeObjective(Objective):
        def value(self, z, tol):
            return 0.5 * float(np.dot(z.array, z.array))

        def gradient(self, g, z, tol):
            g.array[:] = z.array

    z = NumPyVector(np.array([0.0], dtype=np.float64))
    objective = SmokeObjective()
    lower = NumPyVector(np.array([-1.0], dtype=np.float64))
    upper = NumPyVector(np.array([1.0], dtype=np.float64))
    bounds = Bounds(lower, upper)
    problem = Problem(objective, z)
    problem.addBoundConstraint(bounds)
    parameters = ParameterList()
    solver = Solver(problem, parameters)

    # pyrol.version() describes the Python API, not the installed wheel.
    assert version("rol-python") == "2025.9.10.dev1712"
    assert pyrol.version() == "PyROL version: 0.1.0"
    assert objective is not None
    assert problem is not None and solver is not None and bounds is not None
    assert isinstance(parameters, ParameterList)
    assert z.dimension() == 1 and z.array.dtype == np.float64
