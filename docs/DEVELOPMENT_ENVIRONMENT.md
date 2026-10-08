# Validated local macOS development environment

This is the validated local macOS setup for the baseline/ROL milestone.  It is
not a portable CI contract.  From a fresh shell, use configurable roots rather
than machine-specific absolute paths:

```sh
export DIMSWE_VENV_ROOT="${DIMSWE_VENV_ROOT:-$HOME/venvs/dimswe-firedrake-2026.4.1-py312}"
export DIMSWE_BUILD_ROOT="${DIMSWE_BUILD_ROOT:-$HOME/venvs/dimswe-firedrake-2026.4.1-py312-build}"

source "$DIMSWE_VENV_ROOT/bin/activate"
export PETSC_DIR="${PETSC_DIR:-$DIMSWE_BUILD_ROOT/petsc}"
export PETSC_ARCH="${PETSC_ARCH:-arch-firedrake-default}"
export HDF5_MPI="${HDF5_MPI:-ON}"
export HDF5_DIR="${HDF5_DIR:-$PETSC_DIR/$PETSC_ARCH}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
```

JAX must use 64-bit values because the validated Firedrake/PETSc installation
uses double-precision scalars.  The optional JAX smoke test enables and checks
x64 before it imports the Firedrake JAX bridge, then strictly checks that a JAX
scalar has `float64` dtype.

`tests/conftest.py` assigns each pytest process a unique temporary PyOP2/TSFC
cache root before any test module can import Firedrake.  To choose the root
explicitly (for example in another restricted runner), set
`DIMSWE_TEST_CACHE_DIR` before invoking pytest; the two Firedrake cache
variables will be placed below that root.

## Validated versions

The local environment was validated on 2026-08-03 and re-audited on
2026-08-04 with:

| Component | Version |
| --- | --- |
| Python | 3.12.13 |
| Firedrake | 2026.4.1 |
| PETSc / petsc4py | 3.25.0 / 3.25.0 |
| MPI / mpi4py | Open MPI 5.0.9 / 4.1.2 |
| JAX | 0.11.0 |
| rol-python distribution | 2025.9.10.dev1712 |
| PyROL Python API | 0.1.0 |

The installed PyROL wheel version is obtained with
`importlib.metadata.version("rol-python")`.  `pyrol.version()` reports the
independent Python API string and is not a substitute for the distribution
version check.

The validated PyROL imports are:

```python
import pyrol
from pyrol import Bounds, Objective, ParameterList, Problem, Solver
from pyrol.vectors import NumPyVector
```

After x64 is enabled, the validated Firedrake/JAX bridge import is:

```python
from firedrake.ml.jax.ml_operator import JaxOperator, ml_operator
```

## Scope, CI policy, and caveats

This development milestone is serial-only.  Construct meshes with
`firedrake.COMM_SELF` where the API permits it, run `pytest` directly, and do
not use `mpiexec` or another multi-rank launcher.

The existing project CI installs Firedrake and pytest but does not install
`rol-python` or guarantee the Firedrake JAX extra.  ROL and JAX therefore
remain optional there until the project dependency policy is updated.  Their
smoke and adapter tests use precise `pytest.importorskip` guards; both execute
without skipping in this validated local environment.  Matplotlib likewise
remains optional and is checked only by the separate plotting-import smoke.

The PETSc configuration advertises MUMPS (including mixed-precision MUMPS),
but that is not evidence that a distributed MUMPS factorization works in the
current runtime.  Multi-rank MUMPS is outside this milestone and remains
unvalidated.  In the restricted macOS runner, initializing Open MPI can also
print a TCP-listener permission warning even for a one-rank process; the
validated serial `COMM_SELF` run succeeds outside a sandbox that denies host
identity system calls.

The environment smoke tests stop after imports and object construction.  They
do not run an optimization or a production simulation.
