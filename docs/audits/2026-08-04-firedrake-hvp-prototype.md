# Audit: isolated Firedrake exact HVP prototype

Date: 2026-08-04

Branch: `dev/firedrake-hvp-prototype`

Starting HEAD: `c82f487126e71a23bf75a89c62c835603b227b99`

## Repository gate and environment

The repository root was
`/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615`.
The required branch was checked out and all tracked files were clean before
editing.  The only initial untracked files were `.DS_Store` and
`docs/.DS_Store`; both remained untouched.  No commit was created.

Installed serial environment:

| Component | Installed value |
| --- | --- |
| Python executable | `/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/python` |
| Python | 3.12.13 |
| Firedrake distribution | 2026.4.1 |
| petsc4py | 3.25.0 |
| PETSc runtime | 3.25.0 |

Firedrake and PETSc imports emit an Open MPI TCP bind diagnostic in this
sandbox.  All meshes explicitly use `COMM_SELF`; no MPI launcher or MPI test
was run.

## Exact files changed

- `firedrake_hvp_prototype/__init__.py`
- `firedrake_hvp_prototype/core.py`
- `tests/test_firedrake_hvp_prototype.py`
- `docs/FIREDRAKE_HVP_PROTOTYPE.md`
- `docs/audits/2026-08-04-firedrake-hvp-prototype.md`

## Scope audit

The implementation is a new isolated package.  No file in `dimswe/`,
`ode_adjoint/`, `hamiltonian_neural_odes/`, CI, dependency/environment files,
or production configuration was edited.  No production timestepper,
optimizer, PyROL adapter, MTSWE physics, JAX physics, or neural-network path
was connected to the prototype.

## Implementation audit

- All primal states, tendencies, tangents, and reverse auxiliary fields are
  Firedrake `Function` objects on \(V_h\).
- Terminal, stage, and incoming ordinary/incremental adjoints are genuine
  `Cofunction` objects on \(V_h^*\).
- Terminal derivatives are assembled one-forms, not primal residual fields.
- The L2 Riesz conversion is explicitly named and implemented with a mass
  `LinearSolver`; no mass inverse is formed.
- Primal and tangent stage equations are variational mass solves.
- Tangent, ordinary reverse, and incremental reverse contractions use UFL
  `derivative` on one-forms or already-contracted scalar forms.
- No finite difference occurs inside the HVP and no third-order derivative
  tensor is constructed.
- Euler and classical RK4 share one `ExplicitRungeKutta` implementation.
- Timesteps and stages reverse in exact decreasing graph order, accumulating
  the shared scalar gradient and HVP.
- Reaction-diffusion uses the sign convention
  \((\eta,K)+\kappa(\nabla\eta,\nabla Y)-p^2(\eta,Y)=0\) with homogeneous
  endpoint Dirichlet conditions.
- The full timestep Jacobian is not assembled.  A tiny dense interior mass
  and stiffness reference exists only in tests.
- Inputs and reverse right-hand sides are copied.  Stage and trajectory caches
  own independent Firedrake storage.

## Installed dual-space API findings

Local probes established that:

- assembling a one-form returns `Cofunction(V.dual())`;
- `assemble(action(cofunction, function))` evaluates the natural pairing;
- `Cofunction.riesz_representation("L2")` maps dual to primal and
  `Function.riesz_representation("L2")` maps primal to dual;
- `LinearSolver.solve(Function, Cofunction)` is supported directly;
- `Cofunction.copy()` is deep by default, while `Function.copy()` requires
  `deepcopy=True` for owned storage;
- scalar multiplication of a `Cofunction` can yield a symbolic `FormSum`, so
  concrete dual sums use PETSc AXPY;
- `DirichletBC.apply` rejects a `Cofunction`, but one-form assembly with the
  boundary conditions supplied produces the required zero constrained dual
  entries.

The last two points were the only material API surprises.  They did not
require weakening the dual-native convention.

## Certified cases and measured errors

Reaction certification includes:

- one Euler step: `K=18`, `u_plus=3.8`, `J=3.92`, `w_plus=0.6`,
  `gradient=3.36`, `Hq=1.28`, and `Hessian=2.56`;
- reverse auxiliaries `psi=0.28` and `delta_psi=0.06`, with HVP terms `0.56`,
  `0`, and `0.72`;
- two Euler steps: states `[2, 3.8, 7.22]`, `J=19.3442`,
  `gradient=28.3632`, `Hq=19.6024`, and `Hessian=39.2048`;
- one, two, and five Euler and RK4 steps against a direct scalar Hessian
  formula.

Reaction-diffusion certification includes one and three Euler steps and one
and three RK4 steps against an independently assembled tiny dense serial
interior oracle.  States, tangents, objective, gradient, and combined
state/parameter HVP agree within `5e-13`.

Dual pairing identities agree within `3e-15`.  Reaction and constrained
reaction-diffusion Riesz round-trips recover the original dual within
`2e-15`; primal Riesz representatives agree in L2 within `2e-14`.

Centered-difference errors for epsilon `[0.04, 0.02, 0.01, 0.005]` were:

| Check | Errors |
| --- | --- |
| Reaction RK4 objective to gradient | `[2.9546895e-4, 7.3853825e-5, 1.8462618e-5, 4.6156020e-6]` |
| Reaction RK4 gradient to HVP | `[6.3308652e-5, 1.5826570e-5, 3.9566055e-6, 9.8914909e-7]` |
| Diffusion Euler combined gradient to HVP | `[4.4373654e-6, 1.1093302e-6, 2.7733186e-7, 6.9332922e-8]` |
| Diffusion RK4 combined gradient to HVP | `[4.7594494e-6, 1.1898469e-6, 2.9746076e-7, 7.4365127e-8]` |

Every sequence displays the expected factor-of-four reduction under epsilon
halving before roundoff.  The scalar reaction tests also compare `Hq/q` with
the direct full scalar Hessian.

Input Functions and Cofunctions are unchanged by forward/reverse evaluation;
caches remain unchanged after input mutation; repeated runs are bitwise
identical.

## Test and command record

- Final focused prototype suite: `20 passed, 4506 warnings in 15.71s`.
- Accepted NumPy reference suite, `python -m pytest -q ode_adjoint`:
  `28 passed, 1 xfailed, 6 warnings in 9.31s`.
- Full repository suite, `python -m pytest -q`: `70 passed, 1 skipped,
  1 xfailed, 16700 warnings in 467.62s`.

The focused warnings are Firedrake/PyOP2 NumPy 2.5 deprecations from installed
code plus eight occurrences at the test-only PETSc dense extraction call
site.  The full-suite warning families additionally include the repository's
existing SciPy optimizer notices, finite-element naming/quadrature warnings,
and corresponding Firedrake/PyOP2 deprecations.

Final repository checks:

- `git diff --check`: passed;
- `git status --short`: only the two original `.DS_Store` files and the five
  allowed new prototype files/directories are untracked;
- `git diff --stat`: no output because the milestone consists entirely of
  new untracked files;
- a separate trailing-whitespace scan over all five new files: passed;
- tracked worktree and index: clean;
- final branch and HEAD: `dev/firedrake-hvp-prototype` at
  `c82f487126e71a23bf75a89c62c835603b227b99`;
- generated `__pycache__` directories and `.pyc` files: removed.

## Unresolved limitations

The prototype is serial, scalar, fixed-step, explicit-RK, and terminal-
least-squares only.  The spatial operator is symmetric and linear with fixed
homogeneous Dirichlet data.  Mixed spaces, nonsymmetric and nonlinear
operators, running costs, parameter-dependent mass matrices/BCs, variable
steps, distributed cache behavior, checkpointing, and production
preconditioners are not certified.

No DIMSWE production HVP, PyROL `hessVec`, JAX physics, or neural-network work
was started.
