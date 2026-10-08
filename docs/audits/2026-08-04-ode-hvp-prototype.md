# Audit: isolated NumPy ODE exact HVP prototype

Date: 2026-08-04

Branch: `dev/ode-hvp-prototype`

Starting HEAD: `5bb9a5b211bdec8a74fd264b240c37ec1eebea07`

## Repository gate

The repository root was
`/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615`.
The branch matched the required branch and the tree was clean before editing.
No commit was created.

## Scope audit

Changes are confined to `ode_adjoint/`, this focused development note, and
this audit.  No Firedrake production module, DIMSWE timestepper or optimizer,
PyROL adapter, MTSWE physics, JAX or neural-network code, CI file, or
production configuration was modified.

## Mathematical and implementation audit

- Added contracted directional-transpose dynamics actions for
  \([f_{xx}[w]+f_{xp}[q]]^T\lambda\) and
  \([f_{px}[w]+f_{pp}[q]]^T\lambda\).
- Added copied primal and tangent stage caches to the existing generic
  explicit RK implementation.
- Added exact ordinary and incremental reverse stage traversals in strict
  reverse graph order.  Euler, RK4, SSPRK3, and SSPRK43 share the algorithm.
- Added multistep exact HVPs for terminal least squares without constructing
  dense timestep Jacobians.
- Added combined initial-state/parameter directions and output blocks.
- Added a two-child generic composition prototype.
- Added a composition gradient-only API whose child forward and ordinary
  reverse passes are independent of the HVP and incremental adjoint.
- Added a clearly separate terminal least-squares Gauss--Newton HVP.
- Preserved the historical sign/output convention of `take_adjoint_step` and
  retained all existing dynamics and optimization tests.
- Hardened `take_adjoint_step` so it verifies cached step size, reconstructed
  start time, and strictly shaped parameters before reverse evaluation.  Each
  inconsistency raises a field-specific `ValueError`.

Array-shape audit: states and adjoints are `(nx,)`, parameters and parameter
HVPs are `(nparams,)`, trajectories and tangents are `(nsteps+1,nx)`, and each
RK stage cache contains `(nx,)` arrays.  Runtime checks reject mismatched
directions.  Tests confirm floating output, no input mutation, and exact
repeatability.

## Independent verification

Certified exact cases:

- one-step Euler scalar analytic reference;
- two-step Euler scalar analytic reference;
- multidimensional nonlinear Euler direct algebra;
- generic Euler, RK4, SSPRK3, and SSPRK43 stage graphs;
- one, two, three, six, and seven timestep paths;
- parameter-only and combined initial-state/parameter directions;
- multidimensional parameter Hessian symmetry;
- two-child operator composition with a parameter-independent nonlinear
  child;
- composition centered differences using the gradient-only ordinary adjoint;
- exact and Gauss--Newton comparison at zero and nonzero residual.

Centered gradient differences converge quadratically from approximately
`eps=1e-1` through `1e-4`.  Best errors occur near `1e-5` (`2.8e-12` to
`9.9e-12` in the reported cases); roundoff growth starts near `1e-6`.
Three deterministic symmetry comparisons have absolute errors
`0`, `3.469e-18`, and `3.469e-18`.

At zero residual the measured exact and Gauss--Newton HVPs are identical.  At
nonzero residual their 2-norm difference is `7.5299e-2`, demonstrating the
residual-weighted model-curvature term in the exact HVP.

## Test and final command record

- Pre-change NumPy ODE baseline: `9 passed, 1 xfailed`.
- Focused HVP and hardening tests: `19 passed`.
- Required final NumPy ODE suite after hardening: `28 passed, 1 xfailed, 6
  warnings` in 9.02 seconds.
- `git diff --check`: passed.
- The authoritative completed outside full suite is
  `/private/tmp/dimswe-ode-hvp-full-suite.log`: `46 passed, 1 skipped, 1
  xfailed, 12200 warnings in 450.57s`.

The repository-wide result comes from the completed outside command; the
affected NumPy ODE suite was rerun after the hardening correction as recorded
above.

The test process emitted the environment's pre-existing Open MPI TCP bind
diagnostic while importing the SciPy-linked stack.  No MPI command or MPI test
was invoked, and pytest completed successfully.  The six pytest warnings are
the existing SciPy optimizer warnings about unused `hessp` and `disp` options.

## Remaining limitations

The prototype is explicit-RK and terminal-least-squares only.  It does not
cover running objectives, variable steps, implicit methods, event maps,
Firedrake object lifetimes, or production checkpointing.  The legacy SciPy
optimizer `hessp` stub is intentionally not connected to the standalone
prototype.

No Firedrake DIMSWE HVP, PyROL `hessVec`, JAX physics, or neural-network work
was started.
