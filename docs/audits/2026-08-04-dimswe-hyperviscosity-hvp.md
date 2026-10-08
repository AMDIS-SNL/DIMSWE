# DIMSWE production hyperviscosity child HVP audit

## Repository and scope

- Branch: `dev/dimswe-production-hvp`
- Starting HEAD: `8581a81c1ccd0fd424f35592453418e123db5c95`
- Starting tracked tree and index: clean
- Pre-existing untracked files left untouched: `.DS_Store` and
  `docs/.DS_Store`

This milestone is limited to the production `GeneralRK` explicit-Euler child
selected by `terms=["hyperviscosity"]`, a scalar physical `c0` direction, and
fixed physical `s`.  The legacy forward and adjoint methods are unchanged.
Audit status: **COMPLETE**.  Every requested focused, targeted, NumPy-reference,
and full-suite verification completed without failures.

## Historical debugging chronology (superseded)

The following failed focused runs document the debugging sequence.  They were
superseded by the final passing external certification recorded below.

An authoritative external focused run of the initial milestone reported
`3 passed, 6 failed`.  Five reverse/HVP failures reached Firedrake assembly
with a canonical domainless UFL structural zero.  The remaining parameter-only
tangent check used `epsilon=2e-6` and measured relative error `1.766e-7`,
consistent with subtraction/solve roundoff for this bilinear map.  Those
results prompted the tangent and structural-zero corrections below.

The authoritative r2 and r3 runs both reported `4 passed, 5 failed`: the
tangent correction passed, and no reverse/HVP numerical assertion executed.
The earlier diagnosis was incomplete.  The raw delayed derivative reached
Firedrake as a rank-zero object; `ZeroFormAssembler` denotes functional
assembly and is not proof that its input is mathematically zero.  This result
prompted the derivative-expansion correction.

The authoritative r4 run reported `5 passed, 4 failed`.  Derivative expansion
fixed the original structural-zero assembly failure and the reverse pairing
test passed.  One remaining cause was a test-only assumption that a raw exact
zero must have zero arguments; the installed object is a one-form with an
identically zero expanded integrand.  The other was a false-positive syntactic
`c0` containment check on the mixed `[s,c0]` coefficient Function.

The authoritative r5 run reported `6 passed, 3 failed`.  The remaining guard
incorrectly required canonical structural zero for pure `c0-c0` curvature.
Installed UFL instead retains a valid rank-zero scalar form with zero arguments,
three integrals, and one domain.  Its mathematical value must be established by
scalar assembly and numerical certification, not symbolic shape alone.

The corrected authoritative focused production run then completed with
`9 passed, 245 warnings in 43.99s`.  All focused tests passed.

Final authoritative verification completed with:

- focused production hyperviscosity HVP: `9 passed, 245 warnings in 43.99s`;
- targeted Firedrake regressions: `30 passed, 16205 warnings in 193.65s`;
- NumPy ODE reference: `28 passed, 1 xfailed, 6 warnings in 9.25s`;
- full repository: `79 passed, 1 skipped, 1 xfailed, 16946 warnings in
  524.20s`;
- `git diff --check`: passed.

The warning families are non-blocking installed-library and existing-repository
warnings.  The reported skip and expected failure are likewise non-failures.

## Files changed

- `dimswe/timestepping.py`
- `dimswe/hyperviscosity_hvp.py` (new)
- `tests/test_production_hyperviscosity_hvp.py` (new)
- `docs/DIMSWE_HYPERVISCOSITY_HVP.md` (new)
- `docs/audits/2026-08-04-dimswe-hyperviscosity-hvp.md` (new)

No file was staged or committed.  `dimswe/optimize.py` and
`dimswe/rol_adapter.py` were not modified.

## Exact configuration

The focused test uses `tests/tswe_rol_small.cfg`:

- 2 by 2 periodic rectangle, quadrilateral cells, order 3;
- dry TSWE mixed state `[v,h,S]`;
- production forcing list containing hyperviscosity;
- production Lie child list containing RK4 model and Euler
  hyperviscosity children;
- semantic child selection by exact term list rather than child index;
- physical `c0 = 0.14`;
- fixed physical `s = 3.2`;
- physical direction `delta_c0 = 0.035`;
- applied child `dt = 100.0`;
- deterministic initialized double-vortex state and deterministic mixed state
  direction;
- direct serial linear solvers and `COMM_SELF` mesh construction;
- no MPI invocation.

## Implemented graph

The cached forward calls the unchanged production Euler child and owns copies
of its incoming state, diagnostic, tendency, and outgoing state together with
the exact child start time, applied `dt`, physical `c0`, and fixed `s`.

The tangent reconstructs forms with the deployed
`compute_aux_expressions` and `-model.rhs` builders, differentiates the
diagnostic and main one-forms with UFL, and applies separate state and
diagnostic mass solves.  The ordinary and incremental reverses use genuine
mixed-space `Cofunction` objects, explicit primal mass/Riesz solves, and
differentiated contracted production forms.  The physical HVP exposes the
incremental-adjoint mixed term, state-direction mixed term, assembled
pure-control term, and their total.

The new assembly boundary first applies installed UFL 2025.3
`expand_derivatives` to eliminate delayed coefficient/variable derivative
nodes.  It then removes only integrals with canonical UFL `Zero` integrands.
Only `ZeroBaseForm`, scalar `Zero`, or an expanded form with no remaining
nonzero integrals is structural zero.  A zero dual becomes a named, explicitly
zeroed `Cofunction`; a zero scalar becomes `0.0`.  A nonzero dual must have
exactly one UFL argument and a nonzero scalar exactly zero arguments before
assembly.  Arity failures report type, argument spaces, integral count, and
domain count.  Domainlessness alone is never treated as proof of zero.

The direct main state pullback is structurally zero because the production main
RHS depends directly on the diagnostic and physical `c0`; its state dependence
is reversed only through the separate diagnostic solve.  The runtime path now
verifies that state is absent from the actual contracted form, constructs and
expands the derivative, and requires exact normalized zero before bypassing
assembly.  The test requires the resulting direct main dual to be exactly zero
and the diagnostic state dual to be nonzero and equal to the full nonidentity
pullback.  Raw arity remains recorded but is not used as a zero criterion.  For
pure `c0-c0` curvature, the ordinary first derivative must remain nonzero; the
second unit-`c0` directional derivative is constructed and expanded.  If it is
canonical structural zero, assembly is bypassed; otherwise it must be a valid
scalar form with zero arguments and at least one domain and is assembled.  The
actual assembled scalar is retained in the debug record and returned as the
pure-control HVP contribution.  No split-component syntax search or hard-coded
pure value is used, and dependence on fixed `s` remains valid.

## Independent checks encoded in the focused test

The new focused test encodes:

1. cached-forward versus unchanged legacy-forward equivalence;
2. the deployed sign and a test-only dense mixed mass/stiffness oracle;
3. parameter-only and combined tangent centered differences;
4. physical gradient comparison with the unchanged legacy gradient;
5. explicit L2 Riesz comparison with the legacy primal adjoint;
6. the tangent/reverse pairing identity;
7. exact HVP comparison with the dense oracle;
8. nonzero, separately checked mixed-curvature contributions;
9. a centered-difference sequence of independent legacy physical gradients;
10. the incoming incremental state dual directly against the independent dense
    identity
    `A(c0).T @ mu_plus_star - dt*delta_c0*H.T @ lambda_plus_star`;
11. centered differences of unchanged legacy incoming primal L2 adjoints,
    with simultaneous incoming-state, physical-`c0`, and terminal-residual-
    adjoint changes, against the explicit L2 Riesz representative of the new
    incremental state dual;
12. caller and dual-input non-mutation, cache ownership, no aliasing, scratch
    independence, and repeatability.

Production code does not call a dense oracle or finite difference.

The corrected tangent oracle uses `epsilon = [1e-2,1e-3,1e-4,1e-5]` for both
directions and records all errors.  Since the child is bilinear, centered
differences have no truncation term and no factor-of-four regime is expected.
Moderate epsilon, including `1e-3`, must meet the original intended tolerance;
smaller values document the onset of subtraction/solve roundoff.  The strict
independent dense tangent tolerance is unchanged.

## Execution environments and authoritative verification

Codex's managed sandbox could not initialize PETSc.  Importing `firedrake`
called `PetscGetHostName`, and the sandbox denied the underlying
`getdomainname` system call with `Operation not permitted` and PETSc error code
88.  OpenMPI also could not bind a local TCP socket.  This was an execution-
environment limitation, not a repository failure.

Within that sandbox, the following commands stopped during Firedrake/PETSc
collection before their test bodies could execute:

- `python -m pytest -q tests/test_production_hyperviscosity_hvp.py`
- `python -m pytest -q tests/test_timestepping_coeff_gradients.py`
- `python -m pytest -q tests/test_timestepping_ic_gradients.py`
- `python -m pytest -q tests/test_rol_adapter.py`
- `python -m pytest -q tests/test_firedrake_hvp_prototype.py`
- `python -m pytest -q`

The sandbox full-suite attempt reported 8 collection errors, all caused by the
same PETSc hostname/domain initialization limitation.

The independent non-Firedrake sandbox command completed:

- `python -m pytest -q ode_adjoint`: 28 passed, 1 xfailed, 6 warnings in
  9.28 seconds.

The warnings were the existing SciPy optimizer warnings concerning `hessp`
with L-BFGS-B and an unknown `disp` option.

All three changed Python sources were parsed with Python `compile` without
executing imports.  `git diff --check` passed during the static verification.

The user subsequently ran the authoritative tests from their normal serial
Firedrake terminal, where PETSc initialized normally.  Those external runs
completed successfully:

- focused production hyperviscosity HVP: `9 passed, 245 warnings in 43.99s`;
- targeted Firedrake regressions: `30 passed, 16205 warnings in 193.65s`;
- NumPy ODE reference: `28 passed, 1 xfailed, 6 warnings in 9.25s`;
- full repository: `79 passed, 1 skipped, 1 xfailed, 16946 warnings in
  524.20s`;
- `git diff --check`: passed.

Thus the sandbox limitation did not prevent ultimate runtime certification.

## Measurements

The local sandbox failed before Firedrake test collection, so it supplied no
production numerical measurements itself.  Authoritative external execution
subsequently supplied the passing focused, targeted, NumPy-reference, and
full-suite results recorded above.  No unreported numerical values have been
inferred or fabricated.

The initial external `3 passed, 6 failed` result and its `1.766e-7`
tiny-epsilon tangent error diagnosed the first correction.  The r2 result of
`4 passed, 5 failed` certified the tangent correction but showed that the prior
zero-type guard did not cover the installed zero-integral form representation.
The identical r3 result disproved that representation-only diagnosis and
motivated derivative expansion.  The r4 result certified that structural-zero
repair while exposing the raw-arity test assumption and mixed-component syntax
guard described above.  The r5 result certified those two corrections and
established that the nested pure derivative remains a nonempty scalar
functional.  The final focused run certified its assembly and scale-aware
dense-zero check and reported `9 passed, 245 warnings in 43.99s`.

That passing run establishes:

- the noncanonical pure-`c0` second derivative was assembled, not hard-coded;
- its numerical value passed both the mixed-term-scale/machine-precision zero
  tolerance and the independent dense oracle's exact-zero result;
- both mixed HVP contributions passed separately and their returned sum passed;
- the incremental incoming state adjoint passed the dense dual-vector oracle
  and the centered-difference legacy incoming-adjoint ladder;
- all nine focused production tests passed, including the gradient, Riesz,
  HVP, pairing, ownership, non-mutation, and repeatability checks.

External numerical certification passed for:

- cached forward equivalence;
- parameter-only and combined tangents;
- dual reverse versus the legacy physical gradient;
- the explicit L2 Riesz state pullback versus the legacy primal adjoint;
- the pairing identity;
- the exact physical-`c0` HVP;
- both mixed-curvature terms independently;
- assembled pure-control curvature numerically zero under the scale-aware
  check and equal to the dense oracle's exact-zero result;
- the complete dense mass/stiffness oracle;
- centered differences of independent legacy physical gradients;
- the incremental incoming state adjoint against both the dense dual-vector
  and centered-difference legacy-adjoint oracles;
- cache ownership, non-aliasing, input preservation, and repeatability.

The successfully executed focused tests record or check the following
quantities:

- dense diagnostic, tendency, and forward relative errors;
- parameter-only and combined tangent relative errors at
  `epsilon = [1e-2,1e-3,1e-4,1e-5]`;
- legacy and dense gradient and state-pullback errors;
- both mixed HVP terms, pure-control term, total, and dense relative error;
- dense incoming incremental-state-dual relative error;
- centered HVP values, errors, and error ratios, plus centered legacy incoming-
  adjoint relative errors and ratios, for
  `epsilon = [0.04, 0.02, 0.01, 0.005]`.

Expected centered-difference behavior is quadratic while truncation error
dominates: each successive error ratio must exceed 3.8, and the final relative
HVP error and final incremental-state-pullback relative error must be below
`2e-5`.  Both ladders passed in the authoritative focused run.

## Certified-scope limitations

- Only the dry `tests/tswe_rol_small.cfg` active blocks `[v,h,S]` are in the
  focused test; moist inactive blocks are outside this milestone.
- Only a single production Euler child step is implemented and tested.
- Fixed `s` has no tangent, gradient, or HVP API.
- There is no full Lie, RK4, SSPRK43, moist-physics, multistep, checkpoint, MPI,
  or PyROL propagation.

No full Lie HVP, PyROL `hessVec`, normalized-control scaling, JAX physics, or
neural-network work was started.

Audit status: **COMPLETE**.
