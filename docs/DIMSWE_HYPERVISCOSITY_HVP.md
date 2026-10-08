# DIMSWE production hyperviscosity child HVP

## Scope

This implementation covers exactly one production `GeneralRK` child configured
as explicit Euler with `terms=["hyperviscosity"]`.  The direction is the scalar
physical coefficient `c0`; the physical exponent `s` is fixed.  The code does
not implement RK4, SSPRK43, Lie-split, moist-physics, PyROL, normalized-control,
checkpointing, MPI, JAX, or neural-network HVPs.

The legacy `take_forward_step` and `take_adjoint_step` methods are unchanged.
The new methods are lazy, separate APIs that reject any uncertified tableau or
term list before constructing derivative forms.

## Production state, diagnostic, and weak forms

The production state is a mixed Firedrake `Function`.  In dry TSWE it contains
`[v,h,S]`; in moist TSWE it additionally contains the inactive hyperviscosity
blocks `[Qv,Qc,Qr]`.  Hyperviscosity acts only on `[v,h,S]`.

For each active field `a`, define the production mass and stiffness actions by

\[
  (M_a u,\eta)=\int_\Omega \eta u\,dx,
  \qquad
  (K_a u,\eta)=\int_\Omega \nabla\eta\mathbin\cdot\nabla u\,dx.
\]

`Hyperviscosity.compute_aux_expressions` supplies the deployed diagnostic
equation

\[
  M_a q_a=-K_a x_a.
\]

`Hyperviscosity.rhs` supplies the negative weak hyperviscosity form.  The
production `GeneralRK` convention sets `B=-model.rhs`, so its actual stage solve
is

\[
  M_a F_a=c_0r^sK_aq_a
  =-c_0r^sK_aM_a^{-1}K_ax_a,
\]

where

\[
  r=\max(\mathit{mesh.dx}/\mathit{order},
         \mathit{mesh.dy}/\mathit{order}).
\]

Euler applies

\[
  x_+=x+\Delta tF.
\]

The cached forward method calls the unchanged production Euler method, then
copies its actual diagnostic and tendency scratch.  The derivative helper
independently reconstructs its forms through the production
`model.compute_aux_expressions(..., terms=["hyperviscosity"])` and
`-model.rhs(..., terms=["hyperviscosity"])` APIs.  It does not contain a second
hand-written hyperviscosity forward form.

## Physical control

The cache records the physical `R0` values of `c0` and fixed `s` used by the
child.  Tangent input `delta_c0`, returned gradients, and returned HVPs are
physical Python floating scalars.  No PyROL normalization or scaling is present.

Changing the trainable coefficient uses the existing coefficient `Function`.
The state and diagnostic mass matrices do not depend on `c0`, so their solver
objects are reusable.

## Primal and dual conventions

The following objects are primal Firedrake `Function` instances:

- state, diagnostic, and tendency;
- their tangent directions;
- the reverse state-mass auxiliary fields;
- the reverse diagnostic-mass auxiliary fields.

The following objects are genuine Firedrake `Cofunction` instances:

- incoming and outgoing ordinary state adjoints;
- incoming and outgoing incremental state adjoints;
- tendency adjoints;
- diagnostic adjoints.

Natural pairings use `assemble(action(dual, primal))`.  Reverse mass solves copy
their dual right-hand side and solve into a new primal `Function`.  Cofunction
coefficients are never copied into primal storage.

The legacy adjoint instead stores primal L2/Riesz representatives.  The narrow
API exposes an explicit mixed mass map and mixed Riesz solve so tests compare
the two conventions as

\[
  Mz=\lambda^\star,
\]

not by comparing or copying coefficient arrays across primal and dual spaces.

## Owned caches

`HyperviscosityPrimalCache` owns deep copies of:

- exact child start time and applied `dt`;
- physical `c0` and fixed physical `s`;
- incoming state;
- production auxiliary diagnostic;
- production stage tendency;
- outgoing state.

`HyperviscosityTangentCache` references its owned primal cache and owns deep
copies of:

- physical `delta_c0`;
- incoming state direction;
- diagnostic direction;
- tendency direction;
- outgoing state direction.

Reverse results likewise own all returned primal and dual data.  No cached field
aliases caller inputs, `GeneralRK` scratch, or another cached field.  Frozen
dataclasses make the data contract explicit; Firedrake objects remain mutable,
so ownership rather than Python-level immutability is the operative guarantee.

## Forward tangent

The code directionally differentiates both production one-forms before their
corresponding mass solves:

\[
  M_a\,\delta q_a=-K_a\,\delta x_a,
\]

\[
  M_a\,\delta F_a=
  \delta c_0r^sK_aq_a+c_0r^sK_a\,\delta q_a,
\]

\[
  \delta x_+=\delta x+\Delta t\,\delta F.
\]

Both the parameter-only direction and a combined state/control direction are
part of the focused certification test.

Because this child map is bilinear in `(x,c0)`, its centered directional
difference has no truncation term: the even quadratic term cancels exactly.
The focused oracle therefore uses the deterministic sequence
`[1e-2,1e-3,1e-4,1e-5]`, requires the moderate values (including `1e-3`) to
meet the intended tolerance, and records every error.  It does not demand a
factor-of-four regime; degradation at the smaller steps is subtraction and
linear-solve roundoff, which is why an artificially tiny single epsilon is
inappropriate here.  The independent dense tangent tolerance is unchanged.

## Ordinary dual reverse

Given `lambda_plus_star`, the Euler update first gives

\[
  \bar F^\star=\Delta t\,\lambda_+^\star.
\]

The code solves a mixed state-mass system

\[
  M\psi=\bar F^\star
\]

and contracts the actual production main right-hand-side one-form with `psi`.
UFL derivatives of that contracted scalar produce the physical `c0` gradient
and the diagnostic dual.  A second primal solve

\[
  M_{aux}\zeta=\bar q^\star
\]

is followed by differentiation of the contracted production diagnostic
right-hand side.  The incoming dual is the identity contribution
`lambda_plus_star` plus the main and diagnostic state pullbacks.

For this hyperviscosity form, the contracted main right-hand side depends
directly on the diagnostic and `c0`, not on the incoming state.  Thus its
direct state derivative is exactly zero.  The raw delayed UFL derivative can
still present as a one-form whose expanded integrand is identically zero.
Conversely, `ZeroFormAssembler` denotes rank-zero/functional assembly and is
not evidence of mathematical zero.  Raw UFL arity therefore proves neither
zero nor nonzero.  The helper first applies `expand_derivatives`, then removes
only integrals whose expanded integrand is canonical UFL `Zero`.  Only
`ZeroBaseForm` or a resulting form with no nonzero integrals is structural
zero.  It becomes an explicitly zeroed concrete `Cofunction`; every nonzero
dual form must have exactly one UFL argument before assembly and still receives
the strict dual-space check.  All nonidentity state pullback continues to enter
through reversal of the diagnostic equation.

## Incremental dual reverse and mixed curvature

The incremental Euler edge gives

\[
  \delta\bar F^\star=\Delta t\,\mu_+^\star,
  \qquad M\,\delta\psi=\delta\bar F^\star.
\]

The implementation differentiates the ordinary contracted main pullbacks in
the cached combined direction and solves the resulting incremental diagnostic
dual through the auxiliary mass matrix.  It then differentiates the ordinary
contracted diagnostic pullback.  No dense timestep Jacobian, inverse matrix,
third-order tensor, internal finite difference, pyadjoint operation, or
standalone prototype call is used.

Although the child map is separately linear in state and `c0`, it is bilinear
jointly.  Its physical HVP contains two nonzero contributions:

\[
  h_{c_0}^{(\mu)}
  =-\Delta t\,\langle\mu_+,Hx\rangle,
\]

\[
  h_{c_0}^{(\delta x)}
  =-\Delta t\,\langle\lambda_+,H\delta x\rangle.
\]

The production result reports these separately.  Syntactic containment checks
on `c0` are invalid because physical `s` and `c0` are split components of one
mixed coefficient Function: the nonzero first `c0` derivative legitimately
retains dependence on fixed `s` and may retain mixed parent/split syntax.  Pure
curvature is instead certified by constructing the second UFL directional
derivative in the unit `c0` direction and expanding it.  Mathematical zero does
not require UFL to canonicalize that derivative: the installed stack retains a
nonempty rank-zero scalar form.  A canonical structural zero bypasses assembly;
otherwise the form must have zero arguments and at least one domain and is
assembled normally.  Its actual scalar value is returned—no pure-control value
is hard-coded.  Tests require the first physical-`c0` pullback to be nonzero,
compare the assembled pure value with the dense oracle's exact zero using a
machine-precision tolerance scaled by both mixed terms, and require the total
to equal all three returned contributions while differing from either mixed
term alone.

The incoming incremental state dual is also covered by the encoded
certification.  For the test-only dense representation

\[
  A(c_0)=I-\Delta t\,c_0H,
\]

an independent mass/stiffness construction checks its dual coefficient vector
directly against

\[
  \mu_-^\star=A(c_0)^T\mu_+^\star
  -\Delta t\,\delta c_0H^T\lambda_+^\star.
\]

This is a test oracle only; the production reverse continues to differentiate
the deployed variational graph.  A second check simultaneously perturbs the
incoming state, physical `c0`, and terminal residual adjoint, centered-
differences the unchanged legacy incoming primal L2 adjoint, and compares it
with an explicit L2 Riesz representative of `incremental_state_adjoint_in`.

## Independent verification design

`tests/test_production_hyperviscosity_hvp.py` uses
`tests/tswe_rol_small.cfg`, selects the hyperviscosity child by its semantic
term list, and constructs its mesh on `COMM_SELF`.  It checks:

1. cached output against unchanged legacy Euler output;
2. parameter-only and combined tangents against centered legacy forwards;
3. new physical gradient against the unchanged legacy coefficient gradient;
4. the explicit L2 Riesz representative of the new incoming dual against the
   legacy primal adjoint;
5. exact HVP against centered differences of independent legacy gradients for
   a terminal least-squares objective;
6. a test-only tiny dense mixed mass/stiffness oracle for the diagnostic,
   tendency, forward map, tangent, gradient, state pullback, HVP, and incoming
   incremental state dual;
7. centered differences of unchanged legacy incoming L2 adjoints under
   simultaneous state, physical-`c0`, and terminal-adjoint perturbations;
8. the reverse/tangent dual-pairing identity;
9. caller non-mutation, cache ownership, dual-input preservation, and repeated
   results.

Production code never calls the dense oracle or finite-difference checks.

The ordinary reverse test additionally requires an exactly zero direct main
state adjoint and a nonzero diagnostic state adjoint that accounts for the
entire nonidentity state pullback.

Private debug records retain the exact contracted form and the raw, expanded,
and normalized derivatives created inside the reverse path, together with
arity/integral/domain metadata.  The focused test inspects these runtime
objects and also verifies that passing a nonzero rank-zero functional to dual
assembly raises the arity error before Firedrake assembly.

## Authoritative certification

Milestone status: **COMPLETE**.

The authoritative external results are:

- focused production hyperviscosity HVP: `9 passed, 245 warnings in 43.99s`;
- targeted Firedrake regressions: `30 passed, 16205 warnings in 193.65s`;
- NumPy ODE reference: `28 passed, 1 xfailed, 6 warnings in 9.25s`;
- full repository: `79 passed, 1 skipped, 1 xfailed, 16946 warnings in
  524.20s`;
- `git diff --check`: passed.

External numerical certification passed for:

- cached forward equivalence;
- parameter-only and combined tangents;
- dual reverse against the legacy physical gradient;
- the explicit Riesz state pullback against the legacy adjoint;
- the reverse/tangent pairing identity;
- the exact physical-`c0` HVP;
- the noncanonical second physical-`c0` derivative was assembled as a valid
  scalar functional rather than replaced by a hard-coded zero;
- its assembled value passed the scale-aware numerical-zero check and the
  independent dense oracle;
- the incremental-adjoint and state-direction mixed HVP terms each passed
  their independent checks;
- `incremental_state_adjoint_in` passed both the direct dense dual-vector
  oracle and centered differences of unchanged legacy incoming L2 adjoints;
- the centered legacy-gradient HVP ladder;
- cache ownership, non-aliasing, input preservation, and repeatability.

The warning families are non-blocking installed-library and existing-repository
warnings.  No warning was treated as a certification failure.

## Limitations before Lie propagation

- Only one explicit Euler stage and `terms=["hyperviscosity"]` are accepted.
- `s` is fixed; there is no `s` tangent, gradient, or HVP.
- The helper requires the production coefficient order `[s,c0]` and one
  trainable hyperviscosity term.
- The dry-core RK4, DG SSPRK43, moist-physics Euler, and Lie composition have no
  tangent or incremental reverse in this milestone.
- There is no multistep trajectory/checkpoint policy.
- There is no PyROL `hessVec` or normalized-control scaling.
- There is no MPI certification.
- The legacy Lie reverse time-argument behavior is untouched.  New primal
  caches retain the exact child start time instead.
