# DIMSWE production dry Lie HVP

## Scope and certified configuration

This implementation transfers the exact discrete-HVP machinery to the
production dry three-field TSWE timestep in `tests/tswe_rol_small.cfg`.  One
complete timestep is the deployed forward Lie composition

1. classical four-stage RK4 with `terms=["model"]`;
2. explicit Euler with `terms=["hyperviscosity"]`.

Both children use one subcycle, so the production `LieSplittingIntegrator`
passes the parent start time `t_n` and full parent step `dt` to both children.
The forward child order is therefore dry RK4 then hyperviscosity Euler; the
reverse order is Euler then RK4.  Runtime guards require the exact tableaus,
term lists, `[1,1]` subcycles, and dry state block `[v,h,S]`.  An alternative
RK tableau, SSPRK43, a moist state, or a changed split is rejected rather than
silently treated as certified.

The only physical scalar control is `c0`.  The exponent `s` remains fixed.
The dry child has no direct dependence on either trainable hyperviscosity
coefficient, so its direct control action, gradient, and HVP contribution are
exactly zero.  It still transports state tangents and both ordinary and
incremental adjoints.

## Dry RK4 primal stage graph

Write the production dry one-form as

\[
  B(x,t;\widehat x)=-R_{\rm model}(x,t;\widehat x),
\]

which is the same sign convention used by `GeneralRK`.  With the mixed L2 mass
map `M`, the classical RK4 graph is

\[
  X_i=x_n+\Delta t\sum_{j<i}a_{ij}F_j,
  \qquad M F_i=B(X_i,t_n+c_i\Delta t),
\]

\[
  x_{n+1}=x_n+\Delta t\sum_i b_iF_i.
\]

The coefficients are the exact deployed classical tableau:

\[
A=\begin{bmatrix}
0&0&0&0\\
1/2&0&0&0\\
0&1/2&0&0\\
0&0&1&0
\end{bmatrix},\quad
b=(1/6,1/3,1/3,1/6),\quad
c=(0,1/2,1/2,1).
\]

`DryRK4PrimalCache` owns exact scalar `t0` and `dt` and deep copies of the
incoming state, all four stage states, all four stage tendencies, and the
outgoing state.  Cached forward execution calls the unchanged production
forward child; stage states are materialized from the copied stage tendencies
using the deployed tableau.

The deployed UFL graph does **not** contain a standalone stage-state
`Function` `X_i`.  `GeneralRK` constructs and retains one exact form object
`B_i` per stage by replacing the splits of its production-owned base-state
`Function`, `xk`, with
`xk + dt*sum_j A[i,j]*Fi[j]`.  Consequently the UFL coefficients of the exact
stage form include `xk` and each live predecessor tendency `Fi[j]` on a
nonzero tableau edge, together with production `t`, `dt`, and shared fixed
model fields.  The exact objects are the same objects used to construct the
deployed stage residuals and solvers.

The former HVP path independently called `model.rhs` over a helper-owned mixed
state `Function`.  That reconstructed form genuinely depended on its helper
state, but it did not contain production `xk` or predecessor `Fi[j]`
coefficients by identity.  Materializing their values into a copied `X_i`
preserved the primal field but froze the exact UFL coefficient/evaluation
graph.  This is the first structural dependency mismatch: `xk` is replaced at
stage 0, and the first live predecessor edge `Fi[0]` is absent beginning at
stage 1.  The authoritative diagnostics first resolved a numerical reverse
discrepancy at the stage-3 RHS pullback.

## Dry tangent

For an incoming state direction `w_n`, the cached materialized direction still
satisfies:

\[
  W_i=w_n+\Delta t\sum_{j<i}a_{ij}G_j,
\]

\[
  M G_i=D_xB(X_i,t_i)[W_i],
\]

\[
  w_{n+1}=w_n+\Delta t\sum_i b_iG_i.
\]

The active implementation differentiates the exact production coefficient
graph without replacing it by a separately reconstructed `B(X_i)`:

\[
 M G_i=D_{xk}B_i[w_n]
       +\sum_{j<i}D_{F_j}B_i[G_j].
\]

The `D_{F_j}` form already contains the deployed `dt*A[i,j]` edge.  This is
the exact UFL derivative of the form evaluated by `GeneralRK`, including its
coefficient identities and operation ordering.  Only the mixed mass system
is solved; no full-stage or full-step Jacobian is assembled.
`DryRK4TangentCache` continues to own copies of every materialized `W_i`, every
`G_i`, the incoming direction, and the outgoing direction together with its
owned primal cache.  The independently reconstructed form is retained only
for stage-local diagnostic comparison.

## Ordinary dual reverse

Adjoints are genuine mixed `Cofunction` objects.  Given the outgoing dual
`lambda_plus_star`, stage tendency duals are accumulated in strict order
`3,2,1,0` using exact stored-form edges:

\[
 \bar F_i^\star=\Delta t\,b_i\lambda_+^\star
 +\sum_{j>i}D_{F_i}\langle B_j,\psi_j\rangle.
\]

For each stage, the implementation explicitly solves a primal Riesz problem

\[
  M\psi_i=\bar F_i^\star
\]

and contracts the deployed one-form with `psi_i`.  The incoming-state
transpose action is the UFL derivative with respect to the exact production
`xk` coefficient:

\[
  \bar x_{k,i}^\star=D_{xk}\langle B_i,\psi_i\rangle.
\]

The incoming dual is

\[
  \lambda_n^\star=\lambda_+^\star+\sum_i\bar x_{k,i}^\star.
\]

Reverse results own copied tendency duals, primal reverse auxiliaries, and
incoming-state and predecessor-tendency edge duals.  The legacy
`stage_state_adjoint` field is retained as the exact `xk` pullback, preserving
the public result API.  The direct dry `c0` gradient is zero.

## State-curvature incremental reverse

The dry map is nonlinear, so differentiating only the transported incremental
adjoint would omit the discrete state curvature.  Let `mu_plus_star` be the
outgoing incremental dual.  The incremental stage tendency dual uses the
derivatives of the exact predecessor edge pullbacks:

\[
 \delta\bar F_i^\star=\Delta t\,b_i\mu_+^\star
 +\sum_{j>i}\delta\left(
   D_{F_i}\langle B_j,\psi_j\rangle\right),
\]

followed by `M delta_psi_i = delta_bar_F_i_star`.  The exact incremental stage
pullback is

\[
 \delta\bar X_i^\star=
  D_{xk}\langle B_i,\delta\psi_i\rangle
 +D_{(xk,F_{<i})}\left(
   D_{xk}\langle B_i,\psi_i\rangle
  \right)[w_n,G_{<i}].
\]

The same full-graph directional derivative is applied to every predecessor
`D_Fj` pullback.  These are the exact `(D A_dry[w])^T lambda` terms, constructed
with nested UFL directional derivatives of the stored production forms.  The
incoming result is

\[
 \mu_n^\star=\mu_+^\star+\sum_i\delta\bar X_i^\star.
\]

No finite difference, dense Jacobian, third-order tensor, or pyadjoint object
appears in production code.  The direct dry physical-`c0` HVP is zero.

## Two-child Lie composition

`DryLiePrimalCache` owns the parent-step metadata, parent input and output, and
the two owned child caches.  Its tangent accepts the combined direction
`(delta_x_in, delta_c0)` and evaluates

\[
 w_d=A_{dry}\,\delta x_{in},\qquad
 w_+=A_{hv}\,w_d+B_{hv}\,\delta c_0.
\]

This covers physical-`c0`-only, initial-condition-only, and combined
directions.  Ordinary reversal first applies the certified hyperviscosity
Euler reverse and then the dry RK4 reverse.  It returns the incoming mixed
state dual and the physical scalar `c0` gradient.  Incremental reversal uses
the same reverse-child order and returns the incoming incremental mixed dual
and physical scalar `c0` HVP.  Only the Euler child contributes a direct
scalar; the RK4 child transports and curves the state information received
from it.

## Natural pairing and Riesz representatives

State derivatives live in the algebraic dual `V_h*`, not in `V_h`.  For a
mixed primal direction `v` and dual `ell_star`, the natural pairing is

\[
  \langle\ell^\star,v\rangle,
\]

evaluated by UFL `action` and scalar assembly.  A field-valued L2
representative is obtained only through the explicit solve

\[
  M z=\ell^\star.
\]

The new initial-condition gradient and HVP outputs remain `Cofunction`s.
Explicit Riesz conversion is used only when comparing them with the legacy
primal initial-condition adjoint or when a caller explicitly requests a field
representative.

## Multistep reduced terminal HVP

For one or more complete Lie timesteps and terminal target `d`, the reduced API
uses

\[
 J=\tfrac12\int\|x_N-d\|^2\,dx,
\]

\[
 \lambda_N^\star=M(x_N-d),\qquad
 \mu_N^\star=Mw_N.
\]

All primal and tangent timestep caches are retained.  Reversal visits complete
timesteps `N-1,...,0`, and within each timestep visits hyperviscosity then dry
RK4, with RK stages `3,2,1,0`.  Scalar gradient and HVP contributions are
summed over hyperviscosity children.  The final reduced blocks are

\[
 \nabla_{x_0}J=\lambda_0^\star,\qquad
 H_{x_0}q=\mu_0^\star,
\]

together with the physical `c0` gradient and physical `c0` HVP component.
There is no normalized-control scaling and no PyROL adapter in this API.

## Mixed Hessian blocks

The off-diagonal blocks use the natural scalar/dual pairing.  For a state
direction `delta_x0` and scalar direction `delta_c0`, symmetry is checked as

\[
 \langle H_{x_0,c_0}\,\delta c_0,\delta x_0\rangle
 \simeq
 \delta c_0\,H_{c_0,x_0}\,\delta x_0.
\]

This comparison does not identify primal and dual coefficient vectors and does
not insert an implicit mass inverse.

## Verification ladder

`tests/test_production_dry_lie_hvp.py` uses the deterministic serial
`tests/tswe_rol_small.cfg` case and encodes:

1. dry cached forward against unchanged legacy forward;
2. dry tangent against a centered-forward epsilon ladder;
3. dry ordinary dual reverse against the independently executed legacy
   initial-condition adjoint after explicit L2 Riesz conversion;
4. dry tangent/reverse pairing;
5. dry incremental incoming dual against centered differences of independent
   legacy incoming adjoints, including a differentiated pairing check;
6. complete Lie cached forward against unchanged production Lie forward;
7. Lie tangents for `c0`-only, IC-only, and combined directions;
8. one- and three-step reduced scalar/field gradients against independent
   legacy reversal;
9. all scalar and field HVP blocks against centered legacy-gradient ladders;
10. mixed-block Hessian symmetry in the natural pairing;
11. exact forward-child, reverse-child, and reverse-stage order;
12. input preservation, cache ownership, non-aliasing, scratch independence,
    and bitwise repeatability.

The epsilon ladder is `[0.04, 0.02, 0.01, 0.005]`.  Tests require an observed
quadratic centered-difference regime whenever the selected nonlinear path has
a truncation term.  A one-step `c0`-only forward map is affine in `c0`, and its
one-step reduced gradients have centered differences with no truncation term;
those cases require agreement at multiple moderate epsilons and record the
smaller-step roundoff trend instead of inventing a factor-of-four requirement.
The accepted focused hyperviscosity certification remains unchanged and is
rerun separately.

## Focused diagnostic protocol

An authoritative external run of the original 16 focused cases reported
`3 passed, 13 failed`.  The supplied failure characterization showed a flat
dry-tangent finite-difference error near `6.958e-8`, a flat complete-Lie state
tangent error near `6.817e-8`, a new-versus-legacy dry incoming-adjoint
discrepancy near `9.541e-6`, a flat incremental-versus-centered-legacy
discrepancy near `1.067e-5`, and strong three-step amplification at `dt=100`.
The four subsequent authoritative diagnostics all passed.  They established
that cached and deployed primal graphs are bitwise identical, the reconstructed
tangent/reverse graph is internally transposed to roundoff, the legacy scalar
gradient follows centered deployed objectives to about `1e-12`, and the
reconstructed derivative retains a `7e-10` to `1.2e-9` scalar-gradient plateau
and a `7e-8` to `9e-8` tangent floor.  They localized the first measured reverse
difference to the stage-3 RHS pullback and showed that the three-step `dt=100`
discrepancy is amplification of that local derivative mismatch, not a separate
multistep-order defect.

The original four diagnostic tests are:

1. `test_diagnostic_dry_rk4_independent_tangent_and_adjoint_oracles` uses
   three deterministic mixed-state probes and the wider epsilon ladder
   `[0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]`.  It records the tangent
   centered-forward errors without a rate assertion, the exact new
   tangent/dual pairing, the corresponding legacy-primal pairing, centered
   scalar objective derivatives, both scalar-gradient errors, the
   new-versus-legacy field discrepancy, and an explicit mass/Riesz roundtrip.
2. `test_diagnostic_dry_rk4_stage_graph_and_local_pairings` records the exact
   tableau, stage times, final weights, reverse edge coefficients, `B=-R`
   sign, identity edge, stage order, reconstruction errors, aliases, integral
   metadata, and solver settings.  It independently repeats the legacy
   forward, compares every copied stage state and tendency, checks every new
   local pullback pairing, compares every new tendency dual and reverse
   auxiliary with the corresponding legacy `mu_i`, and separately checks all
   reverse accumulations.
3. `test_diagnostic_dry_rk4_incremental_independent_new_reverse_oracle`
   center-differences the new ordinary dual reverse with the matching
   perturbed terminal duals, as well as the legacy primal reverse.  It also
   differentiates both the new ordinary-reverse pairing and the independently
   evaluated terminal tangent pairing.
4. `test_diagnostic_multistep_dt_amplification` records one- and three-step
   terminal norms, objectives, physical-`c0` gradients, and IC-gradient norms
   and differences for `dt=[100,50,25,12.5]`.

The correction adds
`test_diagnostic_dry_rk4_stage_local_exact_production_form_oracle`.  For every
stage and two deterministic stage directions it:

- extracts every UFL coefficient and records Python object identity, UFL
  count, name, type, and matches against production `xk`, `t`, `dt`, every
  `Fi[j]`, and the reconstructed helper coefficients;
- records the exact stored and reconstructed form-object identities and
  integral metadata;
- repeats the exact deployed stage solve at the cached graph and after
  perturbing only production `xk`, leaving predecessor `Fi[j]` values and the
  symbolic stage expression intact;
- compares centered exact-stage solves with the derivative of the stored
  production form and with the former reconstructed derivative;
- records exact-production versus reconstructed tangent differences.

The authoritative corrected first-order diagnostic passed.  Whole-child dry
tangent errors recover factor-of-four centered convergence before reaching a
`4e-13` to `7e-13` floor.  The new incoming Riesz adjoint differs from legacy
by `2.487998645380611e-16`, and the new dual differs from the mass-mapped
legacy dual by `2.30493604865847e-16`.  Tangent/adjoint pairings are zero or
roundoff, and new and legacy scalar directional gradients agree to displayed
precision.  This externally confirms the exact production-form correction at
first order.

The first corrected stage-local execution reached the numerical floor
immediately: its exact-form relative errors began
`[8.6666e-15, 4.1453e-14, 5.7032e-14, 1.9660e-13, ...]`.  The former mandatory
factor-of-four assertion therefore stopped the test before all four stages and
both directions could be recorded; it was an inappropriate oracle for an
already roundoff-limited derivative, not a production derivative failure.

The completed stage-local payload subsequently showed two valid regimes.
Stage 0 starts at the immediate floor.  Stages 1--3 exhibit at least three
clean consecutive ratios in `[3.8,4.2]` (for example approximately
`[4.00002,3.99985,4.00042]`) before reaching a `1e-12`-scale floor.  The
reconstructed diagnostic tangents remain wrong by approximately `6e-6` to
`3.5e-5`; they remain diagnostic-only.

Every diagnostic emits a single sorted JSON record prefixed with
`DRY_LIE_DIAGNOSTIC` when pytest is run with `-s`; it also stores the same JSON
as a pytest `record_property`.  The diagnostic tests assert only graph shape,
the exact certified tableau/order, required coefficient identities, and cache
non-aliasing.  Stage-local exact tangents use a two-branch certification: an
observed regime requires at least three consecutive ratios in `[3.8,4.2]`,
monotonic error reduction across that window, and at least one error at or
after the window below `1e-11`.  It does not require the first four errors to
have reached the floor.  Otherwise all first four largest/moderate errors must
already be below `1e-11`, which certifies an immediate roundoff floor without
ratio requirements.  The start, inclusive end, ratios, monotonic flag, minimum
subsequent error, threshold, and certification result are recorded per stage
and direction.  Numerical certification is asserted only after all eight
records are emitted.  No original whole-child certification tolerance,
convergence-rate assertion, deployed weak form, or `dt=100` stress case was
changed.

## Final external certification

The authoritative serial certification is complete.  The final focused run
passed all 21 cases, including the floor-aware stage-local classifier.  The
stage graph, exact-form tangent and reverse, exact-form incremental reverse,
one- and three-step reduced derivatives, all three control-direction classes,
mixed-block symmetry, ownership, non-aliasing, and repeatability checks are
therefore certified in the stated dry scope.  In particular, the `dt=100`
three-step stress case passes without changing the deployed weak forms,
choosing a smaller timestep, or loosening a certification tolerance.

The exact external results were:

| Run | Result | Warnings | Time |
| --- | --- | ---: | ---: |
| Production dry RK4/Lie focused suite | 21 passed | 5,377 | 207.92 s (0:03:27) |
| Accepted Firedrake regression set | 39 passed | 16,448 | 198.07 s (0:03:18) |
| `ode_adjoint` | 28 passed, 1 xfailed | 6 | 9.32 s |
| Full repository suite | 100 passed, 1 skipped, 1 xfailed | 22,314 | 618.61 s (0:10:18) |

The full-suite warnings are nonblocking and comprise 21,891 NumPy-2.5
shape-assignment deprecations from PyOP2 and the existing dense-oracle test
helpers, 25 FINAT notices that discontinuous Lagrange on quadrilaterals is
represented by DQ, 392 UFL metadata-stringification notices for tensor-product
or ordinary quadrature rules, and six existing SciPy optimizer notices.  The
SciPy notices state that L-BFGS-B ignores `hessp` and that `disp` is an unknown
option.  PETSc also reports the pytest `-q` option as an unused database option
after successful Firedrake runs; it is not a pytest warning or test failure.
The single skip is the established optional Matplotlib plotting import, and
the expected failure is the pre-existing combined ODE parameter/initial-state
optimization case.

## Limitations before later transfers

The certified scope is dry three-field TSWE, classical RK4 plus one
hyperviscosity Euler child, physical `c0`, terminal least squares, and stored
one- or multistep caches.  It does not cover DG SSPRK43, moist physics,
six-field MTSWE, checkpoint scheduling, MPI, normalized controls, a PyROL
initial-condition vector, or PyROL `hessVec`.  No JAX or neural-network path is
used or introduced.
