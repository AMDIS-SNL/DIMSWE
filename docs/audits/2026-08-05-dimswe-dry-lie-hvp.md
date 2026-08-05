# DIMSWE production dry Lie HVP audit

## Repository gate and status

- Repository root:
  `/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615`
- Branch: `dev/dimswe-dry-lie-hvp`
- Starting HEAD: `9d832406d810b1f47a744944d3151888cd64c15d`
- Starting tracked tree: clean
- Starting index: clean
- Pre-existing untracked files left untouched: `.DS_Store` and
  `docs/.DS_Store`
- Audit status: **COMPLETE**

The implementation and certification suite were authored through the complete
multistep reduced-HVP scope.  The managed Codex sandbox cannot execute the
Firedrake numerical checks, so all numerical execution was performed by the
authoritative external serial environment.  The corrected exact-form focused
suite, accepted regressions, ODE suite, and full repository suite are green;
the full suite has no failures or unexpected errors.  No file was staged or
committed.

## Exact files changed

- `dimswe/timestepping.py`
- `dimswe/dry_lie_hvp.py` (new)
- `tests/test_production_dry_lie_hvp.py` (new)
- `docs/DIMSWE_DRY_LIE_HVP.md` (new)
- `docs/audits/2026-08-05-dimswe-dry-lie-hvp.md` (new)

`dimswe/hyperviscosity_hvp.py`, `dimswe/optimize.py`, and all legacy objective,
gradient, forward, and adjoint implementations were not modified.

## Exact production configuration

The focused test uses `tests/tswe_rol_small.cfg`:

- 2 by 2 periodic rectangle with quadrilateral cells;
- spatial order 3 and family Q;
- dry TSWE mixed state `[v,h,S]`;
- initialized deterministic double-vortex state;
- forcing list `[hyperviscosity]`;
- production Lie child methods `[RK4,Euler]`;
- production term order `[[model],[hyperviscosity]]`;
- child subcycles `[1,1]`;
- parent and child `dt = 100.0`;
- both deployed children start at the exact parent-step time;
- physical `c0 = 0.14`;
- fixed physical `s = 3.2`;
- physical direction `delta_c0 = 0.035`;
- deterministic non-proportional mixed state direction and deterministic
  mixed probe;
- direct serial linear solver parameters;
- `COMM_SELF` mesh construction;
- one and three complete parent timesteps;
- no MPI invocation.

The centered-difference sequence is
`[0.04, 0.02, 0.01, 0.005]`.  Nonlinear paths require an observed quadratic
regime.  One-step `c0`-only affine/bilinear paths have no centered truncation
term and instead require agreement at multiple moderate epsilons while
recording smaller-step roundoff.

## Implemented dry RK4 graph

The dry helper validates the actual production `RK4` class, exact classical
four-stage tableau, exact `terms=["model"]`, dry state `[v,h,S]`, and absence
of dry stage diagnostics.  It constructs the deployed one-form with the same
`B=-model.rhs` sign as `GeneralRK` and verifies that the resulting dry form has
no trainable-coefficient dependency.

Cached forward calls the unchanged child forward implementation.  The owned
primal cache contains scalar `t0`, scalar applied `dt`, and deep copies of the
incoming state, four materialized stage states, four production stage
tendencies, and outgoing state.  The corrected tangent uses the exact four
stored `GeneralRK` stage form objects.  For stage `i` it differentiates every
live production coefficient edge: `xk` in the incoming direction and each
predecessor `Fi[j]` in its cached tangent direction.  It applies explicit mixed
mass solves and owns copies of the incoming direction, all four materialized
stage-state directions, all four stage-tendency directions, and the outgoing
direction.  There is no direct `c0` tangent action in this child.

Ordinary reverse uses genuine mixed `Cofunction` stage and state adjoints.  It
visits stage indices `3,2,1,0`, explicitly solves the mixed primal mass/Riesz
problem for each stage-tendency dual, and assembles transpose actions from UFL
derivatives of contracted deployed forms.  Later-stage-to-earlier-stage
contributions are the direct `D_Fi <B_j,psi_j>` derivatives of the exact form,
not a separately factored reconstructed pullback.  Returned stage tendency
duals, primal reverse auxiliaries, incoming-state duals, and predecessor-edge
duals are copied.  The incoming state dual includes the identity edge and all
four direct `xk` pullbacks.  The dry physical `c0` gradient is exactly zero.

Incremental reverse independently performs the ordinary reverse and then
visits stages `3,2,1,0` for the incremental pass.  Besides propagation of the
outgoing incremental dual, it includes the nested UFL state-curvature term
`(D A_dry[W])^T lambda` over `xk` and every live predecessor `Fi[j]` edge at
every stage.  The incremental predecessor-edge duals are copied and reversed
directly.  The direct dry physical `c0` HVP is zero; the incoming incremental
mixed state dual is returned and owns its data.

No full timestep Jacobian, dense production matrix, internal finite
difference, pyadjoint object, or changed weak form is used.

## Implemented two-child and multistep graph

The complete step validates exactly two deployed children in the forward
order dry RK4 then hyperviscosity Euler, with exact term lists and `[1,1]`
subcycles.  Both children receive the production time `t_n` and full `dt`.
The accepted `ProductionHyperviscosityEulerHVP` implementation is reused
unchanged.

`DryLiePrimalCache` and `DryLieTangentCache` own parent input/output data and
the corresponding owned child caches.  The tangent accepts
`(delta_x_in,delta_c0)` and supports:

- physical-`c0`-only directions;
- initial-condition-only directions;
- combined initial-condition/physical-`c0` directions.

Ordinary and incremental reverses visit hyperviscosity Euler before dry RK4.
The ordinary result returns an incoming state `Cofunction` and physical scalar
`c0` gradient.  The incremental result returns an incoming incremental state
`Cofunction` and physical scalar `c0` HVP.  Only the Euler child contributes a
direct scalar block; dry RK4 transports and curves the received state data.

The separate reduced API accepts any positive number of complete timesteps.
For terminal least squares it constructs exactly
`lambda_N_star=M(x_N-d)` and `mu_N_star=M w_N`, reverses timesteps in descending
order, and returns:

- objective value;
- physical `c0` gradient;
- initial-condition gradient as a `Cofunction`;
- physical `c0` HVP component;
- initial-condition HVP as a `Cofunction`;
- owned primal and tangent timestep caches;
- copied terminal duals, trajectory states/directions, and reverse results.

No PyROL normalization is present.

## Cache ownership and mutation policy

All public cache/result dataclasses are frozen containers.  Every Firedrake
state, direction, tendency, ordinary dual, incremental dual, and reverse
auxiliary stored at a cache boundary is deep-copied.  Parent Lie caches and
multistep trajectories own data independently of caller inputs and mutable
`GeneralRK`/Lie scratch.  The tests explicitly check distinct `dat` objects,
scratch-reset independence, caller input preservation, repeated-run cache
independence, and bitwise repeatability.  Frozen containers express the graph
contract; the contained Firedrake objects remain mutable, so ownership is the
operative guarantee.

## Independent checks encoded in the focused test

The focused file contains the original 16 milestone cases plus five independent
diagnostic/oracle cases.  Its milestone cases encode:

1. dry RK4 cached forward versus unchanged legacy dry forward;
2. dry RK4 tangent versus a centered legacy-forward ladder;
3. dry dual reverse versus the independent legacy incoming adjoint after an
   explicit L2 Riesz conversion;
4. dry tangent/reverse natural pairing;
5. dry incremental incoming dual versus centered differences of independent
   legacy incoming adjoints;
6. the directional derivative of the dry tangent/adjoint pairing;
7. complete cached Lie forward versus unchanged legacy Lie forward;
8. complete Lie tangents for `c0`-only, IC-only, and combined directions;
9. independent reduced ordinary gradients for one and three timesteps;
10. physical `c0` gradient versus the legacy coefficient gradient;
11. physical `c0` HVP versus centered legacy coefficient-gradient ladders;
12. initial-condition gradient versus the legacy IC gradient;
13. initial-condition HVP versus centered legacy IC-gradient ladders;
14. both output blocks for combined directions;
15. mixed-block Hessian symmetry in the natural primal/dual pairing;
16. exact child/stage order, ownership, no aliasing, non-mutation, and bitwise
    repeatability.

The managed sandbox did not measure these values.  The authoritative external
runs and the diagnostic history are recorded below without treating the
legacy path as the sole ground truth.

## Final source review

The final read-only source audit confirmed:

- every public primal, tangent, reverse, incremental-reverse, and reduced
  result boundary stores copied Firedrake data; the frozen cache containers do
  not alias caller inputs or mutable `GeneralRK`/Lie scratch;
- `GeneralRK.production_stage_rhs_forms[i]` is the identical `rhs_Fi` object
  inserted into `production_stage_residuals[i]` and used to construct the
  deployed stage solver; coefficient-identity diagnostics cover production
  `xk`, `t`, `dt`, and every live predecessor `Fi[j]` edge;
- active dry tangent, ordinary reverse, and incremental reverse operations all
  use `_production_stage_rhs`; `_reconstructed_stage_rhs` occurs only in
  coefficient/form diagnostics and the explicitly named reconstructed
  stage-tangent comparator;
- complete-step reversal is hyperviscosity Euler then dry RK4, multistep
  reversal descends in timestep index, and both dry reverse passes use stage
  order `3,2,1,0`;
- the dry child contributes exactly zero direct physical-`c0` derivative,
  while the hyperviscosity child supplies the scalar physical-`c0` block and
  dry RK4 transports and curves the field blocks;
- initial-condition gradients and HVPs, terminal duals, and incoming state
  adjoints are genuine mixed `Cofunction`s; scalar `c0` outputs remain physical
  unnormalized values;
- the mixed mass map is assembled explicitly, reverse auxiliaries and Riesz
  representatives are obtained by explicit mass solves, and natural
  dual/primal pairings use `action`;
- all legacy `take_forward_step`, `take_adjoint_step`, objective, Jacobian, and
  scalar-objective value/gradient entry points remain unchanged; the new
  cached and reduced APIs are additive and lazy; and
- the accepted hyperviscosity implementation and `dimswe/optimize.py` remain
  unchanged.

## Sandbox execution barrier

The first direct import attempt

```text
python -c "import firedrake; print('firedrake import ok')"
```

failed while PETSc initialized.  `PetscGetHostName()` called
`getdomainname()`, which the sandbox denied with `Operation not permitted` and
PETSc error code 88.  OpenMPI also reported that its TCP component could not
bind a local socket.

The later pytest processes progressed through collection and mesh/model setup,
but all Firedrake numerical fixtures then stopped when Loopy/Pytools attempted
to write its persistent compilation cache at
`/Users/arjunsharma/Library/Caches/pytools/...sqlite`.  That location is
read-only under the managed filesystem policy, producing
`sqlite3.OperationalError: attempt to write a readonly database`.  No cache or
network workaround was attempted, in accordance with the staged execution
rule.  These setup errors occurred before the requested numerical assertions.

## Focused and regression results

The final authoritative serial results are:

| Command/run | Result | Warnings | Time |
| --- | --- | ---: | ---: |
| `python -m pytest -q tests/test_production_dry_lie_hvp.py` | **21 passed** | 5,377 | 207.92 s (0:03:27) |
| Accepted Firedrake regression set | **39 passed** | 16,448 | 198.07 s (0:03:18) |
| `python -m pytest -q ode_adjoint` | **28 passed, 1 xfailed** | 6 | 9.32 s |
| `python -m pytest -q` | **100 passed, 1 skipped, 1 xfailed** | 22,314 | 618.61 s (0:10:18) |

The accepted Firedrake regression set is the production hyperviscosity HVP,
coefficient-gradient, initial-condition-gradient, ROL-adapter, and isolated
Firedrake HVP prototype files requested by the staged plan.  Its 16,448
warnings comprise 16,308 PyOP2 NumPy-2.5 shape-assignment deprecations, nine
FINAT quadrilateral-DQ notices, 113 UFL tensor-product quadrature-metadata
notices, and 18 equivalent NumPy shape-assignment deprecations in existing
dense-oracle test helpers.  The focused run comprises 5,311 PyOP2
deprecations, two FINAT notices, and 64 UFL metadata notices.  The ODE run has
three SciPy notices that L-BFGS-B ignores `hessp` and three that `disp` is an
unknown option.

Across the full run, the 22,314 nonblocking warnings comprise 21,891 NumPy
shape-assignment deprecations from PyOP2 and existing dense-oracle helpers, 25
FINAT quadrilateral-DQ notices, 392 UFL metadata-stringification notices for
tensor-product or ordinary quadrature rules, and the six SciPy optimizer
notices.  PETSc additionally reports pytest's `-q` as an unused database
option after successful Firedrake execution; it is outside pytest's warning
count and is not a failure.  The one skip is the established optional
Matplotlib plotting import.  The one expected failure is the pre-existing ODE
combined parameter/initial-condition optimization case.  There were no test
failures or unexpected errors.

The final focused run confirms:

- cached production and legacy forwards agree bitwise at all four stage
  states, all four stage tendencies, and the final state;
- exact-production stage derivatives exhibit either an immediate strict
  roundoff floor or factor-of-four convergence followed by a `1e-12`-scale
  floor, while whole-child tangent errors show factor-of-four convergence
  before a `4e-13` to `7e-13` floor;
- the new incoming Riesz adjoint and mass-mapped dual agree with the legacy
  results at `2.487998645380611e-16` and `2.30493604865847e-16` relative,
  respectively, and local/whole-child pairings are at roundoff;
- the corrected incremental reverse converges to centered corrected ordinary
  reverses and its differentiated pairing reaches roundoff; and
- one- and three-step initial-condition and physical-`c0` gradients agree at
  roundoff, including the primary `dt=100` stress case.

For historical traceability, the first external focused execution and the
diagnostics that led to the exact-form correction are retained below.

The authoritative external execution of the original focused file reported:

- `tests/test_production_dry_lie_hvp.py`: **3 passed, 13 failed**;
- accepted Firedrake regressions: **39 passed**;
- `ode_adjoint`: **28 passed, 1 xfailed**.

The externally supplied failure classes are:

- dry RK4 centered-tangent errors
  `[6.9579423e-8, 6.9579195e-8, 6.9579108e-8, 6.9579077e-8]`, with a similar
  complete-Lie state-direction floor near `6.817e-8`; the original
  factor-of-four assertion does not describe that sequence;
- new dry incoming dual after L2 Riesz conversion versus the legacy primal
  incoming adjoint: approximately `9.5407e-6`;
- new dry incremental incoming adjoint versus centered legacy adjoints:
  approximately `1.0666e-5`, flat over the tested epsilons;
- three-step physical `c0` gradient near `-1.61e16` at `dt=100`, with a
  new-versus-legacy relative difference near `2.13e-5`.

All four diagnostic cases subsequently passed in authoritative external
execution.  Their combined conclusions are:

- cached and legacy stage states, stage tendencies, and final dry RK4 state
  are bitwise identical;
- the reconstructed tangent/reverse pairings are internally exact at about
  `1e-16`, with local stage pairings at zero to `1.7e-16` and accumulation at
  about `1e-17`;
- that reconstructed tangent is not the exact derivative of the deployed
  forward evaluation graph: centered forward errors stay near `7e-8` to
  `9e-8` over the seven-epsilon ladder;
- centered scalar objectives approach the legacy gradient to about `1e-12`,
  whereas the reconstructed gradient generally plateaus near `7e-10` to
  `1.2e-9`;
- the first measured reverse discrepancy is the stage-3 RHS pullback, not the
  RK weights, order, identity edge, or tendency-adjoint accumulation;
- the incremental reverse is internally consistent with the reconstructed
  ordinary reverse and therefore inherits its first-order mismatch;
- the three-step `dt=100` difference is amplification of the local dry
  derivative mismatch and decreases strongly down the timestep ladder.  It is
  not a separate multistep accumulation-order defect.

The accepted regressions remained green.  No tolerance was loosened and no
smaller timestep was selected.

## Exact production-form dependency audit and correction

There is no standalone production stage-state `Function`.  The exact stage
form object is `GeneralRK.production_stage_rhs_forms[i]`, the same `rhs_Fi`
object used in the deployed residual `inner(test,Fi)*dx-rhs_Fi` and stage
solver.  Its stage expression is formed by UFL replacement as
`xk + dt*sum_j A[i,j]*Fi[j]`.  UFL coefficient extraction therefore contains,
by Python object identity:

- the production base-state `GeneralRK.xk[0]` at every stage;
- the exact predecessor `GeneralRK.Fi[j][0][0]` on each nonzero tableau edge;
- production `t` and `dt` wherever they remain after UFL simplification;
- the shared fixed model fields used by the dry form;
- no production trainable-coefficient `Function` for `terms=["model"]`.

The former comparator is a distinct form object returned by a separate
`model.rhs` call.  It contains the helper-owned
`dry_rk4_hvp_form_state` and helper time coefficient, plus shared fixed model
fields.  It contains neither production `xk` nor any production predecessor
`Fi[j]` by identity.  Its derivative variable is live in the reconstructed
form, but the production coefficient graph was materialized and frozen into a
copied stage state.  Thus the first identity mismatch is `xk` at stage 0; the
first missing live predecessor dependency is `Fi[0]` at stage 1.  The first
externally measurable pullback mismatch was stage 3.

The correction differentiates each exact stored form over all of its live
state edges:

```text
M G_i = D_xk B_i[w_n] + sum_{j<i} D_Fj B_i[G_j]
```

The exact `D_Fj` already contains its deployed `dt*A[i,j]` expression.  The
ordinary reverse is the direct transpose of those same forms:

```text
bar_F_i* = dt*b_i*lambda_plus* + sum_{j>i} D_Fi <B_j,psi_j>
lambda_in* = lambda_plus* + sum_i D_xk <B_i,psi_i>
```

The incremental reverse differentiates each corrected `xk` and predecessor
edge pullback along the corrected tangent coefficient directions.  Genuine
mixed `Cofunction`s, explicit mass/Riesz solves, strict stage order `3,2,1,0`,
owned caches/results, and all existing public APIs are retained.  The accepted
hyperviscosity child is unchanged.

A fifth diagnostic now evaluates every exact deployed stage solve after
perturbing only production `xk` while keeping the original predecessor `Fi`
coefficients and symbolic stage expression.  For two deterministic directions
and all seven moderate epsilons it compares the centered stage solve with both
the stored-form derivative and the reconstructed comparator.  It also records
all coefficient object identities and integral metadata.  The focused file
now contains 21 cases.  All original certification assertions and tolerances
remain unchanged.

The authoritative corrected first-order diagnostic passed and measured:

- whole-child centered-tangent factor-of-four convergence followed by a
  `4e-13` to `7e-13` floor;
- new-versus-legacy incoming Riesz discrepancy
  `2.487998645380611e-16`;
- new dual versus mass-mapped legacy dual discrepancy
  `2.30493604865847e-16`;
- tangent/adjoint pairings at zero or roundoff;
- new and legacy scalar directional gradients equal to displayed precision.

The first stage-local corrected run failed only in its test oracle.  Its exact
production-form errors began
`[8.6666e-15, 4.1453e-14, 5.7032e-14, 1.9660e-13, ...]`, so an unconditional
factor-of-four assertion was invalid at an immediate roundoff floor and
prevented the remaining stages/directions from being recorded.

The stage-local rule is now floor-aware.  It first looks for a factor-of-four
window among the largest/moderate epsilons and, if found, also requires the
sequence to reach below `1e-11`.  If no such window exists, all first four
largest/moderate errors must be below `1e-11`.  Each result records
`factor_of_four_then_roundoff_floor` or `immediate_roundoff_floor`, the ratios,
window, strict threshold, and certification boolean.  The final numerical
assertion is deferred until all four stages and both directions have been
recorded.  The reconstructed derivative remains
diagnostic-only.  Production mathematics and original whole-child assertions
were not changed.

The authoritative corrected incremental diagnostic also passed.  Centered
corrected-new ordinary-reverse errors decreased from `7.412e-8` to
`2.390e-12` over the seven-epsilon ladder; centered legacy Riesz-adjoint errors
decreased from `1.095e-7` to `3.007e-12`.  The differentiated pairing reached
roundoff-scale accuracy.  This strongly certifies the corrected exact-form
incremental reverse.

The second stage-local run completed all four stages, both directions, form
identity checks, certification classifications, and payload construction.  It
reached `_diagnostic_emit` and failed only because `json.dumps` cannot directly
serialize a `numpy.bool_`; no mathematical assertion failed.  Diagnostic
emission now recursively normalizes NumPy booleans, integer scalars, floating
scalars, and arrays to native JSON booleans, numbers, and lists.  Numeric and
Boolean values are not stringified, and emission remains mandatory.

The subsequent authoritative stage-local payload showed exact production-form
stages 1--3 with a clean three-ratio factor-of-four window followed by a
`1e-12`-scale floor.  One representative sequence was
`[1.1187e-9,2.7968e-10,6.9922e-11,1.7479e-11,2.8367e-12,8.8314e-13,3.2885e-12]`,
with initial ratios approximately `[4.00002,3.99985,4.00042]`.  Stage 0 retains
the immediate-floor classification.  The reconstructed diagnostic path remains
wrong by approximately `6e-6` to `3.5e-5`.

The previous factor branch incorrectly required one of the first four errors
to be below `1e-11`, even though the valid floor occurs at the fifth or sixth
entry.  The corrected factor branch now requires three consecutive ratios in
the strict interval `[3.8,4.2]`, monotonic reduction across that window, and a
minimum error at or after the window below `1e-11`.  It records the inclusive
window start/end, window ratios, monotonic flag, subsequent minimum, threshold,
and certification result.  The immediate-floor branch is unchanged.

The stage-graph, corrected incremental reverse, and multistep diagnostics all
passed.  Before the final classifier correction the focused result was
**20 passed, 1 failed**, with the sole failure in stage-local regime
classification.  After the specified factor-window correction, the final
focused result is **21 passed**.  The production and reconstructed tangent
values, identity checks, epsilon ladder, derivative mathematics, and original
whole-child tests were not changed.

### Earlier managed-sandbox attempts

Commands were attempted exactly once without MPI:

- `python -m pytest -q tests/test_production_dry_lie_hvp.py`:
  16 setup errors, 9 warnings in 3.19 seconds; numerical checks did not execute
  because of the read-only Loopy cache and were later satisfied externally.
- `python -m pytest -q tests/test_production_hyperviscosity_hvp.py`:
  9 setup errors, 9 warnings in 2.37 seconds; accepted numerical checks did not
  execute because of the same cache barrier and later passed externally.
- `python -m pytest -q tests/test_timestepping_coeff_gradients.py`:
  2 reported failures, 39 warnings in 5.18 seconds; both failures are the same
  cache-write exception during Firedrake setup, not executed numerical
  assertions.
- `python -m pytest -q tests/test_timestepping_ic_gradients.py`:
  2 reported failures, 39 warnings in 5.12 seconds; both failures are the same
  cache-write exception during Firedrake setup.
- `python -m pytest -q tests/test_rol_adapter.py`:
  6 setup errors, 14 warnings in 2.83 seconds; numerical adapter checks did not
  execute because of the same cache barrier and later passed externally.
- `python -m pytest -q tests/test_firedrake_hvp_prototype.py`:
  20 setup errors, 8 warnings in 2.50 seconds; numerical prototype checks did
  not execute because of the same cache barrier and later passed externally.
- `python -m pytest -q ode_adjoint`:
  **28 passed, 1 xfailed, 6 warnings in 9.07 seconds**.
- `python -m pytest -q`:
  32 passed, 1 skipped, 1 xfailed, 12 reported failures, 51 setup errors, and
  176 warnings in 28.94 seconds.  The failures/errors are in Firedrake paths
  stopped by the same sandbox cache-write barrier; the later external full
  repository run passed.

The six ODE warnings are the existing SciPy notices that L-BFGS-B ignores
`hessp` and that `disp` is an unknown solver option.  Firedrake warning counts
include existing installed-library warnings and do not constitute numerical
results.

## Static verification

The changed Python files compile with `python -m py_compile`.  `git diff
--check` passes.  Final status and diff-stat checks are recorded after removal
of generated `__pycache__`/`.pyc` artifacts.  The two pre-existing `.DS_Store`
files remain untouched, and no file is staged.

## Remaining limitations

There is no unresolved numerical issue within the stated dry production scope.
The milestone status is **COMPLETE**.  No deployed weak form, certification
tolerance, or certified timestep was changed, and `dt=100` remains the primary
multistep stress case.

No DG SSPRK43, moist physics, six-field MTSWE, PyROL `hessVec`, PyROL
initial-condition vector adapter, normalized control scaling, checkpoint
scheduling, MPI, JAX, or neural-network work was started.
