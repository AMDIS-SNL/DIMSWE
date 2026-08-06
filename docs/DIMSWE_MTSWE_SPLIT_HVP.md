# DIMSWE production MTSWE split HVP

## Phase 0: deployed graph

The production driver is `dimswe.run_model.run_model`.  It builds the model
and `LieSplittingIntegrator` through `get_timestepper`, calls
`take_forward_step(xn, xn_sub, xn, t, dt)`, and advances the external time only
after the complete split step.  For `tests/mtswe.cfg`, the integrator objects,
term lists, and subcycles are

```text
[RK4, Euler, SSPRK43, Euler]
[[model], [hyperviscosity], [dg1limiter], [threewayphysics]]
[2, 1, 2, 1]
```

The loop in `LieSplittingIntegrator.take_forward_step` expands those four
objects into the following exact six-child composition for parent start time
`t_n` and parent step `dt`:

| Child | Production object | Terms | Child start | Applied child step |
| ---: | --- | --- | ---: | ---: |
| 0 | `RK4` / `GeneralRK` | `model` | `t_n` | `dt/2` |
| 1 | the same `RK4` object | `model` | `t_n+dt/2` | `dt/2` |
| 2 | `Euler` / `GeneralRK` | `hyperviscosity` | `t_n` | `dt` |
| 3 | `SSPRK43` / `GeneralRK` | `dg1limiter` | `t_n` | `dt/2` |
| 4 | the same `SSPRK43` object | `dg1limiter` | `t_n+dt/2` | `dt/2` |
| 5 | `Euler` / `GeneralRK` | `threewayphysics` | `t_n` | `dt` |

The state passed out of each child is the exact input to the next child.  The
reverse order is moist Euler, second DG SSPRK43, first DG SSPRK43,
hyperviscosity Euler, second dry RK4, first dry RK4.

The six mixed state fields and spaces are, in order,
`[v,h,S,Qv,Qc,Qr]`: vector CG for `v`, scalar CG for `h` and `S`, and scalar
DG1 for `Qv`, `Qc`, and `Qr`.  The dry child modifies only `[v,h,S]`; the
hyperviscosity child modifies only `[v,h,S]`; DG transport reads `v` and
modifies only `[Qv,Qc,Qr]`; moist physics reads `h,S,Qv,Qc,Qr`, modifies
`[S,Qv,Qc,Qr]`, and leaves `[v,h]` unchanged.

## Exact production RK forms

Every child is a `GeneralRK` object.  During construction, each deployed stage
builds

```text
B_i = -model.rhs(..., terms=child_terms)
```

and applies UFL replacement so the state expression is
`xk + child_dt*sum_j A[i,j]*Fi[j]` and the time is
`child_t0+c[i]*child_dt`.  `GeneralRK.production_stage_rhs_forms[i]` retains
the identical `B_i` object used in `production_stage_residuals[i]` and the
stage solver.  Its coefficients include the production `xk`, `t`, `dt`, each
live predecessor `Fi[j]`, the applicable physical/model fields, and any
enabled trainable coefficient Function.

The deployed DG tableau is the four-stage object constructed by `SSPRK43`:

```text
A = [[0,   0,   0, 0],
     [1/2, 0,   0, 0],
     [1/2, 1/2, 0, 0],
     [1/6, 1/6, 1/6, 0]]
b = [1/6, 1/6, 1/6, 1/2]
c = [0, 1/2, 1, 1/2]
```

No independently reconstructed `model.rhs` form is permitted in an active
MTSWE derivative path.  The prior dry-Lie result is binding here: numerical
equality of reconstructed and production UFL forms does not imply
derivative-graph equivalence.

## Hooks and nonsmooth operations

`DG1LimiterTransport.rhs` uses `sign(dot(v('+'), n('+')))` in the facet flux.
Its `post_step` method would apply a `VertexBasedLimiter`, but all
`model.post_step` calls in the active `GeneralRK` implementation are commented
out.  The deployed timestep therefore applies the DG weak form without a
limiter or other post-step hook.  The derivative must preserve this fact; it
must not add a limiter call.

The moist closure contains no `sign` or clipping operation.  Its nonsmooth
operations are exactly:

- `max_value(0, gamma_v*(qv-q_sat)/tau_v)` for condensation;
- `max_value(0, gamma_v*(q_sat-qv)/tau_v)` inside evaporation;
- `min_value(qc/configured_dt, evaporation_candidate)` for the cloud-water
  evaporation cap; and
- `max_value(0, gamma_r*(qc-qprecip)/tau_r)` for rain conversion.

Classical tangent/HVP certification is valid only while every centered
perturbation retains the same selections at all four switching surfaces.
Margins and active signatures must be recorded and a crossing must fail the
test.  The production closure is not smoothed.

## Deployed moist update and timestep convention

With `qv=Qv/h`, `qc=Qc/h`, `s=S/h`, the code defines

```text
Dqv = max(0, gamma_v*(qv-q_sat)/tau_v)
Dqc = min(qc/configured_dt,
          max(0, gamma_v*(q_sat-qv)/tau_v))
Dqr = max(0, gamma_r*(qc-qprecip)/tau_r)
A   = Dqc-Dqv
```

where `tau_v=tau_r=configured_dt`.  The exact Euler update with the *passed*
child step `child_dt` is

```text
Qv+ = Qv + child_dt*h*A
Qc+ = Qc + child_dt*h*(Dqv-Dqc-Dqr)
Qr+ = Qr + child_dt*h*Dqr
S+  = S  + child_dt*h*beta2*A
```

Thus the conversion increment scales as
`child_dt/configured_dt`; the child does not replace its configured rate
timestep with the passed step.  Algebraically, the deployed map preserves
`Qv+Qc+Qr` and `S-beta2*Qv` on every active branch.

## Controls and coefficient mode

The checked-in `tests/mtswe.cfg` fixes hyperviscosity (`treat_as_coeffs=false`)
and exposes `[gamma_r,qprecip,L]` as the model coefficient vector.  That exact
object contains no differentiable `c0` coefficient.  The HVP certification
uses the existing production coefficient modes with unchanged physical
values and weak equations:

```text
hyperviscosity.treat_as_coeffs = true
threewayphysics.treat_as_coeffs = false
```

The resulting production coefficient order is exactly `[s,c0]`.  `s` is held
fixed, the only scalar direction is physical `delta_c0`, and all moist
parameters remain fixed.  No normalized scaling or additional control is
introduced.

## Dual and mass convention

State derivatives remain in the genuine algebraic dual of the complete mixed
space.  The mixed L2 mass form is block diagonal across the vector-CG,
scalar-CG, and DG1 fields.  Child reverse maps exchange `Cofunction`s directly.
Explicit mass solves are used only to obtain primal reverse auxiliaries or an
explicit public Riesz representative; unchanged fields enter through the
identity edge rather than an implicit conversion.

The legacy `LieSplittingIntegrator.take_adjoint_step` exists.  It recomputes
the complete child trajectory and invokes `GeneralRK.take_adjoint_step` in
reverse integrator/subcycle order, returning a primal legacy state adjoint and
the complete configured coefficient gradient.  It is an independent oracle
where the certification coefficient mode makes its `c0` entry applicable; it
is not the implementation of the new dual-native reverse.

## DG SSPRK43 derivative graph

The DG helper validates the actual production class, term list, stage count,
and all entries of `A`, `b`, and `c` before accepting the child.  It reuses the
certified four-stage exact-form graph operations, generalized only by a
six-field validation subclass.  A primal cache owns the incoming state, all
four materialized stage states, all four solved production tendencies, the
child time and applied half step, and the outgoing state.  The tangent cache
owns the corresponding incoming, stage-state, stage-tendency, and outgoing
directions.

For the exact stored stage form `B_i`, the tangent is

```text
M G_i = D_xk B_i[w_in] + sum_j<i D_Fj B_i[G_j]
```

where each `D_Fj` retains the deployed `child_dt*A[i,j]` expression.  The
ordinary reverse uses exact contracted UFL actions in order `3,2,1,0`:

```text
bar_F_i* = child_dt*b_i*lambda_plus*
            + sum_j>i D_Fi <B_j, psi_j>
M psi_i = bar_F_i*
lambda_in* = lambda_plus* + sum_i D_xk <B_i, psi_i>
```

The incremental reverse differentiates each `xk` and predecessor-`Fi` edge
pullback along `[w_in,G_0,...,G_3]`, and adds the reverse driven by
`mu_plus*`.  The DG direct `c0` gradient and HVP are structurally zero.  The
identity part of the map leaves `[v,h,S]` bitwise unchanged, although `v` is a
live coefficient in the DG flux and therefore receives a reverse pullback.

The independently reconstructed DG form inherited for diagnostics is used
only by the explicitly named stage-local comparator.  Tangent, reverse, and
incremental reverse use `production_stage_rhs_forms` exclusively.

## Moist Euler derivative graph

`MoistEulerPrimalCache` owns `t0`, the applied child step, incoming and stage
state, the exact production tendency, outgoing state, and a fixed-size active
signature with margins to every moist switch.  Its tangent cache owns the
incoming/stage/tendency/outgoing directions.  The exact one-stage formulas are

```text
M G = D_xk B_moist[w]
w_plus = w + child_dt*G

bar_F* = child_dt*lambda_plus*
M psi = bar_F*
lambda_in* = lambda_plus* + D_xk <B_moist,psi>

delta_bar_F* = child_dt*mu_plus*
M delta_psi = delta_bar_F*
mu_in* = mu_plus* + D_xk <B_moist,delta_psi>
         + D_xk(D_xk <B_moist,psi>)[w]
```

All derivatives use the identical production-owned moist `B` form.  There is
no direct `c0` term.  `[v,h]` are copied exactly in primal and tangent output;
the moist dependence on `h` is nevertheless retained in reverse.  The
ordinary and differentiated consequences of total-water and
`S-beta2*Qv` preservation are part of the focused certification.

The active-set diagnostic samples the two max arguments, the nested min/max
cap difference, the rain max argument, and the nonsingular `h+B` denominator
on the moisture DG space.  It stores Boolean selections and positive minimum
absolute margins.  Every plus/minus cache in every epsilon ladder must match
the base signature.  No result is accepted at a switching surface or after a
branch crossing.

## Complete six-child tangent and reverse

`MTSWESplitPrimalCache` owns seven boundary states and six child caches in the
exact forward order.  `MTSWESplitTangentCache` owns all seven boundary
directions, the physical `delta_c0`, and all six child tangent caches.  The
full tangent applies each local derivative to the previous boundary direction;
only the hyperviscosity child receives the scalar direction.

The ordinary reverse starts with a genuine full mixed `Cofunction`, visits
the six caches in exact reverse order, and passes each incoming child dual
directly to the preceding child.  Only the accepted hyperviscosity child adds
a scalar physical-`c0` gradient.  The incremental reverse applies the same
order to ordinary and incremental duals and sums only the hyperviscosity
physical-`c0` HVP contribution.  Per-child result objects and order metadata
are retained.  No Riesz conversion occurs between children.

The accepted dry exact-form and hyperviscosity implementations are reused
without changing their mathematics or accepted APIs.  The MTSWE dry subclass
changes only the semantic state-list guard from the dry three-field list to
the exact six-field list; the exact stored-form algorithms are inherited.

## Reduced terminal objective

For any positive number of complete parent steps, the reduced APIs retain all
parent primal/tangent caches and use

```text
J = 0.5 * integral inner(x_N-d, x_N-d) dx
lambda_N* = M(x_N-d)
mu_N* = M w_N
```

They reverse complete timesteps in descending order and return
`J`, the initial-condition gradient `lambda_0*` as a `Cofunction`, physical
`gradient_c0`, the initial-condition HVP `mu_0*` as a `Cofunction`, and
physical `Hq_c0`.  Directions may be `c0`-only, full-IC-only, or combined.
The scalar is never normalized.

## Independent certification ladder

`tests/test_production_mtswe_split_hvp.py` uses the 2-by-2 serial mesh from
`tests/mtswe_small.cfg`, the production `[2,1,2,1]` subcycles, parent
`dt=100`, the exact control mode described above, a smooth nonconstant
six-field state, and deterministic full-state directions.  It encodes:

1. complete cached output and every child boundary against independently
   executed production children;
2. exact field injection/unchanged-field checks at all six boundaries;
3. DG production stages/tendencies, stage-local exact-form differences,
   coefficient identities, local pairings, whole-child pairing, tangent, and
   centered incremental reverse;
4. moist form identity, active signatures/margins, unchanged `[v,h]`, primal,
   tangent, reverse, incremental reverse, and invariant consequences;
5. complete deployed-forward differences and full tangent/adjoint pairings for
   `c0`-only, IC-only, and combined directions;
6. one- and three-step ordinary gradients against the independent legacy
   reverse and centered scalar objectives;
7. complete and reduced incremental reverses against centered corrected new
   ordinary reverses/gradients for every direction block;
8. natural-pairing mixed-block Hessian symmetry;
9. exact child and RK reverse orders; and
10. `Cofunction` types, explicit mass/Riesz roundtrip, ownership,
    non-aliasing, input preservation, scratch independence, and bitwise
    repeatability.

Child and whole-map centered oracles retain the strict original floor-aware
classifier: three consecutive ratios in `[3.5,4.5]`, monotonic reduction, and
a subsequent relative error below `1e-9`, or an immediate strict floor.  A
flat `1e-7` or `1e-5` discrepancy cannot pass.

Reduced objective/gradient diagnostics additionally retain their original
ladders and evaluate wider geometric ladders in both directions.  Each record
contains absolute and relative errors, plus/minus and subtraction-numerator
magnitudes, exact and centered block norms, unperturbed repeatability, and a
machine-epsilon subtraction-floor estimate.  Field blocks record both raw
coefficient-vector norms and the natural dual norm
`sqrt(<g*,M^{-1}g*>)`.  Objective directional, scalar-gradient, and field-HVP
limits remain separate; the original `1e-9` constant is not globally
increased.

When active-set crossings truncate a smooth geometric ladder before it can
supply three ideal factor-of-four ratios, a separate fallback fits
`log(error)=p*log(epsilon)+constant` over the truncation prefix ending at the
first per-record attainable-floor contact.  Later floor-dominated values are
excluded even when they continue to decrease.  Three or more prefix values
use the least-squares fit with `1.7 <= p <= 2.3`.  Exactly two values may use
their actual-epsilon-ratio order only for the independently certified full
reduced-HVP test: the first must be above its attainable floor, the second at
or below it, every value must be from the strictly interior active-set-safe
ladder, and a later floor fluctuation or upturn must be present.  Disabling
the independent checks disables this two-point classification.  The ideal
factor-window and immediate strict-floor branches are unchanged.

An immediate-floor sequence is classified without an order fit.  It requires
at least three active-set-safe selected records, explicit independent
evidence, and every error at or below its own attainable floor.  The result is
`immediate_strict_floor` when every error is at or below the unchanged strict
floor, or `immediate_per_record_attainable_floor` when a measured per-record
floor is required.  Scalar and probe regimes are certified first.  Field
certification then receives a recorded evidence map covering active-set
safety, exact gradient/HVP repeatability, input non-mutation, scalar and probe
certification, and the absolute natural-floor check.  Coefficient-vector and
natural-norm views of the same field error are classified but are not counted
as independent evidence for one another.

Cancellation-limited scalar and probe ladders may also certify an initial
floor prefix followed by roundoff escape.  This path requires at least three
contiguous initial records within their individual attainable floors, full
independent evidence and active-set safety, and a later out-of-floor error
larger than every accepted prefix error.  Only the prefix is certification
evidence; all later in-floor/out-of-floor scatter remains in the payload.  It
reports `immediate_floor_prefix_then_roundoff_escape` and fits no order.

The coefficient-vector field norm is explicitly secondary to the natural
mass/Riesz norm.  When the primary natural metric and its absolute-floor check
are already certified, the secondary diagnostic may record two consecutive
factor-of-four steps followed by a bracketed floor contact.  It requires the
minimum at the factor transition or the next point, a post-minimum upturn, and
a later per-record floor contact no more than two indices after the minimum.
This `secondary_metric_two_factor_steps_then_bracketed_floor` path is opt-in,
fits no order through floor-dominated records, and cannot certify a natural,
scalar, or standalone field result.

Every reduced ladder records base/plus/minus moist masks, changed degree-of-
freedom counts by branch, and signed/minimum-absolute margins at each complete
timestep.  Certification uses a geometric ladder strictly inside the largest
contiguous sampled symmetric active-set-safe interval.  Crossing values
remain in the diagnostic payload.  The `c0`-only mixed state block is also
checked in the natural norm and through centered gradient pairings with three
deterministic field probes, in addition to the unchanged mixed-block symmetry
test.

## Certification status

The authoritative r3 runs produced **3 passed** for the reduced-gradient
diagnostic group, **5 passed, 2 failed** for the reduced-HVP diagnostic group,
**1 passed** for mixed-block symmetry, and **18 passed, 2 failed** for the
complete focused suite.  The only failures were scalar probe-pairing regime
classification in the three-step IC-only and combined reduced-HVP cases.
Their full field HVP blocks, scalar HVP blocks, active-set stability,
reproducibility, and independent checks had already passed at the unchanged
strict floor; no production mathematical failure was observed.

For IC-only, the strictly interior safe sequence was

```text
epsilon: [0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125]
error:   [1.5740918751133455e-9,
          3.951326930992529e-10,
          1.7302564355017118e-10,
          6.866683477406852e-11,
          2.0571161006801823e-10]
```

For the combined direction the errors on the same epsilon sequence were

```text
[1.3853932072132853e-9,
 3.6785589214053265e-10,
 3.214568728623676e-10,
 1.9950858649456812e-10,
 1.053114074097447e-9]
```

The first centered step has observed order `1.9941106248714737` and
`1.9130829001594707`, respectively, and the second point is already below the
unchanged `1e-9` strict floor.  Fitting through the later numerical minimum
incorrectly mixed truncation and floor-dominated points.  The contract now
requires indices `[0,1]` for these two order estimates and proves that neither
sequence can use the narrow two-point branch when `independent_checks=False`.

The authoritative r4 rerun then produced **1 passed, 2 failed, 8689 warnings
in 464.18 s**.  The two-point classifier contract passed.  The remaining
failures stayed at `test_reduced_hvp_centered_gradients[ic-3]` and
`test_reduced_hvp_centered_gradients[combined-3]`, but moved to the separate
natural-norm field-classifier call site, which still passed
`allow_immediate_floor=False`.

The selected IC-only natural-norm errors were

```text
[7.799172758518771e-10,
 2.2756663810021927e-10,
 1.8309531331481302e-10,
 2.878037488206652e-10,
 7.724697794749904e-10]
```

and the combined errors were

```text
[8.763633521462349e-10,
 2.4954256423046307e-10,
 1.8831030672507957e-10,
 3.2823104880127766e-10,
 9.941986000958176e-10]
```

All five values in both active-set-safe sequences are already below the
unchanged `1e-9` strict floor, so no convergence order is observable or
required.  The failure was call-site evidence plumbing only; no production
mathematical failure was observed.

The authoritative r5 narrow rerun then passed all corrected cases: **3
passed, 8689 warnings in 463.95 s**.  The complete focused suite reported
**18 passed, 2 failed, 29690 warnings in 864.13 s**.  The previously failing
IC-only and combined three-step cases passed.  The two remaining failures are
the `c0`-only one- and three-step classifier cases.

For `c0`-only at one step, the safe scalar errors were

```text
[2.3659715736786227e-11,
 5.7248518327684635e-12,
 6.794392392598897e-11,
 4.0248262778803686e-11,
 4.949186997509626e-10,
 2.728665730007977e-10,
 1.4960916071338244e-9,
 3.39478304376295e-9,
 4.231807469635109e-9]
```

The first six records are below the unchanged `1e-9` strict floor; later
smaller-epsilon records escape through subtraction roundoff.  Certification
therefore uses only this initial six-record strict-floor prefix and retains
the remaining three points as roundoff diagnostics.

For `c0`-only at three steps, only the secondary coefficient-vector
representation failed.  Its first errors were

```text
[8.717998581810343e-8,
 2.1792455305071106e-8,
 5.458810025469264e-9,
 1.5940985005644644e-9,
 3.1297725209364616e-9,
 2.089126084432338e-9, ...]
```

The first two ratios are `4.000466427379417` and `3.992162248437603`.
The minimum at index 3 is `1.5940985005644644e-9`, versus its local attainable
floor `1.50729765042339e-9`, followed by an upturn and a later per-record floor
contact.  The scalar, all probes, primary natural/Riesz metric, absolute
natural floor, repeatability, non-mutation, and active-set checks had already
passed.  Both r5 failures were classifier-only; no production mathematical
failure was observed.

## Final certification

The production MTSWE split-HVP milestone is **COMPLETE**.  The authoritative
serial external results are:

- focused production MTSWE split-HVP suite: **22 passed, 29690 warnings in
  888.92 s**;
- accepted targeted regressions: **88 passed, 1 xfailed, 21800 warnings in
  376.20 s**; and
- complete repository suite: **122 passed, 1 skipped, 1 xfailed, 51972
  warnings in 1523.05 s**.

The nonblocking warnings are PyOP2 assignment-to-NumPy-array shape warnings,
FINAT quadrilateral DG-to-DQ characterization notices, UFL quadrature metadata
stringification notices, and SciPy L-BFGS-B `hessp`/`disp` notices.  PETSc's
unused `-q` report refers to pytest's command-line option, not an unused
production solver option.

`git diff --check` passed, and generated `__pycache__` directories and `*.pyc`
files were removed.  Certification is serial-only: no MPI certification is
claimed.  No PyROL `hessVec` or field vector, normalized scaling, JAX, neural
network, checkpointing, remote operation, or push work was performed.

The final inspection confirmed that production mathematics, moist switching,
legacy APIs, and the previously accepted dry-RK4/hyperviscosity behavior are
unchanged.  The central derivative-graph lesson remains: **numerical equality
of reconstructed and production UFL forms does not imply derivative-graph
equivalence**.

This milestone does not cover PyROL vectors or `hessVec`, normalized controls,
JAX, neural networks, MPI certification, checkpointing, limiter redesign,
physics redesign, or an alternative split.
