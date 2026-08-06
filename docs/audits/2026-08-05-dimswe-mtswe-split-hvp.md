# DIMSWE production MTSWE split HVP audit

## Repository gate

- Root: `/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615`
- Branch: `dev/dimswe-mtswe-split-hvp`
- Starting HEAD: `849b38399db4a22b93c4bf30318570d7da125f9e`
- Starting tracked tree: clean
- Starting index: clean
- Protected untracked files: `.DS_Store`, `docs/.DS_Store`
- Push URL: `DISABLE_PUSH` (unchanged)
- Status: **COMPLETE** after authoritative serial external certification

No remote operation, push, or protection change was made.  Finalization uses
only the three requested scoped local commits.

## Phase 0 production-graph findings

The exact discovered forward order is:

```text
dry RK4(t_n, dt/2)
dry RK4(t_n+dt/2, dt/2)
hyperviscosity Euler(t_n, dt)
DG SSPRK43(t_n, dt/2)
DG SSPRK43(t_n+dt/2, dt/2)
moist-physics Euler(t_n, dt)
```

The reverse must visit those exact executions in the opposite order.  The
four underlying production objects, exact field effects, exact stored-form
coefficient graph, DG tableau, legacy reverse, mixed CG/DG mass convention,
nonsmooth closure operations, inactive limiter hook, and configured-versus-
applied moist timestep distinction are documented in
`docs/DIMSWE_MTSWE_SPLIT_HVP.md`.

The certification parameter copy will retain the production split and all
physical values, set subcycles to `[2,1,2,1]`, enable the existing trainable
hyperviscosity representation, and fix the moist coefficients.  Its only
scalar control is physical `c0`; `s` has zero direction.

## Files changed

- `dimswe/timestepping.py`
- `dimswe/mtswe_split_hvp.py` (new)
- `tests/test_production_mtswe_split_hvp.py` (new)
- `docs/DIMSWE_MTSWE_SPLIT_HVP.md` (new)
- `docs/audits/2026-08-05-dimswe-mtswe-split-hvp.md` (new)

No accepted dry-Lie or hyperviscosity source/test file, optimizer source, or
deployed physics/transport form was modified.

## Implemented scope

- exact six-field dry RK4 transfer through the accepted production-stage graph;
- exact DG SSPRK43 owned primal/tangent caches, genuine-dual reverse, and exact
  state-curvature incremental reverse;
- exact production-form moist Euler cache, tangent, genuine-dual reverse,
  state-curvature incremental reverse, active-set signatures/margins, and
  invariant interfaces;
- owned seven-boundary/six-child complete-step primal and tangent caches;
- exact reverse child order with per-child ordinary and incremental metadata;
- physical `c0`-only, full-IC-only, and combined blocks;
- one- and multistep terminal least-squares value, gradients, and HVPs with
  state blocks returned as genuine `Cofunction`s; and
- explicit complete mixed CG/DG mass maps, Riesz solves, and natural pairings.

The only supporting production-file change is additive lazy wrapper methods in
`dimswe/timestepping.py`.  Existing forward, adjoint, dry-Lie, and
hyperviscosity entry points are unchanged.

## Exact-form and ownership audit

The active DG and moist derivative paths reference only the identical stored
`GeneralRK.production_stage_rhs_forms`.  DG's independently reconstructed
form is inherited solely for the named stage-local diagnostic.  Every cache
boundary deep-copies state, direction, tendency, dual, and reverse auxiliary
data.  Complete child/timestep containers are frozen and own their boundary
data independently of caller inputs and integrator scratch.

Dry and DG ordinary/incremental reverses use stage order `3,2,1,0`; moist uses
stage `0`; the complete reverse order is moist, DG-1, DG-0, hyperviscosity,
dry-1, dry-0.  State derivatives remain `Cofunction`s between children.

## Focused certification authored

The new focused file contains independent production-child trajectories,
stage-state/tendency comparisons, centered deployed forwards, independent
legacy reduced gradients, centered corrected-new reverse/gradient HVP
oracles, pairings, active-set checks over every ladder, supported primal and
differentiated moist invariants, mixed-block symmetry, order/type checks, and
ownership/repeatability checks.  Child/whole-map classifiers retain the strict
`1e-9` rule.  Reduced checks now retain that strict category limit while also
requiring measured scale-aware subtraction-floor evidence, active-set-safe
geometric ladders, and independent mixed-block pairings; broad flat
discrepancies still do not pass.

## Numerical execution

### First authoritative external focused run

The externally supplied focused result is **10 passed, 8 failed**.  No timing
or warning count was supplied with that result.  Its authoritative failure
classification is:

- combined scalar objective-directional checks at one and three steps show
  clean factor-four convergence but bottom out near `2.51e-9` and `1.66e-9`,
  respectively;
- one-step IC-only state-HVP errors are
  `[7.601e-8, 1.898e-8, 4.746e-9, 1.181e-9]`;
- one-step combined state-HVP errors are
  `[6.682e-8, 1.669e-8, 4.177e-9, 1.064e-9]`;
- one-step `c0`-only state-HVP errors are
  `[2.204e-9, 5.210e-9, 1.041e-8, 2.479e-8]`, worsening as epsilon shrinks;
- three-step `c0`-only state-HVP errors are
  `[2.179e-8, 5.459e-9, 1.594e-9, 3.130e-9]`, showing initial convergence and
  a subsequent upturn; and
- the current three-step IC-only and combined ladders cross moist switching
  surfaces on the minus trajectory.

Before the scalar-objective classifier failed, ordinary new-versus-legacy IC
and physical-`c0` gradients had already passed at the unchanged `2e-10`
tolerances.  The result does not identify a production-mathematics defect.

### Diagnostic follow-up authored

The reduced tests now emit, for both their retained original ladders and wider
geometric ladders:

- epsilon, absolute/relative error, consecutive ratios, exact/centered/plus/
  minus/subtraction-numerator magnitudes, and unperturbed repeatability;
- coefficient-vector and natural mass/Riesz norms for every dual field block;
- a scale-aware floor model computed from machine epsilon, repeated-evaluation
  error, plus/minus magnitudes, and the epsilon attaining minimum error;
- base/plus/minus moist signatures, signed extrema and minimum absolute
  margins for every switch at every complete timestep, branch-specific
  changed-DOF counts, and the largest contiguous symmetric safe interval;
- a strictly interior geometric active-set-safe certification sub-ladder; and
- the `c0`-to-state mixed-block natural absolute error and three independent
  centered field-probe pairings.

The mixed-block symmetry payload now includes both absolute and relative
discrepancies and the natural norm of the `c0`-to-state block.  Original
new-versus-legacy, pairing, active-set equality on certified values,
invariant, `Cofunction`, ownership, and nonmutation checks are unchanged.
`STRICT_FLOOR=1e-9` was not broadened globally, and no production or moist
closure code was changed in this follow-up.

### Second authoritative external diagnostic run

The targeted diagnostics produced **2 passed, 7 failed** and the complete
focused suite produced **11 passed, 7 failed**.

The three-step ordinary reduced gradient is independently certified:

- new-versus-legacy IC Riesz relative error:
  `2.676598561941038e-16`;
- new-versus-legacy physical-`c0` relative error:
  `2.890449973618449e-16`; and
- repeated objective, IC gradient, and physical-`c0` gradient were exactly
  reproducible.

Its largest symmetric active-set-safe epsilon was `0.025`.  Strictly inside
that interval, the epsilon/error pairs were

```text
epsilon: [0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125]
error:   [1.1752523129319e-8,
          2.9379549891086098e-9,
          5.744368650193705e-10,
          1.974488495805344e-10,
          1.4970337730376229e-9]
```

A least-squares log-log fit over indices `0,1,2,3`, ending at the minimum,
gives order approximately `2.0041`.  The minimum is below the unchanged
`1e-9` floor and the final point is a roundoff upturn.  The failed assertion
was therefore classifier-only.  The ideal factor-window rule remains intact;
the added active-set-truncated fallback requires at least four monotonically
decreasing values, fitted order in `[1.7,2.3]`, strict/measured floor
attainment, and any available post-minimum upturn.

All six reduced-HVP cases failed before constructing their diagnostic records:
`_dual_fd_diagnostic` returned a dictionary, while the caller attempted to
unpack it into two positional values.  No HVP ladder from this run is claimed
to pass or fail mathematically.  The helper now returns a frozen named
`DualFDDiagnosticResult` retaining the diagnostic record and the centered,
numerator, and error duals.  The caller uses `.record`, and pure contract tests
cover both the named result fields and the authoritative classifier sequence.

The mixed-block symmetry test passed with:

- relative discrepancy: `1.2915653012458413e-16`;
- absolute discrepancy: `1.52587890625e-05`;
- `c0`-to-state natural norm: `303481.53621484`; and
- `c0`-to-state coefficient-vector norm: `2.884151622916343e11`.

The mixed field block is non-negligible and symmetric to roundoff.  No
production code or moist switching operation was changed, and the strict
`1e-9` floor remains unchanged.

### Authoritative r3 external diagnostic run

The r3 results were:

- reduced-gradient diagnostic group: **3 passed**;
- reduced-HVP diagnostic group: **5 passed, 2 failed**;
- mixed-block symmetry: **1 passed**; and
- complete focused suite: **18 passed, 2 failed**.

The only failures were
`test_reduced_hvp_centered_gradients[ic-3]` and
`test_reduced_hvp_centered_gradients[combined-3]`.  In both cases the full
field HVP block, scalar HVP block, active-set checks, exact reproducibility,
and independent checks had already passed at the unchanged strict floor.  The
failure occurred only while classifying a scalar direction-probe pairing; no
production mathematical failure was observed.

The IC-only three-step probe had the strictly interior active-set-safe
sequence

```text
epsilon: [0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125]
error:   [1.5740918751133455e-9,
          3.951326930992529e-10,
          1.7302564355017118e-10,
          6.866683477406852e-11,
          2.0571161006801823e-10]
```

The combined three-step probe used the same epsilons and had errors

```text
[1.3853932072132853e-9,
 3.6785589214053265e-10,
 3.214568728623676e-10,
 1.9950858649456812e-10,
 1.053114074097447e-9]
```

Their first-step orders are `1.9941106248714737` and
`1.9130829001594707`.  In each sequence index 1 is the first point below the
unchanged `1e-9` strict floor.  The former classifier fit through the global
minimum and therefore included later floor-dominated values, producing
misleading orders of approximately `1.475` and `0.858`.

The corrected active-set-truncated rule computes a separate attainable floor
for every record and ends its truncation prefix at the first floor-reaching
index.  A prefix of at least three values retains the least-squares log-log
fit and `[1.7,2.3]` interval.  A two-value prefix is accepted only for the
independently certified full reduced-HVP context, with a strictly interior
active-set-safe ladder, one quadratic step from above the floor into it, and
a recorded later floor fluctuation or upturn.  The two authoritative probe
contract cases use only indices `[0,1]` and must fail when
`independent_checks=False`.  The three-step gradient contract remains and now
fits indices `[0,1,2]`, ending at its first strict-floor contact.  The ideal
three-ratio factor-of-four branch, immediate strict-floor branch, all epsilon
values, directions, and strict floors are unchanged.

### Authoritative r4 external rerun

The r4 targeted rerun reported **1 passed, 2 failed, 8689 warnings in
464.18 s**.  The active-set-truncated classifier contract passed, certifying
the preceding two-point correction.  The only failures remained
`test_reduced_hvp_centered_gradients[ic-3]` and
`test_reduced_hvp_centered_gradients[combined-3]`, now at a distinct field
natural-norm classifier call site.

The selected active-set-safe IC-only natural-norm relative errors were

```text
[7.799172758518771e-10,
 2.2756663810021927e-10,
 1.8309531331481302e-10,
 2.878037488206652e-10,
 7.724697794749904e-10]
```

The combined sequence was

```text
[8.763633521462349e-10,
 2.4954256423046307e-10,
 1.8831030672507957e-10,
 3.2823104880127766e-10,
 9.941986000958176e-10]
```

Every value in both five-record sequences is below the unchanged `1e-9`
strict floor.  The traceback reached the generic fallback because the field
classifier was still called with `allow_immediate_floor=False`; no convergence
order is observable or required after the first selected point has already
entered the strict floor.  This was a call-site evidence-plumbing failure, not
a production mathematical failure.

The test-only correction now certifies scalar and probe regimes before either
field representation.  It constructs and emits an explicit field evidence
map requiring selected-record active-set safety, exact gradient/HVP and scalar
repeatability, probe-pairing repeatability, local input non-mutation and `c0`
restoration, certified scalar and probe regimes, and the certified absolute
natural-floor assessment.  Both the natural-norm and coefficient-vector
classifiers receive this same evidence flag with immediate-floor handling
enabled; neither representation is treated as independent evidence for the
other.

Immediate-floor classification now requires at least three selected records,
active-set safety, independent evidence, and every error within its own
per-record attainable floor.  It reports `immediate_strict_floor` only when
all records are below the strict floor and otherwise reports
`immediate_per_record_attainable_floor`; it never fits an order.  Contract
coverage includes both authoritative natural-norm sequences, rejection when
independent evidence or active-set safety is absent, and rejection of a
sequence containing a value above both its strict and measured attainable
floor.  Epsilon ladders, active-set logic, directions, tolerances, strict
floors, and production code are unchanged.

### Authoritative r5 external runs

The narrow rerun produced **3 passed, 8689 warnings in 463.95 s**.  The
complete focused suite produced **18 passed, 2 failed, 29690 warnings in
864.13 s**.  The former IC-only and combined three-step failures passed.  The
only remaining failures were
`test_reduced_hvp_centered_gradients[c0-1]` and
`test_reduced_hvp_centered_gradients[c0-3]`.

For the one-step `c0` direction, failure first occurred in the scalar regime.
The active-set-safe relative-error sequence was

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

The first six values are below the unchanged `1e-9` strict floor.  The final
three smaller-epsilon values escape that floor through subtraction roundoff.
The new `immediate_floor_prefix_then_roundoff_escape` path accepts only the
contiguous initial floor prefix and retains every later record as diagnostic
scatter.  It requires independent checks, active-set safety, at least three
prefix records, a later out-of-floor record, and a post-prefix error larger
than every prefix error; it infers no convergence order.  Contracts include
the authoritative scalar sequence, an irregular probe sequence that moves in
and out of its floors, and every specified rejection case.

For the three-step `c0` direction, failure occurred only in the secondary
coefficient-vector field representation.  Its leading errors were

```text
[8.717998581810343e-8,
 2.1792455305071106e-8,
 5.458810025469264e-9,
 1.5940985005644644e-9,
 3.1297725209364616e-9,
 2.089126084432338e-9, ...]
```

The first ratios, `4.000466427379417` and `3.992162248437603`, are inside the
unchanged factor interval.  The index-3 minimum is
`1.5940985005644644e-9`; its local attainable floor is
`1.50729765042339e-9`, giving a ratio of approximately `1.0576`.  An upturn
follows, and a later record reaches its own per-record floor.  The scalar,
all probe, primary natural mass/Riesz, absolute natural-floor, exact
repeatability, non-mutation, and active-set checks had already passed.

The coefficient classifier now receives
`primary_metric_certified = state_natural_regime["certified"] and
absolute_natural_floor["certified"]` explicitly.  The opt-in
`secondary_metric_two_factor_steps_then_bracketed_floor` path requires that
primary evidence, two consecutive factor steps, the minimum at or directly
after their transition, a later upturn, and a per-record floor contact within
two indices.  It is marked secondary, reports its factor and floor-bracketing
metadata, and fits no floor-dominated order.  It cannot be used by natural,
scalar, probe, or standalone field certification.  Contracts reject missing
primary or independent evidence, unsafe active sets, inadequate factor
steps, absent upturn/contact, and a contact more than two indices after the
minimum.

These were classifier-only failures.  Production code, HVP mathematics,
directions, active-set logic, epsilon ladders, strict floors, tolerances, and
`ROUNDOFF_SAFETY_FACTOR=64` are unchanged.  No production mathematical
failure was observed.

### Final authoritative external certification

The completed authoritative serial certification is:

```text
python -m pytest -q tests/test_production_mtswe_split_hvp.py
22 passed, 29690 warnings in 888.92s

python -m pytest -q \
  tests/test_production_dry_lie_hvp.py \
  tests/test_production_hyperviscosity_hvp.py \
  tests/test_timestepping_coeff_gradients.py \
  tests/test_timestepping_ic_gradients.py \
  tests/test_rol_adapter.py \
  tests/test_firedrake_hvp_prototype.py \
  ode_adjoint
88 passed, 1 xfailed, 21800 warnings in 376.20s

python -m pytest -q
122 passed, 1 skipped, 1 xfailed, 51972 warnings in 1523.05s
```

The focused result certifies all 22 MTSWE split-HVP checks, including
physical-`c0`-only, full six-field IC-only, and combined directions for one
and three complete timesteps.  The accepted targeted regressions and full
repository suite contain no failures or unexpected errors.

The 51972 full-suite warnings fall into the known nonblocking categories:
PyOP2 assignment-to-NumPy-array shape warnings, FINAT quadrilateral DG-to-DQ
characterization notices, UFL quadrature metadata stringification notices,
and SciPy L-BFGS-B `hessp`/`disp` notices.  PETSc's unused `-q` report is
pytest's command-line option and is not an unused production solver option.

`git diff --check` passed.  Generated `__pycache__` directories and `*.pyc`
files were removed.  Certification is serial-only; no MPI certification is
claimed.  No PyROL `hessVec` or field vector, normalized scaling, JAX, neural
network, checkpointing, remote, or push work was performed.

### Managed-sandbox attempts

The focused command was attempted once:

```text
python -m pytest -q tests/test_production_mtswe_split_hvp.py
```

Result: **18 setup errors, 9 warnings in 2.86 s**.  No mathematical assertion
ran.  The production fixture reached construction of `DG1LimiterTransport`
and its deployed `VertexBasedLimiter`; a Loopy cache miss then attempted to
write

```text
/Users/arjunsharma/Library/Caches/pytools/
  pdict-v5-loopy-memoize-cache-preprocess_program-...sqlite
```

and failed with `sqlite3.OperationalError: attempt to write a readonly
database`.  OpenMPI independently printed `bind() failed: Operation not
permitted (1)` for its TCP listener.  The nine nonblocking warnings comprised
five PyOP2/NumPy shape deprecations and four FINAT quadrilateral-DG notices.
PETSc also reported pytest's `-q` as an unused database option.

The relevant accepted regressions were then attempted together once:

```text
python -m pytest -q \
  tests/test_production_dry_lie_hvp.py \
  tests/test_production_hyperviscosity_hvp.py \
  tests/test_timestepping_coeff_gradients.py \
  tests/test_timestepping_ic_gradients.py \
  tests/test_rol_adapter.py \
  tests/test_firedrake_hvp_prototype.py \
  ode_adjoint
```

Result: **4 failed, 28 passed, 1 xfailed, 56 setup errors, 111 warnings in
26.36 s**.  The 56 setup errors in dry-Lie (21), hyperviscosity (9), ROL
adapter (6), and the Firedrake prototype (20) have the same read-only Loopy
cache cause.  The four reported failures were the two legacy coefficient-
gradient tests and two legacy initial-condition-gradient tests; the retained
truncated sandbox output did not include their assertion bodies, so no cause
is inferred.  The pure `ode_adjoint` portion supplied the 28 passes and one
expected failure.

The 111 warnings were 77 PyOP2/NumPy shape deprecations, 11 FINAT
quadrilateral-DG notices, 17 UFL metadata-string notices, three SciPy
`RuntimeWarning`s that L-BFGS-B ignores `hessp`, and three SciPy
`OptimizeWarning`s for the `disp` option.  These warning categories are
nonblocking; the cache and socket restrictions are blocking.

These managed-sandbox limitations are retained as execution history only.
They are superseded by the complete authoritative serial certification above
and must not be interpreted as MPI certification.

## Final source review

- Every stored state, tangent, tendency, boundary, dual, and reverse auxiliary
  is copied into an owned cache/result object; the focused suite contains
  non-aliasing, scratch-reset, input-preservation, and repeatability checks.
- Active dry and DG stage derivatives consume the exact object identities in
  `GeneralRK.production_stage_rhs_forms`; the independently reconstructed form
  remains diagnostic-only.  The moist derivative consumes its exact single
  production-owned stage form.
- Reverse child order is moist, second DG, first DG, hyperviscosity, second
  dry, first dry.  Both four-stage children reverse `3,2,1,0`; moist reverses
  stage `0`.
- Only the accepted hyperviscosity child contributes the physical `c0`
  gradient/HVP.  All state blocks remain genuine mixed `Cofunction`s between
  children and at the reduced public interface.
- Mixed CG/DG mass maps and Riesz solves are explicit.  No implicit Riesz
  conversion is inserted between child reverse maps.
- All new methods in `timestepping.py` are additive lazy wrappers.  Legacy
  forward/adjoint methods and accepted dry/hyperviscosity APIs are unchanged.
- No reconstructed-form production derivative path is present in the new
  composition.

## Static checks

The final changed-Python compilation completed successfully for
`dimswe/timestepping.py`, `dimswe/mtswe_split_hvp.py`, and
`tests/test_production_mtswe_split_hvp.py`.  `git diff --check` completed with
no findings.  All generated `*.pyc` files and empty `__pycache__` directories
were removed after compilation and test collection.  Neither protected
`.DS_Store` file was read, changed, removed, staged, or committed.

## Excluded work

No PyROL `hessVec` or field vector, normalized scaling, JAX, neural network,
MPI certification, checkpointing, limiter/physics redesign, alternate split,
aerosol/chemistry feature, or production optimization run has begun.
