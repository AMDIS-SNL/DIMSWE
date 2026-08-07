# DIMSWE learned-physics experiments

## Purpose and scope

J4A provides an opt-in laboratory for comparing learned or inferred physics
without choosing a neural moist model.  The framework separates the feature
map, arbitrary parameterized model, and output/physics map:

```text
state, context --Phi--> features --N_theta--> raw output
       baseline physics, state, context, raw output --D--> deployed physics
```

`dimswe.learned_physics.LearnedPhysicsModel` implements this composition with
pure JAX numerical inputs.  It assumes neither an MLP nor a fixed feature set,
parameter tree, output representation, residual form, or replacement form.
Every numerical leaf must be `float64`.  Input pytrees are copied before user
callbacks and outputs are copied on return.  A Firedrake object never crosses
this pure-JAX boundary; a deployment adapter owns Firedrake interpolation,
assembly, solves, and coefficient restoration.

This is a strict runtime contract: learned-physics operations require
`jax_enable_x64=True`, do not silently enable it during import, and reject
float32 execution.  Every JAX validation process must be started with both

```bash
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
```

An earlier framework-test process omitted `JAX_ENABLE_X64=True` and produced
nine expected precondition failures.  That was not a mathematical failure.
The subsequent x64 process passed all 12 framework tests.

The companion pytree utilities are `tree_copy`, `tree_zeros`, `tree_dot`,
`tree_norm`, `tree_axpy`, and `tree_all_finite`.  They preserve arbitrary JAX
pytree structure and require float64 leaves.

The package initializer imports no Firedrake or PyROL dependency.  The
production hidden-c0 adapter is separately opt-in as `dimswe.hidden_c0`.
Existing UFL remains the default moist backend, and the certified JAX moist
backend remains opt-in.

## Four training modes

The stable `TrainingMode` values and their APIs are intentionally separate.

### `apriori_offline`: operator/apriori offline

The generic API name remains `apriori_offline`.  For Test 1A its precise
scientific label is **operator/apriori offline**: its target already contains
the weak finite-element spatial operator, so it is not literally spatially
discretizer-unaware.  It does not apply the deployed mass inverse, Euler
timestep, or complete state-transition operator, and it makes no complete
solver call during optimization.

For hidden c0, the target at every externally supplied truth state is the
actual assembled production weak hyperviscosity right-hand side.  The adapter
obtains it as `M * tendency` from the unchanged production child, which is
algebraically the assembled weak residual.  Candidate c0 scales this fixed
operator because the production form is linear in physical c0 at fixed state.
The loss ignores the mixed mass inverse, Euler timestep, all other split
children, and state propagation.  It is therefore an operator-coefficient
loss, not a fabricated continuous target.

### `discrete_offline`: deployed-discrete offline

`discrete_offline` evaluates each example independently at its supplied truth
state.  Learned physics passes through a fixed deployed discrete map, but a
predicted state is never used by the next example.

For hidden c0, the target is the actual production hyperviscosity Euler-child
increment

```text
dt * M^-1 * b_hyperviscosity(x_truth; c0_truth).
```

It includes weak assembly, the complete mixed mass solve, and the child Euler
update.  Those observations are precomputed once; optimization scales the
fixed child increments and invokes no full solver.  This is discretizer-aware
but still offline because every observation remains anchored to an external
truth state and there is no recursive model trajectory.

### `truth_reset`

`truth_reset` starts every window from its trusted state, runs the unchanged
complete production split recursively for the configured internal horizon,
compares to trusted targets, and then discards the prediction before the next
window.  J4A uses the exact J3 physical-c0 gradient and HVP through each
window.  Certified hidden-c0 horizons are one or three complete timesteps.
Terminal-only and accumulated-within-window observations are explicit.

### `rollout`

`rollout` starts from truth only at the first state.  Every later state is the
previous model prediction.  It has no truth reset inside the rollout.  Its
loss accumulation is explicitly `terminal` or `accumulated`; J4A defaults to
an accumulated one-, two-, and three-step autonomous-prefix loss.  Each
prefix starts from the same first truth state and recursively traverses the
unchanged production solver.  Exact J3 gradients and HVPs are used.

The generic objective tests instrument these dataflows directly: a-priori
does not call a solver, discrete-offline sees each external truth state,
truth-reset restarts every window, and rollout sees its own predictions.

## Common experiment and data contracts

`ExperimentDefinition` records the benchmark; truth, baseline, and model
configuration; training mode; observation definition; rollout horizon; seed;
optimizer configuration; and evaluation metric names.  It is immutable and
has canonical, sorted JSON serialization.

Truth generation is an explicit operation, never an implicit side effect of
training.  `TruthDataset` writes one compressed NPZ containing `states` and
`times`, plus one JSON sidecar.  `TruthMetadata` records:

- solver/backend and exact split configuration;
- timestep and number of complete steps;
- initial-condition definition;
- physical parameters and hidden physical c0;
- UFL or JAX moist backend;
- random seed;
- state-field flattening and control convention; and
- serial/accelerator/checkpointing scope.

Snapshots are owned float64 arrays and become read-only on construction or
load.  The JSON metadata and NPZ payload are validated together on load.

`ExperimentResult` records initial/final parameters in normalized and
physical coordinates, histories, evaluation counts, HVP counts, complete
solver-call counts, timing, deployment metrics, success, and failure reason.
Canonical JSON writing and compact summary helpers are separate from the
optimizer.  Plotting is deliberately absent from the optimization core.

## Test hierarchy

- **Test 1A:** 2-by-2 hidden-c0 plumbing/integration certification.
- **Test 1B-0:** resolved-flow c0 identifiability and resolution pilot.
- **Test 1B:** selected 16-by-16, dt=100 `doublevortex` hidden-c0 inverse
  experiment, executed through four explicit review gates.
- **Test 2:** full learned replacement of moist physics.
- **Test 3:** learned correction of configurable misspecified moist physics.

Test 2 does not begin until Test 1B has been understood.  Test 1B remains a
correctly specified scalar inverse problem: truth and learner differ only in
c0.  It isolates the behavior of the four training formulations in nonlinear,
resolved flow before introducing neural representation or model error.
Resolved case discovery, diagnostics, inference equations, and staged external
commands are in `docs/RESOLVED_HIDDEN_C0.md`; the exact selected ladder is in
`docs/TEST1B_SELECTED_EXECUTION.md`.

The canonical Test 1B fitting interval is states 0 through 80.  Both offline
objectives use starts 0 through 79.  Truth-reset uses 16 non-overlapping
five-step windows with starts `0,5,...,75`; full rollout uses one autonomous
80-step training trajectory.  Both accumulated solver losses contain target
states 1 through 80 exactly once and use the same target-mass normalizer
`||X*_n||_M^2`.  Their intended distinction is therefore how long generated
states feed back before truth is reintroduced: five steps versus all 80.
Held-out targets 81 through 160 are inaccessible to scans and fitting.
J4B evaluates each accumulated trajectory with one cached forward traversal,
one reverse traversal for gradients, and one tangent plus incremental-reverse
traversal for HVPs.  Canonical objective-only forward work is therefore 80
deployed steps for reset and 80 for rollout.  This accounting does not imply
equal total fit time or equal optimizer iterations.

## Test 1A: hidden-c0 plumbing/integration benchmark

### Exact production configuration and truth

The configuration is `dimswe/configs/hidden_c0_tiny.cfg`.  The initial
condition class supplies the doubly periodic physical domain

```text
Omega = [0, 5,000,000) x [0, 5,000,000) metres.
```

`PeriodicRectangleMesh(2, 2, Lx, Ly, quadrilateral=True)` constructs four
quadrilateral cells with periodic x and y boundaries.  Family `Q`, order 3,
and lumped mass give vector spectral CG(3) for `v`, scalar spectral CG(3) for
`h` and `S`, and scalar spectral DG(1) for each of `Qv`, `Qc`, and `Qr`.  The
configured volume measure uses the order-3 tensor GLL rule.  Topography is
exactly `B=0`, and the inherited constant Coriolis value is
`f=0.00006147`.

The native double-vortex state is deliberately overwritten, after model
initialization, by the following deterministic projected state.  With

```text
mx(x) = sin(2*pi*x/Lx),       my(y) = cos(2*pi*y/Ly),
H(x,y) = 750 + 4*mx + 3*my,  g = 9.80616,
```

the six prognostic fields are exactly

```text
v  = (25 + 1.5*my, 17 + mx),
h  = H,
S  = H*g*(1.02 + 0.0015*mx - 0.0010*my),
Qv = 0.0030*H,
Qc = 0.0010*H,
Qr = 0.0002*H.
```

The timestep is `dt=100`; default truth contains four complete transitions
and five snapshots at times `(0,100,200,300,400)`.  The focused external test
fixture uses three transitions for cost.  Physical truth is `c0_truth=0.14`,
the starting guess is `c0_initial=0.07`, exponent `s=3.2`, and the immutable
control map is `c0=0.07 z`.  The default moist backend is UFL; JAX moist
remains opt-in.

The complete Lie split has configured integrators
`[RK4,Euler,SSPRK43,Euler]` and subcycles `[2,1,2,1]`.  Its actual six-child
order is

```text
dry_rk4_0 -> dry_rk4_1 -> hyperviscosity_euler
          -> dg_ssprk43_0 -> dg_ssprk43_1 -> moist_euler.
```

Each dry and DG child advances `dt/2`; hyperviscosity and moist children each
advance `dt`.  This is a real execution of the complete production six-field
MTSWE split, but a 2-by-2 mesh is not scientifically resolved flow.  Test 1A
certifies derivatives, optimization, dataflow, and training-mode semantics.
It supports no conclusion about the superiority of offline or solver-in-loop
training.

Truth and learner use the same initial condition, dry dynamics,
hyperviscosity formulation, transport, moist formulation/backend, solver,
timestep, and all physical parameters.  They differ only in c0.  Both values
lie inside the existing production `[0.01, 2.0]` physical bounds.

Physical c0 and normalized z are both recorded.  The adapter snapshots every
child coefficient, applies an owned physical coefficient, and restores all
children in `finally`; caller truth states and coefficient templates are not
production scratch.

### Exact Test-1A objectives

Let `X*_k` be the stored truth states, `c*=0.14`, and
`b_k=b_hv(X*_k;c*)` be the actual assembled production weak
hyperviscosity right-hand side.  Let

```text
Delta_k = dt M^-1 b_k
||Y||_M^2 = integral_Omega inner(Y,Y) dx_GLL.
```

The operator/apriori objective uses coefficient-vector Euclidean norm after
flattening the assembled weak dual:

```text
J_op(c) = mean_k 1/2 ||(c/c* - 1)b_k||_2^2 / ||b_k||_2^2.
```

It uses the real weak FE spatial operator and fixed truth states, but ignores
`M^-1`, child `dt`, every other split child, and propagation.

The deployed-discrete offline objective is

```text
J_disc(c) = mean_k 1/2 ||(c/c* - 1)Delta_k||_2^2 / ||Delta_k||_2^2.
```

`Delta_k` is obtained from the actual production hyperviscosity Euler child,
including weak assembly, mixed mass inverse, and timestep.  It remains offline
because every input is a fixed truth state, observations are precomputed, no
candidate state becomes another example, and optimization makes zero complete
solver calls.

For solver observations define the fixed initial-guess normalizer

```text
N(k,p) = ||F_cinit^p(X*_k) - X*_(k+p)||_M^2.
```

Test 1A truth-reset uses terminal loss, actual horizon `p=1`, and resets at
every possible truth state:

```text
J_reset(c) = mean_k 1/2 ||F_c(X*_k)-X*_(k+1)||_M^2 / N(k,1).
```

For the four-transition default truth there are four windows; the three-step
test fixture has three.  The next window always starts from truth.

Test 1A rollout uses actual horizon 3 and accumulated loss:

```text
J_roll(c) = (1/3) sum_(p=1)^3
            1/2 ||F_c^p(X*_0)-X*_p||_M^2 / N(0,p).
```

Each prefix begins at `X*_0` and is recursively autonomous inside the prefix;
no truth reset occurs at an internal step.  The implementation separately
executes prefixes of lengths 1, 2, and 3, which is mathematically the stated
accumulated autonomous-prefix objective.  Exact J3 physical-c0 gradients and
HVPs are scaled as `g_z=0.07 g_c` and
`H_z q=0.07 H_c(0.07 q)`.

### Optimizer, stopping, and accounting

Each offline residual is normalized by the squared norm of its fixed target.
Each solver-in-loop terminal residual is normalized by the corresponding
initial-guess residual.  These fixed scales improve conditioning without
moving the exact minimum.

All modes use the same deterministic scalar optimizer in normalized z.
Physical optimizer bounds are `[0.01,0.30]` (inside the production
`[0.01,2.0]` bounds), the budget is eight Newton iterates and six line-search
trials, gradient tolerance is `1e-9`, step tolerance `1e-11`, and minimum
accepted positive curvature `1e-12`.  A positive exact HVP supplies the Newton
step; otherwise a positive secant or deterministic bounded gradient fallback
is used.  Proposals are clipped to the bounds and backtracking halves the step
until the objective strictly decreases.  Success requires the gradient
tolerance—not merely objective reduction.  Stagnation, line-search failure,
nonfinite values, and iteration limit have distinct termination reasons.

One local, cold-cache-aware three-step smoke run produced:

| mode | recovered physical c0 | objective history | objective / gradient / HVP evaluations | complete solver calls |
| --- | ---: | --- | --- | ---: |
| operator/apriori offline | `0.14` | `0.125 -> 0` | `3 / 2 / 1` | `0` |
| deployed-discrete offline | `0.14` | `0.125 -> 0` | `3 / 2 / 1` | `0` |
| truth-reset, terminal horizon 1 | `0.14000000000000107` | `0.5 -> 1.22e-11 -> 9.18e-29` | `5 / 3 / 2` | `21` |
| rollout, accumulated horizon 3 | `0.13999999999987023` | `0.5 -> 3.35e-5 -> 1.54e-13 -> 2.32e-24` | `7 / 4 / 3` | `60` |

The accepted Newton-step counts were respectively 1, 1, 2, and 3; the final
gradient evaluation certifies convergence.  A combined value/gradient call
increments both counters once.  Each HVP call increments the HVP counter once.
`solver_calls` counts complete forward timesteps traversed by values,
gradients, and HVPs; it does not relabel internal adjoint linear solves as
complete forward calls.  For the three-step smoke fixture, reset costs three
complete steps per aggregate call and accumulated rollout costs `1+2+3=6`.

Preprocessing is reported separately.  Four-transition default truth uses
four hyperviscosity-child calls, four reset-normalizer forward steps, and six
rollout-normalizer forward steps.  The three-transition smoke uses three,
three, and six.  Wall time uses `perf_counter` around optimization.  Dense
landscape scans have their own objective and solver counts.

### Common deployment evaluation

Every recovered parameter is evaluated with the same protocol:

1. physical absolute and relative c0 error;
2. one-step complete-state prediction error;
3. short autonomous rollout error;
4. final-state error;
5. accumulated trajectory error;
6. objective values under all four training losses, alongside the initial
   guess values;
7. per-field block errors for `(v, h, S, Qv, Qc, Qr)`;
8. finite-state check, explicitly not a stand-alone numerical-stability proof;
9. exact repeated-state-vector repeatability; and
10. objective/gradient/HVP/solver-call/preprocessing/wall-time costs.

Mixed-state errors use the deployed mixed L2 measure.  Field errors use the
corresponding field L2 blocks.  Cross-evaluation intentionally uses the same
stored truth regardless of fitting mode.

## Test 2 and Test 3 extension contracts

`BENCHMARK_CONTRACTS` reserves two model-agnostic extensions.

`learned_moist_replacement` defines truth as PDE plus certified moist physics
and deployment as PDE plus an output-map-provided learned closure.  Possible
representations include direct rates, subprocess rates, invariant-null-space
sources, or another output map.  All four training APIs accept it.  The
contract fixes no features, normalization, architecture, NN library, NN size,
or output representation.

`misspecified_moist_correction` defines separate truth and configurable nearby
baseline formulations, then applies a learned correction through the same
composition.  Metadata can later describe gamma-r multipliers, precipitation
thresholds, relaxation times, saturation parameters, rain subprocesses, or
evaporation-cap changes.  Possible correction outputs include residual rates,
residual subprocesses, physical-parameter corrections, or another correction
map.  None is selected as the scientific benchmark in J4A.

## J4A boundaries

J4A is serial CPU only.  It implements no MPI, accelerator claim,
checkpointing, large sweep, or long training campaign.  It does not choose or
train a neural moist model, final features, normalization, residual versus
replacement strategy, architecture, or NN size.  It does not modify accepted
J1/J2/J3 mathematics, the default UFL semantics, JAX opt-in semantics, or c0
normalization.

## Authoritative J4A external validation

All JAX executions used `JAX_ENABLE_X64=True`; production runs also used
`OMP_NUM_THREADS=1`.

- Pure framework: `12 passed in 0.72s`.
- Hidden-c0 Test 1A: `21 passed, 17508 warnings in 631.62s`.
- J3/J2 regression: `45 passed, 11477 warnings in 335.46s`.
- J1 plus production MTSWE HVP regression:
  `58 passed, 29908 warnings in 629.39s`.
- Complete repository: `293 passed, 1 skipped, 1 xfailed, 89516 warnings in
  2407.05s (0:40:07)` with no FAILED or ERROR section.

The complete-repository result certifies J4A before J4B-PREP changes.  It is
not to be rerun inside Codex.

## Ordered external validation

Use a writable cold-cache root and run only the focused sequence until every
J4A check is green:

```bash
cd /Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export DIMSWE_TEST_CACHE_DIR="$(mktemp -d /tmp/dimswe-j4a.XXXXXX)"
export XDG_CACHE_HOME="$DIMSWE_TEST_CACHE_DIR/xdg"

# 1. framework-only tests
python -m pytest -q tests/test_learned_physics_framework.py

# 2. hidden-c0 truth generation
python -m pytest -q tests/test_hidden_c0_benchmark.py -k truth_generation

# 3. c0 objective scans
python -m pytest -q tests/test_hidden_c0_benchmark.py -k dense_objective_scan

# 4. operator/apriori offline recovery
python -m pytest -q tests/test_hidden_c0_benchmark.py -k 'each_mode_recovers_hidden_c0_strictly and apriori_offline'

# 5. deployed-discrete offline recovery
python -m pytest -q tests/test_hidden_c0_benchmark.py -k 'each_mode_recovers_hidden_c0_strictly and discrete_offline'

# 6. truth-reset recovery
python -m pytest -q tests/test_hidden_c0_benchmark.py -k 'each_mode_recovers_hidden_c0_strictly and truth_reset'

# 7. rollout recovery
python -m pytest -q tests/test_hidden_c0_benchmark.py -k 'each_mode_recovers_hidden_c0_strictly and rollout'

# 8. complete focused Benchmark-1 file
python -m pytest -q tests/test_hidden_c0_benchmark.py

# 9. J3 PyROL regression
python -m pytest -q tests/test_mtswe_rol_adapter.py tests/test_mtswe_rol_state_adapter.py tests/test_mtswe_rol_combined_adapter.py

# 10. J2 JAX derivative/full-split regression
python -m pytest -q tests/test_jax_moist_derivatives.py tests/test_jax_moist_full_split.py

# 11. J1 local/adapter regression
python -m pytest -q tests/test_jax_moist_local.py tests/test_jax_moist_firedrake.py

# 12. production MTSWE HVP regression
python -m pytest -q tests/test_production_mtswe_split_hvp.py
```

These commands reproduce the already certified focused sequence.  Do not rerun
the complete repository suite for J4B-PREP.
