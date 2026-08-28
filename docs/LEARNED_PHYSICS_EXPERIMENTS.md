# DIMSWE learned-physics experiments

Test-2A Method-3/Method-4 trajectory preparation and the matched
Method-1/Method-2 long-fit contract are documented in
[`TEST2A_M3_M4_PREP_AND_FAIR_LONGFIT.md`](TEST2A_M3_M4_PREP_AND_FAIR_LONGFIT.md).
The final Method-3/Method-4 metric, horizon, reset schedule, and
endpoint-versus-accumulated choice remain scientifically unfrozen.

The secondary operator-pretrained deployed-discrete workflow diagnostic is
documented separately in
[`TEST2A_M1_TO_M2_FINETUNE.md`](TEST2A_M1_TO_M2_FINETUNE.md). It does not
replace the matched seed-zero Method-1/Method-2 comparison.

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
- **Test 2 activity audit:** quantitative A/R signal screen on states 0..80.
- **Test 2A-1:** local neural operator learning for A only, with original R
  retained.
- **Test 2A-2:** opt-in embedding and derivative certification of the frozen
  neural A while retaining original deployed R.
- **Later Test 2 stages:** deployed-discrete and solver-in-loop comparisons
  after the Test 2A-2 external certification gate passes.
- **Test 3:** learned correction of configurable misspecified moist physics.

Test 2 does not begin until Test 1B has been understood.  Test 1B remains a
correctly specified scalar inverse problem: truth and learner differ only in
c0.  It isolates the behavior of the four training formulations in nonlinear,
resolved flow before introducing neural representation or model error.

After Test 1B closure, Test 2 begins with a quantitative moist-activity audit,
not network implementation.  `dimswe.test2_moist_activity` evaluates A, R, and
their structurally coupled sources at the certified cell-local 4-by-4 GLL
points for truth states 0 through 80 only.  It measures rate support,
intermittency, source increments, actual moist-child mass-norm updates, process
contributions, and existing-input support before any architecture, feature, or
normalization choice.  States 81 through 160 remain untouched future Test-2
deployment data.  The detailed contract and external command are in
`docs/TEST2_MOIST_ACTIVITY.md`.

The accepted activity classification is `A_ACTIVE_R_WEAK`: A is active
throughout the training trajectory, while R is exactly zero at all 331,776
training samples.  Test 2A-1 consequently learns only A from the exact local
inputs `(h,S,Qv,Qc,B)`, but its hybrid output map always retains the original
deployed R law.  The selected configurable 5-32-32-1 tanh model, training-only
normalization, full-batch operator objective, arbitrary-pytree PyROL adapter,
and external training command are documented in
`docs/TEST2A_OPERATOR_LEARNING.md`.  This is a scoped architecture freeze for
the first operator proof, not a general scientific choice for later Test 2
models.

The subsequent fixed-problem optimizer study compares common-start ROL
L-BFGS memory 10/20 and trust-region truncated CG with exact local JAX HVPs.
Longer L-BFGS runs substantially improve operator accuracy, but all methods
terminate at their iteration limits without convincing stationarity.  The
classification is `OPTIMIZATION_AND_MODEL_BOTH_UNRESOLVED`, so that endpoint
was not ready for solver embedding. See `docs/TEST2A_OPTIMIZER_STUDY.md`.

A checkpointed memory-20 continuation then ran for 1,500 additional accepted
iterations from the exact optimizer-study pytree.  It reached relative RMS
`0.205` and correlation `0.979`, but remained at `MAXITER`, with sign accuracy
`0.813` and an 8.05-percent objective reduction over its last 100 accepted
steps.  The result is `CONTINUING_OPTIMIZER_LIMITED`; further same-method
optimization, not Test 2A-2 embedding or architecture modification, is the
next controlled step.  Details are in `docs/TEST2A_CONTINUATION.md`.

The bounded convergence-to-plateau continuation from that exact `+1500`
endpoint then completed another 5,000 accepted memory-20 iterations. It
reached normalized MSE `0.00946960`, relative RMS(A) `0.0973119`, and
correlation `0.995243`. Sign accuracy was `0.92494`, `0.98937`, and `1.0` on
the `1e-3`, `1e-2`, and `1e-1` activity strata, respectively. The result is
still `STILL_OPTIMIZER_LIMITED`: the final 100 steps reduced the objective by
`1.064%`, the gradient ratio was `0.402`, and the relative parameter step was
`4.49e-5`, so the joint practical-plateau screen was not met. See
`docs/TEST2A_PLATEAU_CONTINUATION.md`.

A read-only residual-structure audit of that frozen endpoint found no
saturation-switch-localized failure: `|qv-q_sat|/q_sat <= 1e-6` accounts for
only `0.244%` of residual squared energy. Instead, `83.89%` lies at
`|A| > 1e-2 max|A|`, with moderate weighting toward sub-saturated evaporation
branches and early states. Activity-stratified sign accuracy reaches `0.92494`,
`0.98937`, and `1.0` at the `1e-3`, `1e-2`, and `1e-1` thresholds. Because the
residual is distributed rather than a sharp representation failure and the
source fit remained nonstationary, the diagnostic recommendation is
`CONTINUE_OPTIMIZATION`. See `docs/TEST2A_RESIDUAL_STRUCTURE.md`.

The subsequent accepted memory-20 continuation is frozen as the practical
Test 2A operator baseline. Its 5-32-32-1 float64 parameter artifact has 1,281
parameters and reaches normalized MSE `0.004285912836972889`, relative RMS
error `0.0654668835135207`, correlation `0.9978490330152804`, and active
`1e-3` sign accuracy `0.9557392085657594`. ROL did not meet the nominal
mathematical gradient tolerance; Test 2A-2 does not continue optimization.

Test 2A-2 inserts that frozen network only inside the existing local JAX moist
evaluation. It retains the analytical rain law at the current deployed state,
the exact coupled sources, broken-CG3 4-by-4 GLL packing, weak assembly, mixed
mass solve, and Euler update. The original UFL backend remains the default and
the analytical JAX path remains the no-provider behavior. Pure-network parity
on all 331,776 state-0..80 samples is bitwise exact. Complete-child state and
parameter JVP/VJP, joint differentiated-VJP, weak primal, and full-split smoke
certifications are defined for the external Firedrake environment in
`docs/TEST2A_EMBEDDED_NEURAL_A.md`. No state after 80 is accessed.

Test 2A-3A defines the deployed-discrete offline neural objective without
training it. At each fixed truth state, shared original R cancels and the
neural error follows the exact linear map
`G_k=M^-1 W H_k` from local A error to the mass-solved moist tendency. The
global mixed-mass loss therefore weights pointwise operator error by
`G_k^* M G_k`; it contains no recursive state feedback. Pointwise exact fits
minimize both offline losses, but projection null spaces and finite-network
weighting can produce different discrete optima. The actual value/gradient
comparison is an external certification gate documented in
`docs/TEST2A_DEPLOYED_DISCRETE_OFFLINE.md`. Canonical optimization, if later
authorized, starts from the same seed-0 initial pytree as Test 2A-1 rather
than the operator-trained artifact.

The external Test 2A-3A comparison confirmed that the two offline objectives
are strongly distinct on this benchmark. At seed 0 their gradient cosine is
`-0.17895866992372664` with nonproportional residual
`0.9838565924255072`; at the frozen operator fit the corresponding values are
`0.1828428001277335` and `0.9831421618674736`. A separate fixed-state
deployed-discrete fit is therefore justified.

Test 2A-3B prepares an evaluation-only, 80-step autonomous deployment of the
frozen a-priori network from `X*_0` over the already used training-support
interval. It records mixed/field/energy/enstrophy errors, evaluates neural and
analytical A on every model-generated state, retains and audits original R,
and imposes no automatic scientific pass threshold. Test 2A-3C prepares a
same-seed, memory-20 ROL L-BFGS deployed-discrete fit. Its production-oracle
certified sparse cache removes repeated Firedrake setup while preserving the
exact fixed-state weak/mass objective; it forms no dense `G_k` or `K_k`.
Periodic CG3 indexing is obtained from cell topology rather than interpolated
seam coordinates. The resulting 48-by-48 ordering reconstructs the production
mass matrix to float64 roundoff and performs no Firedrake/PETSc solve in the
training hot loop. The completed direct-production Method-2 fit has
`J_disc=0.0017427829635521567`, parameter SHA-256
`4db13a84f52d6fcf66f0345ce5c677aff2b46a84350228a78bc05b073ff6b12a`, and
`J_op=0.0020819762080123453`. This accepted artifact is a comparison result,
not an accelerated warm start.
Commands and restart contracts are in `docs/TEST2A_3B_3C.md`.

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
Canonical Gate 2 is only a nine-point scalar objective-landscape sanity check,
so its selected derivative level is objective-only.  General scans retain
explicit objective-plus-gradient and objective-plus-gradient-plus-Hessian
levels.  Gate 3 remains the actual parameter-learning stage and continues to
use the exact cached gradient and HVP machinery.

Gate 3 recovered c0=0.14 to reported precision under all four objectives.
Post-fit Gate 4 is therefore a deterministic workflow certification: each
successful fit supplies its own recovered c0, trusted state 80 is the sole
held-out initializer, and the full six-field model evolves autonomously for 80
steps before comparison with truth states 81 through 160.  The gate reports
per-time mixed/fieldwise mass errors and energy/enstrophy mismatches with
explicit zero-reference handling.  It is not framed as difficult
extrapolation or machine-learning generalization.

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
trials.  All stopping and safeguard tolerances are dimensionless: the gradient
must decrease by `1e-9` relative to its initial nonzero magnitude, a parameter
step is small at `1e-11` relative to `max(1,abs(z))`, and positive curvature is
screened at `1e-12` relative to the first nonzero curvature magnitude.  An
exactly zero initial gradient is handled separately.  A positive exact HVP
supplies the Newton step; otherwise a positive secant or deterministic bounded
gradient fallback is used.  Negative curvature is rejected by sign regardless
of magnitude.  Proposals are clipped to the bounds and backtracking uses the
scale-homogeneous Armijo condition with constant `1e-4`.  Success may follow
relative gradient reduction, a small relative parameter step, or the scalar
projected-gradient condition at a bound—not merely objective reduction.
Line-search failure, nonfinite values, and iteration limit have distinct
termination reasons.

Multiplying any objective, gradient, and Hessian consistently by `alpha>0`
therefore leaves Newton directions, safeguards, accepted iterates, stopping
decisions, and termination reasons unchanged up to floating-point roundoff.
Absolute objective and gradient histories remain useful reported quantities,
but they do not control convergence.

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
