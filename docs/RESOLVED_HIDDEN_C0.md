# Resolved hidden-c0 preparation: Test 1B-0 and Test 1B

## Scope and authorization boundary

J4B-PREP added two stages.  External pilot review has now selected the first
scientific production case:

- **Test 1B-0** runs paired complete-production flows to ask whether c0 is
  identifiable at a candidate resolution and duration.
- **Test 1B** is the selected 16-by-16 `doublevortex` case with `dt=100`, 160
  steps, and every-step truth output.  Execution remains gated: truth, scans,
  fits, and held-out evaluation stop for review between stages.

The authoritative selection and exact commands are in
`dimswe/configs/test1b_selected_plan.json` and
`docs/TEST1B_SELECTED_EXECUTION.md`.  Generic templates later in this document
are retained as design history and are superseded by that exact ladder.

Test 1B remains a correctly specified scalar inverse problem.  Truth and
learner use identical equations, mesh, discretization, initial condition,
forcing terms, moist physics, timestep, split, and seed; they differ only in
physical c0.  The immutable control remains `c0=0.07 z`, with defaults
`c0_truth=0.14` and `c0_initial=0.07`.

Every command below starts with

```bash
cd /Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
```

The drivers are serial CPU tools.  They claim no MPI or accelerator support.
Per-state restart snapshots are experiment/state restartability, not generic
adjoint checkpointing, revolve, or a production checkpointing claim.

The cheap preparation checks are

```bash
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
python -m py_compile \
  dimswe/resolved_hidden_c0.py \
  dimswe/resolved_hidden_c0_driver.py \
  dimswe/analyze_resolved_hidden_c0.py \
  dimswe/resolved_hidden_c0_inference.py \
  dimswe/hyperviscosity_stability.py \
  tests/test_hyperviscosity_stability.py \
  tests/test_resolved_hidden_c0_prep.py
python -m pytest -q \
  tests/test_hyperviscosity_stability.py \
  tests/test_resolved_hidden_c0_prep.py
```

They do not launch a resolved flow.

## Completed exact Euler-child stability gate

Further resolved Test-1B-0 runs were suspended until the exact-child stability
table was generated and reviewed.  That external review is now complete and
selected the 16-by-16 dt=100 case recorded above; no additional pilot or
64-by-64 run is requested.  The external 32-by-32
`doublevortex` pair at `dt=400` was finite, but the `c0=0.14` member developed
late growth after initially dissipative behavior: the hyperviscosity tendency
norm reached about `3.57e6`, kinetic energy and projected enstrophy grew, and
the sampled high-wavenumber fraction reached about `1.28e-5`.  The `c0=0.07`
member remained well behaved through `t=16000`.  In this setting the high-k
signal is not accepted as evidence of physically populated small scales.

No production form is changed.  For each active space
`i in {v,h,S}`, the unchanged forms and `GeneralRK` sign convention give

```text
M_i Q_i = -K_i x_i,
M_i F_i = c0 r^s K_i Q_i,
x_i^+ = x_i + dt F_i
      = [I - dt c0 r^s (M_i^-1 K_i)^2] x_i,
r = max(mesh.dx/order, mesh.dy/order).
```

Here `M_i` is the exact configured GLL mass matrix and `K_i` is the exact
weak stiffness matrix assembled on the active production space.  The velocity
operator is assembled on vector spectral CG(3), including both components;
h and S use scalar spectral CG(3).  Qv, Qc, and Qr are inactive in this child.
If

```text
K_i phi = mu_i M_i phi,
lambda_i = r^s mu_i^2,
```

then the exact explicit-Euler condition for the positive-semidefinite child is

```text
sigma_i = dt c0 lambda_i,max <= 2.
```

The largest-mode amplification estimate is `|1-sigma_i|`; the full operator
also has an undamped null mode, so its amplification is at least one.  The
reported conservative Euler amplification bound uses the independently
bounded eigenvalue and is
`max(1, |1-dt*c0*lambda_i,upper|)`.  The table reports
`dt_max=2/(c0*lambda_i,max)`, the conservative bound
`2/(c0*lambda_i,upper)`, and a recommended timestep equal to a documented
default safety factor `0.8` times that conservative bound.

`dimswe.hyperviscosity_stability` assembles PETSc AIJ matrices and extracts
serial sparse CSR.  It checks that the deployed GLL mass is diagonal and the
stiffness is symmetric, then applies the exact similarity transform

```text
B_i = M_i^-1/2 K_i M_i^-1/2.
```

Symmetric ARPACK Lanczos (`scipy.sparse.linalg.eigsh`) estimates the largest
eigenvalue and reports the Ritz residual.  The maximum absolute row sum of
`B_i` supplies an independent conservative upper bound.  Production matrices
are never densified.  Dense conversion is permitted only by the explicitly
bounded tiny-oracle path.

The 2-by-2 oracle passed.  Vector v had 72 dofs and scalar h/S each had 36.
Sparse and dense largest eigenvalues agreed to at most `1.36e-16` relative
error.  Applying the matrix formula to an owned random state agreed with the
actual deployed hyperviscosity child to at most `1.64e-16` relative error in
v/h/S; Qv/Qc/Qr were unchanged exactly.  This verifies the implemented
operator and sign on the tiny production case, not the stability of a
resolved run.

### Preserved exact stability-audit commands

These commands reproduce the completed stability audit if independently
needed.  They do not authorize another pilot.  The table command assembles
operators and solves sparse eigenproblems but advances no DIMSWE state.

```bash
cd /Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export DIMSWE_STABILITY_CACHE_ROOT="$(mktemp -d /tmp/dimswe-hv-stability.XXXXXX)"
export PYOP2_CACHE_DIR="$DIMSWE_STABILITY_CACHE_ROOT/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$DIMSWE_STABILITY_CACHE_ROOT/tsfc"
export XDG_CACHE_HOME="$DIMSWE_STABILITY_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$DIMSWE_STABILITY_CACHE_ROOT/matplotlib"
export STABILITY_ROOT="$PWD/external-results/test1b0-stability"

python -m pytest -q tests/test_hyperviscosity_stability.py

python -m dimswe.hyperviscosity_stability verify-tiny \
  --grid 2x2 --dt 100 --c0 0.14 --s 3.2 \
  --output "$STABILITY_ROOT/tiny_dense_oracle.json"

python -m dimswe.hyperviscosity_stability table \
  --case doublevortex --grid 16x16 32x32 64x64 \
  --dt 400 --c0-values 0.07 0.14 --s 3.2 \
  --safety-factor 0.8 \
  --output "$STABILITY_ROOT/euler_stability_n16_n32_n64.json" \
  --csv "$STABILITY_ROOT/euler_stability_n16_n32_n64.csv"
```

Review every v/h/S row for both c0 values.  A converged Ritz value with
`sigma>2` proves that the requested timestep is outside the Euler interval.
`sigma_upper<=2` certifies it inside the conservative bound.  The intermediate
classification is inconclusive and requires tighter spectral analysis; it is
not permission to run.  Do not automatically launch another pilot after this
table.  A candidate dt must be selected from the worst active-field and c0
row, and its scientific run remains a separate human-reviewed decision.

## Repository case discovery

The repository has more dispatcher names than complete cases.  The following
is based on the actual source, not the names alone.

### 1. `doublevortex`: primary pilot candidate

Sources are `dimswe/initial_conditions.py:DoubleVortex`, the root `mtswe.cfg`,
and related test configurations.  It is the only nontrivial moist case already
configured as a production run.

- Domain: `[0,5,000,000)^2` m, periodic in x and y under the configured
  `rectangle-periodic` mesh.
- Existing mesh/time convention: 50-by-50 quadrilaterals, Q family order 3,
  GLL-lumped mass, `dt=400`, 20 steps, output every 2 steps.
- Spaces: vector CG(3) velocity; scalar CG(3) h and S; DG(1) Qv, Qc, Qr.
- Geometry: flat topography, constant `f=0.00006147`, no external forcing.
  The configured split terms are dry dynamics, hyperviscosity, DG1 limiter,
  and three-way moist physics.
- Initial height is the mean-zero-adjusted sum of two negative Gaussian-like
  periodic vortices.  With centres `(0.4Lx,0.4Ly)` and `(0.6Lx,0.6Ly)`, widths
  `sigma_x=sigma_y=3Lx/40`, and periodic sine coordinates `x'_i,y'_i`,

  ```text
  h = H0 - dh [exp(-(x'_1^2+y'_1^2)/2)
               + exp(-(x'_2^2+y'_2^2)/2)
               - 4*pi*sigma_x*sigma_y/(Lx*Ly)],
  H0=750, dh=75.
  ```

  Velocity is the repository's geostrophic pair formed from the corresponding
  double-angle periodic coordinates, proportional to `g*dh/(f*sigma)`.  The
  exact executable equations remain in the cited source rather than being
  duplicated in the driver.
- Thermal/moist initialization is

  ```text
  s = g [1 + 0.05 exp(-((x-Lx/2)^2+(y-Ly/2)^2)/((1/3)^2(Lx/2)^2))],
  S = h s,
  Qv = h qsat(h,s,0,q0=0.002,H0,g),  Qc=Qr=0.
  ```

The two vortices evolve and interact, and their finite widths contain more
spatial structure than a balanced zonal state.  The risk is that the initial
condition is still smooth: a short or coarse run may not populate enough high
modes for c0 to be identifiable.  That uncertainty is exactly what the pilot
measures.

### 2. `TC5`: conditional secondary pilot candidate

Source is `dimswe/initial_conditions.py:TC5`; no dedicated runnable repository
configuration exists.  The J4B driver applies the same complete MTSWE
production configuration used for the other pilot cases.

- Domain: `[0,2*pi*a)^2`, `a=6,371,120` m, with the pilot's doubly periodic
  quadrilateral mesh.
- Base state: `u=20 cos(y/a)`, `v=0`, constant `f=0.00006147`, and

  ```text
  h = H0 - a f u0/g sin(y/a) - B,  H0=5960,
  s = g [1 + 0.05 H0^2/h^2],       S=h s.
  ```

- Topography: a conical mountain of height 2000 and radius `pi*a/9`, centred
  at `(Lx/3,2Ly/3)`.
- Moisture: `Qv=h qsat(h,s,B,q0=0.007,H0,g)`, `Qc=Qr=0`.
- No external forcing; the mountain/flow interaction can generate wave and
  smaller-scale structure.

This is potentially richer for c0 identification, but it is less mature in
this repository.  The mountain is narrow on a coarse grid, and the planar,
doubly periodic adaptation requires visual and resolution checks.  It ranks
behind `doublevortex` until those checks pass.

### Complete but unsuitable, or incomplete, cases

- `TC2` is the smooth balanced zonal-flow base of TC5 with `B=0`.  It is
  complete, but near-steady large scales are a weak c0-identification target.
- `densitywave` is a dry two-dimensional state and does not populate the six
  moist prognostic fields required by this experiment.
- `isolatedvortices` references attributes never initialized by its class.
- `geostrophicturbulence`, `gravitywave`, and `galewsky` have only incomplete
  constructors and no executable `get_value` implementation.
- No complete repository shear-instability, Galewsky jet, or turbulence-like
  moist configuration was found.

The external pilot therefore has at most two candidates: `doublevortex`
first, and TC5 only if doublevortex evidence is inadequate or a contrasting
wave-generating flow is needed.

## Pilot execution contract

`dimswe.resolved_hidden_c0_driver` starts from
`dimswe/configs/resolved_hidden_c0_pilot.cfg` and exposes case, nx, ny, dt,
nsteps/final time, output stride, c0, s, moist backend, seed, spectral sampling,
and output directory.  It changes configuration values but duplicates no
production equation.  The actual execution is the unchanged production
six-child MTSWE split.

`run-pair` constructs two configurations and verifies that their serialized
physics configuration differs only in c0.  Output paths differ operationally,
but equations, IC, discretization, mesh, dt, physics, seed, and output times do
not.

Each saved output step contains:

- a Firedrake HDF5 checkpoint containing the mesh, mixed state, and named
  `v,h,S,Qv,Qc,Qr` fields;
- a flat float64 NPY state array for deterministic same-configuration restart
  and mass-metric comparison;
- a JSON diagnostic record; and
- an NPZ shell spectrum.

`metadata.json` records domain, boundary conditions, spaces/order, timestep,
steps and times, c0, s, moist backend, IC configuration, split identity, seed,
field layout, diagnostics, status, wall time, and Git branch/checkpoint when
obtainable.  Writes are atomic.  A valid complete directory is skipped; an
incomplete directory resumes from its latest valid saved state.  Ctrl-C marks
the run interrupted.  Restart is at the configured output cadence, not every
internal solver step.

## Pilot diagnostics

The paired mixed separation is assembled during analysis with the same
configured GLL mixed mass measure:

```text
D_X(t) = ||X_a(t)-X_b(t)||_M / ||X_b(t)||_M.
```

The same formula is assembled for each of v, h, S, Qv, Qc, and Qr.  The
analysis never advances or changes a saved solver trajectory.

Single-run diagnostics contain:

- kinetic energy `0.5 integral h |v|^2 dx_GLL`;
- the L2 norm and half squared norm (projected enstrophy) of the CG(3) L2
  projection of the repository `curl2D(v)` diagnostic;
- the mixed mass norm of the actual deployed hyperviscosity-child tendency;
- the mixed norm of the hyperviscosity child update and child-only kinetic
  energy change as secondary proxies;
- minimum height coefficient and finite-state status; and
- velocity high-wavenumber content.

The hyperviscosity quantities are contribution proxies.  The child kinetic
energy change is not asserted to be a sign-definite physical dissipation law
for the complete mixed state.

The spectrum never FFTs a Firedrake coefficient vector.  Physical velocity is
evaluated on a configurable uniform, cell-centred periodic grid.  A component-
wise two-dimensional FFT uses `norm='forward'`; modal energy is
`0.5 sum_components |v_hat|^2`, so its sum equals grid-mean kinetic energy by
Parseval.  Integer cycle modes use physical wavenumbers `2*pi*k/L`.  Modes are
grouped by nearest-integer radial shell, and both shell sums and shell means
are saved.  The default high-mode fraction uses radii above two thirds of the
largest sampled radius.  This is a diagnostic only and does not enter the
solver or objective.

Finite-state status is now deliberately separate from numerical-stability
interpretation.  Finite coefficients and positive h do not classify a run as
stable.  Analysis JSON retains a finite-state check and separately reports
non-conclusive late-time-growth heuristics for kinetic energy, projected
enstrophy, hyperviscosity tendency norm, and high-wavenumber fraction.  Each
heuristic compares the median of the final quarter of saved samples with the
median of the first half.  Default warning factors are respectively 1.25, 2,
10, and 10; all window fractions, factors, and absolute floors are serialized.
A warning is neither a necessary nor sufficient proof of instability, and no
warning is not a stability certificate.

Threshold flags now say only that the high-wavenumber threshold was exceeded;
they do not label those modes dynamically or physically populated.  The exact
Euler-child spectral audit, the growth warnings, and the finite-state check
must be inspected together.  None automatically selects a scientific case.

## Suspended historical Test 1B-0 workflow

The commands in this section describe the staged pilot design but are
currently **not authorized for execution**.  The exact Euler-child stability
gate above must be reviewed first, and a new timestep must not be inferred
silently from the old `dt=400` recipes.

The first sequence uses existing `doublevortex` conventions but does not claim
that 16, 32, or 64 is adequate.  Relative unknown counts scale approximately
as nx times ny: 32 squared has about 4 times and 64 squared about 16 times the
spatial unknowns of 16 squared.  Assembly/solve wall time can grow faster, so
no absolute runtime is claimed without external timing.

### A1. Historical low-cost 16-squared pair — do not run now

```bash
export PILOT_ROOT="$PWD/external-results/test1b0"
python -m dimswe.resolved_hidden_c0_driver run-pair \
  --case doublevortex --nx 16 --ny 16 --dt 400 --nsteps 40 \
  --output-stride 2 --c0-a 0.07 --c0-b 0.14 --s 3.2 \
  --moist-backend ufl --seed 0 \
  --output-directory "$PILOT_ROOT/doublevortex_n16"
```

### A2. Historical analysis command

```bash
python -m dimswe.analyze_resolved_hidden_c0 pair \
  "$PILOT_ROOT/doublevortex_n16/doublevortex_n16x16_c0_0.07" \
  "$PILOT_ROOT/doublevortex_n16/doublevortex_n16x16_c0_0.14" \
  --output "$PILOT_ROOT/doublevortex_n16_summary.json" \
  --plot-directory "$PILOT_ROOT/doublevortex_n16_plots"
```

Check the exact-child stability table, finite status, heuristic growth
warnings, separation onset, and high-mode history.  Threshold exceedance alone
does not establish physically populated high modes.

### A3. Historical refinement recipe — do not run now

```bash
python -m dimswe.resolved_hidden_c0_driver run-pair \
  --case doublevortex --nx 32 --ny 32 --dt 400 --nsteps 40 \
  --output-stride 2 --c0-a 0.07 --c0-b 0.14 --s 3.2 \
  --moist-backend ufl --seed 0 \
  --output-directory "$PILOT_ROOT/doublevortex_n32"

python -m dimswe.analyze_resolved_hidden_c0 pair \
  "$PILOT_ROOT/doublevortex_n32/doublevortex_n32x32_c0_0.07" \
  "$PILOT_ROOT/doublevortex_n32/doublevortex_n32x32_c0_0.14" \
  --output "$PILOT_ROOT/doublevortex_n32_summary.json" \
  --plot-directory "$PILOT_ROOT/doublevortex_n32_plots"

python -m dimswe.analyze_resolved_hidden_c0 resolutions \
  "$PILOT_ROOT/doublevortex_n16_summary.json" \
  "$PILOT_ROOT/doublevortex_n32_summary.json" \
  --output "$PILOT_ROOT/doublevortex_resolution_trend.json"
```

This historical recipe did not run 64 squared automatically.  Any future
resolution refinement requires a newly selected timestep after the stability
table is reviewed.  A TC5 screen would likewise be a distinct, separately
authorized candidate, not a fallback launched from these old commands:

```bash
python -m dimswe.resolved_hidden_c0_driver run-pair \
  --case TC5 --nx 32 --ny 32 --dt 400 --nsteps 40 \
  --output-stride 2 --c0-a 0.07 --c0-b 0.14 --s 3.2 \
  --moist-backend ufl --seed 0 \
  --output-directory "$PILOT_ROOT/TC5_n32"

python -m dimswe.analyze_resolved_hidden_c0 pair \
  "$PILOT_ROOT/TC5_n32/TC5_n32x32_c0_0.07" \
  "$PILOT_ROOT/TC5_n32/TC5_n32x32_c0_0.14" \
  --output "$PILOT_ROOT/TC5_n32_summary.json" \
  --plot-directory "$PILOT_ROOT/TC5_n32_plots"
```

### B. Mandatory scientific stop

After the stability table and any later authorized summaries exist, stop.
Human/ChatGPT review must select the flow,
nx, ny, dt, duration, and output cadence.  Relevant evidence is:

- material separation beyond roundoff and its onset time;
- high-mode content that remains credible after exact-child and growth review;
- finite trajectories, separately assessed numerical stability, and sensible
  height/energy histories;
- a nonzero deployed hyperviscosity proxy;
- no obvious grid-scale-only artifact; and
- a resolution trend that does not reverse the identifiability conclusion.

Software readiness does not authorize Test 1B production inference.

## Prepared Test 1B inference contract

`ResolvedInferenceConfiguration` keeps training transitions half-open
`[training_start,training_stop)` and held-out transitions
`[training_stop,heldout_stop)`.  The boundary truth state initializes held-out
deployment, but no held-out transition enters fitting.  Observation stride is
independent of model dt.  Values 1, 2, 5, and 10 are supported.  A configured
solver horizon must end on an observed state, so it must be divisible by the
observation stride; incompatible combinations fail explicitly.
Training and held-out endpoints must also lie on that cadence so final and
accumulated evaluation refer to an unambiguous observed endpoint.

Horizons such as 1, 3, 5, and 10 are accepted without launching a parameter
campaign.  Terminal and accumulated losses are distinct enum values and CLI
arguments.  Reset-window stride is independent of reset-window length: its
default preserves overlapping-window experiments, while the canonical Test 1B
uses length five and stride five.  Rollout length is independent and is 80 for
the canonical plan.

Let K be fixed training observation states, `b_k(c*)` the assembled weak
hyperviscosity RHS, `Delta_k(c*)=dt M^-1 b_k(c*)`, and

```text
N(k,p)=||F_cinit^p(X*_k)-X*_(k+p)||_M^2.
```

`N(k,p)` is the generic initial-guess-residual convention.  The selected
canonical comparison instead uses the shared target normalizer

```text
T_n=||X*_n||_M^2
```

for both reset and rollout so the same target cannot receive different
effective scaling merely because its prediction starts at a different state.
Both choices are explicit `solver_loss_normalization` values; no Test-1A
objective is changed.

The four resolved objectives are exactly:

```text
J_op(c)   = mean_(k in K excluding endpoint)
            1/2 ||(c/c*-1)b_k(c*)||_2^2 / ||b_k(c*)||_2^2,

J_disc(c) = mean_(k in K excluding endpoint)
            1/2 ||(c/c*-1)Delta_k(c*)||_2^2 / ||Delta_k(c*)||_2^2,

J_reset(c)= mean_(window k, selected prefix p)
            1/2 ||F_c^p(X*_k)-X*_(k+p)||_M^2 / N(k,p),

J_roll(c) = mean_(selected prefix p)
            1/2 ||F_c^p(X*_(training_start))
                    -X*_(training_start+p)||_M^2 / N(training_start,p).
```

Operator/apriori and deployed-discrete objectives never propagate candidate
states.  Reset windows recursively differentiate through the complete solver
inside each window and then discard their prediction.  Rollout starts once
and is autonomous for its configured trajectory.  Terminal uses only `p=m`;
accumulated uses observed states through m.  The actual deployed J3
forward-cache, tangent, reverse, and incremental-reverse machinery is retained
for every complete solver step.

For the selected Test 1B, substitute `T_(k+p)` for `N(k,p)`, use reset starts
`k=0,5,...,75` and prefixes `p=1,...,5`, and use rollout prefixes
`p=1,...,80`.  Thus both solver objectives contain the target multiset
`{1,...,80}` exactly once.  The mixed metric is
`integral_Omega inner(Y,Y) dx_GLL`, the natural unit-weighted sum of all six
state blocks.  The only intended semantic difference is that feedback is reset
after five steps in the reset objective and persists for 80 steps in rollout.

Accumulated evaluation constructs each forward state exactly once.  For a
window of length `m`, the adjoint starts with the local loss dual at `m`; each
reverse step transports that dual to the preceding state, where the preceding
local loss dual is added directly in the state dual space.  The HVP constructs
one tangent trajectory and applies the certified incremental reverse once per
step, adding `M delta_X_n/N_n` from the local quadratic loss at each target.
There is no intermediate Riesz solve.  Consequently the canonical reset and
rollout objective-only evaluations each use 80 deployed forward steps rather
than replaying every prefix independently.  Gradient and HVP records expose
forward, reverse, tangent, and incremental-reverse counts separately.  This
removes artificial replay overhead; it does not assert equal optimization
iterations, memory use, or wall time between modes.

Every fit uses the same deterministic bounded normalized-scalar Newton method
as Test 1A and retains exact gradient/HVP calls.  Landscape scans have an
explicit derivative level: `objective_only`, `objective_gradient`, or
`objective_gradient_hessian`.  The selected Gate-2 policy is objective-only at
all nine prescribed c0 values; it records J, finite status, forward work, wall
time, and failure reason without reverse, tangent, or incremental-reverse work.
Derivative-capable scans remain explicit generic opt-ins.

Landscape records are written atomically after each scalar point and resume by
skipping completed points.  The serialized scan configuration includes its
derivative level.  Completed points with different derivative-level metadata
are rejected rather than silently resumed.  Fit and evaluation records detect
a completed matching intent; after interruption they are deterministically
rerun from the beginning because generic adjoint checkpointing is outside this
milestone.

Common evaluation starts a training autonomous deployment and a separate
held-out autonomous deployment at the truth boundary.  It records c0 error,
one-step, training and held-out trajectory, final, accumulated, and fieldwise
mass errors; kinetic-energy, high-wavenumber, and hyperviscosity-proxy
mismatches; finite state; costs; and cross-evaluation under all four objective
families.

## Canonical external execution

The production case and corrected target-coverage semantics are now selected.
To prevent stale generic flags from restoring the superseded five-step rollout,
the canonical scan, fit, and evaluation commands load
`dimswe/configs/test1b_selected_plan.json` and reject scientific indexing
overrides.  Use only the gated commands in
`docs/TEST1B_SELECTED_EXECUTION.md`.  The general CLI remains available for
explicit later ablations, but it is not the source of truth for the canonical
comparison.
