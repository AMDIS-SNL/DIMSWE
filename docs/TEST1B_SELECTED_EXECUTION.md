# Selected Test 1B: exact external execution ladder

## Scientific selection

The first production Test 1B is now selected:

```text
case             doublevortex
nx, ny           16, 16
dt               100
nsteps           160
final_time       16000
output_stride    1
c0_truth         0.14
c0_initial       0.07
control          c0 = 0.07 z
s                3.2
moist_backend    ufl
seed             0
```

The machine-readable source of truth is
`dimswe/configs/test1b_selected_plan.json`.  The canonical truth path is
`external-results/test1b-production/truth_c0_0.14`.  The earlier
output-stride-8 pilot is not canonical production truth.

The stable 16-by-16, dt=100 pilot had maximum mixed c0-pair separation
`1.643758942633589e-4`, final velocity separation `7.174971e-3`, final
kinetic-energy mismatch `7.791521e-3`, final projected-enstrophy mismatch
`1.532189e-2`, maximum Qc separation `6.784130e-2`, and no numerical-stability
heuristic warning.  The stable 32-by-32, dt=100 case was not selected because
c0 sensitivity was about one order of magnitude weaker under the deployed
`r^s`, `s=3.2` scaling for this smooth flow.  No 64-by-64 run is requested.

The 32-by-32, dt=400, c0=0.14 run is rejected as truth or training data.  Its
exact Euler-child audit gave `sigma=2.5734163936944783>2`.  It remains only
diagnostic evidence of explicit-Euler instability.

## Data boundary and exact indexing

States `X_0,...,X_160` occur at `t_n=100n`.

- Training states are `X_0,...,X_80`, corresponding to `0<=t<=8000`.
- Training transitions have start indices `n=0,...,79` and map
  `X_n -> X_(n+1)`.
- State `X_80` is both the final training state and the trusted initializer for
  post-fit held-out deployment.
- Held-out target states are `X_81,...,X_160`, corresponding to
  `8000<t<=16000`.
- Held-out transitions have start indices `n=80,...,159`.

Scan and fit commands load only `X_0,...,X_80`.  States `X_81,...,X_160` do
not enter targets, fitted normalization, landscapes, gradients, HVPs, line
searches, or optimizer stopping.  They are loaded only by the post-fit
`evaluate` command.

## Canonical four-method comparison

All modes use observation stride one.  Operator/apriori and
deployed-discrete offline objectives use fixed training inputs
`X_0,...,X_79` and never propagate a candidate state.

Truth-reset uses 16 disjoint length-five windows with starts
`k=0,5,...,75`.  Rollout spans all 80 training transitions from state zero.
Both use accumulated mismatch and the same target-specific normalizer:

```text
N_n = ||X*_n||_M^2,

J_reset(c) = (1/80) sum_(q=0)^15 sum_(j=1)^5
             1/2 ||F_c^j(X*_(5q))-X*_(5q+j)||_M^2 / N_(5q+j),

J_roll(c)  = (1/80) sum_(n=1)^80
             1/2 ||F_c^n(X*_0)-X*_n||_M^2 / N_n.
```

Here `||Y||_M^2 = integral_Omega inner(Y,Y) dx_GLL` is the existing mixed-state
metric: the natural sum over `v,h,S,Qv,Qc,Qr`, with no extra field scaling.
Each target `X*_1,...,X*_80` has outer coefficient `1/80`, appears once in each
objective, and uses the bit-identical denominator `N_n` in both.  Different
targets can have different `N_n`, but this target-dependent weighting is shared
exactly between reset and rollout.

Within each reset window, predictions are recursive for five steps; the
terminal prediction is discarded and the next window starts from truth.
Rollout has no truth reintroduction over its 80-step training trajectory.  The
intentional distinction is therefore recursion depth five versus 80, not
training interval, target coverage, accumulation, metric, or normalization.

The generic J4B API still supports overlapping reset windows, terminal loss,
short rollouts, and initial-guess-residual normalization as later ablations.
The selected plan fixes none of those alternatives globally.  Accumulated
objectives now cache each complete deployed state and production split cache
once per trajectory.  Reset constructs 16 independent five-step trajectories;
rollout constructs one 80-step trajectory.  Their reverse recursions add the
local target loss at every state before continuing through the preceding
cached split step.  The exact tangent and incremental-reverse traversal does
the same for local loss Hessian actions.  No intermediate Riesz map is used.

## Shared environment

Run each gate from a fresh shell with:

```bash
cd /path/to/DIMSWE-collaborator
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export TEST1B_PLAN="$PWD/dimswe/configs/test1b_selected_plan.json"
export TEST1B_ROOT="$PWD/external-results/test1b-production"
export TRUTH_RUN="$TEST1B_ROOT/truth_c0_0.14"
```

Do not combine the gates into an unattended campaign.

## GATE 1 — production truth and audit

Validate the immutable selected plan, then generate a new every-step truth:

```bash
python -m dimswe.selected_test1b validate-plan \
  --plan "$TEST1B_PLAN"

python -m dimswe.resolved_hidden_c0_driver run \
  --case doublevortex \
  --nx 16 --ny 16 \
  --dt 100 --nsteps 160 --final-time 16000 \
  --output-stride 1 \
  --c0 0.14 --s 3.2 \
  --moist-backend ufl --seed 0 \
  --output-directory "$TRUTH_RUN"
```

Audit the completed truth without advancing a solver:

```bash
python -m dimswe.selected_test1b audit-truth \
  --plan "$TEST1B_PLAN" \
  --truth-run "$TRUTH_RUN" \
  --output "$TEST1B_ROOT/gate1_truth_audit.json"
```

The audit requires 161 restart arrays, Firedrake checkpoints, diagnostic JSON
files, and spectra; exact metadata; finite arrays; strictly positive minimum h;
c0=0.14; output stride one; times 0,100,...,16000; no late-time growth warning;
and the exact training/held-out indices above.

**STOP after Gate 1.** Review `gate1_truth_audit.json`; do not begin scans
unless `passed` is true and the trajectory is scientifically accepted.

## Selected-plan-driven inference

After Gate 1 is accepted:

```bash
COMMON_TEST1B_ARGS=(
  --truth-run "$TRUTH_RUN"
  --selected-plan "$TEST1B_PLAN"
)

python -m dimswe.resolved_hidden_c0_inference plan \
  "${COMMON_TEST1B_ARGS[@]}" \
  --output "$TEST1B_ROOT/canonical_inference_index_plan.json"
```

## GATE 2 — scalar objective-landscape sanity check

The nine-point interval `[0.04,0.20]` contains c0 truth exactly at 0.14.  Every
canonical Gate-2 point records only `J(c0)`, finite/nonfinite status, forward
model-step count, wall time, and any failure reason.  The selected plan fixes
`derivative_level=objective_only`; no gradient, reverse, tangent, HVP, or
incremental-reverse work is requested by these commands.

```bash
for MODE in apriori_offline discrete_offline truth_reset rollout; do
  python -m dimswe.resolved_hidden_c0_inference scan \
    --mode "$MODE" \
    "${COMMON_TEST1B_ARGS[@]}" \
    --output "$TEST1B_ROOT/gate2_objective_scan_${MODE}.json" || break
done
```

The selected plan owns the interval and nine grid points; conflicting CLI
overrides, including derivative-level overrides, are rejected.  Each command
records fixed-target preprocessing separately from point costs.
Truth-target-mass normalizers need no solver
steps.  Objective-only evaluation is 80 forward deployed steps for either
canonical solver loss: `16*5` for reset and one 80-step rollout.  Reverse,
tangent, and incremental-reverse counts must all be zero.  The two offline
landscapes likewise evaluate only their scalar objective.

An earlier external exact truth-reset spot check at `c0=0.04` is retained as
diagnostic evidence:

```text
J          7.927002933447668e-11
dJ/dc0    -1.70886319543831e-09
d2J/dc0²   2.108131445611791e-08
wall time  approximately 656.5 seconds
work       forward 160, reverse 80, tangent 80, incremental reverse 80
```

That process was manually interrupted during the next point's exact HVP
assembly while active at full CPU.  It was not a correctness failure.  This
spot check is not part of the canonical objective-only nine-point scan.

The subsequently completed canonical Gate-2 scan found the sampled minimum of
all four objectives exactly at `c0=0.14`.  For this scalar normalized benchmark,
the operator/apriori and deployed-discrete curves coincide exactly.  Near
`c0=0.12` their objectives are about `1.02e-2`, whereas truth reset is about
`2.81e-12` and rollout about `1.49e-10`.  Rollout is therefore substantially
more sensitive than five-step truth reset on this landscape, but absolute
objective magnitude is neither training quality nor a convergence test.  Gate
3 uses scale-invariant bounded-Newton decisions precisely so these definition-
dependent magnitudes cannot cause premature stopping or curvature rejection.

**STOP after Gate 2.** Inspect all four objective landscapes, finite status,
and forward cost.  The expected minimizer is 0.14, but software does not
hard-code that outcome as success.

## GATE 3 — actual parameter-learning optimization with exact derivatives

Only after all landscapes are accepted:

```bash
for MODE in apriori_offline discrete_offline truth_reset rollout; do
  python -m dimswe.resolved_hidden_c0_inference fit \
    --mode "$MODE" \
    "${COMMON_TEST1B_ARGS[@]}" \
    --output "$TEST1B_ROOT/gate3_fit_${MODE}.json" || break
done
```

Every fit starts at c0=0.07 and uses the same deterministic bounded-Newton
configuration.  Gate 3's derivative path remains unchanged: it requests exact
gradients and exact HVPs from the cached production trajectory derivatives.
Each result
records recovered c0, relative error, accepted
steps, objective/gradient/HVP evaluations, complete solver steps, wall time,
and the termination reason.  Objective decrease alone is not called
convergence.  Gradient stopping is relative to the initial nonzero gradient;
step stopping is relative to `max(1,abs(z))`; positive curvature magnitude is
relative to the first nonzero curvature reference while negative curvature is
always rejected; and backtracking uses a scale-homogeneous Armijo test.  These
rules, the parameter bounds, and the optimizer budget are identical for all
four modes.

The accepted Gate-3 results are:

| mode | starting c0 | learned c0 | accepted steps | objective / gradient / HVP evaluations | fit wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| operator/apriori offline | `0.07` | `0.14` | 1 | `3 / 2 / 1` | `0.00170 s` |
| deployed-discrete offline | `0.07` | `0.14` | 1 | `3 / 2 / 1` | `0.00161 s` |
| truth reset | `0.07` | `0.13999999999997986` | 4 | `9 / 5 / 4` | `1721.56 s` |
| full rollout | `0.07` | `0.13999999999998547` | 5 | `11 / 6 / 5` | `2372.70 s` |

All four values equal the generating `0.14` to the reported scientific
precision.  These fits used DIMSWE's custom safeguarded bounded scalar Newton
solver.  They did not use SciPy or ROL.  Exact gradients and HVPs came from the
production discrete derivative machinery.  Fit inputs stopped at state 80;
held-out states 81 through 160 were not loaded during Gate 3 and could not
affect line searches, stopping, or normalization.

## GATE 4 — common held-out autonomous deployment

After all four fits are accepted:

```bash
for MODE in apriori_offline discrete_offline truth_reset rollout; do
  python -m dimswe.resolved_hidden_c0_inference evaluate \
    "${COMMON_TEST1B_ARGS[@]}" \
    --fit-result "$TEST1B_ROOT/gate3_fit_${MODE}.json" \
    --output "$TEST1B_ROOT/gate4_evaluate_${MODE}.json" || break
done
```

Every recovered c0 uses the same autonomous training deployment and a separate
held-out autonomous deployment initialized from trusted `X_80`.  Outputs
include relative c0 error; one-step, training, held-out, final, accumulated,
and six field-block errors; kinetic-energy and projected-enstrophy mismatch;
hyperviscosity and high-wavenumber mismatch; finite/height and non-conclusive
growth status; costs; and objective values under all four training definitions.

Before advancing, `evaluate` independently requires a complete, successful,
configuration-compatible fit and reads its recovered c0 from that JSON record;
the evaluator never substitutes the known truth c0.  The canonical held-out
path is exactly

```text
Xhat_80 = X*_80,
Xhat_(n+1) = F_crecovered(Xhat_n),  n=80,...,159.
```

It constructs all 80 predicted states recursively without a truth reset.  Only
after this prediction exists does it compare against `X*_81,...,X*_160`.
Per-time output includes mixed and six fieldwise relative mass errors with
explicit zero-reference status; absolute and relative kinetic-energy and
projected-enstrophy mismatches; maxima and final values; finite state, minimum
height, and existing growth heuristics.  The canonical held-out deployment
step count is 80.  The broader common-evaluation record separately accounts
for its retained training-interval deployment and training-objective
cross-evaluations.

Gate-4 certification requires relative c0 error at most `1e-12`, required
held-out relative errors at most `1e-10`, finite states, positive height, no
undefined nonzero-over-zero relative error, and no late-time growth warning.
These are deterministic numerical-precision workflow checks, not fitted force
scales or scientific-performance thresholds.  A violation marks the output
failed and stops the command ladder for diagnosis.

After all four evaluations pass, write the compact plotting summary and three
minimal figures:

```bash
python -m dimswe.test1b_gate4 \
  --result "apriori_offline=$TEST1B_ROOT/gate4_evaluate_apriori_offline.json" \
  --result "discrete_offline=$TEST1B_ROOT/gate4_evaluate_discrete_offline.json" \
  --result "truth_reset=$TEST1B_ROOT/gate4_evaluate_truth_reset.json" \
  --result "rollout=$TEST1B_ROOT/gate4_evaluate_rollout.json" \
  --output "$TEST1B_ROOT/gate4_aggregate_summary.json" \
  --plot-directory "$TEST1B_ROOT/gate4_plots"
```

The summary contains the starting and learned c0, maximum/final mixed error,
maximum/final absolute and relative KE and projected-enstrophy mismatches,
held-out complete-step count, evaluation wall time, and pass/fail for each
strategy.  Its plot data support mixed error versus time and truth/deployed KE
and projected enstrophy.  Nearly coincident curves are the expected result.

Once all four external records and the aggregate pass, Gate 4 closes Test 1B.
This is deterministic end-to-end certification of the correctly specified
inverse workflow, not evidence of difficult extrapolation or machine-learning
generalization.  Horizon or observation-cadence sweeps remain a later,
separately authorized study.
