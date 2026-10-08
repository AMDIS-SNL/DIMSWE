# Test 2A-1: local neural operator learning for A

## Scientific scope

The accepted Test-2 activity audit classified the selected double-vortex
training interval as `A_ACTIVE_R_WEAK`.  Test 2A-1 is therefore a local
representability proof for the phase-change rate A.  It does not embed a
network in the timestepper, run deployed-discrete training, differentiate a
solver trajectory, or inspect truth states 81 through 160.

Only A is replaced.  The future hybrid child must evaluate the original
deployed rain physics at its current state and preserve

```text
Qv_t =  h A_theta
Qc_t = -h (A_theta + R_original)
Qr_t =  h R_original
S_t  =  h beta2 A_theta.
```

R happened to be zero on the complete operator-training dataset, but it is
not encoded as zero.  This matters because an imperfect A model can move an
autonomous trajectory into a region where the original R law becomes active.

## Reused infrastructure

The implementation extends rather than duplicates the existing layers:

- `LearnedPhysicsModel` still composes a `FeatureMap`, arbitrary-pytree
  `ParameterizedModel`, and `OutputMap` under the strict x64/immutability
  contract.
- `JAXMoistEulerPrimal` supplies the certified J1 cell-local packing and
  `cache.rates["A"]`; it remains an opt-in replica of the default UFL child.
- the deployed representation is the broken-CG3, cell-major tensor-product
  4-by-4 GLL array, with the first physical coordinate varying fastest.
  Shared physical CG points are repeated exactly as the local child sees them.
- JAX supplies objective gradients and the local certification HVP.
- J2 already provides `moist_source_jvp`, `moist_source_vjp`, and
  `moist_source_differentiated_vjp` for the unchanged analytical local source,
  with the surrounding certified Firedrake maps in `jax_moist_hvp`.  Test
  2A-1 does not modify or claim that these analytical-source callbacks already
  embed the neural pytree; that is a later deployment task.
- the existing ROL line-search/limited-memory-BFGS parameter-list convention
  is reused.  `JAXPytreeObjective` is only the missing serial
  pytree-to-`NumPyVector` bridge; it does not introduce a second learned-
  physics framework.

The existing PyROL adapters before Test 2A were scalar-c0 or production-state
specific.  They could not represent a 1,281-parameter nested neural pytree.
The repository's `dimswe/nn_train.py` is an old standalone Firedrake-adjoint
Burgers demonstration whose neural portion consists only of planning comments;
it contains no reusable network, dataset, or optimizer implementation.
The new adapter flattens and reconstructs arbitrary x64 JAX pytrees, gives ROL
exact reverse-mode gradients, and exposes exact HVP actions for certification.
Canonical Test 2A-1 uses line-search L-BFGS and does not ask ROL for HVPs.
Neither SciPy nor Optax/Adam is a hidden fallback.

## Frozen selected operator experiment

The machine-readable source of truth is
`dimswe/configs/test2a_selected_operator.json`:

```text
truth states             0,...,80 only
samples                  81 * 256 cells * 16 GLL = 331,776
sample order             state, cell, local GLL point
features                 h, S, Qv, Qc, B (in exactly that order)
target                   one physical A per deployed GLL sample
model                    dense 5 -> 32 -> 32 -> 1
activation               tanh
dtype                    float64
initialization           Glorot uniform, seed 0, zero biases
trainable parameters     1,281
batching                 deterministic full batch
optimizer                PyROL/ROL line-search L-BFGS, exact JAX gradient
L-BFGS secant memory     10 vector pairs
```

Depth, widths, activation, and seed are configuration values.  No other core
code assumes two hidden layers or width 32.  Initialization does not evaluate,
approximate, or encode the analytical A formula.

There is no random pointwise train/validation split.  Adjacent and duplicated
cell-local GLL samples would make such a split misleading.  The primary metric
is representability on the complete physical training support.  Any later
within-0..80 time-block diagnostic must be labeled as such and cannot be
described as future generalization.

## Training-only normalization and loss

Every statistic is fitted from the 331,776 samples in states 0 through 80.
Each input uses its training mean and population standard deviation.  A
constant or diagnostically zero-variation input receives scale 1 and an
explicit metadata flag; after centering it is identically zero.

The physical target is not centered.  Its positive scale is

```text
a_scale = RMS_training(A).
```

The sidecar also records `std(A)` and `max_abs(A)`.  RMS is selected because
the operator objective is mean squared error, so it conditions the typical
quantity being minimized and maps normalized zero exactly back to physical
`A=0`.  Standard deviation would introduce an unnecessary mean convention;
max-absolute scaling would compress the bulk of an intermittent distribution
relative to its tails.  An exactly zero/degenerate target has an explicit
scale-1 fallback and cannot be mistaken for active data.

The network predicts normalized A and the output map recovers physical units
exactly as `A_theta = a_scale * raw_output`.  The objective is

```text
J(theta) = (1/N) sum_i ((A_theta(z_i) - A_i*) / a_scale)^2,
N = 331,776.
```

Every deployed state/cell/local-point sample appears once.  Repeated shared
physical points are intentionally not deduplicated or reweighted.

## Metrics and baselines

The result records normalized MSE, physical RMSE and MAE, relative RMS error,
maximum absolute error, correlation when defined, and sign accuracy outside
`1e-6 max_abs(A)`.  RMSE/MAE/relative RMS are repeated on active strata
`|A| > 1e-3 max_abs(A)` and `|A| > 1e-6 max_abs(A)`.

Deterministic zero, training-mean constant, and five-input affine least-
squares predictions are reported as diagnostic baselines.  Before solver
embedding, the selected MLP must be reviewed and must materially outperform
the best trivial baseline.  The selected operational screen records whether
its relative RMS error improves by at least 10 percent; this is a readiness
screen, not a universal claim about neural closures or held-out behavior.

## Derivative contract

For the pure local full-batch loss, JAX provides the exact gradient with
respect to every pytree leaf.  A focused tiny test compares its directional
derivative to centered differences.  A second test obtains the exact HVP by
differentiating the gradient and compares it to a centered gradient
difference.  A separate analytic PyROL test certifies pytree roundtrip,
gradient and HVP callbacks, and an actual line-search L-BFGS solve.  The solve
uses gradients only, as intended for the production operator fit.

## External operator-learning ladder

Run this in the certified serial Firedrake/JAX/PyROL environment.  Data
preparation reads existing restart states and executes the local certified
moist child; it does not integrate a new truth trajectory.

```bash
cd /path/to/DIMSWE-collaborator
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export TEST2A_CACHE_ROOT="$(mktemp -d /tmp/dimswe-test2a.XXXXXX)"
export PYOP2_CACHE_DIR="$TEST2A_CACHE_ROOT/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$TEST2A_CACHE_ROOT/tsfc"
export XDG_CACHE_HOME="$TEST2A_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$TEST2A_CACHE_ROOT/matplotlib"

export TEST1B_TRUTH="$PWD/external-results/test1b-production/truth_c0_0.14"
export TEST1B_PLAN="$PWD/dimswe/configs/test1b_selected_plan.json"
export TEST2A_CONFIG="$PWD/dimswe/configs/test2a_selected_operator.json"
export TEST2A_ROOT="$PWD/external-results/test2a"

mkdir -p "$TEST2A_ROOT/dataset" "$TEST2A_ROOT/plots"

python -m dimswe.test2a_operator prepare-data \
  --truth-run "$TEST1B_TRUTH" \
  --selected-plan "$TEST1B_PLAN" \
  --configuration "$TEST2A_CONFIG" \
  --output "$TEST2A_ROOT/dataset/doublevortex_A_operator.npz"

python -m dimswe.test2a_pyrol train \
  --configuration "$TEST2A_CONFIG" \
  --dataset "$TEST2A_ROOT/dataset/doublevortex_A_operator.npz" \
  --output "$TEST2A_ROOT/operator_fit_result.json" \
  --parameter-output "$TEST2A_ROOT/trained_A_parameters.npz" \
  --plot-directory "$TEST2A_ROOT/plots"
```

Generated, untracked outputs are:

```text
external-results/test2a/
  dataset/doublevortex_A_operator.npz
  dataset/doublevortex_A_operator.json
  operator_fit_result.json
  trained_A_parameters.npz
  trained_A_parameters.json
  plots/operator_prediction_vs_truth.png
  plots/operator_training_history.png
```

The dataset sidecar contains normalization, provenance, exact sample
accounting, a content hash, and baseline metrics.  The fit result contains the
architecture, objective/gradient/HVP callback counts, ROL termination,
training history summaries, full-support metrics, baseline comparison, and
readiness screen.  The parameter NPZ and architecture sidecar reconstruct the
exact trained pytree without pickle.

## Boundary before Test 2A-2

No deployed neural moist child is added in Test 2A-1.  Before
deployed-discrete embedding is authorized, the external full-data result must
show finite deterministic optimization, acceptable active-region errors, and
material improvement over the trivial/affine baselines.  The next stage must
then reuse the certified weak GLL source assembly and preserve original R,
including its JVP/VJP/differentiated-VJP path.  Truth-reset, autonomous
rollout, and states 81 through 160 remain outside this task.

The controlled optimizer study has now been executed and is documented in
`TEST2A_OPTIMIZER_STUDY.md`.  Memory-20 L-BFGS is the current accuracy/cost
leader, but every tested method terminated at its iteration limit and none
reached credible stationarity.  The accepted classification is
`OPTIMIZATION_AND_MODEL_BOTH_UNRESOLVED`; the current operator is not ready for
solver embedding.

The subsequent memory-20 continuation materially improved the same fixed
model to relative RMS `0.205`, correlation `0.979`, and sign accuracy `0.813`.
It still terminated at `MAXITER`, and the last 100 accepted steps reduced the
objective by about 8 percent.  The decision is therefore
`CONTINUING_OPTIMIZER_LIMITED`, not convergence or embedding readiness.  See
`docs/TEST2A_CONTINUATION.md`.

Later controlled continuations of the same fixed problem produced the frozen
practical baseline selected for Test 2A-2. Its normalized MSE is
`0.004285912836972889`, relative RMS error is `0.0654668835135207`, and
correlation is `0.9978490330152804`. This handoff does not retroactively label
ROL mathematically converged; it freezes the accepted artifact without more
training. Its verified embedding contract is documented in
`docs/TEST2A_EMBEDDED_NEURAL_A.md`.
