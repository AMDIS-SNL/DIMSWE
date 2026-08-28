# Test 2A-3A: deployed-discrete offline neural objective

## Scientific boundary

Test 2A-3A defines and certifies the deployed-discrete offline objective. It
does not train the network in this gate, propagate a model-generated state,
truth-reset a multi-step window, run a rollout, or read any truth state after
80. Every example begins and ends at one trusted state `X_k*`, for
`k=0,...,80`.

The canonical future fit starts from the same seed-0 5-32-32-1 parameter
pytree that began Test 2A-1. Its SHA-256 fingerprint is
`6d4ac7bafe775b90a70e2a199ef3305308c3d02333f7daeac1c870b6f18e0975`.
The frozen operator-trained artifact is evaluated only as a comparison probe;
it is not a warm start.

## Exact fixed-state map

At a trusted state, let

```text
P_k       = exact state-to-cell-local 4x4 GLL packing,
H_k a     = structural source densities (h beta2 a, h a, -h a, 0),
W         = existing weak mixed-space source assembly,
M         = existing complete mixed mass matrix,
G_k       = M^-1 W H_k.
```

The analytical and neural children both evaluate `R_original(X_k*)`. Their
source difference is therefore exactly

```text
s_theta(X_k*) - s_*(X_k*)
  = H_k [A_theta(P_k X_k*) - A_*(P_k X_k*)].
```

This cancellation does not assume that R is zero. The implementation calls
the original rain law on both sides and requires the resulting R arrays to
agree to a scale-relative float64 roundoff bound at each fixed state. (The
two separately JIT-compiled analytical and neural kernels may differ by an
evaluation-order ULP.) Consequently, the mass-solved tendency difference is

```text
T_theta,k - T_*,k = G_k [A_theta,k - A_*,k].
```

`G_k` is fixed with respect to neural parameters at this trusted state. It
contains the exact h-dependent structural source map, weak assembly, mixed
mass inverse, function spaces, and production quadrature.

## Why the objective uses tendency

The complete moist Euler updates satisfy

```text
M_theta(X_k*) - M_*(X_k*) = dt [T_theta,k - T_*,k].
```

The unchanged incoming state cancels. Comparing the increment gives the same
difference. Because the selected objective normalizes by a target response
constructed with the same `dt`, the common `dt^2` also cancels. The
mass-solved tendency is therefore exactly equivalent while avoiding an
unchanged six-field state and an arbitrary timestep scale in the reported
loss.

The canonical objective is

```text
N_A = sum_(k=0)^80 ||G_k A_*,k||_M^2,

J_disc(theta)
  = [sum_(k=0)^80 ||G_k(A_theta,k-A_*,k)||_M^2] / N_A,

||u||_M^2 = integral_Omega inner(u,u) dx_GLL
           = <M u, u>.
```

`G_k A_*,k` is obtained by taking the analytical A already returned by the
certified target child, constructing only its structurally coupled A source,
and passing that source through the same `W` and `M^-1`. It is a normalization
response, not an independently discretized regression target. The actual
target remains the complete analytical-A/original-R deployed tendency.

One global training-support normalizer is used. There is no inverse
per-state-activity weighting, so nearly dry states do not receive enormous
weight merely because their local target increment is small. The normalizer
also excludes the shared R contribution, which cannot inform A.

## Relation to operator regression

The Test 2A-1 operator objective can be written

```text
J_op(theta)
  = [sum_k ||A_theta,k-A_*,k||_2^2]
    / [sum_k ||A_*,k||_2^2].
```

The discrete objective replaces the pointwise identity weighting with the
positive-semidefinite operator

```text
K_k = G_k^* M G_k.
```

Thus Test 2A-3A is not a recursive objective; it is a discretization-induced
weighted operator regression.

If the selected network can reproduce every A target exactly, that parameter
is a zero-loss minimizer of both objectives. The converse need not hold:
`G_k` can have a null space because local broken-GLL errors are weakly
projected into finite-dimensional mixed spaces. A discrete zero can therefore
hide an A error in `null(G_k)`. Even when `G_k` is injective on the realized
error manifold, finite-capacity least-squares optima generally differ because
`K_k` is not a scalar identity. The objectives have proportional gradients
only if this weighting acts as one positive scalar on the relevant residual
and network-Jacobian subspaces.

The external comparison evaluates both values and gradients at seed 0, the
frozen operator fit, and deterministic positive/negative perturbations. It
reports gradient norms, cosine similarity, best scalar proportional fit, and
the remaining nonproportional gradient residual. No distinction is assumed in
advance.

## Exact gradient and HVP

With `r_k=G_k(A_theta,k-A_*,k)`, the exact gradient is

```text
grad_theta J_disc
  = (2/N_A) sum_k (D_theta A_theta,k)^* H_k^* W^* r_k.
```

The mass factor cancels against the symmetric mixed mass inverse in the
dual-native chain; the implementation applies the certified mass norm to the
residual and the existing weak-assembly transpose `W*`. It then invokes JAX's
all-parameter VJP through normalization, the network, original R evaluation,
and structural source construction. No coefficient-vector Euclidean state
metric, dense Jacobian, or intermediate Riesz map is introduced.

An exact optional HVP is exposed for certification. It includes both the
network second derivative acting on the residual covector and the
Gauss-Newton term obtained by propagating the parameter tangent through
`W`, `M^-1`, the mixed metric, and `W*`. Canonical L-BFGS training is
gradient-only and does not request this HVP.

## Implementation and certification

`dimswe.test2a_discrete_offline` layers the objective on the certified
Test-2A-2 provider and J2 Firedrake operators. Analytical targets are
precomputed once from states 0..80. Objective calls evaluate 81 independent
neural predictions and never feed one prediction into another example.

Cheap algebraic tests certify global normalization, fixed-state dataflow,
nonzero shared-R cancellation, arbitrary-pytree gradients, directional finite
differences, exact HVPs, and objective-comparison diagnostics. The tiny
Firedrake tests independently compare VJP gradients with both centered
parameter differences and a forward JVP/mixed-mass chain.

## External certification commands

```bash
cd /Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export TEST2A3A_CACHE_ROOT="$(mktemp -d /tmp/dimswe-test2a3a.XXXXXX)"
export PYOP2_CACHE_DIR="$TEST2A3A_CACHE_ROOT/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$TEST2A3A_CACHE_ROOT/tsfc"
export XDG_CACHE_HOME="$TEST2A3A_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$TEST2A3A_CACHE_ROOT/matplotlib"
export PYTHONPATH="$PWD"
mkdir -p external-results/test2a/deployed-discrete-offline

# A. deployed analytical target, neural prediction, R cancellation, W/M metric
python -m pytest -q tests/test_test2a_discrete_offline_firedrake.py \
  -k primal_target_prediction_and_mass_metric_are_deployed

# B. all-parameter gradient, independent JVP chain, and optional exact HVP
python -m pytest -q tests/test_test2a_discrete_offline_firedrake.py \
  -k 'exact_all_parameter_gradient or exact_deployed_discrete_hvp'

# C. actual 0..80 operator-vs-discrete value/gradient comparison
python -m dimswe.test2a_discrete_offline compare \
  --configuration dimswe/configs/test2a_deployed_discrete_offline.json \
  --truth-run external-results/test1b-production/truth_c0_0.14 \
  --selected-plan dimswe/configs/test1b_selected_plan.json \
  --operator-dataset external-results/test2a/dataset/doublevortex_A_operator.npz \
  --output external-results/test2a/deployed-discrete-offline/objective_comparison.json
```

After the comparison is reviewed and optimization is separately authorized,
the prepared canonical seed-0, memory-20 ROL command is:

```bash
python -m dimswe.test2a_discrete_offline train \
  --configuration dimswe/configs/test2a_deployed_discrete_offline.json \
  --truth-run external-results/test1b-production/truth_c0_0.14 \
  --selected-plan dimswe/configs/test1b_selected_plan.json \
  --operator-dataset external-results/test2a/dataset/doublevortex_A_operator.npz \
  --output external-results/test2a/deployed-discrete-offline/fit_result.json \
  --parameter-output external-results/test2a/deployed-discrete-offline/final_parameters.npz
```

Do not run this training command as part of Test 2A-3A certification. The
initial 500-iteration ceiling is a reviewable first budget, not an inherited
52,000-iteration assumption and not a convergence claim.

## Gate before optimization

The external objective/gradient comparison must establish whether the actual
double-vortex `K_k` weighting produces materially nonproportional parameter
gradients. Only then should the same-seed deployed-discrete fit be authorized.
Truth-reset, rollout, and states 81..160 remain outside this stage.

## Subsequent accepted direct-production fit

After this objective gate was certified, the separately authorized
direct-production memory-20 L-BFGS run reached 50,000 accepted iterations and
`J_disc=0.0017427829635521567` with parameter SHA-256
`4db13a84f52d6fcf66f0345ce5c677aff2b46a84350228a78bc05b073ff6b12a`.
The same certified operator dataset gives
`J_op=0.0020819762080123453` for that artifact.  This historical direct result
is immutable comparison evidence; the exact acceleration and its periodic
CG3 topology repair are documented in `docs/TEST2A_3B_3C.md`.
