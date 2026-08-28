# Test 2A-3B/3C: autonomous a-priori evaluation and discrete training

## Scientific separation

These gates retain the frozen five-input, 5-32-32-1 tanh architecture,
training-only normalization, original analytical rain law, and truth support
0..80. They do not access states 81..160 and do not implement truth-reset or
rollout training.

Test 2A-3B is evaluation only. Test 2A-3C fits independent fixed truth-state
examples with the deployed-discrete metric. The latter remains offline even
though it uses production weak assembly and the mixed mass inverse: no
predicted state is ever reused as an objective input.

## Accepted Test 2A-3A distinction result

The external production comparison established a strong distinction:

| parameters | J_op | J_disc | gradient cosine | nonproportional residual |
|---|---:|---:|---:|---:|
| seed 0 | 0.9135568693989472 | 1.2027413730332317 | -0.17895866992372664 | 0.9838565924255072 |
| frozen operator fit | 0.00428591283697289 | 0.00794193542678781 | 0.1828428001277335 | 0.9831421618674736 |

Objective magnitude is not a ranking of scientific quality. These values show
that the parameter gradients are not scalar rescalings, so an independent
deployed-discrete fit is justified.

The deployed map is denoted

```text
G_k = M^-1 W H_k,
delta tendency_k = G_k delta A_k.
```

`B` remains reserved for physical topography.

## Test 2A-3B autonomous training-support diagnostic

The selected configuration is
`dimswe/configs/test2a_apriori_autonomous.json`. The default artifact has
pytree SHA-256
`f7d2fd9577ba2a824d3df6c1f8c90a425e6964e1fb5962508315509b247bed56`.
An arbitrary artifact is accepted only when its architecture, leaf shapes,
dtype, and frozen normalization are compatible.

The evaluator converts trusted `X*_0` into the neural case once, constructs
all states

```text
Xhat_(n+1) = F_neural-A,original-R(Xhat_n), n=0,...,79,
```

and only then consults truth targets 1..80. The production child order is
unchanged: two dry RK4 half steps, hyperviscosity Euler, two DG SSPRK43 half
steps, and neural-A/original-R moist Euler. Physical c0 is 0.14.

At every neural state, the same deployed GLL inputs are passed both to the
frozen network and the analytical A formula. Per-time and aggregate relative
RMS error and active sign agreement are recorded. This is an off-truth-manifold
law diagnostic because both A values are evaluated at `Xhat_n`, not at
`X*_n`.

Rain is always the original analytical law. Three notions are separated:

1. exact nonzero (`R != 0`);
2. above a float64-scale tolerance, `64 eps` times the local A/R comparison
   scale;
3. physically meaningful, which additionally requires
   `|dt h R| > 1e-12 RMS(Qr)` at the same deployed GLL state.

No automatic scientific classification threshold is imposed. The JSON lists
the four permitted classifications and requires review after the trajectory
exists.

External command:

```bash
cd /path/to/DIMSWE-collaborator
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export TEST2A3_CACHE_ROOT="$(mktemp -d /tmp/dimswe-test2a3.XXXXXX)"
export PYOP2_CACHE_DIR="$TEST2A3_CACHE_ROOT/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$TEST2A3_CACHE_ROOT/tsfc"
export XDG_CACHE_HOME="$TEST2A3_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$TEST2A3_CACHE_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$TEST2A3_CACHE_ROOT/pycache"
export PYTHONPATH="$PWD"

python -m dimswe.test2a_apriori_autonomous \
  --configuration dimswe/configs/test2a_apriori_autonomous.json \
  --parameter-file external-results/test2a/optimizer-study/continuation-m20-plus45000/continuation_final_parameters.npz \
  --expected-pytree-sha256 f7d2fd9577ba2a824d3df6c1f8c90a425e6964e1fb5962508315509b247bed56 \
  --output-directory external-results/test2a/apriori-autonomous-training-support
```

This is training-support deployment, not held-out validation or ML
generalization.

## Test 2A-3C exact fixed-state cache

Calling Firedrake assembly and a mass solve for 81 states on every ROL trial
would add avoidable implementation overhead. At fixed truth states, the only
parameter-dependent quantity is A. The cached objective retains exactly

```text
e_k       = A_theta - A_star,
d_q,k     = W_Q diag(h_k) e_k,
d_S,k     = W_S diag(beta2 h_k) e_k,

J_disc = sum_k [
    d_S,k^T M_S^-1 d_S,k
  + 2 d_q,k^T M_Q^-1 d_q,k
] / N_A.
```

The factor two is the identical Qv and Qc A-error contribution. Original R is
not set to zero: it cancels exactly because both children evaluate the same
parameter-independent R at the same fixed state.

The cache stores normalized GLL inputs, analytical A targets, h, beta2,
sparse production weak matrices, an exact tensor-factorized CG3 mass inverse,
exact sparse inverses of local DG1 mass components, and the accepted global
normalizer. It does not form dense `G_k` or `K_k`. The implementation certifies
the CG tensor factorization and the local DG identity `M M^-1 - I` against the
assembled production mass blocks.

### Periodic CG3 topology repair

The first external cache attempt failed because it tried to infer a global
Cartesian tensor grid by interpolating `x,y` into the seam-identified periodic
CG3 space.  On the production 16x16 mesh the space has 2304 DOFs, but those
interpolated coordinate arrays contain 60 and 64 distinct floating-point
values: their Cartesian product is 3840, not 2304.  A periodic global DOF does
not own one unambiguous nodal Cartesian coordinate, so this was an indexing
error rather than a failure of tensor-product mass structure.

The repaired path constructs the permutation from the CG and broken-GLL
cell-node maps.  It ranks cells and their four local GLL nodes per axis, wraps
the cell-boundary nodes through the periodic topology, and obtains an exact
48x48 bijection.  In this ordering the production matrix agrees with the
Kronecker reconstruction to `4.50e-16` relative error.  Three deterministic
mass-action probes agree with the assembled matrix within `4.17e-16`; forward
and transpose inverse actions agree with a production PETSc preonly/LU solve
within `1.10e-16`.  No dense global inverse is formed.

Before the cache is accepted, values and all-parameter gradients are compared
with the Test 2A-3A Firedrake oracle at seed 0, the frozen operator fit, the
completed direct-production Method-2 fit, and two deterministic perturbations.
It also reproduces the accepted cross-objective values. Training refuses an
uncertified or fingerprint-incompatible cache.

The local certification run obtained objective relative differences no larger
than `6.56e-16`. Gradient relative errors ranged from `1.36e-15` to
`2.13e-13`, with cosine similarity one to displayed float64 precision. The
largest direct-fit gradient relative error reflects its much smaller gradient
norm; its absolute gradient discrepancy was `9.40e-16`.

The completed direct-production Method-2 artifact has SHA-256
`4db13a84f52d6fcf66f0345ce5c677aff2b46a84350228a78bc05b073ff6b12a` and

```text
J_op(theta_disc)   = 0.0020819762080123453
J_disc(theta_disc) = 0.0017427829635521567.
```

The accepted training-support autonomous evaluation also established that
this Method-2 artifact reduced final/max/accumulated mixed-state error from
`1.45348e-6 / 1.88803e-6 / 1.62801e-6` for the a-priori artifact to
`6.95139e-7 / 1.03913e-6 / 8.85287e-7`. Its off-manifold A relative RMS was
`0.0515077`, versus `0.0664628` for a-priori. These are training-support
deployment diagnostics, not held-out generalization results.

### Cache construction and timing gate

```bash
mkdir -p external-results/test2a/deployed-discrete-offline

python -m dimswe.test2a_discrete_training prepare-cache \
  --configuration dimswe/configs/test2a_deployed_discrete_50k.json \
  --cache external-results/test2a/deployed-discrete-offline/fixed_operator_cache.npz

python -m dimswe.test2a_discrete_training benchmark \
  --configuration dimswe/configs/test2a_deployed_discrete_50k.json \
  --cache external-results/test2a/deployed-discrete-offline/fixed_operator_cache.npz \
  --output external-results/test2a/deployed-discrete-offline/performance_estimate.json \
  --repeats 7 \
  --production-repeats 3
```

Review `performance_estimate.json` before launching 50k. It reports first-JIT
and steady value/value-gradient timings plus direct-production timings.  The
local result was:

| quantity | wall seconds |
|---|---:|
| one-time cache construction and five-probe certification | 52.805 |
| first JIT value | 0.1577 |
| first JIT value+gradient | 0.2407 |
| steady cached value | 0.01729 |
| steady cached value+gradient | 0.03572 |
| production value | 0.56045 |
| production gradient after cached value | 0.23570 |
| production fresh value+gradient | 0.71827 |

This is a 32.4x objective speedup and 20.1x fresh value-gradient speedup. Using
the completed run's actual 104234 objective and 50001 gradient evaluations,
the base estimate is 3588 seconds (59.8 minutes), excluding checkpoint
diagnostics, or 10.46x below the accepted direct run's 37523-second wall time.
The cached arrays occupy 20,001,792 bytes uncompressed and
11,246,260 bytes in the NPZ. The steady optimization loop performs zero
Firedrake/PETSc solves.

A noncanonical 20-accepted-iteration smoke run from the exact canonical
seed-0 pytree reduced `J_disc` from `1.202741373033232` to
`0.9849326307534226` in 1.61 seconds, using 43 objective evaluations, 21
gradient evaluations, and zero HVPs. It terminated by the deliberately small
smoke `MAXITER`; it is not a replacement Method-2 fit or a convergence claim.

### Checkpointed 50k run

The fit starts from seed-0 pytree SHA-256
`6d4ac7bafe775b90a70e2a199ef3305308c3d02333f7daeac1c870b6f18e0975`.
It uses ROL line-search L-BFGS, memory 20, exact gradients, and zero production
HVPs. Checkpoints are written at 1000, 2500, 5000, and every selected later
milestone through 50000. Each contains J_disc, J_op, gradient norm, relative
parameter step, physical A metrics, and parameter fingerprints.

```bash
mkdir -p external-results/test2a/deployed-discrete-offline/fit-50k

nohup python -u -m dimswe.test2a_discrete_training train \
  --configuration dimswe/configs/test2a_deployed_discrete_50k.json \
  --cache external-results/test2a/deployed-discrete-offline/fixed_operator_cache.npz \
  --output-directory external-results/test2a/deployed-discrete-offline/fit-50k \
  > external-results/test2a/deployed-discrete-offline/fit-50k/train.log 2>&1 &
echo $!
```

After interruption, resume from the most recent verified parameter checkpoint:

```bash
nohup python -u -m dimswe.test2a_discrete_training train \
  --configuration dimswe/configs/test2a_deployed_discrete_50k.json \
  --cache external-results/test2a/deployed-discrete-offline/fixed_operator_cache.npz \
  --output-directory external-results/test2a/deployed-discrete-offline/fit-50k \
  --resume \
  >> external-results/test2a/deployed-discrete-offline/fit-50k/train.log 2>&1 &
echo $!
```

Resume restores parameters exactly, but not ROL's process-local secant history.
This is experiment restartability, not adjoint checkpointing.

## After training

The direct-production fit and its training-support autonomous diagnostic are
accepted evidence.  The repaired accelerator has not been used for another
long fit in this change.  A future accelerated run must start again from the
canonical seed-0 artifact; it is not a continuation of the direct fit.
