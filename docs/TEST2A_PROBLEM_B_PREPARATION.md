# Test 2A Problem B: frozen contract, implementation, and production preparation

Date: 2026-08-10  
Repository HEAD audited: `d2f5d66ecb5500aad24eca37280f8a52e22a250f`  
Branch: `dev/dimswe-learned-physics-framework`

## 1. Purpose and relation to frozen Problem A

Problem A, frozen in `docs/TEST2A_PROBLEM_A_FINAL_SYNTHESIS.md`, learned a
single scalar `A` while imposing the exact four-source manifold. Problem B is
the controlled output-representation ablation. It holds the resolved input
information and production discretization fixed, but learns four independent
local physical source densities

\[
N_\theta(h,S,Q_v,Q_c,B)=(S_t,Q_{v,t},Q_{c,t},Q_{r,t})_\theta.
\]

No analytical `A` or `R`, shared latent scalar, projection, conservation
correction, zero-rain constraint, or Problem-A source identity enters the
learned source calculation. Those identities are post-hoc diagnostics only.
FIML is deferred until the five core methods have been completed.

Only truth states 0 through 80 are available to preparation, fitting, and
model selection. States 81 through 160 remain locked.

## 2. Frozen input/output and architecture

The input is exactly Problem A's ordered feature vector

```text
h, S, Qv, Qc, B
```

with its exact training mean/population-standard-deviation normalization. `Qr`
is deliberately excluded because this no-rain ablation changes only the
output representation and `Qr` has zero variance on the support. This choice
does not apply to a future rain-active Test 2B.

The network is

```text
5 -> 32 -> 32 -> 4, tanh, float64, seed 0
```

and contains exactly

\[
(5\cdot32+32)+(32\cdot32+32)+(32\cdot4+4)=1380
\]

parameters. The actual initialized pytree contains 1,380 float64 values and
has SHA256

```text
e52dd73e3f97d44adf4d55354b1c8d9a9b252186a17cae4ad09410270b86df1e
```

Initialization uses the accepted Problem-A convention: independent
Glorot-uniform weights per layer and zero biases. Parameter artifacts serialize
named weight/bias leaves and verify architecture, shapes, dtype, and pytree
fingerprint on load.

## 3. Four-output conditioning

Let `M_c` be the accepted broken-CG3/GLL carrier mass. Its assembled production
matrix was found to be diagonal in packed production order (maximum absolute
off-diagonal entry exactly zero). Over all 331,776 samples from states 0..80,

\[
\sigma_S=\operatorname{RMS}_{M_c}(S_t^*)
 =4.465574092866371\times10^{-4},
\]

and

\[
\sigma_Q=\sqrt{\tfrac12(\operatorname{RMS}_{M_c}(Q_{v,t}^*)^2+
                    \operatorname{RMS}_{M_c}(Q_{c,t}^*)^2)}
 =4.553845840641363\times10^{-6}.
\]

The output map is

\[
D=\operatorname{diag}(\sigma_S,\sigma_Q,\sigma_Q,\sigma_Q),
\qquad N_\theta=D\,z_\theta.
\]

Thus the exactly zero truth `Qr_t` still has a positive, physically comparable
penalty. Scale-provenance SHA256:

```text
f9abe6ba1da5dc0499722ab4bd003f64fd8e90c4b262580ab7576a2caa91c9a3
```

The complete fixed-data NPZ SHA256 is
`a869d1dd6801fd06d6f4e981f1ad36bab6f285204c259a1bfe9c19b6d4eb839b`.
Targets were evaluated directly from the analytical production JAX source
oracle; scalar `A` labels were not used to construct the learned output.
Post-hoc target checks give maximum residuals: water `0`, `Qr_t` `0`, and
`S_t-beta2*Qv_t = 1.734723475976807e-18`.

## 4. Accepted deployed timestep and learned source location

The complete step is unchanged:

1. dry RK4 half step at `t_n`;
2. dry RK4 half step at `t_n+dt/2`;
3. hyperviscosity Euler full step;
4. DG SSPRK43 half step;
5. DG SSPRK43 half step;
6. moist Euler full step.

Only child 6 calls `N_theta`. Its four outputs are passed directly through the
existing weak source assembly, mixed mass solve, and Euler update. The original
analytical/UFL and Problem-A neural-A modes remain unchanged and are still the
defaults unless explicitly selected.

## 5. Frozen five-objective ladder

Let `M_4` be four copies of the carrier mass, `M` the complete mixed-state
mass, `G_4=M^{-1}W` the exact map from the four local source fields to the
mass-solved moist tendency, and `P=C_5...C_1`.

### 5.1 M1: direct operator/a-priori regression

\[
J_{op}^B(\theta)=
\frac{\sum_{k=0}^{80}\|D^{-1}[N_\theta(X_k^*)-N^*(X_k^*)]\|_{M_4}^2}
     {\sum_{k=0}^{80}\|D^{-1}N^*(X_k^*)\|_{M_4}^2}.
\]

This is direct tendency regression. It contains no `G_4`, mass inverse,
timestep prefix, or model-generated state.

### 5.2 M2-X: boundary-state deployed-map regression

\[
J_{M2-X}^B(\theta)=
\frac{\sum_{k=0}^{80}\|G_4[N_\theta(X_k^*)-N^*(X_k^*)]\|_M^2}
     {\sum_{k=0}^{80}\|G_4N^*(X_k^*)\|_M^2}.
\]

This fixed cache retains the exact topology-aware periodic CG3 tensor mass
inverse and exact local DG inverse used by Problem A. No dense `G_4` is formed,
and the hot loop performs no Firedrake/PETSc solve.

### 5.3 H1/M2-Y: post-prefix one-step objective

For `Y_k=P(X_k^*)`, `k=0,...,79`,

\[
J_{H1}^B(\theta)=
\frac{\sum_k\|dt\,G_4[N_\theta(Y_k)-N^*(Y_k)]\|_M^2}{D_B},
\quad
D_B=\sum_k\|dt\,G_4N^*(Y_k)\|_M^2.
\]

The measured common denominator is

```text
D_B = 4.090171967662303e12
```

(The equality with Problem A's denominator is expected: it is the same exact
truth source vector, now represented directly.) Every `Y_k`, target, weak map,
mass inverse, and denominator is fixed. H1 is therefore fully offline and
cacheable; it is not recursive solver-in-loop learning.

### 5.4 H2 and H5: recursive dense objectives

For horizon `H`, start each nonoverlapping window from exact truth and recurse
with the complete learned-source step:

\[
\widehat X_k=X_k^*,\qquad
\widehat X_{k+j}=F_\theta(\widehat X_{k+j-1}),
\]

\[
J_H^B(\theta)=\frac{\sum_{k\in S_H}\sum_{j=1}^H
\|\widehat X_{k+j}-X_{k+j}^*\|_M^2}{D_B}.
\]

Schedules are identical to Problem A:

| horizon | starts | windows | target multiplicity |
|---:|---|---:|---|
| 1 | `0,1,...,79` | 80 | each `X_1*...X_80*` once |
| 2 | `0,2,...,78` | 40 | each target once |
| 5 | `0,5,...,75` | 16 | each target once |

H2 is the first objective with a model-generated state entering later dry,
hyperviscosity, DG, and neural moist evaluations. H5 extends the same exact
reverse-time graph to five steps.

## 6. Exact derivative and cache certification

Preparation/certification evidence is under
`external-results/test2a/problem-b/preparation/`.

| gate | result |
|---|---:|
| M1 centered directional relative discrepancy | `5.9756809007e-10` |
| M2-X cached/production value relative difference | `8.7428832344e-16` |
| M2-X cached/production gradient relative difference | `3.2157446398e-16` |
| H1 cached/literal value relative difference | `2.9918193799e-13` |
| H1 cached/literal gradient relative difference | `3.7815024926e-13` |
| H1 literal directional relative discrepancy | `4.6028813639e-7` |
| H2 state tangent/adjoint relative discrepancy | `0` |
| H2 parameter directional relative discrepancy | `2.7602399663e-8` |
| H5 state tangent/adjoint relative discrepancy | `1.3521366789e-16` |
| H5 parameter directional relative discrepancy | `9.7330710285e-8` |

The directional differences are finite-difference sanity checks on the full
mixed solver. Exact tangent/adjoint duality is the stronger discrete check.
H2's recursive gradient differs from two independent H1 gradients with
relative difference `2.01024` and cosine `0.98612`; H5 differs from five H1
gradients with relative difference `9.89193` and cosine `0.93888`. Thus the
new recursive sensitivity is numerically material, not merely a scalar
renormalization.

The provider also exposes exact state JVP/VJP, parameter JVP/VJP, and joint
differentiated-VJP actions through the already certified complete-split stack.
No dense state or parameter Jacobian is formed.

## 7. Structural-discovery diagnostics

The common evaluator reports without feeding back into training:

- `Qv_t+Qc_t+Qr_t` RMS;
- `S_t-beta2*Qv_t` RMS;
- spurious `Qr_t` RMS and maximum;
- componentwise physical RMS errors;
- angular agreement with the truth four-vector;
- distance from `span{(beta2,1,-1,0)}` in the same normalized coordinates
  `D^{-1}N`;
- autonomous total-water and `S-beta2 Qv` drift;
- mixed/fieldwise trajectory errors, kinetic energy, projected enstrophy, and
  rain-source activity.

These quantities are diagnostics only. No projection or penalty enforcing
them appears in any core objective.

## 8. Optimizer-basin and continuation design

Every fit uses PyROL/ROL line-search L-BFGS, memory 20, exact gradients,
float64, gradient tolerance `1e-8`, step tolerance `1e-12`, and no production
HVP. Every stage is a new process with empty secant history. Parameter-only
restart does not restore L-BFGS history.

To separate objective effects from optimizer basins, production retains both:

1. independent M2-X from the same seed-0 network as M1; and
2. M1-to-M2-X warm-start as a distinct diagnostic.

The dense curriculum is `M1 -> H1 -> H2 -> H5`. H5 is initialized from H2,
matching the accepted Problem-A protocol.

Proposed caps are deliberately the matched Problem-A caps pending normal-
Terminal timing confirmation:

| stage | accepted-iteration cap |
|---|---:|
| M1 seed 0 | 200,000 |
| M2-X seed 0 | 200,000 |
| M1 -> M2-X diagnostic | 50,000 |
| H1 from M1 | 50,000 |
| H2 from H1 | 100 |
| H5 from H2 | 100 |

MAXITER is not convergence. Each stage records actual termination, counts,
wall time, SHA256, atomic progress, and scheduled parameter checkpoints.

## 9. Timings and nonscientific smokes

Five-repeat steady fixed-cache timings on this machine were:

| objective | value (s) | value+gradient (s) | hot-loop solves |
|---|---:|---:|---:|
| M1 | `0.0059754` | `0.0288640` | 0 |
| M2-X | `0.0259842` | `0.0496339` | 0 |
| H1 | `0.0271168` | `0.0482670` | 0 |

One-accepted-iteration NONSCIENTIFIC smokes decreased all three fixed
objectives, used two gradients, zero HVPs, and terminated only by the smoke
MAXITER cap.

The full H2/H5 derivative certification executed successfully. Later bounded
benchmark/smoke attempts were blocked at PETSc initialization by the local
Codex sandbox's intermittent `getdomainname(): Operation not permitted`
restriction. This is not a code or mathematical failure. For engineering
planning, the closest accepted same-split Problem-A measurements are about
48 s (H2) and 54 s (H5) for same-tape value+gradient; actual 100-iteration
production walls were 4,018 s and 6,271 s because line-search counts differ.
Problem B should therefore budget approximately 1.1--1.8 hours per 100-step
recursive stage, but the exact external Problem-B benchmark command below is
authoritative before launch. A 500/1000 cap would scale to roughly 5--9/11--18
hours and is not recommended initially.

## 10. Postprocessing and provenance

The prepared evaluator computes M1, M2-X, dense H1/H2/H5, four-source physical
and structural diagnostics, and an autonomous 80-step training-support
rollout for every final artifact. Autonomous results are post hoc and cannot
select checkpoints. The runner aborts on missing/incompatible artifacts and
never overwrites an existing production root.

Generated preparation data are not source-controlled. Historical Problem-A
artifacts remain untouched.

## 11. Manual commands

First run the external recursive timing and one-iteration smoke gate from a
normal Terminal:

```bash
cd /path/to/DIMSWE-collaborator
source /path/to/dimswe-firedrake-environment/bin/activate
export JAX_ENABLE_X64=True OMP_NUM_THREADS=1 PYTHONPATH="$PWD"
export TEST2B_CACHE_ROOT="$(mktemp -d /tmp/dimswe-test2b-gate.XXXXXX)"
export PYOP2_CACHE_DIR="$TEST2B_CACHE_ROOT/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$TEST2B_CACHE_ROOT/tsfc"
export XDG_CACHE_HOME="$TEST2B_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$TEST2B_CACHE_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$TEST2B_CACHE_ROOT/pycache"
python -u -m dimswe.test2a_problem_b_campaign benchmark \
  --configuration dimswe/configs/test2a_problem_b.json \
  --preparation external-results/test2a/problem-b/preparation/problem_b_fixed_data.npz \
  --output external-results/test2a/problem-b/preparation/problem_b_recursive_benchmark.json \
  --repeats 1 --stages H2,H5
python -u -m dimswe.test2a_problem_b_campaign smoke \
  --configuration dimswe/configs/test2a_problem_b.json \
  --preparation external-results/test2a/problem-b/preparation/problem_b_fixed_data.npz \
  --output external-results/test2a/problem-b/preparation/problem_b_recursive_smoke.json \
  --iterations 1 --stages H2,H5
```

After that gate confirms the estimates, the exact production launch is:

```bash
cd /path/to/DIMSWE-collaborator
nohup caffeinate -i bash scripts/run_test2a_problem_b_campaign.sh \
  > external-results/test2a/problem-b/problem_b_campaign_master.log 2>&1 &
```

Codex did not execute either production command.

## 12. Known limits and readiness

- Serial execution only; no MPI or thread-safety claim is made.
- The 200k/50k/100 caps are matched scientific caps, not convergence claims.
- Problem B can violate conservation and create rain; that is the intended
  ablation, not an implementation defect.
- States 81..160 remain locked until all core artifacts are frozen.
- FIML is explicitly deferred.
- Recursive wall-time calibration still requires the supplied normal-Terminal
  gate because of the Codex sandbox hostname restriction.

All mathematical, cache, primal/derivative, optimizer, checkpoint, and
postprocessing machinery is prepared. Production is ready subject to the
explicit external timing/smoke gate immediately preceding launch.

**Preparation status: PROBLEM_B_READY_FOR_PRODUCTION.**

## 13. Production/postprocessing note (2026-08-11)

All six production optimizations completed successfully: M1,
M2-X-independent, M1-to-M2-X, H1, H2, and H5.  Their completed parameter
artifacts and fit records were preserved without modification.  The campaign's
automatic final postprocessor initially stopped before writing its comparison
because it constructed an ad hoc diagnostic configuration containing only
`sampling_shape`; the reused `ResolvedDiagnosticEvaluator` also requires
`high_wavenumber_fraction` for its velocity-spectrum diagnostic.

The complete evaluator/configuration contract is:

| Evaluator attribute | Previously provided? | Canonical source/value | Fix |
|---|---:|---|---|
| `sampling_shape` | yes, but hard-coded incorrectly as `(16, 16)` | `ResolvedPilotConfiguration.sampling_shape = (32, 32)` from stored truth-run metadata | copy through an explicit immutable adapter |
| `high_wavenumber_fraction` | no | `ResolvedPilotConfiguration.high_wavenumber_fraction = 2/3` from the same metadata | copy through the same adapter |

Static inspection confirms that these are the only configuration attributes
read by `ResolvedDiagnosticEvaluator`.  Postprocessing now reconstructs both
from the accepted truth-run `ResolvedPilotConfiguration`, records their
provenance in every autonomous result, and validates all six fit/progress
records, stage identities, final paths, and pytree SHA256s before evaluating
any artifact.  No diagnostic value is independently invented or changed for
Problem B.

The standalone postprocessor then completed without retraining.  Its
authoritative output is
`external-results/test2a/problem-b/production/problem_b_comparison.json`
(SHA256
`c9bba696f957b34e40a1f95d29e645463bbdd4456144abd536b2fed4d24561d2`).
It contains all six networks, the five objective values, normalized physical
and structural source diagnostics, complete 80-step autonomous mixed/field,
kinetic-energy, projected-enstrophy, spurious-rain, and accumulated-invariant
histories.  The fix and rerun performed no training or optimization.

**Postprocessing status: PROBLEM_B_POSTPROCESS_COMPLETE.**
