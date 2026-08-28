# Test 2A H1-H2-H5 horizon curriculum

This document freezes and records the first production horizon curriculum for
Problem A. It prepares three sequential fits, but no production fit was run
during preparation.

## Scientific definition

Let `P = C5 o ... o C1`, `Y_k = P(X_k*)`, and let `F_theta` be the complete
six-child timestep whose only learned law is neural `A_theta` in child 6. The
original analytical rain law `R` remains active. For each horizon, a window
starts at exact truth `X_k*` and is recursive only inside that window:

```text
Xhat_k = X_k*
Xhat_{k+j} = F_theta(Xhat_{k+j-1}), j=1,...,H.
```

The frozen objective is

```text
N_H(theta) = sum_windows sum_{j=1}^H
             ||Xhat_{k+j}(theta) - X*_{k+j}||_M^2

D = sum_{k=0}^{79} ||dt G(Y_k) A*(Y_k)||_M^2

J_H(theta) = N_H(theta) / D.
```

All weights are one. There is no per-target normalization, hidden factor of
one half, or horizon-dependent denominator. `||.||_M` is the production mixed
mass norm. The generated cache gives

```text
dt                         = 100
sum ||G(Y_k) A*(Y_k)||_M^2 = 409017196.7662303
D                          = 4090171967662.3027
D fingerprint              = 10bda77bf2e003802c560ef1218fe28b17531da6b30e3f97cf22fa04a62d4753
```

For H=1 the original `R` law is evaluated on the common `Y_k` and cancels
between the neural and analytical children. Therefore

```text
J_H1 = sum ||dt G(Y_k)(A_theta(Y_k)-A*(Y_k))||_M^2 / D,
```

which is precisely the fixed post-prefix M2-Y objective. H=1 is not described
as recursive solver-in-loop learning.

## Window schedules

The schedules are non-overlapping and every target boundary 1 through 80 is
used exactly once:

| horizon | starts | windows | targets per window | total targets |
|---:|---|---:|---:|---:|
| 1 | 0,1,...,79 | 80 | 1 | 80 |
| 2 | 0,2,...,78 | 40 | 2 | 80 |
| 5 | 0,5,...,75 | 16 | 5 | 80 |

There is no internal reset in an H=2 or H=5 window. The next window resets to
its requested truth origin. No truth state after 80 is loaded.

## H=1 fixed cache and certificate

The H=1 cache reuses the certified periodic-CG3 sparse/Kronecker mass action
from M2-X but replaces its fixed inputs with the 80 genuine post-prefix
states. Analytical targets are evaluated with `local_physics=None`, so this
cannot accidentally select a frozen neural provider. The cache does not form
dense `G`, `K`, or mass-inverse matrices. Its optimization hot loop has zero
Firedrake/PETSc solves.

Cache artifact:

```text
external-results/test2a/horizon-curriculum-h1-h2-h5/h1_postprefix_cache.npz
SHA256 b30bb0fd2c919734ca0ecba44e32a2c9bd40b491fb7abce0a0aa011a5ea99b89
size   11149060 bytes
```

Cached value and all-1281-parameter gradients were compared with the literal
complete H=1 trajectory at five probes:

| probe | cached J | literal J | relative J error | relative gradient error | cosine |
|---|---:|---:|---:|---:|---:|
| seed 0 | 1.14980528361485 | 1.14980528361506 | 1.81e-13 | 7.24e-12 | 1.0 |
| matched M1 200k | 7.04871420122811e-4 | 7.04871420116691e-4 | 8.68e-12 | 1.43e-9 | 1.0 |
| matched M2-X 200k | 1.59255298976456e-3 | 1.59255298977029e-3 | 3.60e-12 | 1.54e-9 | 1.0 |
| M1 to M2-X 50k | 4.94643999674344e-4 | 4.94643999679749e-4 | 1.09e-11 | 1.47e-9 | 1.0 |
| deterministic perturbation | 1.15125860576392 | 1.15125860576419 | 2.31e-13 | 7.33e-12 | 1.0 |

These are float64-consistent differences between the cached exact algebra and
the literal complete-split oracle.

## Recursive information at H=2 and H=5

At matched M1-200k parameters, 80 independent one-step resets have
`J=0.0007048714201166905` and gradient norm `0.2860286876614512`.

| comparison | recursive J | J ratio | gradient cosine | best scale | nonproportional residual |
|---|---:|---:|---:|---:|---:|
| H=2 vs independent H=1 | 0.001120554401295603 | 1.58973 | 0.964983 | 1.40221 | 0.262313 |
| H=5 vs independent H=1 | 0.0022506331541617204 | 3.19297 | 0.905246 | 2.00531 | 0.424889 |

Thus neither recursive gradient is a rescaled independent-H1 gradient.
Parameter dependence from the first moist update propagates through later dry,
hyperviscosity, DG, neural-A, and analytical-R calculations.

The earlier exact trajectory certificates remain applicable: two-step primal
fieldwise differences were zero; H=2 tangent/adjoint relative discrepancy was
`1.74e-16`; H=1/H=2 directional gradient discrepancies were approximately
`1.59e-7` and `2.33e-7` in scale-aware finite-difference tests; fixed-prefix
value/gradient parity and same-theta tape parity were exact. A changed exact
parameter fingerprint invalidates the tape.

## Timings and memory

The following are serial full-production-window measurements at matched
M1-200k parameters. They are engineering measurements, not scientific fits.

| horizon | windows | targets | steady value (s) | fresh value+gradient (s) | same-tape value+gradient (s) | owned tape |
|---:|---:|---:|---:|---:|---:|---:|
| 1 cached | 80 | 80 | 0.0170 | 0.0352 | n/a | 11.15 MB cache |
| 2 recursive | 40 | 80 | 5.1745 | 35.2928 | 48.1397 | 550.2 MB |
| 5 recursive | 16 | 80 | 4.5323 | 54.7356 | 53.8888 | 547.8 MB |

Same-theta reuse removes redundant forward steps exactly. Wall time fluctuated:
it was slower than a separate fresh H=2 value-plus-gradient measurement in
this single sample and slightly faster for H=5, so no universal wall-time
speedup is claimed. The mathematical forward accounting remains 80 complete
steps per value and 80 reverse steps per gradient for H=2 and H=5.

Two-iteration, deliberately nonscientific L-BFGS smokes produced:

| horizon | initial J | final J | objective evals | gradient evals | wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.04871420122811e-4 | 7.01373434093610e-4 | 9 | 3 | 0.636 |
| 2 | 1.12055440129560e-3 | 1.11521749521421e-3 | 9 | 3 | 130.09 |
| 5 | 2.25063315416172e-3 | 2.24092204298788e-3 | 10 | 3 | 182.23 |

All used new optimizers, exact gradients, zero HVPs, and round-tripped their
parameter artifacts. `MAXITER` is expected at two iterations and is not a
convergence claim.

Linear projections from those two-iteration smokes are deliberately coarse:

| horizon | 100 iterations | 500 iterations | 1000 iterations |
|---:|---:|---:|---:|
| 2 | 1.81 h | 9.03 h | 18.07 h |
| 5 | 2.53 h | 12.66 h | 25.31 h |

The selected initial caps are H1=50,000, H2=100, and H5=100. Their combined
smoke-linear estimate is about 8.76 hours. These are caps; canonical gradient
or step termination may stop earlier. Review after this first campaign should
determine whether larger recursive-stage budgets are warranted.

## Optimizer, checkpoints, and continuation

Every stage uses PyROL/ROL line-search L-BFGS, memory 20, exact float64
gradients, gradient tolerance `1e-8`, step tolerance `1e-12`, and no HVP.
The chain is matched M1-200k to H1, H1 final to H2, and H2 final to H5.
Each stage is a new Python/ROL optimizer process with empty secant history.
The scripts validate the upstream configuration, cache, initialization, and
parameter fingerprints before progressing.

Checkpoint schedules are:

- H1: 0, 100, 500, 1000, 5000, 10000, 25000, 50000.
- H2/H5: 0, 5, 10, 25, 50, 75, 100.

A parameter-only `--resume` starts a new optimizer process from the latest
verified checkpoint. It does not restore process-local L-BFGS secant pairs and
is not an exact optimizer-state resume. Progress JSON is atomic and records
evaluation counts, elapsed time, objective diagnostics, and fingerprints.

After all three stages finish, postprocessing evaluates each stage boundary
under H1/H2/H5, M2-X, and the direct operator objective, computes direct-A
metrics, then runs the existing autonomous 0..80 evaluator. Autonomous
metrics are post-hoc only and cannot stop a fit or control horizon progression.

## Manual launch

The single sequential launch is practical as an unattended overnight run and
is protected by stage-completion and fingerprint checks. Stage-wise launches
remain safer if the user wants to inspect each horizon before spending the
next stage's cost.

Sequential normal-Terminal command:

```bash
cd /Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615
nohup caffeinate -i bash scripts/run_test2a_horizon_curriculum_h1_h2_h5.sh \
  > external-results/test2a/horizon-curriculum-h1-h2-h5/curriculum_master.log 2>&1 &
```

Stage-wise commands:

```bash
nohup caffeinate -i bash scripts/run_test2a_horizon_curriculum_h1.sh \
  > external-results/test2a/horizon-curriculum-h1-h2-h5/h1_master.log 2>&1 &

nohup caffeinate -i bash scripts/run_test2a_horizon_curriculum_h2.sh \
  > external-results/test2a/horizon-curriculum-h1-h2-h5/h2_master.log 2>&1 &

nohup caffeinate -i bash scripts/run_test2a_horizon_curriculum_h5.sh \
  > external-results/test2a/horizon-curriculum-h1-h2-h5/h5_master.log 2>&1 &
```

Use `--resume` only after reviewing an incomplete checkpoint. A completed
stage is reused only after validation; an incompatible or missing upstream
stage aborts the pipeline.

## Correction to the earlier H=1 audit

The stored-target/post-prefix equivalence and the H=2 recursive conclusion in
the earlier audit remain valid. Its reported "analytical JAX" discrepancy was
caused by a helper invocation that selected frozen neural A parameters. The
backend-offset audit established no material analytical UFL/JAX mismatch.
This curriculum explicitly constructs its analytical target with
`local_physics=None`; see
`docs/audits/2026-08-09-test2a-ufl-jax-backend-offset.md`.

