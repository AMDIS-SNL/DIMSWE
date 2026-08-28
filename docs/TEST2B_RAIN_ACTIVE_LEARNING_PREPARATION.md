# Test2B rain-active learning: frozen truth and production preparation

## 1. Scope and status

This document freezes the completed 64-by-64 rain-active double-vortex truth
and the scientific/implementation contract for the later matched learning
study.  No truth was generated and no ML objective was optimized during this
preparation.  The accepted production timestep remains the exact six-child
split (dry RK4 half, dry RK4 half, hyperviscosity Euler, DG SSPRK43 half, DG
SSPRK43 half, moist Euler).

Authoritative read-only evidence is under
`external-results/test2b-rain-active-learning/preparation/`:

| artifact | SHA256 | role |
|---|---|---|
| `truth_manifest.json` | `746ae7020093261a4c5292fb37f61c8b12b0c825d5e5fa76a560fc830f37fe40` | complete file-level truth inventory |
| `learning_support_audit.json` | `e83edbfe67e90e04afd78b40249c04344948849a141fbb558c5eb3230b9d383d` | boundary/post-prefix analytical support |
| `fixed_learning_data.npz` | `6e159015234fd94881b0b97888b7481eb049a02dcd96c571078920c0bedc901c` | exact M1/M2-X/H1 fixed data and operators |
| `derivative_certification.json` | `77ff222912071eebc43d0b662f14d1f205ba9ca547c579714355ed5b4d45a1b6` | directional and recursive derivative gates |
| `oracle_certification.json` | `c5d419197f1874cd3483fe2785321cfc0de32685b8d892e7475110a158cddef8` | literal-child cache parity and H1 duality |
| `runtime_benchmark.json` | `bdba393a192c07f00394a63248fe09cd32145619d9efa52577bc78e9f8629c9e` | representative full 64-by-64 costs |

The manifest payload fingerprint is
`b13e75d7cebddbbf23ab222eba7fe67f281443d5044ca26db66186d113bc8e7c`.
It inventories and hashes all 161 restart arrays, 161 Firedrake checkpoints,
161 diagnostic records, and 161 spectra without altering the source run.

## 2. Frozen truth identity

The source is
`external-results/test2b-rain-active-truth/production-n64-zeta-m0p06-dt100-t16000/`.
Its configuration SHA256 is
`a917a1cfb3f39abab39ab6426ab50005aae9ef4102f55b3a0f708e1e85cd3aa2`.

| item | frozen value |
|---|---:|
| periodic quadrilateral mesh | `64 x 64` cells |
| spatial order | production order 3; CG3/DG1 mixed state |
| deployed GLL samples per state | `65,536` |
| timestep / output cadence | `100 / 100` |
| saved states | `0..160` (`t=0..16000`) |
| initial moisture control | `zeta=-0.06` |
| initial `Qc`, `Qr` | exactly zero |
| `qprecip`, `gamma_r` | `1e-4`, `0.001` |
| `g`, `L`, `beta2` | `9.80616`, `10`, `98.0616` |

The exact analytical laws remain

\[
A^*=E-C,\qquad
R^*=\max\{0,\gamma_r(q_c-q_{precip})/\Delta t\},
\]

with the accepted condensation, capped evaporation, saturation, and source
map in `dimswe/jax_moist.py`.  The truth's maximum relative total-water drift
is `1.7742780285355334e-14`.  Maximum local source residuals are
`1.108422020821057e-21` for water and `1.3877787807814457e-17` for
`S-beta2 Qv`.

## 3. Approved rain regimes

The approved criterion is applied unchanged: every saved state in a
continuous interval must have meaningful local `R`, positive integrated rain
production, mean active GLL fraction at least `1e-4`, and positive rain-mass
gain above `128 eps` times total water; the duration must reach 1,000.

Meaningful rain begins at step 51.  The qualifying interval starts there and
is certified after 1,000 units at step 61.  For unambiguous regime labels:

| regime/event | steps | times |
|---|---:|---:|
| PRE_RAIN | `0..50` | `0..5000` |
| first exact `R>0` | `51` | `5100` |
| first meaningful `R>0` | `51` | `5100` |
| ONSET (persistence not yet certified) | `51..60` | `5100..6000` |
| first SUSTAINED_RAIN_ACTIVE certification | `61` | `6100` |
| SUSTAINED_RAIN_ACTIVE | `61..160` | `6100..16000` |

The 11-state qualifying mean active fraction is
`0.006411465731534091`; rain mass grows `538706.5819210776` against a
floating-point floor of `1.0512302194880667`.  Meaningful production remains
positive through the endpoint, so the completed trajectory passes the
approved classification.

## 4. Learning support

Analytical rates were evaluated both at each saved boundary state `X_k` and
at the actual post-prefix state `Y_k=P(X_k)`.  These are independent read-only
replays, not a new truth integration.

| location/regime | A min | A max | A RMS | R max | R RMS | active R fraction |
|---|---:|---:|---:|---:|---:|---:|
| boundary PRE | `-8.52844e-7` | `1.71614e-8` | `1.14100e-7` | 0 | 0 | 0 |
| boundary ONSET | `-4.31653e-8` | `2.01747e-8` | `1.75990e-9` | `2.67164e-11` | `6.07847e-13` | `0.0055969` |
| boundary SUSTAINED | `-3.81083e-8` | `3.70716e-8` | `2.12871e-9` | `5.18254e-11` | `5.79350e-12` | `0.0645374` |
| post-prefix PRE | `-8.52867e-7` | `3.24439e-8` | `1.14131e-7` | 0 | 0 | 0 |
| post-prefix ONSET | `-4.29457e-8` | `3.17988e-8` | `3.23267e-9` | `2.66243e-11` | `6.05432e-13` | `0.0055664` |
| post-prefix SUSTAINED | `-3.80249e-8` | `4.77115e-8` | `3.58868e-9` | `5.17910e-11` | `5.79164e-12` | `0.0645355` |

Active sustained boundary `R` has median `1.8770e-11`, 10th/90th
percentiles `4.1179e-12 / 3.5766e-11`, and range approximately
`8.66e-17..5.18e-11`.  On training boundary states `0..80`, 40,800 of
5,308,416 GLL labels are active (`0.0076859086`).  This is sparse but
sufficient for a first controlled learned-`R` study: it supplies tens of
thousands of active samples, 10 onset states, and 20 sustained states without
synthetic augmentation.  An unscaled all-sample MSE would nevertheless make
the dry majority dominant, hence the active-rate scale below.

## 5. Common input decision

All representations use exactly

\[
x=(h,S,Q_v,Q_c,B).
\]

`Qr` is dynamically nonzero, but neither accepted local constitutive law reads
it: `A` depends on `(h,S,Qv,Qc,B)` and `R` on `(h,Qc)`.  Thus these five fields
are sufficient for the exact Markovian local source law.  Adding `Qr` would
provide a history/partition correlate absent from the truth constitutive law
and would confound the intended output-representation comparison.  `Qr`
remains a prognostic state and a mandatory rollout/partition diagnostic.

Training-only mass-weighted feature offset/scale is

| feature | offset | scale |
|---|---:|---:|
| h | `749.6487720807651` | `16.913638066122523` |
| S | `7376.434989735685` | `133.5602198531373` |
| Qv | `1.4193153609575624` | `0.21326095651874272` |
| Qc | `0.06015957787413514` | `0.012412402653357142` |
| B | `0` | `1` (constant/degenerate convention) |

## 6. Representations and architecture

All use tanh, float64, seed 0, and `5 -> 32 -> 32 -> d`.

| ID | outputs | physical source map | parameters | seed-0 pytree SHA256 |
|---|---|---|---:|---|
| A | `A_theta` | `h A_theta (beta2,1,-1,0) + h R* (0,0,-1,1)` | 1281 | `6d4ac7bafe775b90a70e2a199ef3305308c3d02333f7daeac1c870b6f18e0975` |
| B | `(A_theta,R_theta)` | `h(beta2 A,A,-A-R,R)` | 1314 | `cfadd9f3ee02a78c5b3a946b88c039d9f7ed34e719325ff22c92e1fe4afac056` |
| C | `(S_t,Qv_t,Qc_t,Qr_t)` | direct independent outputs | 1380 | `e52dd73e3f97d44adf4d55354b1c8d9a9b252186a17cae4ad09410270b86df1e` |

Representation A retains the analytical `R`; B learns both physical rates and
conserves water and `S-beta2 Qv` algebraically; C has no analytical correction,
projection, nonnegativity enforcement, or conservation repair.

## 7. Output scaling and M1

All zeros remain in every loss.  Scales condition coordinates but neither
discard dry samples nor impose structure:

* `sigma_A=9.052258655848717e-8`, all-support mass-weighted RMS;
* `sigma_R=1.9902871261559996e-11`, mass-weighted RMS conditional on active
  training `R>0`;
* four-source scales `(S,Qv,Qc,Qr)` are
  `(0.006671477765500949, 6.803353979030477e-5,
  6.80335397581467e-5, 1.5076498196845062e-8)`, using all-support RMS for
  the first three and active-only RMS for `Qr`.

For output vector `z_r` and its diagonal scale `D_r`, M1 is

\[
J_{op,r}=\frac{\sum_{k=0}^{80}\|D_r^{-1}
(z_{r,\theta}(X_k)-z_r^*(X_k))\|_{M_r}^2}
{\sum_{k=0}^{80}\|D_r^{-1}z_r^*(X_k)\|_{M_r}^2}.
\]

For A, `z=A`; for B, `z=(A,R)`; for C, `z=N`.  Denominators are respectively
`2.025e15`, `2.0401038275824642e15`, and `6.090103827582461e15`.
Cross-representation M1 scalar values are not directly comparable because
their output spaces and conditioning differ.

## 8. Deployed objective ladder

Let `G4(Z)=M^{-1}W` be the exact four-source-to-state tendency map at fixed
state `Z` and let `F_theta` be the accepted complete timestep.

\[
J_{M2-X,r}=\frac{\sum_{k=0}^{80}\|G_4(X_k)
(N_{r,\theta}(X_k)-N^*(X_k))\|_M^2}{D_X},
\quad D_X=9.014768540958347\times10^{10}.
\]

For `Y_k=P(X_k)`, H1 is

\[
J_{H1,r}=\frac{\sum_{k=0}^{79}\|F_\theta(X_k)-X_{k+1}\|_M^2}{D_Y},
\quad D_Y=9.01944200525722\times10^{14}.
\]

It is exactly cacheable because `P` is parameter-independent and all learned
outputs enter only child 6.  The cached form applies `G4(Y_k)` and cancels the
common `dt^2` without approximation.  `D_X` fingerprint is
`fb6d1628b18a641f1077f40e6dfcb1464d7401590ab5b8660e148cebd853b664`;
`D_Y` fingerprint is
`30d5e71613485f7bd08ca8d996fcf241771dfb5f7f6aee28cd495bdbbb173912`.

For `H=2,5`, nonoverlapping windows start at `0,H,...,80-H` and

\[
J_{H,r}=D_Y^{-1}\sum_{k}\sum_{j=1}^H
\|\widehat X_{k+j}-X_{k+j}\|_M^2,
\qquad \widehat X_{n+1}=F_\theta(\widehat X_n).
\]

H2 is the first genuinely recursive objective.  H5 extends the same exact
state/parameter feedback; neither is an independent fixed regression.

## 9. Regime coverage and held-out plan

The canonical schedule is unchanged because target multiplicity is already
matched and training contains useful active support:

| objective | starts | windows | PRE targets/windows | ONSET | SUSTAINED |
|---|---|---:|---:|---:|---:|
| H1 | `0..79` | 80 | 50 | 10 | 20 |
| H2 | `0,2,...,78` | 40 | 25 | 5 | 10 |
| H5 | `0,5,...,75` | 16 | 10 | 2 | 4 |

Every target boundary `1..80` occurs exactly once.  No regime reweighting is
part of the canonical objective.  A separately labelled regime-balanced
diagnostic may later report conditional metrics; it must not replace or tune
the primary objective silently.

The split is frozen before training: `0..80` for fitting/model-selection and
`81..160` held out.  This matches Test2A, preserves onset plus 20 sustained
targets in training, and reserves an 80-state contiguous mature-rain test.
The tradeoff is deliberate: held-out evaluation measures temporal/mature-rain
extrapolation rather than an i.i.d. random split.  Autonomous diagnostics are
post hoc and never stopping/selection criteria.

## 10. Exact derivative and cache certification

The 64-by-64 periodic CG3 mass factorization reconstructs the mass action to
`4.07e-16` relative and its inverse to `1.05e-16` relative versus PETSc/LU.
No dense global inverse is formed.  Fixed M1/M2-X/H1 hot loops execute zero
Firedrake/PETSc solves.

Across A/B/C:

* M2-X cache versus literal deployed moist child: value relative difference
  `2.16e-16..3.27e-15`; all-parameter gradient relative error
  `4.81e-16..1.55e-15`;
* H1 cache versus literal complete trajectory: absolute value difference
  `4.25e-14..4.73e-14`; all-parameter gradient relative error
  `2.01e-14..2.42e-14`;
* state tangent/adjoint duality: H1 `0..2.71e-16`, H2
  `1.35e-16..4.06e-16`, H5 `1.34e-16..1.35e-16` relative;
* directional gradient sanity: M1 `9.15e-10..5.85e-9`, M2-X
  `1.08e-9..7.92e-9`, H1 `1.38e-10..6.03e-10`, H2
  `1.19e-11..1.10e-9`, H5 `8.09e-13..7.75e-10` relative.

For arbitrary network output, B's measured water residual is
`2.51e-21` and thermodynamic residual exactly zero.  C deliberately produces
nonzero residuals (`3.62e-5` water and `7.25e-3` thermodynamic in the probe),
proving that no projection leaked into its source adapter.

## 11. Structural diagnostics

The common postprocessor records physical `A/R` errors when those rates exist,
component source errors, water and `S-beta2 Qv` defects, negative `Qr_t`,
rain/cloud/vapor partition, state errors, KE, projected enstrophy, and
accumulated water/thermodynamic drift.  For C, distance to physical structure
uses the normalized source metric and the two-dimensional basis

\[
v_A=(\beta_2,1,-1,0)^T,\qquad v_R=(0,0,-1,1)^T.
\]

Projection is diagnostic only.  It never feeds the model or loss.  For A the
two structural identities are exact by construction with analytical R; B
also preserves both by construction; C must discover them.

## 12. Measured cost and production budget

Representative B timing (the output dimension changes little of the dominant
Firedrake work) is:

| objective | steady value | steady value+gradient | setup/notes |
|---|---:|---:|---|
| M1 | `0.196 s` | `1.837 s` | pure JAX, 5.31M samples |
| M2-X | `0.620 s` | `1.484 s` | exact cache, zero solves |
| H1 | `0.554 s` | `0.923 s` | exact cache, zero solves |
| H2 | `30.976 s` | `192.405 s` | 40 windows, 80 steps; setup `83.16 s` |
| H5 | `30.384 s` | `119.192 s` | 16 windows, 80 steps; setup `13.22 s` |

The fixed cache is 335 MiB compressed and approximately 586 MiB as arrays.
Recursive tape memory is process-local and substantially larger; only one
representation/stage should run at a time.

Recommended caps are M1 10,000; independent M2-X 10,000; M1-to-M2-X 5,000;
H1-from-M1 5,000; H2 20; H5 20.  These are caps, not forced iteration counts.
Every stage uses a fresh PyROL line-search L-BFGS process, memory 20, exact
gradient, tolerances `1e-8/1e-12`, and empty secant history.  Both independent
M2-X and M1-warm-start M2-X are retained to expose basin effects.

With one gradient and roughly two value calls per accepted iteration, rough
per-representation costs are 6--7 h M1, 7--8 h independent M2-X, 3--4 h
warm M2-X, 2.5--3 h H1, 1.2--1.5 h H2, and 0.8--1 h H5.  Line-search behavior
can change these estimates.  The three representations therefore require
roughly 60--75 serial hours and should be launched stage-wise or at least one
representation at a time, not as one blind monolithic job.

## 13. Checkpoint, evaluation, and launch contract

Checkpoints include iteration 0, 1, 5, 10, 20, 100, 500, 1,000, 5,000,
10,000 and the stage cap where applicable; progress is refreshed every 100
accepted iterations, with parameter pytree SHA256 and atomic JSON.  Passing a
parameter artifact to a later stage starts a new optimizer; process-local
L-BFGS secant history is not restored or transferred.  The postprocessor
evaluates all five objectives plus structural and 80-step autonomous
diagnostics without using them for selection.

Normal-Terminal commands (do **not** run them inside Codex):

```bash
cd /Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615
mkdir -p external-results/test2b-rain-active-learning/logs

nohup caffeinate -i bash scripts/run_test2b_rain_learning_representation.sh A \
  > external-results/test2b-rain-active-learning/logs/representation-A.log 2>&1 &

# Launch B only after A is complete and audited:
nohup caffeinate -i bash scripts/run_test2b_rain_learning_representation.sh B \
  > external-results/test2b-rain-active-learning/logs/representation-B.log 2>&1 &

# Launch C only after B is complete and audited:
nohup caffeinate -i bash scripts/run_test2b_rain_learning_representation.sh C \
  > external-results/test2b-rain-active-learning/logs/representation-C.log 2>&1 &
```

`scripts/run_test2b_rain_learning_campaign.sh` is also a guarded sequential
wrapper and ends with common postprocessing, but stage-wise launches are safer
given the measured multi-day total.  The wrapper must not be invoked until the
user deliberately chooses a fresh production root.

## 14. Readiness

The truth, regimes, support, inputs, normalizations, representations, exact
fixed caches, recursive derivatives, schedules, held-out split, diagnostics,
budgets, and guarded launch paths are frozen and certified.  No mathematical
or implementation blocker remains before manual production.  No production
ML fit was launched during preparation.

**STATUS: TEST2B_RAIN_ACTIVE_LEARNING_READY**
