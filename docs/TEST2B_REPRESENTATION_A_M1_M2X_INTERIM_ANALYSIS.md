# Test2B Representation A: interim M1/M2-X analysis

**INTERIM — 2026-08-13.**  This note covers only the completed Representation-A
M1, independent M2-X, and M1-to-M2-X fits.  H1 production was still running
when this analysis was made; H1 parameters were neither read nor evaluated.
H2/H5 training and the common autonomous postprocessing were not run here.
Consequently this document is a fixed-objective and direct-rate analysis, not
a final deployment comparison.

## 1. Read-only evidence and method

The scientific contract and normalizations are those frozen in
`docs/TEST2B_RAIN_ACTIVE_LEARNING_PREPARATION.md`.  The three final parameter
sidecars were loaded through the fingerprint-validating Representation-A
loader.  For every fit, `fit_result.json` and `fit_progress.json` are bytewise
identical, report `status=complete`, and agree with the final parameter
sidecar on stage, accepted iteration, and pytree SHA256.

The cross-objectives use the certified preparation
`external-results/test2b-rain-active-learning/preparation/fixed_learning_data.npz`
(SHA256 `6e159015234fd94881b0b97888b7481eb049a02dcd96c571078920c0bedc901c`).
M1 uses the frozen carrier-mass weighting.  M2-X and H1 use the cached exact
weak assembly and mass inverses.  This analysis performed no Firedrake/PETSc
solve for those objectives.

Direct-A diagnostics use the same carrier-mass weights and frozen
`sigma_A=9.052258655848717e-8`.  The held-out calculation evaluates only the
analytical local A law and the three MLPs at saved boundary states `81..160`;
it takes no model timestep and performs no autonomous rollout.  Held-out
metrics are post hoc and were not used to choose any model or budget.

## 2. Completed artifacts and budgets

| network | stage | accepted | value/gradient evaluations | wall time | termination | final pytree SHA256 |
|---|---|---:|---:|---:|---|---|
| M1 | M1 | 10,000 | 20,983 / 10,001 | 14,062.131 s | `EXITSTATUS_MAXITER` | `471f3ac8a9b84f68bbe14bdc7dee62e3a025ac5cf61503db6644d5a1fa1bb506` |
| M2-X-independent | M2-X | 10,000 | 21,009 / 10,001 | 37,487.576 s | `EXITSTATUS_MAXITER` | `aca1a5810b1f74516bdb26ae902780050929c3e06238795a6301371491dc3652` |
| M1-to-M2-X | M2-X | 5,000 | 10,471 / 5,001 | 24,112.226 s | `EXITSTATUS_MAXITER` | `0cf7dc9036c84c6ac3a78cf7e16cfa56fb7eb985ba7f460e3dccb139bb826d3e` |

The corresponding NPZ file SHA256s are, in table order,
`e4f8d600fdc9833ab9c0c159a0f7601ec7b162f32c0369d44a82988dbd15908f`,
`b644f72dad39f78bbe42174c03ac1b7f16f8789275eae4672a7636e014eecfa4`,
and `a966a41245712e15bb2872b9a96960eb2e101890b70b8e7aca89f5105b2bea44`.
All three fits are budget-limited; none is claimed to have converged.

## 3. Fixed-objective cross-evaluation

Raw values are comparable **down each column**, not across columns, because
M1, M2-X, and H1 have different definitions and normalizations.

| network | J_M1 | J_M2-X | J_H1 |
|---|---:|---:|---:|
| M1 | `2.989733757570708e-5` | `1.835110319053747e-5` | `1.770084405503893e-4` |
| M2-X-independent | `3.034694643382488e-5` | `1.864768212896719e-5` | `2.002361752101823e-4` |
| M1-to-M2-X | `2.570080153639556e-5` | `1.487584885096875e-5` | `1.617805975503903e-4` |

Percentage change relative to the M1 network within each column is:

| network | J_M1 | J_M2-X | J_H1 |
|---|---:|---:|---:|
| M2-X-independent | +1.504% | +1.616% | +13.122% |
| M1-to-M2-X | -14.036% | -18.938% | -8.603% |

Thus an M2-X objective does not automatically move a finite-budget fit toward
H1: the independent basin is worse than M1 on all three fixed metrics.  The
M1-warm-start basin is better on all three.

## 4. Optimizer-basin result

The identical M2-X objective gives

\[
J_{M2-X}(\theta_{ind})=1.8647682128967192\times10^{-5},\qquad
J_{M2-X}(\theta_{warm})=1.4875848850968752\times10^{-5}.
\]

The warm-start value is 20.2268% below the independent value; equivalently,
the independent value is 25.3554% above the warm value.  Despite using only
half as many accepted iterations, the warm solution also has 15.3101% lower
J_M1 and 19.2051% lower J_H1 than the independent solution.  This is strong
evidence of an optimizer-basin effect under the tested budgets, not evidence
against the M2-X objective itself.  Since both runs terminated at MAXITER,
the result compares attained finite-budget points, not mathematical optima.

## 5. Direct A accuracy on training support

`nRMSE` is physical mass-weighted RMS error divided by the frozen training
`sigma_A`; `relRMS` divides by the target RMS within that row.  Bias is signed
and mass weighted.  PRE_RAIN is `0..50`, ONSET is `51..60`, and the training
SUSTAINED interval is `61..80`.

| network/regime | nRMSE | relRMS | max abs error | signed bias | correlation |
|---|---:|---:|---:|---:|---:|
| M1 / overall | `5.467846e-3` | `5.467846e-3` | `4.519818e-8` | `-8.629942e-13` | `0.9999849` |
| independent / overall | `5.508806e-3` | `5.508806e-3` | `4.501415e-8` | `7.295907e-13` | `0.9999846` |
| warm / overall | `5.069596e-3` | `5.069596e-3` | `4.427765e-8` | `-5.239080e-13` | `0.9999870` |
| M1 / PRE_RAIN | `4.398799e-3` | `3.490500e-3` | `4.314605e-8` | `-1.040167e-11` | `0.9999938` |
| independent / PRE_RAIN | `4.524519e-3` | `3.590260e-3` | `4.276813e-8` | `-6.244101e-12` | `0.9999934` |
| warm / PRE_RAIN | `3.900724e-3` | `3.095271e-3` | `4.257885e-8` | `-6.727223e-12` | `0.9999951` |
| M1 / ONSET | `6.597164e-3` | `5.829644e-1` | `4.519818e-8` | `1.381512e-11` | `0.81279` |
| independent / ONSET | `6.531055e-3` | `5.771227e-1` | `4.501415e-8` | `9.812469e-13` | `0.81676` |
| warm / ONSET | `6.313121e-3` | `5.578647e-1` | `4.427765e-8` | `3.702910e-12` | `0.83174` |
| M1 / SUSTAINED | `7.069786e-3` | `5.986070e-1` | `4.189455e-8` | `1.612158e-11` | `0.80134` |
| independent / SUSTAINED | `7.026809e-3` | `5.949681e-1` | `4.189289e-8` | `1.838668e-11` | `0.80410` |
| warm / SUSTAINED | `6.735027e-3` | `5.702626e-1` | `3.985135e-8` | `1.318114e-11` | `0.82323` |

Relative to M1, independent M2-X changes nRMSE by +0.75% overall,
+2.86% PRE_RAIN, -1.00% ONSET, and -0.61% SUSTAINED.  The warm point changes
it by -7.28%, -11.32%, -4.31%, and -4.74%, respectively.  The rain-active
relative errors look large because A itself becomes very small: target RMS is
`1.0244e-9` at onset and `1.0691e-9` in the training sustained interval,
versus `1.1408e-7` PRE_RAIN.

## 6. Held-out mature-rain direct A

All held-out states `81..160` are in the mature SUSTAINED_RAIN_ACTIVE regime.

| network | nRMSE | relRMS | physical RMS | max abs error | signed bias | correlation |
|---|---:|---:|---:|---:|---:|---:|
| M1 | `3.211510e-2` | `2.233068` | `2.907142e-9` | `1.005311e-7` | `2.447134e-11` | `0.27585` |
| M2-X-independent | `4.974688e-2` | `3.459063` | `4.503216e-9` | `7.285380e-8` | `5.350886e-10` | `0.20319` |
| M1-to-M2-X | `2.860098e-2` | `1.988720` | `2.589035e-9` | `8.420779e-8` | `6.694193e-11` | `0.32732` |

The held-out A target RMS is `1.301860e-9`.  Independent M2-X has 54.90%
higher nRMSE than M1 and a much larger positive bias.  Warm M2-X has 10.94%
lower nRMSE than M1 and the best correlation, although M1 retains the smallest
signed bias.  Warm nRMSE is 42.51% below independent.  This is post-hoc
constitutive-law evidence only; it is not held-out solver-trajectory evidence.

## 7. Is A error redistributed toward low-impact regions?

Simple mass-weighted binning does not support a clean tradeoff story for the
warm solution.  The nRMSEs in ascending mass quartiles of `|A*|` are:

| network | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| M1 | `.001754` | `.000671` | `.001272` | `.010698` |
| independent | `.001656` | `.000651` | `.001354` | `.010788` |
| warm | `.001250` | `.000476` | `.000949` | `.010006` |

The quartile boundaries are `0`, `4.72175e-12`, `4.97594e-11`,
`2.62121e-10`, and `8.52844e-7`.  Warm M2-X reduces absolute A error in all
four bins.  Independent M2-X improves the two smallest-|A*| bins but worsens
the upper two, consistent with redistribution inside a poorer basin rather
than a general physical/deployed-map tradeoff.

By analytical saturation branch, nRMSE is:

| network | condensation | evaporation | inactive/balanced |
|---|---:|---:|---:|
| M1 | `.004838` | `.004207` | `.052865` |
| independent | `.004987` | `.004179` | `.053329` |
| warm | `.004323` | `.003868` | `.050582` |

Warm M2-X again improves all three categories.  In depth quartiles it improves
the first three and is about 8.1% worse than M1 only in the highest-h
quartile:

| network | h Q1 | h Q2 | h Q3 | h Q4 |
|---|---:|---:|---:|---:|
| M1 | `.010845` | `.001071` | `.000417` | `.000804` |
| independent | `.010931` | `.000946` | `.000293` | `.000959` |
| warm | `.010044` | `.001020` | `.000368` | `.000869` |

The h boundaries are `626.0193`, `754.4372`, `755.3003`, `755.3386`, and
`761.5511`.  Independent M2-X improves the middle-depth bins while worsening
the lowest and highest.  Finally, per-state direct-A error and exact M2-X energy
contribution remain strongly correlated (`0.936..0.945`) across all three
networks.  These diagnostics show some allocation freedom but no robust
evidence that the successful warm M2-X fit obtains its gain by sacrificing
direct A accuracy in discretely unimportant regions.  A pointwise causal
interpretation would be unjustified because weak assembly and the mass solve
are nonlocal within the finite-element map.

## 8. Interim scientific conclusions

What is already established:

1. Problem-A-style optimizer-basin sensitivity persists in rain-active
   Representation A.  Initialization changes the finite-budget M2-X result
   more than doubling the accepted-iteration budget did.
2. The M1-warm-started M2-X point is Pareto-better than the completed M1 point
   under all three fixed objectives: J_M1, J_M2-X, and J_H1.
3. M2-X warm-starting already moves 8.60% toward the deployment-location H1
   objective before any H1 optimization.
4. Its direct A improvement is broad across PRE_RAIN, ONSET, training
   SUSTAINED, and held-out mature-rain support.  It is therefore not merely a
   training-dry-regime improvement.
5. The independent M2-X artifact must not be used as evidence that M2-X is
   intrinsically worse: it is worse on its own objective than both the warm
   M2-X point and, slightly, M1.

What must wait:

* the final H1 artifact and matched H1/H2/H5 cross-objectives;
* autonomous rain onset, rain mass, state errors, invariants, KE, and
  enstrophy;
* any claim that fixed-objective improvements improve deployment;
* any final choice among the six Representation-A models.

No training, optimization, recursive H2/H5 evaluation, or autonomous rollout
was performed for this note.  The live H1 production process and its files
were untouched.

**STATUS: REPRESENTATION_A_INTERIM_ANALYSIS_COMPLETE**
