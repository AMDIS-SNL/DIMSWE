# Test2B Representation A: final scientific synthesis

**Frozen evaluation date:** 2026-08-13

**Status:** final evaluation of all six completed Representation-A fits

**Scope:** learned scalar `A_theta`; analytical, state-dependent `R`; no new
training or truth generation

This document supersedes the interim report retained at
`archive/development-history/test2b_representation_a_interim/README.md`. The interim
fixed-objective, optimizer-basin, and direct-`A` findings remain valid; the
present document adds the completed H1/H2/H5 artifacts, the exact 6-by-5
objective matrix, and identical 160-step autonomous evaluations.

The authoritative machine-readable evaluation is
`external-results/test2b-rain-active-learning/production/representation-A/representation_a_final_comparison.json`
(SHA256
`75506a833862b69437a29a7a2b30e64c361d35b8560342bf8abe93302daf5b7f`).
It was produced from immutable final parameters by the evaluation-only module
`dimswe.test2b_representation_a_postprocess`; the result records
`optimizer_instantiated=false` and `truth_generated=false`.

## 1. Frozen Representation-A contract

The truth, input, normalization, objectives, schedules, derivatives, and
optimizer conventions are those frozen in
`docs/TEST2B_RAIN_ACTIVE_LEARNING_PREPARATION.md`.  The common local input is

\[
x=(h,S,Q_v,Q_c,B),
\]

and the float64, seed-0, `5 -> 32 -> 32 -> 1` tanh network has 1,281
parameters.  `Qr` remains prognostic and diagnostic but is not an input: the
accepted local analytical `A` and `R` laws do not read it.  The network
predicts only `A_theta`.  At every deployed moist call the
analytical rain rate is recomputed from the **current model state**, and the
source is

\[
(S_t,Q_{v,t},Q_{c,t},Q_{r,t})
=h\left(\beta_2 A_\theta,A_\theta,-A_\theta-R^*,R^*\right).
\]

No stored truth rain rate is injected into an autonomous trajectory.  The
representation therefore conserves total water and preserves
`S_t-beta2 Qv_t` algebraically for every neural output.

Training and model-selection support is truth states `0..80`.  States
`81..160` form a frozen contiguous held-out mature-rain interval.  The truth
has first exact and physically meaningful rain at step 51 (`t=5100`), peak
specific cloud water `1.0517909572531444e-4` at step 89, maximum analytical
`R=5.179095725314434e-11`, and integrated/final rain mass
`2.30733006980403e8`.  These numbers come from the frozen rain audit with
SHA256 `0302d1cb3808e9543986665eaa05aa3bcd49b1ab70326c2cca9a4d5dc1861b5d`.

The fixed preparation cache is
`external-results/test2b-rain-active-learning/preparation/fixed_learning_data.npz`
(SHA256
`6e159015234fd94881b0b97888b7481eb049a02dcd96c571078920c0bedc901c`).
It supplies the frozen `sigma_A=9.052258655848717e-8`, mass weights, exact
M2-X/H1 maps, denominators, and schedules.

### 1.1 Objective definitions

For direct physical-rate supervision,

\[
J_{M1}(\theta)=
\frac{\sum_{k=0}^{80}\|[A_\theta(X_k^*)-A^*(X_k^*)]/\sigma_A\|_{M_A}^2}
{\sum_{k=0}^{80}\|A^*(X_k^*)/\sigma_A\|_{M_A}^2}.
\]

For the deployed fixed-state boundary objective,

\[
J_{M2-X}(\theta)=
\frac{\sum_{k=0}^{80}\|G_4(X_k^*)
[N_{A,\theta}(X_k^*)-N^*(X_k^*)]\|_M^2}{D_X},
\quad D_X=9.014768540958347\times10^{10}.
\]

For `Y_k=P(X_k^*)`, where `P` is children 1--5 of the accepted six-child
step,

\[
J_{H1}(\theta)=
\frac{\sum_{k=0}^{79}\|F_\theta(X_k^*)-X_{k+1}^*\|_M^2}{D_Y},
\quad D_Y=9.01944200525722\times10^{14}.
\]

H1 is exact, truth-reset, and cacheable: the prefix is fixed and `A_theta`
enters only child 6.  For `H=2,5`, nonoverlapping windows start at
`0,H,...,80-H` and

\[
J_H(\theta)=D_Y^{-1}\sum_k\sum_{j=1}^{H}
\|\widehat X_{k+j}-X_{k+j}^*\|_M^2,
\qquad \widehat X_{n+1}=F_\theta(\widehat X_n).
\]

Every target `1..80` occurs once in each dense horizon objective.  H2 is the
first objective containing model-generated-state feedback.  Objective values
below are comparable **down a column**, not between columns, because their
definitions and normalizations differ.

## 2. Verified final artifacts and optimization budgets

For all six stages, `fit_result.json` and `fit_progress.json` are bytewise
identical, report `status=complete`, and agree with the final sidecar on stage,
iteration, and pytree SHA256.  Every `final_parameters.npz` passed the
fingerprint-validating loader.

| model | stage/start | accepted | value / gradient evals | wall (s) | termination | final pytree SHA256 |
|---|---|---:|---:|---:|---|---|
| M1 | seed 0 | 10,000 | 20,983 / 10,001 | 14,062.131 | `MAXITER` | `471f3ac8a9b84f68bbe14bdc7dee62e3a025ac5cf61503db6644d5a1fa1bb506` |
| M2-X-independent | seed 0 | 10,000 | 21,009 / 10,001 | 37,487.576 | `MAXITER` | `aca1a5810b1f74516bdb26ae902780050929c3e06238795a6301371491dc3652` |
| M1-to-M2-X | M1 final | 5,000 | 10,471 / 5,001 | 24,112.226 | `MAXITER` | `0cf7dc9036c84c6ac3a78cf7e16cfa56fb7eb985ba7f460e3dccb139bb826d3e` |
| H1 | M1 final | 5,000 | 10,446 / 5,001 | 21,126.160 | `MAXITER` | `73f39ca1e345256157890f7c918ac41a43684262e4ad11c9465bd35706b16f83` |
| H2 | H1 final | 20 | 44 / 21 | 2,867.411 | `MAXITER` | `3e9b414ee32f1d702a7998d6a3508c3bbf7097a2c735dbafc3550a4ecf74c5df` |
| H5 | H2 final | 20 | 44 / 21 | 3,632.405 | `MAXITER` | `f2d56a8947f66761371fbef8aa52b3aa545f3e2ca9822fdb27a7e3ef796d20a9` |

The measured total is 30,040 accepted iterations, 62,997 objective
evaluations, 30,046 gradient evaluations, and `103287.90869008424 s`
(`28.6911 h`).  Every fit is budget-limited.  None is a demonstrated
stationary optimum, and the 5,000-versus-20 iteration asymmetry is essential
when interpreting H1/H2/H5.

## 3. Exact common objective matrix

| network | `J_M1` | `J_M2-X` | `J_H1` | `J_H2` | `J_H5` |
|---|---:|---:|---:|---:|---:|
| M1 | `2.9897337575707075e-5` | `1.8351103190537455e-5` | `1.7700844055039091e-4` | `3.6126191349134846e-4` | `1.0016940305325648e-3` |
| M2-X-independent | `3.0346946433824920e-5` | `1.8647682128967239e-5` | `2.0023617521018256e-4` | `4.2385339685735820e-4` | `1.2655023430765348e-3` |
| M1-to-M2-X | **`2.5700801536395519e-5`** | **`1.4875848850968718e-5`** | `1.6178059755039169e-4` | `3.3192327822344111e-4` | `9.4031957029427730e-4` |
| H1 | `1.5503789167577002e-4` | `1.1867286238016006e-4` | `3.0018183435018870e-5` | `5.5223413642435120e-5` | `1.1056536728691643e-4` |
| H2 | `1.5543799479717557e-4` | `1.1898020748438811e-4` | **`3.0002290459653828e-5`** | **`5.5187182806433557e-5`** | `1.1044682789864873e-4` |
| H5 | `1.5728459431822668e-4` | `1.2042262782866915e-4` | `3.0090679978333905e-5` | `5.5284127940956001e-5` | **`1.1019889746677016e-4`** |

The bold value is the best attained model in that fixed objective column.
Warm M2-X is best under both physical M1 and M2-X.  H2 is marginally best
under H1 and H2, while H5 is best only under H5.

Within-column changes are:

| transition | `J_M1` | `J_M2-X` | `J_H1` | `J_H2` | `J_H5` |
|---|---:|---:|---:|---:|---:|
| M1 -> warm M2-X | -14.036% | -18.938% | -8.603% | -8.121% | -6.127% |
| warm M2-X -> H1 | +503.241% | +697.755% | -81.445% | -83.363% | -88.242% |
| M1 -> H1 | +418.568% | +546.680% | -83.041% | -84.714% | -88.962% |
| H1 -> H2 | +0.258% | +0.259% | -0.053% | -0.066% | -0.107% |
| H2 -> H5 | +1.188% | +1.212% | +0.295% | +0.176% | -0.224% |

The warm-M2-X-to-H1 row is a cross-artifact comparison, not a literal
continuation: H1 started from M1, whereas warm M2-X is a separate branch.

### 3.1 M2-X basin result

The independently initialized 10,000-step M2-X fit finishes 25.36% above the
5,000-step warm fit under the same objective.  It is also worse than M1 under
M2-X by 1.62%.  The warm fit is better than both under M1, M2-X, H1, H2, and
H5.  Thus the independent result is an optimizer-basin/budget result; it
cannot be attributed to the M2-X objective.  Warm-starting retained and in
fact improved direct physical `A` accuracy.

## 4. Direct physical A accuracy

`nRMSE` divides physical mass-weighted RMS error by frozen training
`sigma_A`; `relRMS` divides by the target RMS within the reported regime.
Bias is signed and mass weighted.  The exact fixed truth-state results are:

| model / regime | nRMSE | rel RMS | max abs | signed bias | corr. |
|---|---:|---:|---:|---:|---:|
| M1 / train all | `5.4678458e-3` | `5.4678458e-3` | `4.5198180e-8` | `-8.6299425e-13` | `0.9999849` |
| M1 / pre | `4.3987991e-3` | `3.4904998e-3` | `4.3146053e-8` | `-1.0401672e-11` | `0.9999938` |
| M1 / onset | `6.5971638e-3` | `5.8296438e-1` | `4.5198180e-8` | `1.3815121e-11` | `0.8127916` |
| M1 / train rain | `7.0697862e-3` | `5.9860702e-1` | `4.1894555e-8` | `1.6121576e-11` | `0.8013380` |
| M1 / heldout rain | `3.2115095e-2` | `2.2330676` | `1.0053108e-7` | `2.4471343e-11` | `0.2758460` |
| M2-X-independent / train all | `5.5088063e-3` | `5.5088063e-3` | `4.5014146e-8` | `7.2959067e-13` | `0.9999846` |
| M2-X-independent / pre | `4.5245192e-3` | `3.5902602e-3` | `4.2768132e-8` | `-6.2441010e-12` | `0.9999934` |
| M2-X-independent / onset | `6.5310555e-3` | `5.7712266e-1` | `4.5014146e-8` | `9.8124686e-13` | `0.8167568` |
| M2-X-independent / train rain | `7.0268088e-3` | `5.9496807e-1` | `4.1892887e-8` | `1.8386676e-11` | `0.8041031` |
| M2-X-independent / heldout rain | `4.9746881e-2` | `3.4590633` | `7.2853799e-8` | `5.3508860e-10` | `0.2031886` |
| M1-to-M2-X / train all | **`5.0695958e-3`** | **`5.0695958e-3`** | `4.4277653e-8` | `-5.2390800e-13` | `0.9999870` |
| M1-to-M2-X / pre | **`3.9007242e-3`** | **`3.0952714e-3`** | `4.2578847e-8` | `-6.7272233e-12` | `0.9999951` |
| M1-to-M2-X / onset | **`6.3131215e-3`** | **`5.5786473e-1`** | `4.4277653e-8` | `3.7029101e-12` | `0.8317364` |
| M1-to-M2-X / train rain | **`6.7350270e-3`** | **`5.7026257e-1`** | `3.9851346e-8` | `1.3181137e-11` | `0.8232284` |
| M1-to-M2-X / heldout rain | **`2.8600981e-2`** | **`1.9887198`** | `8.4207787e-8` | `6.6941929e-11` | `0.3273201` |
| H1 / train all | `1.2451421e-2` | `1.2451421e-2` | `4.0913171e-8` | `9.2556549e-11` | `0.9999221` |
| H1 / pre | `1.3498267e-2` | `1.0711036e-2` | `3.8462849e-8` | `1.2864152e-10` | `0.9999422` |
| H1 / onset | `1.1653470e-2` | `1.0297695` | `4.0913171e-8` | `8.8538661e-11` | `0.5643365` |
| H1 / train rain | `9.7664534e-3` | `8.2693697e-1` | `3.8277646e-8` | `2.5488149e-12` | `0.6512490` |
| H1 / heldout rain | `4.0603764e-2` | `2.8233125` | `9.0123175e-8` | `-6.2577984e-10` | `0.2209046` |
| H2 / train all | `1.2467477e-2` | `1.2467477e-2` | `4.0896455e-8` | `9.1880224e-11` | `0.9999219` |
| H2 / pre | `1.3526507e-2` | `1.0733446e-2` | `3.8509475e-8` | `1.2825233e-10` | `0.9999419` |
| H2 / onset | `1.1652751e-2` | `1.0297059` | `4.0896455e-8` | `8.8368717e-11` | `0.5643868` |
| H2 / train rain | `9.7501933e-3` | `8.2556021e-1` | `3.8284579e-8` | `8.8710472e-13` | `0.6520188` |
| H2 / heldout rain | `4.0644476e-2` | `2.8261434` | `9.0084115e-8` | `-6.3319228e-10` | `0.2207631` |
| H5 / train all | `1.2541315e-2` | `1.2541315e-2` | `4.0430640e-8` | `9.1276210e-11` | `0.9999209` |
| H5 / pre | `1.3664787e-2` | `1.0843172e-2` | `3.8236383e-8` | `1.2812798e-10` | `0.9999407` |
| H5 / onset | `1.1639041e-2` | `1.0284944` | `4.0430640e-8` | `8.8079020e-11` | `0.5649647` |
| H5 / train rain | `9.6496984e-3` | `8.1705118e-1` | `3.7912775e-8` | `-1.0972164e-12` | `0.6568926` |
| H5 / heldout rain | `4.0816367e-2` | `2.8380954` | `9.0220369e-8` | `-6.3429312e-10` | `0.2197108` |

Warm M2-X is the best fixed-truth physical model in every regime: relative to
M1 its nRMSE is 7.28% lower overall, 11.32% lower PRE_RAIN, 4.31% lower at
ONSET, 4.74% lower in training sustained rain, and 10.94% lower held out.
H1 trades this accuracy for the deployed one-step state objective: versus M1,
training nRMSE increases 127.72% and held-out nRMSE 26.43%.  The 20 H2/H5
steps do not restore fixed-truth accuracy; H5 is 27.09% above M1 held out.

The interim mass-quartile and branch analysis also remains important.  Warm
M2-X reduced direct error in all four `|A*|` quartiles and in condensation,
evaporation, and inactive/balanced branches.  It did **not** attain its
discrete gain by visibly sacrificing low-impact regions.  Independent M2-X
showed mixed redistribution inside its poorer basin.  Weak assembly and mass
inversion prevent a stronger pointwise causal claim.

For the autonomous trajectories, the analytical law can also be evaluated at
each model's own post-prefix state.  The resulting nRMSEs (against `A*` at the
same model state) are:

| model | all | PRE_RAIN | ONSET | train rain | heldout rain |
|---|---:|---:|---:|---:|---:|
| M1 | `.074395` | `.094716` | `.113805` | `.061242` | `.053906` |
| M2-X-independent | `.097689` | `.127860` | `.146650` | `.083612` | `.066580` |
| M1-to-M2-X | `.076567` | `.094145` | `.126734` | `.078966` | **`.051174`** |
| H1 | `.045822` | `.014622` | `.014603` | `.016803` | `.062995` |
| H2 | `.045800` | `.014456` | `.014607` | `.017027` | `.062972` |
| H5 | **`.045553`** | **`.014283`** | `.014752` | **`.016728`** | `.062653` |

This does not contradict the fixed-truth table: H1 changes the state
distribution.  It is far more accurate on the model states encountered
before and during training rain, but less accurate than M1/warm M2-X on the
held-out mature-rain model states.  This is direct evidence of a
deployment-adapted rather than uniformly superior constitutive approximation.

## 5. Autonomous state trajectory

Each model starts from the same `X_0^*` and runs 160 complete six-child steps.
The learned `A` and analytical `R` are evaluated on current model states.
`accumulated` is the square root of the ratio of summed mixed-mass error
energy to summed truth energy on the named interval.

| model | all final | all max | all accum | PRE accum | ONSET accum | train-rain accum | heldout-rain accum |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | `8.3302840e-6` | `9.8804228e-6` | `6.4019509e-6` | `6.1510836e-6` | `6.4419562e-6` | `4.9279547e-6` | `6.8631577e-6` |
| M2-X-independent | `1.0340615e-5` | `1.2164368e-5` | `8.0988404e-6` | `8.0672593e-6` | `7.6393126e-6` | `5.6973656e-6` | `8.6662159e-6` |
| M1-to-M2-X | `8.0251507e-6` | `1.0075468e-5` | `6.3757503e-6` | `6.1542666e-6` | `7.1048632e-6` | `4.9474080e-6` | `6.7256067e-6` |
| H1 | `4.9344506e-6` | **`4.9344506e-6`** | `3.0104279e-6` | `1.1587661e-6` | **`1.5351835e-6`** | `2.0411298e-6` | `4.0056695e-6` |
| H2 | `4.9347221e-6` | `4.9347221e-6` | **`3.0091619e-6`** | `1.1508914e-6` | `1.5423368e-6` | `2.0447289e-6` | **`4.0043998e-6`** |
| H5 | `4.9670135e-6` | `4.9670135e-6` | `3.0109594e-6` | **`1.1453167e-6`** | `1.5463413e-6` | **`2.0268627e-6`** | `4.0102109e-6` |

H1 is the realized deployment gain: relative to M1 it lowers final, maximum,
and accumulated error by 40.76%, 50.06%, and 52.98%.  The gain occurs in
both training and held-out time, despite worse direct `A` regression.

H1 -> H2 changes final error by +0.0055% and accumulated error by -0.0421%;
held-out accumulated error improves only 0.0317%.  H2 therefore adds genuine
recursive gradient information mathematically, but the completed 20-step
fit yields no practically resolved autonomous gain over H1.

H2 -> H5 improves the targeted H5 objective by 0.2245% and training-rain
accumulated state error by 0.874%, but worsens final error by 0.654%, overall
accumulated error by 0.060%, and held-out accumulated error by 0.145%.
Longer horizon is not monotonically better under this 20-step budget.

### 5.1 Final / maximum fieldwise relative errors

Each entry is `final / maximum` over `0..160`.

| model | v | h | S | Qv | Qc | Qr |
|---|---:|---:|---:|---:|---:|---:|
| M1 | `1.045e-4 / 1.045e-4` | `6.883e-6 / 6.907e-6` | `8.343e-6 / 9.926e-6` | `3.297e-4 / 6.241e-4` | `7.368e-3 / 1.456e-2` | `3.101e-2 / 6.796e-1` |
| M2-X-independent | `1.227e-4 / 1.233e-4` | `7.784e-6 / 8.355e-6` | `1.036e-5 / 1.222e-5` | `4.292e-4 / 8.008e-4` | `9.584e-3 / 1.869e-2` | `1.532e-2 / 2.807e-1` |
| M1-to-M2-X | `1.040e-4 / 1.041e-4` | `6.744e-6 / 7.004e-6` | `8.037e-6 / 1.012e-5` | `2.723e-4 / 6.299e-4` | `6.084e-3 / 1.468e-2` | `1.678e-2 / 9.167e-1` |
| H1 | `5.054e-5 / 5.054e-5` | `4.589e-6 / 4.589e-6` | `4.937e-6 / 4.937e-6` | `3.273e-4 / 3.450e-4` | `7.333e-3 / 7.765e-3` | `1.042e-1 / 6.922e-1` |
| H2 | `5.052e-5 / 5.052e-5` | `4.586e-6 / 4.586e-6` | `4.937e-6 / 4.937e-6` | `3.274e-4 / 3.450e-4` | `7.336e-3 / 7.767e-3` | `1.041e-1 / 6.989e-1` |
| H5 | `5.085e-5 / 5.085e-5` | `4.577e-6 / 4.577e-6` | `4.970e-6 / 4.970e-6` | `3.278e-4 / 3.443e-4` | `7.345e-3 / 7.750e-3` | `1.019e-1 / 7.127e-1` |

The H1-family improvement is clearest in velocity, depth, entropy, and the
maximum vapor/cloud errors.  Rain relative error is ill-conditioned near its
zero onset and does not track integrated rain mass; the mass budget below is
the more useful rain diagnostic.

## 6. Rain onset and rain mass

| model | meaningful onset | qc max (step) | max R | integrated R mass | relative mass error | final Qr mass |
|---|---:|---:|---:|---:|---:|---:|
| truth | `5100` | `1.0517910e-4` (89) | `5.1790957e-11` | `2.3073301e8` | -- | `2.3073301e8` |
| M1 | `5100` | `1.0531905e-4` (97) | `5.3190517e-11` | `2.3710431e8` | `+2.761%` | `2.3710431e8` |
| M2-X-independent | `5100` | `1.0513463e-4` (89) | `5.1346282e-11` | `2.3293396e8` | `+0.954%` | `2.3293396e8` |
| M1-to-M2-X | `5100` | `1.0527093e-4` (90) | `5.2709315e-11` | `2.2763437e8` | `-1.343%` | `2.2763437e8` |
| H1 | `5100` | `1.0541315e-4` (105) | `5.4131499e-11` | `2.5485060e8` | `+10.453%` | `2.5485060e8` |
| H2 | `5100` | `1.0541148e-4` (105) | `5.4114797e-11` | `2.5481140e8` | `+10.436%` | `2.5481140e8` |
| H5 | `5100` | `1.0540410e-4` (105) | `5.4040957e-11` | `2.5419573e8` | `+10.169%` | `2.5419573e8` |

All six models predict the exact and meaningful onset at the truth step.  The
fixed objectives therefore preserve the threshold crossing remarkably well.
They differ after onset.  M2-X-independent has the closest rain mass and
exact peak time, while warm M2-X is 1.34% low.  M1 is 2.76% high and peaks
800 time units late.  H1/H2/H5 peak 1,600 units late and overproduce rain by
about 10%.  H2 changes H1 rain mass by only -0.017 percentage point; H5
reduces the overproduction to 10.17% but does not repair it.

Consequently the large H1 mixed-state improvement is **not** a rain-mass
improvement.  The common state metric admits a tradeoff in which bulk state
and flow errors fall while accumulated analytical rain, driven by the
learned cloud/vapor trajectory, becomes less accurate.

## 7. Physical consistency, flow, and boundedness

The neural source structure, not training, enforces both invariants.  In all
six trajectories the maximum pointwise water-source residual is
`1.30e-21..1.66e-21`, the maximum `S-beta2 Qv` source residual is
`1.3877788e-17`, analytical-R recomputation discrepancy is exactly zero, and
relative total-water drift is `1.77e-14..1.90e-14`.  These are the host
discrete numerical floor, not learned conservation errors.  The global
`S-beta2 Qv` state combination drifts by only `1.49e-14..1.91e-14` relative
to its initial mass (absolute `2688..3456` on an initial
`1.8093136636337517e17`).

All state coefficients remain finite.  `h`, `S`, and `Qv` stay positive.
The DG moisture fields are not coefficientwise positivity preserving:
minimum `Qc` coefficients range from `-2.486e-3` to `-1.629e-3`, and minimum
`Qr` coefficients from `-1.783e-6` to `-1.726e-6`.  This stable but real
undershoot should remain visible when comparing Representations B and C.

Flow relative mismatches are:

| model | KE final / max | projected enstrophy final / max | high-k fraction final / max |
|---|---:|---:|---:|
| M1 | `1.4835e-5 / 3.0191e-5` | `8.9905e-6 / 5.2622e-5` | `3.6228e-1 / 7.4370e-1` |
| M2-X-independent | `1.5661e-5 / 4.1137e-5` | `9.4307e-6 / 7.1556e-5` | `2.6570e-1 / 6.8655e-1` |
| M1-to-M2-X | `1.7209e-5 / 3.1816e-5` | `1.5246e-5 / 5.3794e-5` | `2.8388e-1 / 6.5661e-1` |
| H1 | `2.1757e-5 / 2.1757e-5` | `2.6393e-5 / 2.6393e-5` | `3.2973e-1 / 7.5399e-1` |
| H2 | `2.1728e-5 / 2.1728e-5` | `2.6417e-5 / 2.6417e-5` | `3.2963e-1 / 7.5401e-1` |
| H5 | `2.1258e-5 / 2.1258e-5` | `2.6046e-5 / 2.6046e-5` | `3.2923e-1 / 7.5412e-1` |

The high-wavenumber fraction has a very small truth reference and correspondingly
large relative ratios; it is a sensitivity diagnostic, not evidence of an
unstable state.  The H1 family has better mixed-state and velocity error but
slightly worse final KE/enstrophy mismatch than M1.  No model exhibits a
finite-value or gross stability failure.

## 8. Mechanism synthesis

### 8.1 M1 -> M2-X -> H1

The warm M2-X branch is Pareto-better than M1 across all five training
objectives and all fixed-truth `A` regimes.  This confirms that deployed-map
weighting can add useful information without sacrificing the constitutive
law when it begins in the M1 basin.  The independent fit demonstrates why
initialization must remain an explicit experimental factor.

H1 is a much larger change.  It reduces its own objective 83.04% from M1 and
the autonomous accumulated state error 52.98%, while increasing direct
training `A` nRMSE 127.72%, held-out `A` nRMSE 26.43%, and integrated rain
mass error from +2.76% to +10.45%.  This is evidence that post-prefix
deployment consistency learns a solver-effective one-step correction that is
not the best pointwise physical `A` approximation.  Because the source
manifold remains exact, this is a discretization/state-location compensation,
not exploitation of unphysical conservation directions.

### 8.2 H1 -> H2 -> H5

Exact recursive information exists at H2 and H5, but the completed 20-step
continuations make only `O(1e-3)` relative changes to their objective values
and virtually no autonomous change.  H2's best overall/held-out accumulated
errors beat H1 by only 0.042%/0.032%, while final error is 0.006% worse.  H5
makes the best H5 value and training-rain error but modestly worsens final and
held-out deployment.  These results establish the realized benefit under the
tested budget, not the optimum value of recursive training.

### 8.3 Training versus held-out mature rain

H1's state gain persists into held-out time: held-out accumulated error falls
41.63% from M1 (`6.8632e-6 -> 4.0057e-6`).  That does **not** mean the local
law extrapolates better: fixed-state held-out `A` nRMSE rises 26.43%, and rain
mass becomes substantially high.  Thus poor held-out direct-`A` correlation
does not prevent a good bulk state trajectory, but it does appear in a
specific mature-rain budget error.  Reporting only one of state error,
physical-rate error, or rain mass would give an incomplete conclusion.

## 9. Frozen Representation-A conclusions

Certified findings:

1. All six final artifacts are complete and immutable, and all common
   evaluations use the frozen truth, schedules, metrics, and normalization.
2. M2-X is optimizer-basin sensitive.  M1 warm-starting reaches the best
   M1 and M2-X values despite half the independent accepted-iteration budget.
3. H1 supplies the dominant practical state-trajectory improvement.  It is
   fixed/cacheable, so this gain does not require recursive feedback.
4. H1's gain is accompanied by worse physical `A` recovery and approximately
   10% rain overproduction, even though rain onset remains exact.
5. Twenty H2 and twenty H5 accepted iterations yield only marginal changes;
   H5 improves its target but slightly worsens full held-out deployment.
6. Total-water and thermodynamic source consistency remain exact by
   representation in every model.  The observed tradeoff occurs within the
   physically structured source manifold.

Limitations:

* every optimizer stopped at `MAXITER`, so no result is a mathematical
  optimum;
* H1 received 5,000 accepted iterations, versus 20 each for H2 and H5;
* the held-out test is temporal extrapolation on one deterministic truth, not
  an ensemble or new initial condition;
* analytical rain removes one learning challenge and couples rain errors to
  learned-state errors rather than learned-R error;
* the mixed-state metric does not alone guarantee accurate phase partition or
  integrated rain mass.

Questions carried into Representations B and C:

* Can learned `R_theta` reproduce onset, magnitude, and accumulated rain while
  retaining the H1 state gain?
* Does the two-rate structured representation preserve the helpful H1
  correction without increasing rain-mass bias?
* Does a four-output black box exploit nonconservative directions, as in
  no-rain Test2A Problem B, or can rain-active supervision discover the
  two-rate manifold?
* Are recursive H2/H5 corrections larger when `R` is learned, and are 20
  iterations enough to measure them?
* Do the three representations preserve onset on the held-out mature-rain
  interval, and how do state, rate, conservation, and rain-mass metrics trade
  off?

**STATUS: TEST2B_REPRESENTATION_A_FROZEN**
