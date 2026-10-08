# Test2B Representation B: final scientific synthesis

**Frozen evaluation date:** 2026-08-15

**Status:** final evaluation of all six completed Representation-B fits

**Scope:** learned physical rates `(A_theta,R_theta)` with the exact two-rate
source map; no new optimization or truth generation

The authoritative machine-readable evaluation is
`external-results/test2b-rain-active-learning/production/representation-B/representation_b_final_comparison.json`
(SHA256
`6044c0fbd42484e3bd6f0ec53bef91d9a871fa315fc83c193c10f1879813aadd`).
It was generated from immutable final parameters by the evaluation-only
module `dimswe.test2b_representation_b_postprocess`.  The file records
`evaluation_only=true`, `optimizer_instantiated=false`, and
`truth_generated=false`.  Its complete time series are authoritative where
this document reports compact summaries.

## 1. Frozen Representation-B contract

The truth, training support, input normalization, state metrics, objective
schedules, derivatives, and optimizer conventions are frozen in
`docs/TEST2B_RAIN_ACTIVE_LEARNING_PREPARATION.md`.  The local input is

\[
x=(h,S,Q_v,Q_c,B).
\]

`Qr` is prognostic and diagnostic but not an input because neither accepted
local constitutive law depends on it.  The float64, seed-0,
`5 -> 32 -> 32 -> 2` tanh network has 1,314 parameters and predicts the two
independent scaled physical rates `(A_theta,R_theta)`.  The deployed moist
source is exactly

\[
(S_t,Q_{v,t},Q_{c,t},Q_{r,t})
=h(\beta_2 A_\theta,A_\theta,-A_\theta-R_\theta,R_\theta).
\]

Consequently, for arbitrary neural outputs,

\[
Q_{v,t}+Q_{c,t}+Q_{r,t}=0,
\qquad S_t-\beta_2Q_{v,t}=0.
\]

No analytical `A` or `R`, truth-rate injection, conservation repair, or
source projection is used.  The representation does **not** impose the
analytical precipitation threshold, `R>=0`, or a zero-rain pre-onset branch.
Those are learned properties.

Training/model-selection support is truth states `0..80`.  The contiguous
held-out mature-rain interval is `81..160`.  The frozen truth first has exact
and physically meaningful rain at step 51 (`t=5100`), peaks at specific
`Qc=1.0517909572531444e-4` at step 89, has maximum
`R=5.179095725314434e-11`, and accumulates/finalizes at rain mass
`2.30733006980403e8`.  These data come from the frozen rain audit (SHA256
`0302d1cb3808e9543986665eaa05aa3bcd49b1ab70326c2cca9a4d5dc1861b5d`).

The common fixed cache is
`external-results/test2b-rain-active-learning/preparation/fixed_learning_data.npz`
(SHA256
`6e159015234fd94881b0b97888b7481eb049a02dcd96c571078920c0bedc901c`).
Its frozen output scales are
`sigma_A=9.052258655848717e-8` and active-support
`sigma_R=1.9902871261559996e-11`.

### 1.1 Objective definitions

For `z_B=(A,R)` and `D_B=diag(sigma_A,sigma_R)`, the direct physical-rate
objective is

\[
J_{M1}(\theta)=
\frac{\sum_{k=0}^{80}\|D_B^{-1}
[z_{B,\theta}(X_k^*)-z_B^*(X_k^*)]\|_{M_B}^2}
{\sum_{k=0}^{80}\|D_B^{-1}z_B^*(X_k^*)\|_{M_B}^2}.
\]

The deployed fixed-state boundary objective is

\[
J_{M2-X}(\theta)=
\frac{\sum_{k=0}^{80}\|G_4(X_k^*)
[N_{B,\theta}(X_k^*)-N^*(X_k^*)]\|_M^2}{D_X},
\qquad D_X=9.014768540958347\times10^{10}.
\]

Let `P` denote children 1--5 of the accepted six-child timestep and
`Y_k=P(X_k^*)`.  The fixed/cacheable deployed one-step objective is

\[
J_{H1}(\theta)=
\frac{\sum_{k=0}^{79}\|F_\theta(X_k^*)-X_{k+1}^*\|_M^2}{D_Y},
\qquad D_Y=9.01944200525722\times10^{14}.
\]

For `H=2,5`, nonoverlapping windows start at `0,H,...,80-H` and

\[
J_H(\theta)=D_Y^{-1}\sum_k\sum_{j=1}^{H}
\|\widehat X_{k+j}-X_{k+j}^*\|_M^2,
\qquad \widehat X_{n+1}=F_\theta(\widehat X_n).
\]

Every target boundary `1..80` occurs once per dense horizon objective.  H1 is
truth-reset and offline; H2 is the first genuinely recursive objective.
Values are compared **within an objective column**, not across differently
defined columns.  Representation-A and -B M1 scalars are also not directly
comparable because their output spaces and conditioning differ; the deployed
state objectives use the same state metric and denominators.

## 2. Verified artifacts and optimization budgets

For every stage, `fit_result.json` and `fit_progress.json` are bytewise
identical, report `status=complete`, and agree with the final sidecar on
stage, accepted iteration, and pytree fingerprint.  Every
`final_parameters.npz` passed the fingerprint-validating loader.

| model | start | accepted | value / gradient evals | wall (s) | termination | pytree SHA256 | NPZ SHA256 |
|---|---|---:|---:|---:|---|---|---|
| M1 | seed 0 | 10,000 | 20,964 / 10,001 | 16,313.668 | `MAXITER` | `cfc9d3da6a8d07d74ae17e3d9a5beabe434e63b8005e95df4a1925c3a63c609c` | `a740fa161ad5ff954beb73b4b44fda51b705ce34a40718bd8912716425aeb459` |
| M2-X-independent | seed 0 | 10,000 | 20,944 / 10,001 | 45,794.088 | `MAXITER` | `8292c556c31e15a303cc9010d9174149ab46cf31ce6ef385ffcb22d406b9b74a` | `c260591a8073549afbf106d189e3cebdcf88d19af56b9eddcb035f8307712a4d` |
| M1-to-M2-X | M1 final | 5,000 | 10,506 / 5,001 | 22,886.966 | `MAXITER` | `5e1bca9a6345f9bcd0360aa3e7e3feeef2cb2b2ac960e00106a7779e8d7892db` | `bb83dfcac9307f89b8210b2f7aea683ae0ce469877b72bc97595fc5240ee86f2` |
| H1 | M1 final | 5,000 | 10,500 / 5,001 | 20,643.086 | `MAXITER` | `9ebda59a2f74f1522e595b854a06f7c1af01ef9c85a734dc0aab1996c7ac5e03` | `bc622626887cc65acf215c03bf3281eb0fe97447723528415fbc5f8345eccf8e` |
| H2 | H1 final | 20 | 44 / 21 | 2,378.449 | `MAXITER` | `866baac093ffcd94826daa14c13115ea057ce0e07a2b770ba642ccb1f79c44a5` | `b3257a8821b80b90359bce7ca22759e0a48f1785f01b468c60a90c8bf9f41bcc` |
| H5 | H2 final | 20 | 44 / 21 | 3,412.530 | `MAXITER` | `4caffb3b14e290581d2cb4394458746e5f47d43689dc3497c942e23582ae0129` | `575d26dacc4eddd4ef84c564725a8b8beb521dc279fb797eaece34e4de842f74` |

The measured total is 30,040 accepted iterations, 63,002 objective
evaluations, 30,046 gradient evaluations, and `111428.78706287616 s`
(`30.9524 h`).  Every fit is `MAXITER`-limited; none is a demonstrated
stationary optimum.  The 5,000-versus-20 iteration asymmetry is essential to
the H1/H2/H5 interpretation.

## 3. Exact common objective matrix

| network | `J_M1` | `J_M2-X` | `J_H1` | `J_H2` | `J_H5` |
|---|---:|---:|---:|---:|---:|
| M1 | **`4.0727693178163562e-5`** | `2.6983137744699252e-5` | `2.2321605852819912e-4` | `4.6971842827333383e-4` | `1.4028281646974411e-3` |
| M2-X-independent | `1.4751715122501050e-1` | `1.8633732586466780e-5` | `2.1771951248522730e-4` | `4.6379825081291890e-4` | `1.4475034220879873e-3` |
| M1-to-M2-X | `1.4203985431530742e-3` | **`1.4814244410556754e-5`** | `1.8253361158110601e-4` | `3.8974981118315157e-4` | `1.2002049966588278e-3` |
| H1 | `2.4481122892735471e-3` | `7.8164232503741268e-5` | `2.2820014210665148e-5` | `3.9949925274640299e-5` | `7.4285773457921352e-5` |
| H2 | `2.4477556169140817e-3` | `7.7837967432553575e-5` | **`2.2798499371866257e-5`** | **`3.9905932068011353e-5`** | `7.4201382852323971e-5` |
| H5 | `2.4472406368625868e-3` | `7.7558210064442192e-5` | `2.2913030788814663e-5` | `3.9971427978199688e-5` | **`7.3977563019742115e-5`** |

The best attained value in each fixed column is bold.  M1 remains best only
under direct two-rate supervision; warm M2-X is best under M2-X; H2 is best
under H1 and H2; H5 is best under H5.

Within-column changes are:

| transition | `J_M1` | `J_M2-X` | `J_H1` | `J_H2` | `J_H5` |
|---|---:|---:|---:|---:|---:|
| M1 -> warm M2-X | +3387.550% | -45.098% | -18.226% | -17.025% | -14.444% |
| warm M2-X -> H1 | +72.354% | +427.629% | -87.498% | -89.750% | -93.811% |
| M1 -> H1 | +5910.928% | +189.678% | -89.777% | -91.495% | -94.705% |
| H1 -> H2 | -0.0146% | -0.4174% | -0.0943% | -0.1101% | -0.1136% |
| H2 -> H5 | -0.0210% | -0.3594% | +0.5024% | +0.1641% | -0.3016% |

The warm-M2-X-to-H1 row is a cross-branch comparison: H1 starts from M1,
not from warm M2-X.

### 3.1 M2-X basin result

Independent M2-X ends at `1.8633732586466780e-5`; warm M2-X ends at
`1.4814244410556754e-5`.  The independent value is 25.783% above the warm
value despite twice the accepted-iteration budget.  Warm M2-X also improves
M2-X by 45.098% from M1 and is better than independent M2-X under H1, H2,
and H5.  This is a rigorous optimizer-basin/budget effect, not evidence
against the M2-X objective.

The physical-rate qualification matters.  Warm M2-X improves the `A` fit,
but its combined two-rate M1 value is 34.88 times M1's because its learned
`R` is much worse.  It is nevertheless far better under M1 than independent
M2-X (`1.420e-3` versus `1.475e-1`).  Warm-starting therefore preserves much
more of the physical-rate basin, but not M1-level rain-law accuracy.

## 4. Direct physical-rate accuracy on truth states

`nRMSE` uses the frozen training scale for the corresponding physical rate;
`relRMS` uses the target RMS within the named regime.  Held-out metrics are
post hoc and did not select any fit.

### 4.1 A accuracy

| model | train nRMSE | PRE | ONSET | train rain | heldout rain | heldout rel RMS | heldout max abs | heldout bias | heldout corr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | `.006291` | `.005411` | `.007116` | `.007766` | `.041648` | `2.89595` | `6.677e-8` | `-6.715e-10` | `.1995` |
| M2-X-independent | `.005476` | `.004578` | `.006422` | `.006883` | `.049428` | `3.43687` | `1.857e-7` | `-2.416e-10` | `.1920` |
| M1-to-M2-X | **`.005044`** | **`.004099`** | **`.006012`** | **`.006490`** | **`.013647`** | **`.94892`** | `2.868e-8` | `3.294e-11` | **`.6380`** |
| H1 | `.010196` | `.011013` | `.009723` | `.008033` | `.023200` | `1.61316` | `2.281e-8` | `2.709e-10` | `.4436` |
| H2 | `.010177` | `.010984` | `.009713` | `.008038` | `.023042` | `1.60216` | `2.262e-8` | `2.652e-10` | `.4460` |
| H5 | `.010162` | `.011019` | `.009581` | `.007921` | `.022753` | `1.58210` | **`2.254e-8`** | `2.741e-10` | `.4512` |

Warm M2-X has the best fixed-truth `A` nRMSE in every regime: relative to
M1, training and held-out nRMSE are 19.82% and 67.23% lower.  H1 doubles the
training error relative to warm M2-X, yet remains 44.29% better than M1 on
held-out truth states.  H2/H5 slightly improve the H1 held-out `A` error;
H5 is 1.93% below H1.  Thus the H1 state correction trades training-support
`A` fidelity, but it does not sacrifice held-out `A` relative to the B-M1
fit.

### 4.2 Learned R accuracy and sparse activity

| model | train all nRMSE | train-active nRMSE / corr. | heldout-active nRMSE / corr. | PRE false-positive rate | train-rain false-negative rate |
|---|---:|---:|---:|---:|---:|
| M1 | **`.001206`** | **`.009525 / .99984`** | **`.021380 / .99965`** | `.21087` | `0` |
| M2-X-independent | `.385470` | `1.373226 / -.05125` | `1.371069 / -.21672` | `.11168` | `1.00000` |
| M1-to-M2-X | `.037491` | `.447703 / .95632` | `.520345 / .98301` | `.82381` | `.00194` |
| H1 | `.048605` | `.560583 / .97035` | `.628006 / .97839` | `.82481` | `0` |
| H2 | `.048605` | `.560584 / .97035` | `.628012 / .97839` | `.82480` | `0` |
| H5 | `.048603` | `.560543 / .97037` | `.627982 / .97841` | `.82480` | `0` |

M1 learns the analytical rain law extremely accurately on truth states.  The
independent M2-X basin essentially misses active rain and has negligible or
negative active correlation.  Warm M2-X retains activity and correlation but
substantially under-resolves magnitude.  H1/H2/H5 detect the active support
on truth states and retain correlations near `.97` training and `.978`
heldout, while active magnitudes remain poor (`nRMSE .56` training and `.63`
heldout).

All learned networks predict both positive and negative residual `R` on
nominally dry states.  The frozen activity test therefore reports first
positive `R` at `t=0` for all six networks.  Its numerical rate tolerance at
the initial dry state is about `1.21e-20`, so an activity count alone
overstates the significance of the smallest residuals.  The integrated
pre-onset budgets in Section 6 establish which residuals are materially
cumulative.  The learned representation has no threshold or positivity
architecture, so this failure is scientifically real rather than a
postprocessor defect.

### 4.3 Accuracy on each model's own states

Truth-rate evaluation at the exact same model post-prefix state exposes
off-manifold behavior:

| model | A nRMSE all / heldout | active-R nRMSE all / heldout |
|---|---:|---:|
| M1 | `28.358 / 40.105` | `617.63 / 638.36` |
| M2-X-independent | `.1258 / .0630` | `1.432 / 1.441` |
| M1-to-M2-X | `.1046 / .0589` | `.524 / .533` |
| H1 | `.03585 / .04953` | `.672 / .684` |
| H2 | `.03572 / .04935` | `.672 / .685` |
| H5 | **`.03520 / .04861`** | **`.668 / .680`** |

M1's excellent truth-state regression is therefore not dynamically robust:
small early state errors eventually move it into a region where both learned
rates extrapolate catastrophically.  H1-family rates are much less accurate
than M1 on the truth manifold but remain far better behaved on their own
deployed states.  This is a deployment-adaptation result, not recovery of the
analytical `R` law.

## 5. Full 160-step autonomous state trajectory

Every model starts at the same `X_0^*`.  Both neural rates are evaluated at
the current model state at every moist call; neither truth rate is injected.
`accumulated` is the square root of summed mixed error energy divided by
summed truth energy on the named interval.

| model | final | maximum | accumulated | PRE accum | ONSET accum | training-rain accum | heldout-rain accum |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | `7.0248255e-3` | `7.0248255e-3` | `2.2572374e-3` | `8.6798e-6` | `9.0783e-6` | `6.2886e-6` | `3.2020503e-3` |
| M2-X-independent | `1.0321429e-5` | `1.5666261e-5` | `9.4055746e-6` | `1.0350e-5` | `1.0269e-5` | `7.0096e-6` | `9.1807e-6` |
| M1-to-M2-X | `8.9683379e-6` | `1.2424451e-5` | `7.6189089e-6` | `7.9764e-6` | `9.3093e-6` | `6.2273e-6` | `7.4656e-6` |
| H1 | `3.3807730e-6` | `3.8888069e-6` | `2.4320777e-6` | `8.0714e-7` | **`1.1876e-6`** | `1.6172e-6` | `3.2646e-6` |
| H2 | `3.3692799e-6` | `3.8752646e-6` | `2.4255301e-6` | `8.0558e-7` | `1.1880e-6` | `1.6160e-6` | `3.2552e-6` |
| H5 | **`3.3078053e-6`** | **`3.8173374e-6`** | **`2.3914526e-6`** | **`8.0081e-7`** | `1.1926e-6` | **`1.6108e-6`** | **`3.2052e-6`** |

M1 stays finite but becomes physically and dynamically unusable in held-out
time: final/accumulated errors rise to `7.02e-3/2.26e-3`, and its minimum
`Qv` coefficient is `-3.53`.  Warm M2-X avoids that runaway and reduces
M1's accumulated error by 99.66%, although it remains worse than H1.

H1 is the dominant state correction.  Relative to M1 its accumulated error
falls 99.892%; relative to the stable warm branch it falls 68.08%.  H1 -> H2
reduces final/maximum/accumulated errors by only
`0.340%/0.348%/0.269%`; held-out accumulated error improves 0.289%.
H2 -> H5 reduces them by `1.825%/1.495%/1.405%`; held-out improves 1.534%.
Recursive feedback produces a small but measurable realized benefit here,
larger than in Representation A, while remaining modest under the 20-step
budgets.

### 5.1 Final / maximum fieldwise relative errors

Each entry is `final / maximum` over `0..160`.

| model | v | h | S | Qv | Qc | Qr |
|---|---:|---:|---:|---:|---:|---:|
| M1 | `.03494/.03494` | `.002966/.002966` | `.007053/.007053` | `.4957/.4957` | `11.11/11.11` | `2.794/235.0` |
| M2-X-independent | `1.486e-4/1.577e-4` | `1.014e-5/1.087e-5` | `1.032e-5/1.574e-5` | `3.448e-4/1.044e-3` | `.007858/.02441` | `4.313/3.374e5` |
| M1-to-M2-X | `1.194e-4/1.283e-4` | `7.861e-6/9.266e-6` | `8.978e-6/1.248e-5` | `2.830e-4/8.186e-4` | `.006335/.01908` | `.4346/3556` |
| H1 | `3.308e-5/3.308e-5` | `3.126e-6/3.126e-6` | `3.383e-6/3.900e-6` | `2.349e-4/2.573e-4` | `.005285/.005803` | `.4829/9192` |
| H2 | `3.311e-5/3.311e-5` | `3.124e-6/3.124e-6` | `3.371e-6/3.887e-6` | `2.339e-4/2.562e-4` | `.005262/.005780` | `.4821/9192` |
| H5 | **`3.244e-5/3.244e-5`** | **`3.085e-6/3.085e-6`** | **`3.309e-6/3.829e-6`** | **`2.307e-4/2.526e-4`** | **`.005191/.005698`** | `.4867/9192` |

Rain-relative maxima are ill-conditioned when truth rain is zero or tiny and
should not be read as a bulk rain-mass statistic.  Section 6 provides the
physically useful integrated budget.

## 6. Learned rain onset, magnitude, and phase partition

| model | first positive / meaningful R | qc max (step) | max / min R | integrated/final rain mass | relative mass error |
|---|---:|---:|---:|---:|---:|
| truth | `5100` | `1.0517910e-4` (89) | `5.1791e-11 / 0` | `2.3073301e8` | -- |
| M1 | `0` | `7.7586960e-3` (160) | `7.9581e-11 / -2.3438e-11` | `6.8335401e8` | `+196.167%` |
| M2-X-independent | `0` | `1.0537751e-4` (97) | `1.5728e-11 / -1.5585e-11` | `-1.7571370e9` | `-861.546%` |
| M1-to-M2-X | `0` | `1.0512510e-4` (105) | `3.1429e-11 / -1.9916e-12` | `1.4257199e8` | `-38.209%` |
| H1 | `0` | `1.0570836e-4` (106) | `2.8306e-11 / -2.3337e-12` | `1.5126077e8` | `-34.443%` |
| H2 | `0` | `1.0571131e-4` (106) | `2.8328e-11 / -2.3344e-12` | `1.5147256e8` | `-34.352%` |
| H5 | `0` | `1.0567065e-4` (106) | `2.8033e-11 / -2.3328e-12` | `1.5026782e8` | `-34.874%` |

No B model discovers the exact dry branch or onset time.  The H1-family
positive pre-onset integrated source is `1.445e7`, 6.26% of the truth's final
rain mass, so its `t=0` classification is not merely an isolated roundoff
event.  Warm M2-X produces `5.035e6` before truth onset (2.18%); M1's signed
pre-onset integral is small and negative because positive and negative local
rates largely cancel.  Independent M2-X produces a large negative rain mass
and misses essentially all truth-active points, making it scientifically
unusable despite a good M2-X value.

H1/H2/H5 peak cloud water 1,700 time units late and underproduce final rain
by about 34--35%.  Their active-R correlations remain high but their rate
magnitudes and inactive branch are wrong.  H2 changes H1's rain-mass error by
only +0.092 percentage point; H5 worsens it by 0.522 percentage point from
H2.  The recursive improvements in bulk state error therefore do not repair
the learned precipitation law.

Negative learned `R` is permitted by the frozen representation.  It reverses
the cloud-to-rain partition while conserving total water.  Negative rain
mass in independent M2-X is consequently a phase-partition failure, not
water destruction.

## 7. Structural consistency, flow, and stability

All six neural sources preserve the two structural identities to roundoff.
Maximum pointwise water-source residuals are
`1.78e-19..2.80e-19`; maximum `S-beta2 Qv` residual is
`1.3877788e-17`; relative total-water drift is
`1.71e-14..1.96e-14`.  These are the host solver floor.  Representation B
cannot create or destroy total water through its moist source even when its
rain partition is wrong.

All state coefficients are finite, but finite does not imply physically
bounded.  M1 reaches `Qv_min=-3.5297` and `Qc_min=-0.6215`; independent M2-X
ends with negative rain mass and reaches `Qr_min=-1.666e-4`.  Warm/H1/H2/H5
retain positive `Qv` but have DG coefficient undershoots
`Qc_min=-2.716e-3..-1.408e-3` and `Qr_min=-2.947e-6..-1.636e-6`.

| model | KE final / max relative mismatch | enstrophy final / max relative mismatch |
|---|---:|---:|
| M1 | `1.9266e-3 / 1.9266e-3` | `5.6746e-4 / 5.6746e-4` |
| M2-X-independent | `2.8411e-5 / 5.6634e-5` | `2.5718e-5 / 9.6291e-5` |
| M1-to-M2-X | `2.1536e-5 / 4.5859e-5` | `2.3391e-5 / 7.4420e-5` |
| H1 | `1.3815e-5 / 1.3815e-5` | `1.4677e-5 / 1.4677e-5` |
| H2 | `1.3884e-5 / 1.3884e-5` | `1.4734e-5 / 1.4734e-5` |
| H5 | **`1.3560e-5 / 1.3560e-5`** | **`1.4493e-5 / 1.4493e-5`** |

The H1 family is stable in the finite-value sense and has the best flow
diagnostics, but its rain law remains inaccurate.  State, flow, rain, and
constitutive metrics must therefore remain separate.

## 8. Mechanism synthesis

### 8.1 M1 -> M2-X -> H1

M1 warm-starting again supplies the useful M2-X basin.  Warm M2-X reduces
M2-X by 45.10% from M1 and is 20.50% below independent M2-X.  It also improves
fixed-truth `A`, but pays for the discrete state gain by losing much of M1's
excellent `R` law.  The objective therefore reallocates error between the two
physical rates even though it cannot violate source conservation.

H1 is the major deployed-state change.  Relative to M1 it reduces H1 by
89.78% and autonomous accumulated error by 99.89%; relative to stable warm
M2-X it reduces H1 by 87.50% and autonomous accumulated error by 68.08%.
The apparent enormous autonomous percentage from M1 is amplified by M1's
held-out instability, but the warm comparison confirms a large genuine H1
gain.  That gain comes with a training `A` nRMSE 62.08% above M1 and an
active-rain magnitude error about 59 times M1's, plus early rain and 34.44%
rain-mass underproduction.  H1 is a solver-effective two-rate correction,
not the best physical constitutive fit.

### 8.2 H1 -> H2 -> H5

Twenty H2 steps reduce their targeted objective by 0.110%, autonomous
accumulated error by 0.269%, and held-out accumulated error by 0.289%.
Twenty H5 steps reduce H5 by 0.302%, autonomous accumulated error by a further
1.405%, and held-out error by 1.534%.  H5 also slightly improves fixed-truth
and model-path `A`; learned-`R` metrics are essentially unchanged.  Recursive
feedback therefore has a small, resolved state benefit in B, especially at
H5, but it neither discovers onset nor repairs rain mass under the tested
budget.  These results cannot bound what more thoroughly optimized recursive
objectives might achieve.

### 8.3 Physical-law recovery versus trajectory optimization

The comparison cleanly separates three ideas:

1. M1 learns `R` very accurately on truth states but is dynamically unstable
   under temporal extrapolation.
2. H1-family models produce much better state trajectories while learning a
   visibly different, signed, early-onset and low-magnitude rain law.
3. The exact two-rate source map prevents water/thermodynamic violations but
   does not make the inverse problem for `(A,R)` unique, impose precipitation
   threshold physics, or guarantee physically correct phase partition.

Thus source conservation is necessary and valuable representation design,
but it is not sufficient to identify rain microphysics from short-horizon
state loss.  The solver-facing objectives exploit compensating `A/R`
directions **within** the conservative two-rate manifold.

## 9. Matched Representation A versus B

Representation A learns only `A` and recomputes exact analytical `R` on the
current state.  Representation B learns both rates but uses the same input,
hidden architecture family, truth, schedules, state metric, and exact source
structure.  Cross-representation M1 scalars are not compared.  The state
objectives and autonomous state/rain budgets are comparable.

| stage | A final / accum state | B final / accum state | A rain-mass error | B rain-mass error | A/B onset |
|---|---:|---:|---:|---:|---:|
| M1 | `8.330e-6 / 6.402e-6` | `7.025e-3 / 2.257e-3` | `+2.761%` | `+196.167%` | `5100 / 0` |
| warm M2-X | `8.025e-6 / 6.376e-6` | `8.968e-6 / 7.619e-6` | `-1.343%` | `-38.209%` | `5100 / 0` |
| H1 | `4.934e-6 / 3.010e-6` | **`3.381e-6 / 2.432e-6`** | `+10.453%` | `-34.443%` | `5100 / 0` |
| H2 | `4.935e-6 / 3.009e-6` | **`3.369e-6 / 2.426e-6`** | `+10.436%` | `-34.352%` | `5100 / 0` |
| H5 | `4.967e-6 / 3.011e-6` | **`3.308e-6 / 2.391e-6`** | `+10.169%` | `-34.874%` | `5100 / 0` |

The B H1/H2/H5 accumulated state errors are 19.21%, 19.40%, and 20.58%
below their A counterparts, and their common state objectives are also lower
(`J_H1: 2.282e-5` versus `3.002e-5`, `J_H2: 3.991e-5` versus
`5.519e-5`, `J_H5: 7.398e-5` versus `1.102e-4`).  Extra rate freedom can
therefore fit the state metric better.

That flexibility has a clear physical cost.  A preserves exact onset and has
roughly +10% rain bias after H1, whereas B rains from `t=0`, permits negative
rain tendencies, peaks late, and is about 34--35% low in total rain.  A's
structural analytical rain law is therefore much more accurate even though
B has a lower bulk state error.  B's conservation remains exact, so this is
not an A-versus-B water-budget difference; it is an identifiability and phase-
partition difference within the same conservative source map.

B warm/H1/H2/H5 also have better fixed-truth held-out `A` nRMSE than A by
approximately 52.3%, 42.9%, 43.3%, and 44.3%.  This does not make the whole
B constitutive law better: `R` is wrong and the two rates can co-adapt.  The
evidence supports the inference that state training uses B's extra degree of
freedom as a compensating direction.  It is not a theorem of nonuniqueness
from this single deterministic trajectory.

The computational price of learning `R` was modest compared with the
scientific price: B used `111428.8 s` (`30.95 h`) versus A's `103287.9 s`
(`28.69 h`), an increase of 7.88%.  This is a serial, one-problem cost
comparison, not a scalability claim.

## 10. Frozen conclusions and limitations

Certified findings:

1. All six B fits and their fingerprints are complete; all five common
   objectives and six 160-step rollouts use the frozen truth and metrics.
2. M2-X is optimizer-basin sensitive.  The M1 warm start attains a 20.50%
   lower M2-X value than the independent fit with half the iterations.
3. M1 learns both physical laws very accurately on truth states, especially
   `R`, but its autonomous held-out trajectory leaves that support and fails.
4. H1 supplies the dominant stable trajectory gain and substantially lowers
   all dense state objectives, while sacrificing direct `R` magnitude and
   onset physics.
5. H2/H5 supply small realized recursive state gains under 20-iteration
   budgets; H5 is the best autonomous B model but does not improve rain law.
6. Total-water conservation and `S-beta2 Qv` remain at roundoff for every B
   model.  The important error is phase partition and rate identification,
   not water creation/destruction.
7. Compared with A, B achieves lower H1/H2/H5 state error but markedly worse
   onset and rain mass.  Learning `R` buys solver-facing flexibility and
   removes the exact threshold prior.

Limitations:

* all fits stop at `MAXITER`; none is a stationary optimum;
* H1 has 5,000 accepted iterations versus 20 each for H2/H5;
* one deterministic trajectory and a contiguous temporal holdout do not
  establish ensemble generalization;
* the activity threshold is exceptionally sensitive while truth `Qr=0`, so
  pre-onset sample fractions must be read together with integrated mass;
* Representation B does not enforce `R>=0` or analytical activation support;
  adding such constraints would be a new representation/ablation;
* a low mixed-state metric does not uniquely determine `(A,R)` or phase mass.

Questions carried into Representation C are now sharper: does removing even
the conservative two-rate source structure lower the state objective further,
and does it reintroduce total-water/thermodynamic drift on top of B's already
observed phase-partition compensation?  C must be judged jointly by state,
source-manifold, signed-water, rain, and stability diagnostics.

**STATUS: TEST2B_REPRESENTATION_B_FROZEN**
