# Test 2A Problem B final scientific synthesis and freeze audit

Date: 2026-08-11

Repository HEAD audited: `d2f5d66ecb5500aad24eca37280f8a52e22a250f`

Branch audited: `dev/dimswe-learned-physics-framework`

## 1. Scope and frozen contract

Problem B is the controlled output-representation ablation of frozen Problem
A. Both problems use the same ordered, normalized resolved inputs

\[
(h,S,Q_v,Q_c,B).
\]

Problem A learns one scalar `A_theta` and imposes the source map

\[
(S_t,Q_{v,t},Q_{c,t},Q_{r,t})
=hA_\theta(\beta_2,1,-1,0)
\]

on this no-rain support, while retaining the analytical rain law in deployed
code. Problem B instead learns four independent physical tendencies

\[
N_\theta=(S_t,Q_{v,t},Q_{c,t},Q_{r,t})_\theta
\]

with a `5 -> 32 -> 32 -> 4`, tanh, float64 network containing 1,380
parameters. It has no analytical `A` or `R`, shared scalar, conservation
correction, zero-rain constraint, or projection onto the Problem-A source
manifold. The exact no-rain truth nevertheless lies on that manifold.

All objectives and all autonomous comparisons use truth states 0 through 80.
The autonomous integrations are **training-support deployment diagnostics**,
not held-out validation. Nothing here establishes behavior on states 81
through 160, in rain-active flow, on another mesh, or under another network,
normalization, seed, or optimizer budget.

The authoritative machine-readable sources are:

- **PB** — `external-results/test2a/problem-b/production/problem_b_comparison.json`,
  SHA256 `c9bba696f957b34e40a1f95d29e645463bbdd4456144abd536b2fed4d24561d2`;
- **PBF** — the six `fit_result.json`, `fit_progress.json`, and final parameter
  sidecars under `external-results/test2a/problem-b/production/`;
- **PBP** — `docs/TEST2A_PROBLEM_B_PREPARATION.md` and
  `external-results/test2a/problem-b/preparation/problem_b_fixed_data.json`;
- **PA** — `docs/TEST2A_PROBLEM_A_FINAL_SYNTHESIS.md`; and
- **PA-HC** —
  `external-results/test2a/horizon-curriculum-h1-h2-h5/postprocess/horizon_curriculum_report.json`.

Every Problem-B number below comes from PB unless explicitly marked as a
derived ratio or attributed to PBF/PBP. Problem-A comparisons come from PA-HC
and were cross-checked against PA. The audit found no disagreement among these
sources.

## 2. Verified artifacts and objective definitions

### 2.1 Final model verification

The comparison JSON, every final parameter sidecar, and a fresh load/hash of
every parameter pytree agree:

| model | role | final parameter pytree SHA256 |
|---|---|---|
| M1 | direct four-tendency regression | `ed8fb21167bf5f0b56c59201c7dc771473d5d946d6d304cd26b57ac983a56516` |
| M2-X-independent | independent boundary-state deployed-map fit | `eb447dadf00519f105f887e80d507a7fd9d568412ddd086b1828e7e0985dd0e2` |
| M1-to-M2-X | M2-X continuation from final M1 | `db7631e1fd885bc24a25412e9c7c056c52268dca054d35f21c8d17b7f9dd522d` |
| H1 | post-prefix one-step continuation from M1 | `282ff45c40844f90f1c1d7e3602a0989bcd56b88ae425ae2414bcc0045166afb` |
| H2 | recursive two-step continuation from H1 | `bf0233109c26d2892b202819f228f8002de162dd52b78fb5bf7bcfbadc5c8cc5` |
| H5 | recursive five-step continuation from H2 | `9afcd50bb6f3caadba30140b2ea714d049a1b4f29cb5c0b162f1e878f934ccce` |

Both fit records report `complete` for all six models. Every run terminated at
its prescribed `MAXITER`: 200,000 accepted iterations for M1 and independent
M2-X, 50,000 for M1-to-M2-X and H1, and 100 for H2 and H5 [PBF]. Thus these
are matched practical endpoints, not demonstrated mathematical minima.

### 2.2 Loss conditioning

Problem B uses

\[
D_N=\operatorname{diag}(\sigma_S,\sigma_Q,\sigma_Q,\sigma_Q),
\]

where

\[
\sigma_S=4.465574092866371\times10^{-4},\qquad
\sigma_Q=4.553845840641363\times10^{-6}.
\]

The shared water scale gives the exactly zero truth `Qr_t` a positive penalty.
The target-label audit found water and `Qr_t` residuals exactly zero and a
maximum truth `S_t-beta2*Qv_t` residual of
`1.734723475976807e-18` [PBP].

### 2.3 Five objectives

With `M_4` denoting four copies of the accepted carrier mass and `G_4` the
exact production weak-assembly/mixed-mass-inverse map, M1 is

\[
J_{M1}^B(\theta)=
\frac{\sum_{k=0}^{80}\|D_N^{-1}[N_\theta(X_k^*)-N^*(X_k^*)]\|_{M_4}^2}
     {\sum_{k=0}^{80}\|D_N^{-1}N^*(X_k^*)\|_{M_4}^2}.
\]

M2-X is the fixed boundary-state deployed-map loss

\[
J_{M2-X}^B(\theta)=
\frac{\sum_{k=0}^{80}\|G_4[N_\theta(X_k^*)-N^*(X_k^*)]\|_M^2}
     {\sum_{k=0}^{80}\|G_4N^*(X_k^*)\|_M^2}.
\]

For the six-child step, define the parameter-independent prefix

\[
P=C_5\circ C_4\circ C_3\circ C_2\circ C_1,\qquad Y_k=P(X_k^*).
\]

H1/M2-Y evaluates the learned source where production actually calls it:

\[
J_{H1}^B(\theta)=
\frac{\sum_{k=0}^{79}\|dt\,G_4[N_\theta(Y_k)-N^*(Y_k)]\|_M^2}{D_B},
\]

where `D_B = 4.090171967662303e12`. H1 remains a fixed-state, fully
cacheable objective and contains no recursive model-state feedback.

For `H` equal to 2 or 5,

\[
\widehat X_k=X_k^*,\qquad
\widehat X_{k+j}=F_\theta(\widehat X_{k+j-1}),
\]

\[
J_H^B(\theta)=
\frac{\sum_{k\in S_H}\sum_{j=1}^{H}
\|\widehat X_{k+j}-X_{k+j}^*\|_M^2}{D_B}.
\]

The nonoverlapping schedules and common denominator are identical to Problem
A; every target boundary 1 through 80 occurs once. H2 is the first genuinely
recursive objective. These state objectives are directly comparable across A
and B. The scalar Problem-A `J_op` and four-output Problem-B `J_M1` are **not**
directly comparable because their output spaces and normalizations differ.

## 3. Headline Problem-B results

| model | `J_M1` | `J_M2-X` | `J_H1` | `J_H2` | `J_H5` | autonomous final | maximum | accumulated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | 5.517974e-4 | 5.146943e-4 | 5.169328e-4 | 8.572764e-4 | 1.878852e-3 | 3.680097e-7 | 9.214634e-7 | 5.408812e-7 |
| M2-X-independent | 5.115121e-1 | 5.873864e-4 | 6.590782e-4 | 2.867694e-2 | 5.887683e-1 | 2.144401e-4 | 2.144401e-4 | 1.304609e-4 |
| M1-to-M2-X | 1.857627e-3 | 4.385096e-4 | 4.676385e-4 | 1.017633e-3 | 4.550763e-3 | 4.296968e-6 | 4.296968e-6 | 2.733542e-6 |
| H1 | 1.442874e-3 | 5.069440e-4 | 3.984244e-4 | 7.660054e-4 | 2.710693e-3 | 6.205057e-6 | 6.205057e-6 | 3.206946e-6 |
| H2 | 1.488986e-3 | 5.421373e-4 | 4.231610e-4 | 6.962490e-4 | 2.039038e-3 | 2.442689e-6 | 2.556093e-6 | 2.041916e-6 |
| H5 | 1.447821e-3 | 7.393383e-4 | 5.689359e-4 | 7.839912e-4 | 1.620667e-3 | 6.709918e-5 | 6.709918e-5 | 3.098304e-5 |

The lowest trained short-horizon state objective does not consistently identify
the best autonomous model. M1 is best autonomously among these Problem-B
networks despite not minimizing H1, H2, or H5. This is the central result to
explain, not merely a ranking anomaly.

## 4. M2-X optimizer-basin result

The three relevant values are

\[
J_{M2-X}^B(\theta_{M1})=5.146942883989670\times10^{-4},
\]

\[
J_{M2-X}^B(\theta_{M2X,ind})=5.873863838421324\times10^{-4},
\]

\[
J_{M2-X}^B(\theta_{M1\rightarrow M2X})
=4.3850961099866574\times10^{-4}.
\]

The independent 200,000-iteration fit is 14.12% worse than final M1 under its
own M2-X objective. The 50,000-iteration M1 warm start is 14.80% better than
M1 and 25.35% better than the independent M2-X endpoint. All three numbers
are recomputed by the same fixed cache in PB; the independent and M1 fits both
started from the same seed and reached `MAXITER` [PBF].

This rigorously establishes optimizer-basin confounding. It does **not** show
that the M2-X objective is scientifically inferior: the warm-start branch
demonstrates a lower attainable value in the good M1 basin. Nor does it
establish the global M2-X minimum, since neither branch met a mathematical
convergence condition.

The warm-start outcome also reveals a representation problem. Its 14.80%
M2-X improvement is accompanied by a 3.37-fold increase in `J_M1`, a roughly
11.7-fold autonomous-final-error increase, a normalized manifold residual of
`3.5186e-2` versus M1's `1.5792e-4`, and a thermodynamic source defect of
`1.9245e-5` versus `7.2333e-8`. Thus M2-X contains useful deployed-map
information, but in four independent output directions that information alone
does not regularize the constitutive law.

## 5. M1 to H1: deployment consistency opens compensating directions

Relative to M1, H1 changes its targeted objective by

\[
1-\frac{3.984244402958454\times10^{-4}}
        {5.169328070913088\times10^{-4}}
=22.9253\%.
\]

It also lowers `J_H2` by 10.65% and slightly lowers `J_M2-X` by 1.51%, but
raises the direct four-tendency objective `J_M1` by 161.49%. Its `J_H5`
increases by 44.27%.

The autonomous consequences have the opposite sign from the targeted
one-step result:

| metric | M1 | H1 | H1/M1 |
|---|---:|---:|---:|
| final mixed-state error | 3.680097e-7 | 6.205057e-6 | 16.861 |
| maximum mixed-state error | 9.214634e-7 | 6.205057e-6 | 6.734 |
| accumulated mixed-state error | 5.408812e-7 | 3.206946e-6 | 5.929 |
| KE final mismatch | 4.468994e-8 | 3.185731e-6 | 71.28 |
| enstrophy final mismatch | 4.254904e-8 | 2.439795e-6 | 57.34 |

The source diagnostics identify the mechanism. M1, although unconstrained,
nearly discovers the exact source manifold under direct label supervision.
H1 obtains a lower one-step state loss while departing strongly from it:

| fixed truth-support diagnostic | M1 | H1 | H1/M1 |
|---|---:|---:|---:|
| water-source RMS | 5.093569e-10 | 7.480988e-10 | 1.469 |
| `S_t-beta2 Qv_t` RMS | 7.233251e-8 | 1.535490e-5 | 212.28 |
| spurious `Qr_t` RMS | 3.207271e-10 | 1.371486e-10 | 0.428 |
| normalized manifold residual RMS | 1.579228e-4 | 2.806787e-2 | 177.73 |
| normalized vector cosine with truth | 0.9997241 | 0.9993915 | — |

The physical component errors are:

| model | `S_t` RMS error | `Qv_t` | `Qc_t` | `Qr_t` |
|---|---:|---:|---:|---:|
| M1 | 1.048975e-5 | 1.069726e-7 | 1.069704e-7 | 3.207271e-10 |
| H1 | 1.134026e-5 | 1.954799e-7 | 1.953940e-7 | 1.371486e-10 |

Relative to the frozen component scales, M1's `S`, `Qv`, and `Qc` errors are
all about 2.35%. H1 has 2.54% `S` error and about 4.29% `Qv/Qc` error. The
large new defect is therefore not spurious rain: H1 actually reduces fixed-
support `Qr_t` error. It breaks the `S_t = beta2 Qv_t` coupling and moves in
non-manifold water-component directions that can compensate after weak
assembly and the mixed state metric. Its maximum accumulated autonomous
`S-beta2 Qv` drift rises from `2.0148e9` to `1.3329e12`, a factor 661.5.

The supported conclusion is precise:

> H1 deployment consistency is not intrinsically harmful. In the black-box
> output space, however, its state loss is underconstrained with respect to the
> constitutive source vector. It can reduce the one-step deployed defect using
> directions that direct physical supervision strongly suppresses.

## 6. H1 to H2 to H5: recursive information is real but not self-regularizing

### 6.1 H1 to H2

H2 reduces its targeted `J_H2` by 9.1065% and `J_H5` by 24.78% relative to
H1. It partially repairs autonomous behavior:

- final mixed error decreases 60.63%;
- maximum error decreases 58.81%;
- accumulated error decreases 36.33%; and
- maximum autonomous thermodynamic drift decreases 77.60%.

This supports the inference that recursive state feedback exposes and partly
suppresses the most damaging one-step thermodynamic compensation. It does not
restore direct physical accuracy: `J_M1` worsens by another 3.20%, and the
`S`, `Qv`, and `Qc` component errors all increase slightly.

The structural change is mixed, not monotone. H2 lowers the fixed-support
thermodynamic defect by 3.29% and the normalized manifold residual by 3.02%,
while its water-source defect rises from `7.4810e-10` to `4.2269e-8` (factor
56.5). The corresponding maximum accumulated water drift rises from
`1.5853e7` to `7.4106e9` (factor 467). Recursive H2 therefore shifts the
compensation among source directions; it does not discover conservation by
itself.

### 6.2 H2 to H5

H5 reduces its targeted `J_H5` by 20.5181%, but worsens `J_H1` by 34.45% and
`J_H2` by 12.60%. Its 80-step autonomous result degrades sharply:

| metric | H2 | H5 | H5/H2 |
|---|---:|---:|---:|
| final mixed error | 2.442689e-6 | 6.709918e-5 | 27.469 |
| maximum mixed error | 2.556093e-6 | 6.709918e-5 | 26.251 |
| accumulated mixed error | 2.041916e-6 | 3.098304e-5 | 15.174 |
| maximum learned `Qr_t` on autonomous states | 1.889549e-8 | 2.283861e-6 | 120.87 |
| maximum accumulated `S-beta2 Qv` drift | 2.985119e11 | 1.065464e13 | 35.69 |

The fixed truth-support diagnostics alone do not look uniformly worse: H5
reduces the thermodynamic defect by 19.79%, reduces normalized manifold
distance by 13.42%, and slightly improves `J_M1`. It increases the water-source
defect by 48.84%. Fixed-support `Qr_t` RMS is essentially unchanged, yet its
maximum on recursively generated autonomous states rises by two orders of
magnitude.

This distinction matters. The evidence does not support a simple statement
that every longer horizon produces a less physical source on truth states.
Rather, the H5 window loss admits a parameterization that is acceptable over
its reset five-step windows but extrapolates poorly after repeated departure
from truth. Its large autonomous rain-component tendency and thermodynamic
drift are evidence of horizon-specific/off-manifold compensation. Because H5
ended after only 100 accepted iterations at `MAXITER`, this is a result for the
tested curriculum endpoint, not the mathematical optimum of `J_H5`.

## 7. Full structural-discovery audit

### 7.1 Instantaneous fixed-support structure

| model | water defect RMS | beta defect RMS | `Qr_t` RMS | normalized manifold residual | vector cosine |
|---|---:|---:|---:|---:|---:|
| M1 | 5.093569e-10 | 7.233251e-8 | 3.207271e-10 | 1.579228e-4 | 0.9997241 |
| M2-X-independent | 1.521863e-7 | 3.945540e-4 | 6.683275e-8 | 7.149887e-1 | 0.7078345 |
| M1-to-M2-X | 1.372304e-9 | 1.924499e-5 | 5.517266e-10 | 3.518578e-2 | 0.9991123 |
| H1 | 7.480988e-10 | 1.535490e-5 | 1.371486e-10 | 2.806787e-2 | 0.9993915 |
| H2 | 4.226860e-8 | 1.485020e-5 | 1.394445e-10 | 2.721921e-2 | 0.9993652 |
| H5 | 6.291421e-8 | 1.191191e-5 | 1.397823e-10 | 2.356604e-2 | 0.9993576 |

For scale, the normalized M1 water, beta, and `Qr_t` defects are respectively
`1.12e-4 sigma_Q`, `1.62e-4 sigma_S`, and `7.04e-5 sigma_Q`. Direct
supervision therefore makes the independent network nearly recover all three
truth identities without enforcing them. The solver-facing fits retain high
global vector cosine because the dominant tendency direction is still
correct, but their orthogonal residuals are two orders of magnitude larger.

Independent M2-X is the extreme nonidentifiability/basin example: a deployed
objective of `5.87e-4` coexists with `J_M1 = 0.5115`, normalized manifold
residual `0.715`, and vector cosine `0.708`. Its component errors show that the
deployed map can be insensitive to badly wrong local source decompositions:

| model | `S_t` RMS error | `Qv_t` | `Qc_t` | `Qr_t` |
|---|---:|---:|---:|---:|
| M1 | 1.048975e-5 | 1.069726e-7 | 1.069704e-7 | 3.207271e-10 |
| M2-X-independent | 1.274133e-5 | 4.025403e-6 | 3.949335e-6 | 6.683275e-8 |
| M1-to-M2-X | 1.078060e-5 | 2.274861e-7 | 2.274451e-7 | 5.517266e-10 |
| H1 | 1.134026e-5 | 1.954799e-7 | 1.953940e-7 | 1.371486e-10 |
| H2 | 1.163182e-5 | 1.999932e-7 | 1.963827e-7 | 1.394445e-10 |
| H5 | 1.295674e-5 | 1.889149e-7 | 1.921608e-7 | 1.397823e-10 |

### 7.2 Autonomous drift and learned rain-component activity

| model | max total-water drift | max `S-beta2 Qv` drift | max autonomous `|Qr_t|` | steps with physically meaningful `Qr_t` |
|---|---:|---:|---:|---:|
| M1 | 1.036024e7 | 2.014820e9 | 1.165225e-8 | 80/80 |
| M2-X-independent | 1.137368e10 | 3.491959e13 | 1.279840e-6 | 80/80 |
| M1-to-M2-X | 1.592873e7 | 1.896523e11 | 6.793292e-8 | 80/80 |
| H1 | 1.585266e7 | 1.332899e12 | 1.893203e-8 | 80/80 |
| H2 | 7.410596e9 | 2.985119e11 | 1.889549e-8 | 80/80 |
| H5 | 3.616293e9 | 1.065464e13 | 2.283861e-6 | 80/80 |

The common initial integrals are `3.489327848442662e13` for total water and
`1.8113666780642352e17` for `S-beta2 Qv`. Relative to those scales, M1's
maximum drifts are about `2.97e-7` and `1.11e-8`; H1's are `4.54e-7` and
`7.36e-6`; H2's are `2.12e-4` and `1.65e-6`; and H5's are `1.04e-4` and
`5.88e-5`. These are state-integral diagnostics, distinct from the local
instantaneous source defects above.

Problem B has no analytical rain correction. “Rain activity” here means the
learned `Qr_t` component, whose truth is zero. All networks produce nonzero
and physically thresholded `Qr_t` at every autonomous step, although M1's and
H1/H2's amplitudes remain small. This is qualitatively different from Problem
A, where the analytical `R` remains exactly zero and `Qr_t` is structurally
zero.

### 7.3 Signed water budget: creation/destruction versus spurious rain

The post-hoc signed audit is
`external-results/test2a/problem-b/production/problem_b_signed_water_budget.json`;
its complete boundary/source time series is
`problem_b_signed_water_budget_timeseries.csv`. It replays the six immutable
final artifacts only for evaluation and exactly reproduces every previously
stored model total-water integral. No optimizer is constructed.

The accepted discrete integral is

\[
W_n=\operatorname{assemble}\!\left[(Q_v+Q_c+Q_r)\,dx\right].
\]

Truth has `W_0 = 3.489327848442662e13` and
`W_80 = 3.4893278484426562e13`. Its largest signed excursion in absolute
value is `-1.015625e-1` at step 2, only `2.910661e-15` of `W_0`. This is the
observed numerical conservation floor; truth `Qr` mass is identically zero.
Problem A imposes zero moist-source water defect by representation and uses
the same truth/numerical floor.

The following drift subtracts the already tiny truth drift. Positive values
mean artificial net water creation; negative values mean artificial net water
destruction.

| model | final signed drift | final drift / `W_0` | largest creation (step) | largest destruction (step) | final integrated `Qr` | maximum `|integrated Qr|` |
|---|---:|---:|---:|---:|---:|---:|
| M1 | -8.616389e5 | -2.469355e-8 | +1.036024e7 (43) | -8.616389e5 (80) | +3.624657e6 | 5.390387e6 |
| M2-X-independent | +1.137368e10 | +3.259561e-4 | +1.137368e10 (80) | none | -3.835109e9 | 3.835109e9 |
| M1-to-M2-X | +1.352095e7 | +3.874943e-7 | +1.592873e7 (66) | none | +3.336712e7 | 3.336712e7 |
| H1 | -1.585266e7 | -4.543185e-7 | none | -1.585266e7 (80) | -4.175607e6 | 4.175607e6 |
| H2 | +7.410596e9 | +2.123789e-4 | +7.410596e9 (80) | none | +8.183314e5 | 2.507459e6 |
| H5 | -2.687280e9 | -7.701425e-5 | none | -3.616293e9 (52) | +1.486404e9 | 1.486404e9 |

For each complete transition, the audit also extracts the actual child-6
post-prefix source retained in the certified step cache and evaluates

\[
\overline C_{water,n}=\int_\Omega
(Q_{v,t}^\theta+Q_{c,t}^\theta+Q_{r,t}^\theta)\,d\Omega
\]

with the accepted broken-CG3 GLL carrier mass weights. Direct Firedrake
assembly agrees with the weighted integral to at most `3.027e-9`.

| model | final applied `Cbar_water` | `dt sum Cbar_water` | `dt sum |Cbar_water|` | signed/absolute ratio |
|---|---:|---:|---:|---:|
| M1 | -6.503037e3 | -8.616390e5 | 2.187623e7 | 0.0394 |
| M2-X-independent | +1.588609e6 | +1.137368e10 | 1.137368e10 | 1.0000 |
| M1-to-M2-X | -3.971649e3 | +1.352095e7 | 1.833652e7 | 0.7374 |
| H1 | -5.996319e3 | -1.585266e7 | 1.585266e7 | 1.0000 |
| H2 | +1.053595e6 | +7.410596e9 | 7.410596e9 | 1.0000 |
| H5 | +4.148488e5 | -2.687280e9 | 4.545307e9 | 0.5912 |

The cumulative source integral differs from final truth-relative state drift
by at most `5.2e-2`, below the truth conservation floor. Prefix transport and
moist mass-solve operation-order residuals therefore do not explain these
budgets; they are the integrated learned source defects.

The signed interpretation sharpens the earlier RMS result:

- **M1:** local/spatial source errors cancel strongly. Its final small net
  destruction is only 3.94% of its time-integrated absolute source defect.
  Its nonzero `Qr` is mainly an unphysical phase partition, separate from the
  much smaller final total-water loss.
- **H1:** every globally integrated applied defect is negative. H1 therefore
  destroys water systematically, while also ending with an unphysical
  negative `Qr` mass.
- **H2:** its 56.5-fold larger fixed-support RMS water defect does **not**
  cancel globally. The applied global defect is positive at every step, and
  H2 creates `7.411e9` water units (`2.124e-4` of initial water). Its final
  `Qr` mass is only `8.18e5`, so the bias is predominantly net creation, not
  redistribution into rain.
- **H5:** positive and negative source defects partially cancel, but leave
  net destruction of `2.687e9`. Simultaneously, `1.486e9` is placed in the
  nonexistent rain component. H5 therefore exhibits both large spurious
  phase partition and large total-water destruction; spurious rain alone
  does not account for its budget.

### 7.4 Does a better state objective systematically mean worse structure?

No single monotone relationship is supported:

- M1 to H1 strongly supports the compensation mechanism: `J_H1` improves
  while thermodynamic/manifold defects and autonomous behavior worsen.
- H1 to H2 improves the recursive and autonomous objectives and slightly
  improves thermodynamic/manifold defects, but greatly worsens water
  conservation.
- H2 to H5 improves `J_H5` and the fixed-support thermodynamic/manifold
  measures, while worsening water defect and off-manifold autonomous rain and
  thermodynamic drift.
- Independent M2-X shows that optimization basin and deployed-map weak
  directions can coexist; its endpoint should not be interpreted as the
  objective's scientific optimum.

The general finding is therefore underdetermination, not a universal negative
correlation. Different state objectives can exploit different combinations of
the four source directions unless the representation or additional physical
information makes those directions identifiable.

## 8. Matched Problem A versus Problem B

### 8.1 Common dense state objectives

The A/B dense objectives below use the same state metric, target boundaries,
window schedules, and denominator, so they are comparable.

| stage | representation | `J_H1` | `J_H2` | `J_H5` |
|---|---|---:|---:|---:|
| M1 | structured A | 7.048714e-4 | 1.120554e-3 | 2.250633e-3 |
| M1 | black-box B | 5.169328e-4 | 8.572764e-4 | 1.878852e-3 |
| H1 | structured A | 4.510806e-4 | 7.339235e-4 | 1.511100e-3 |
| H1 | black-box B | 3.984244e-4 | 7.660054e-4 | 2.710693e-3 |
| H2 | structured A | 4.524130e-4 | 7.314762e-4 | 1.498248e-3 |
| H2 | black-box B | 4.231610e-4 | 6.962490e-4 | 2.039038e-3 |
| H5 | structured A | 4.606034e-4 | 7.356325e-4 | 1.484799e-3 |
| H5 | black-box B | 5.689359e-4 | 7.839912e-4 | 1.620667e-3 |

Black-box M1 is not intrinsically inaccurate: it has lower dense state losses
than structured M1 on all three horizons. Black-box H1 also attains an 11.67%
lower targeted H1 loss than structured H1, and black-box H2 attains a 4.82%
lower targeted H2 loss than structured H2. These facts make the autonomous
comparison especially informative: the issue is not failure to reduce the
trained state loss.

### 8.2 Autonomous behavior

| stage | representation | final | maximum | accumulated |
|---|---|---:|---:|---:|
| M1 | structured A | 3.763657e-7 | 6.215496e-7 | 4.692863e-7 |
| M1 | black-box B | 3.680097e-7 | 9.214634e-7 | 5.408812e-7 |
| H1 | structured A | 3.883592e-7 | 5.022919e-7 | 4.229449e-7 |
| H1 | black-box B | 6.205057e-6 | 6.205057e-6 | 3.206946e-6 |
| H2 | structured A | 3.861652e-7 | 5.022232e-7 | 4.210796e-7 |
| H2 | black-box B | 2.442689e-6 | 2.556093e-6 | 2.041916e-6 |
| H5 | structured A | 3.800073e-7 | 4.962602e-7 | 4.147441e-7 |
| H5 | black-box B | 6.709918e-5 | 6.709918e-5 | 3.098304e-5 |

Problem-B M1 is indeed of the same order as structured M1: its final error is
2.22% lower, while its maximum and accumulated errors are 48.25% and 15.26%
higher. Direct full-source supervision is sufficient for the black-box model
to approximately discover the manifold on this dense no-rain support.

After solver-facing training, the representations separate sharply. Relative
to structured Problem A, black-box H1 has 15.98/12.35/7.58 times the
final/maximum/accumulated errors; H2 has 6.33/5.09/4.85 times; and H5 has
176.6/135.2/74.7 times. Problem A's exact source structure blocks the
compensating directions diagnosed above, so deployment-aware optimization can
reweight the scalar law without violating water, thermodynamic coupling, or
rain structure.

### 8.3 KE, enstrophy, physical law, and invariants

| stage | representation | KE final / max | enstrophy final / max |
|---|---|---:|---:|
| M1 | structured A | 3.674e-8 / 3.674e-8 | 4.368e-8 / 4.945e-8 |
| M1 | black-box B | 4.469e-8 / 4.469e-8 | 4.255e-8 / 1.221e-7 |
| H1 | structured A | 1.341e-8 / 1.341e-8 | 4.287e-8 / 5.603e-8 |
| H1 | black-box B | 3.186e-6 / 3.186e-6 | 2.440e-6 / 2.440e-6 |
| H2 | structured A | 1.024e-8 / 1.251e-8 | 4.221e-8 / 4.813e-8 |
| H2 | black-box B | 2.644e-6 / 2.644e-6 | 2.427e-6 / 2.427e-6 |
| H5 | structured A | 6.277e-9 / 1.706e-8 | 3.275e-8 / 4.572e-8 |
| H5 | black-box B | 4.325e-7 / 8.363e-7 | 1.404e-6 / 1.404e-6 |

Problem A's scalar relative RMS(A) is 0.01931, 0.02441, 0.02454, and
0.02480 at M1, H1, H2, and H5 [PA-HC]. Problem B's normalized four-output
relative norms, `sqrt(J_M1)`, are 0.02349, 0.03799, 0.03859, and 0.03805.
These numbers describe analogous constitutive-error trends but are not the same
mathematical metric and must not be ranked directly.

The structural comparison is exact. Every Problem-A autonomous source has
water residual zero, maximum `S_t-beta2 Qv_t` residual at most
`1.734723475976807e-18`, and analytical `R = 0` on all 80 steps [PA-HC].
Problem B has the nonzero defects and rain-component activity in Section 7.
Conservation/source structure is therefore a representation-design decision
with direct optimization and extrapolation consequences, not merely an
optional diagnostic penalty.

## 9. Scientific interpretation

The completed evidence supports the following statement:

> In the structured representation, deployment-consistent objectives reweight
> one physical scalar without opening unphysical source directions. In the
> black-box representation, deployed state objectives are underdetermined with
> respect to the four local tendencies. They can reduce the selected short-
> horizon state loss through compensating source directions that direct
> physical supervision suppresses, and those compensations need not remain
> benign under long autonomous recursion.

This is more precise than “deployment-aware training is beneficial only when
constrained.” Deployment-aware information is still real in Problem B: H1
reduces H1, H2 reduces H2 and partially repairs H1's autonomous defect, and H5
reduces H5. The problem is identifiability. A state metric observes the image
of four source fields through weak assembly, mass inversion, and finite-time
dynamics; it does not uniquely demand the physical decomposition. The exact
source manifold removes those directions a priori.

Direct M1 supervision acts as a strong physical regularizer in Problem B. It
nearly discovers water conservation, thermodynamic coupling, and zero rain
from dense local labels, and it produces the best autonomous Problem-B model.
H2 recursive feedback partially suppresses the H1 thermodynamic compensation,
but transfers error into water-conservation directions. H5 demonstrates that
a longer trained horizon does not necessarily improve an 80-step rollout:
reset-window accuracy, off-truth-manifold stability, and constitutive
identifiability remain distinct.

“Horizon-specific compensating physics” is therefore a supported inference,
not a theorem. Its evidence is the simultaneous reduction of the selected
state objective, worsening of nonselected horizon/autonomous metrics, movement
among structural defects, and large H5 off-manifold `Qr_t` activation. The
limited 100-iteration recursive budgets and absence of held-out evaluation
preclude a global or generalization claim.

## 10. Problem-B FIML decision

A black-box Problem-B FIML campaign would answer a genuinely new secondary
question: whether flexible field inversion followed by compression into one
shared four-output function acts as an implicit constitutive regularizer and
damps the compensating directions seen in direct H5. Problem-A FIML suggests
that this can happen, but does not answer it for an unconstrained source
vector.

Recommendation: **RUN LATER as a targeted regularization/identifiability
ablation, not as a missing core rung and not before the rain-active Test 2B
core comparison.** The five completed Problem-B methods already establish the
main representation effect, so Problem-B FIML is not required to freeze this
no-rain study. If run later, it should report both raw field-inversion
nonuniqueness and whether Stage-2 compression recovers or violates the source
manifold; symmetry with Problem A is not sufficient motivation.

## 11. Established findings, interpretations, and limitations

### Certified/established

- All six final artifacts and PB are fingerprint-verified and complete.
- M2-X has severe basin sensitivity: the independent endpoint is worse than
  M1 under M2-X, while M1-to-M2-X reaches a lower M2-X value.
- H1 is fixed-state/cacheable; H2 is the first recursive objective.
- M1 nearly discovers the no-rain source manifold without architectural
  enforcement.
- H1 reduces its target while greatly increasing thermodynamic/manifold defect
  and autonomous error.
- H2 partially repairs autonomous behavior but increases water-source defect.
- H5 reduces its target while severely degrading the 80-step autonomous
  trajectory and activating large off-manifold `Qr_t`.
- Structured Problem A preserves its source identities exactly/at roundoff and
  remains autonomous at order `1e-7` across M1/H1/H2/H5; black-box Problem B
  does not after solver-facing training.

### Supported interpretations

- Direct physical supervision is an effective constitutive regularizer for
  the black-box output space.
- Solver-facing state objectives expose weak/compensating source directions;
  the particular compensation changes with horizon.
- H2 recursive information suppresses some, but not all, damaging H1
  compensation.
- Correct conservation/coupling is consequential representation design, not
  simply a post-hoc property or an optional soft penalty.

### Limitations/not established

- No fit reached demonstrated mathematical stationarity; all ended at
  `MAXITER`.
- The independent M2-X endpoint cannot be used to rank the M2-X objective
  because of basin confounding.
- No result is held-out generalization or rain-active evidence.
- Only one seed, architecture family, mesh, normalization, and set of budgets
  was tested.
- The four-output model has 1,380 parameters versus Problem A's 1,281; the
  ablation holds inputs and hidden width fixed but cannot hold output dimension
  or parameter count identical.
- No soft-conservation, projected-output, alternative weighting, or Problem-B
  FIML ablation was performed.

## 12. Frozen no-rain Problem-B conclusions

1. **Removing the source manifold does not prevent accurate direct learning.**
   Dense M1 source labels make the black-box network approximately discover
   the correct structure and yield autonomous accuracy comparable in order to
   structured M1.

2. **The representation changes what deployed objectives identify.** H1 can
   lower the exact one-step state objective while breaking thermodynamic
   coupling and worsening autonomous behavior by factors of 6–17. This is not
   observed in structured Problem A.

3. **Recursion supplies new information but not automatic physical
   regularization.** H2 materially repairs H1's rollout while shifting error
   into water-conservation directions. H5's longer objective improves its
   reset-window target but produces a much worse 80-step trajectory.

4. **The practical value in Problem A depended on both objective design and
   representation design.** Deployment-aware training was useful because the
   structured scalar output made the inverse problem sufficiently identified;
   the same state objectives over independent tendencies admit compensation.

5. **Problem B is frozen for its stated no-rain, training-support scope.** Its
   principal result is not simply that “structured is better,” but that dense
   physical supervision can discover structure whereas state-only deployed
   losses do not uniquely preserve it.

## 13. Exact handoff to rain-active Test 2B

Test 2B should compare, under matched rain-active states and objectives:

1. structured `A_theta` with analytical `R`;
2. structured `(A_theta,R_theta)` with the exact conservative source map; and
3. an independent four-tendency black-box model.

The most important questions are now:

- Does learning `R` as a scalar rate retain the stability benefit of the
  structured source map while removing the analytical-rain advantage?
- Can direct four-output supervision still discover conservation and the
  coupled `(A,R)` source geometry when `Qr_t` is nonzero and multiple physical
  directions are active?
- Do M2-X/H1/H2 improve the two structured representations without opening the
  compensation seen here, and do they again destabilize the black-box model?
- Does an H2 improvement reflect rain-state feedback or merely a different
  local source decomposition?
- How do spurious precipitation, total-water drift, thermodynamic drift, and
  off-truth-manifold rain activation evolve under autonomous deployment?

The no-rain five-input decision must not be inherited automatically. `Qr`
becomes informative when rain is active, so Test 2B must freeze its input
contract independently and match it across representations. Output scaling
must also use a nonzero rain-active scale rather than the no-rain workaround.

Both independent and M1-warm-start deployed fits should be retained to expose
optimizer-basin effects. Autonomous diagnostics must remain post hoc, and a
longer horizon must not be assumed superior. Problem-B FIML should follow only
after these three core rain-active representations establish whether
amortization addresses a new identifiability problem.

**Freeze decision:** Problem B is complete for its independent-four-tendency,
no-rain, states-0-through-80 scope. The optimizer, stationarity, and held-out
limitations above are part of the frozen conclusion rather than unresolved
artifact conflicts.
