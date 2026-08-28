# Test2B B versus B_TP versus B_TPL: M1 preparation

**Preparation date:** 2026-08-19
**Status:** both structural variants prepared; no production launched

## 1. Focused scientific comparison

The first follow-up to frozen rain-active Representation B is an M1-only,
one-factor-at-a-time comparison:

1. `B`: unrestricted signed learned `R`;
2. `B_TP`: known threshold plus nonnegative sign;
3. `B_TPL`: known threshold, nonnegative sign, and an imposed linear
   normalized exceedance factor.

This separates the minimal threshold/positivity prior from the stronger
linear-onset prior.  All three use the same truth, samples, weights,
normalization, features `(h,S,Qv,Qc,B)`, `5 -> 32 -> 32 -> 2` float64 tanh
network, seed-zero pytree, 1,314 parameters, M1 objective, PyROL L-BFGS
settings, and 10,000-accepted-iteration cap.  Only the physical map of the
raw rain head differs.

After both M1 fits finish, all three final artifacts will receive the same
post-hoc autonomous 160-step evaluation.  No M2-X/H1/H2/H5 fit is authorized
until that comparison is reviewed.

## 2. Exact maps

Both variants retain

\[
A_\theta=\sigma_A a_{\rm raw},\qquad
(S_t,Q_{v,t},Q_{c,t},Q_{r,t})
=h(\beta_2 A_\theta,A_\theta,-A_\theta-R,R).
\]

Consequently, both retain exact structural total-water conservation and the
exact thermodynamic source identity.

### 2.1 B_TP: threshold and positivity only

For `qc=Qc/h`,

\[
R_{TP}=\begin{cases}
0, & q_c\le q_{\rm precip},\\
\sigma_R\operatorname{softplus}(r_{\rm raw})/\log 2,
& q_c>q_{\rm precip}.
\end{cases}
\]

The implementation is a hard JAX `where`.  It is exactly zero on and below
the threshold, nonnegative above it, and intentionally discontinuous at the
gate.  It contains no `delta_q_scale`, exceedance multiplier, analytical
`gamma_r`/`tau_r`, analytical coefficient, or imposed onset law.  Derivatives
are certified separately on the two smooth sides; no finite difference is
taken at the gate.

### 2.2 B_TPL: threshold, positivity, and linear exceedance

The implementation formerly described generically as `BPLUS` is
scientifically B_TPL:

\[
R_{TPL}=\sigma_R
\max\left(0,{q_c-q_{\rm precip}\over\Delta q_{\rm scale}}\right)
{\operatorname{softplus}(r_{\rm raw})\over\log 2}.
\]

`BTPL` is the canonical CLI identifier.  `BPLUS` remains a backward-compatible
legacy identifier so the already generated preparation and certification
artifacts remain valid.  B_TPL's frozen mass-weighted positive-exceedance RMS
is `1.9902871261559997e-6`.  B_TP does not use this scalar.

## 3. Immutable preparation records

| variant | preparation file SHA256 | payload SHA256 |
|---|---|---|
| B_TP | `b7d960c340957d211753214914743e3d825d1c889807670c6ed1d267e1d8b4cb` | `298bc9043129b4984387305a41b50fff07524af0186f41014c6b8fc91004e1c9` |
| B_TPL (legacy BPLUS artifact) | `2e898bb4ff130b89039fee2c39ccea89a6085dcf4ab6af6609ce67a4f867acfa` | `2623b81cda5b9f0b7e6dec0b54db72eee9b6c2d45c8756028d85244726778842` |

B_TP metadata explicitly records `delta_q_scale_used=false` and
`linear_exceedance_factor_used=false`.  Both records bind the unchanged fixed
training cache with SHA256
`6e159015234fd94881b0b97888b7481eb049a02dcd96c571078920c0bedc901c`.
Neither record accesses held-out states.

All B, B_TP, B_TPL, and legacy-BPLUS seed-zero pytrees have SHA256
`cfadd9f3ee02a78c5b3a946b88c039d9f7ed34e719325ff22c92e1fe4afac056`.
The A-head output is bitwise identical at common parameters and state.

## 4. B_TP certification

The evaluation-only B_TP certificate has SHA256
`c68ad6d27d60b39d04ba92f283bc2efe1aeecaccc2c44b366bb01e1ee9721b03`.
It records no optimizer, no production training, no truth generation, and no
held-out access.

### 4.1 Local map and source structure

- below-threshold maximum `|R_TP|`: exactly `0`;
- at-threshold maximum `|R_TP|`: exactly `0`;
- above-threshold minimum `R_TP`: `2.1261551623581708e-11`;
- fixed-raw `qc` derivative below threshold: exactly `0`;
- fixed-raw `qc` derivative above threshold: exactly `0`;
- parameter tangent/adjoint relative discrepancy: `2.21e-16`;
- local parameter directional-FD relative discrepancy: `8.36e-7`;
- water-source maximum residual: `1.14e-22`;
- thermodynamic-source maximum residual: `4.34e-19`;
- A head bitwise equal to B: true.

The B_TP discontinuity is an accepted architectural choice.  No derivative
claim is made at `qc=qprecip`.

### 4.2 Objective and trajectory derivatives

| certificate | relative/absolute result |
|---|---:|
| active-state M1 directional derivative | `1.26e-10` relative |
| active-state M2-X directional derivative | `9.18e-12` relative |
| H1 cache/literal value | `9.87e-16` absolute |
| H1 cache/literal all-parameter gradient | `5.47e-13` relative |
| H1 state tangent/adjoint | `0` relative |
| H1 parameter directional derivative | `1.76e-9` relative |
| H2 state tangent/adjoint | `1.35e-16` relative |
| H2 parameter directional derivative | `1.20e-9` relative |
| H5 state tangent/adjoint | `0` relative |
| H5 parameter directional derivative | `2.40e-9` relative |

The same complete six-child implementation and fixed/recursive objectives
used by B are retained.  These H1/H2/H5 checks certify future compatibility;
they do not authorize production beyond M1.

The existing B_TPL certificate remains unchanged at SHA256
`341bcbd4313cab4e7fc275815cd5ed0437b0b8e11875fb0e397263045fc45ee2`.

## 5. M1-only guarded runner

The common runner accepts exactly one of `BTP` or `BTPL`, creates a separate
immutable production root, refuses an existing root, checks the appropriate
output-map preparation, and launches only the frozen-B M1 stage with a
10,000-accepted-iteration cap.

### B_TP

```bash
cd /Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615
mkdir -p external-results/test2b-rain-active-learning/logs
nohup caffeinate -i bash scripts/run_test2b_rain_learning_btp_btpl_m1.sh BTP \
  > external-results/test2b-rain-active-learning/logs/representation-BTP-M1.log 2>&1 &
```

### B_TPL

```bash
cd /Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615
mkdir -p external-results/test2b-rain-active-learning/logs
nohup caffeinate -i bash scripts/run_test2b_rain_learning_btp_btpl_m1.sh BTPL \
  > external-results/test2b-rain-active-learning/logs/representation-BTPL-M1.log 2>&1 &
```

Neither command was executed during preparation.

## 6. Decision gate after M1

The first scientific decision will use the matched B/B_TP/B_TPL M1 fits and
the same autonomous 160-step evaluation to isolate:

- the effect of threshold and sign alone (`B -> B_TP`);
- the incremental effect of the imposed linear onset factor
  (`B_TP -> B_TPL`).

Only after this comparison will one structured variant, if warranted, be
advanced through M2-X/H1/H2/H5.

**Readiness:** `TEST2B_BTP_BTPL_M1_COMPARISON_PREPARED`.
