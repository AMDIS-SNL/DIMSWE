# Test2B Representation B_TPL (legacy BPLUS): preparation and certification

**Preparation date:** 2026-08-19

**Status:** prepared and derivative-certified; production not launched

## 1. Scientific question

Does supplying the known precipitation threshold, nonnegative rain sign, and
an imposed linear normalized exceedance factor repair frozen Representation
B's deployed rain behavior while retaining flexibility in the learned rain
amplitude?

The implementation originally called `BPLUS` is now classified scientifically
as **B_TPL**: threshold + positivity + linear exceedance factor.  `BPLUS`
remains a backward-compatible CLI/artifact identifier, while `BTPL` is the
canonical new CLI name.  This distinction is necessary because the map does
more than impose threshold and positivity.  The existing preparation and
certification artifacts below remain immutable and are not relabeled.

B_TPL is a controlled ablation of frozen Representation B.
The truth, fixed training cache, support split, inputs, feature normalization,
rate scaling, architecture, initial parameters, optimizer family, tolerances,
objectives, schedules, iteration budgets, finite-element solver, timestep, and
rollout horizons are unchanged.  Only the physical map applied to the raw
second network output changes.  Frozen A/B/C artifacts and syntheses are not
modified.

## 2. Exact output parameterization

The common local inputs remain

\[
x=(h,S,Q_v,Q_c,B),
\]

and the float64 seed-0 network remains `5 -> 32 -> 32 -> 2`, tanh hidden
activations, with 1,314 trainable parameters.  Let its dimensionless outputs
be `(a_raw,r_raw)`.  The A head is exactly the B head,

\[
A_\theta=\sigma_A a_{\rm raw}.
\]

For

\[
q_c=Q_c/h,\qquad \Delta q=q_c-q_{\rm precip},
\]

B_TPL defines

\[
R_+=\sigma_R
\max\left(0,\frac{\Delta q}{\Delta q_{\rm scale}}\right)
\frac{\operatorname{softplus}(r_{\rm raw})}{\log 2}.
\]

The hard `max` is deliberate: `R_+=0` exactly for
`q_c<=q_precip`, is nonnegative for every parameter vector, and approaches
zero continuously from above.  It is not smoothed.  No analytical
`gamma_r`, `tau_r`, rain amplitude, or analytical rain-law correction appears.
Above threshold the network still determines the amplitude through `r_raw`.

The physical source map remains exactly

\[
(S_t,Q_{v,t},Q_{c,t},Q_{r,t})
=h(\beta_2A_\theta,A_\theta,-A_\theta-R_+,R_+).
\]

Therefore, for arbitrary parameters and states,

\[
Q_{v,t}+Q_{c,t}+Q_{r,t}=0,
\qquad S_t-\beta_2Q_{v,t}=0
\]

to ordinary float64 operation-order roundoff.

## 3. Frozen conditioning scale

The authoritative output-map metadata is
`external-results/test2b-rain-active-learning/preparation/representation_bplus_output_map.json`
(file SHA256
`2e898bb4ff130b89039fee2c39ccea89a6085dcf4ab6af6609ce67a4f867acfa`,
payload SHA256
`2623b81cda5b9f0b7e6dec0b54db72eee9b6c2d45c8756028d85244726778842`).

Using only boundary truth states `0..80` from the immutable
`fixed_learning_data.npz` cache, define

\[
\Delta q_{\rm scale}=
\left[
\frac{\sum_{k,i:\Delta q_{ki}>0}w_i\Delta q_{ki}^2}
     {\sum_{k,i:\Delta q_{ki}>0}w_i}
\right]^{1/2}.
\]

The frozen value is

\[
\boxed{\Delta q_{\rm scale}=1.9902871261559997\times10^{-6}}.
\]

There are 40,800 positive-exceedance samples among 5,308,416 deployed GLL
samples.  The weights are the accepted carrier mass weights.  No held-out
state was read.  The fixed cache SHA256 is
`6e159015234fd94881b0b97888b7481eb049a02dcd96c571078920c0bedc901c`.
The unchanged scales are
`sigma_A=9.052258655848717e-8` and
`sigma_R=1.9902871261559996e-11`.

The RMS positive exceedance is preferred over a robust quantile because it is
positive, finite, mass-weighted in the already accepted metric, uses the same
active training support as `sigma_R`, and makes a zero raw rain coordinate
produce `R_+=sigma_R` at one RMS exceedance.  No alternative scale was needed.

## 4. Controlled-ablation and initialization audit

`RainMLPConfiguration("B")`, `RainMLPConfiguration("BPLUS")`, and
`RainMLPConfiguration("BTPL")` all have
layer dimensions `(5,32,32,2)` and 1,314 parameters.  Every seed-zero pytree
leaf is bitwise equal.  Both initial pytrees have SHA256

`cfadd9f3ee02a78c5b3a946b88c039d9f7ed34e719325ff22c92e1fe4afac056`.

The frozen-B provider path retains its accepted multiply-by-`(sigma_A,
sigma_R)` output map.  A deterministic regression reconstructs that prior map
independently and compares all four source arrays exactly; it passes.  The
BPLUS A output is also bitwise equal to B's A output for identical parameters
and state.

BPLUS uses the existing Test2B configuration and fixed NPZ rather than a new
dataset.  The BPLUS metadata loader recomputes the positive-exceedance scale
and checks the fixed NPZ/sidecar hashes, scale values, parameter count, seed
fingerprint, and metadata payload before constructing any objective.

## 5. Objective ladder and runner semantics

The objectives are mathematically identical to frozen B after replacing only
`R_theta` with `R_+`:

- `M1`: direct normalized `(A,R)` regression on boundary truth states `0..80`;
- `M2-X`: fixed boundary-state deployed source-map regression;
- `H1`: fixed post-prefix one-step truth-reset objective at `Y=P(X*)`;
- `H2`: dense accumulated recursive horizon two;
- `H5`: dense accumulated recursive horizon five.

The M1 denominator and target scaling remain those of B.  M2-X/H1/H2/H5 use
the identical state metric, denominators, nonoverlapping schedules, and target
indices.  H1 remains exactly cacheable; H2 remains the first objective with
model-generated-state feedback.

The guarded BPLUS runner creates the distinct root
`external-results/test2b-rain-active-learning/production/representation-BPLUS/`
and refuses an existing root.  It records the BPLUS output-map payload
fingerprint in every parameter sidecar and progress/final record.  Every stage
starts a new ROL process with empty L-BFGS history.  The frozen-B caps and
continuation graph are retained:

| stage | initialization | accepted-iteration cap |
|---|---|---:|
| M1 | common seed 0 | 10,000 |
| M2-X-independent | common seed 0 | 10,000 |
| M1-to-M2-X | BPLUS M1 final | 5,000 |
| H1 | BPLUS M1 final | 5,000 |
| H2 | BPLUS H1 final | 20 |
| H5 | BPLUS H2 final | 20 |

Based on frozen B, a rough serial reference is about 31 hours, but BPLUS
line-search behavior and the extra threshold map can change that.  This is not
a runtime promise.

## 6. Numerical certification

The authoritative evaluation-only certificate is
`external-results/test2b-rain-active-learning/preparation/representation_bplus_certification.json`
(SHA256
`341bcbd4313cab4e7fc275815cd5ed0437b0b8e11875fb0e397263045fc45ee2`).
It records `optimizer_instantiated=false`,
`production_training_launched=false`, `truth_generated=false`, and
`heldout_accessed=false`.  The deployed active-rain window begins at step 60.

### 6.1 Local physics

- below-threshold maximum `|R_+|`: exactly `0`;
- at-threshold maximum `|R_+|`: exactly `0`;
- above-threshold minimum `R_+`: `2.137679775652601e-11`;
- below-threshold state derivative: exactly `0`;
- above-threshold derivative versus analytical formula: relative error `0`;
- local parameter tangent/adjoint relative discrepancy: `0`;
- local parameter directional-FD relative discrepancy: `8.57e-7`;
- water-source residual: `2.08e-22` maximum;
- thermodynamic residual: `4.34e-19` maximum;
- A head bitwise equal to B: true.

Finite differences are not asserted at `qc=qprecip`, where the intended hard
threshold is nondifferentiable.

### 6.2 Fixed and trajectory objectives

| certificate | result |
|---|---:|
| active-state M1 directional-gradient relative error | `1.05e-10` |
| active-state M2-X directional-gradient relative error | `4.48e-11` |
| H1 cached/literal value absolute difference | `9.86e-16` |
| H1 cached/literal all-parameter gradient relative error | `5.46e-13` |
| H1 state tangent/adjoint relative error | `0` |
| H1 parameter directional-gradient relative error | `1.77e-9` |
| H2 state tangent/adjoint relative error | `1.35e-16` |
| H2 parameter directional-gradient relative error | `1.37e-9` |
| H5 state tangent/adjoint relative error | `1.35e-16` |
| H5 parameter directional-gradient relative error | `2.63e-9` |

These checks use the existing complete six-child trajectory implementation;
no second solver or approximate tangent was introduced.  The hard threshold
is differentiated by JAX away from the kink.

## 7. One-factor-at-a-time study

B_TPL and the separately prepared B_TP are two competing forms of the first
of five planned independent interventions against frozen B:

1. threshold/positivity structure — B_TP, with B_TPL as the separate
   linear-onset comparison;
2. additional rainy truth coverage;
3. rain/onset-balanced sampling or weighting;
4. network-capacity increase;
5. model-visited-state or rollout-data augmentation.

Items 2--5 are not implemented here.  They should first be tested separately,
not cumulatively, so their effects remain attributable.

## 8. Historical launch sketches

The unaccepted M1-only BTP/BTPL launcher and the unexecuted full-ladder B+
launcher are retained under
`archive/development-history/test2b_constrained_rain_variants/`. They are not
collaborator-facing reproduction entry points because neither establishes a
canonical completed B+ campaign. Reinstating or running either requires a new,
explicit scientific authorization.

**Readiness:** `TEST2B_REPRESENTATION_B_TPL_M1_PREPARED`.
