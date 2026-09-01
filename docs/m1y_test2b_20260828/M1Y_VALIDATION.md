# Test-2B M1-Y pre-training validation

Status: **PASSED**.

## Historical M1-X production path

The trace is from executable code and frozen configuration, not synthesis
prose:

1. `dimswe/configs/test2b_rain_active_learning.json` freezes truth support
   0..80, feature order `(h,S,Qv,Qc,B)`, the A/B/C architectures, seed 0,
   output scaling conventions, and PyROL line-search L-BFGS.
2. `dimswe/test2b_rain_learning_campaign.py::prepare_data` loads restart
   states 0..80 as timestep-boundary truth states and calls
   `_analytical_arrays` directly on those states to produce `x_features`,
   `x_A`, and `x_R`.
3. `_analytical_arrays` calls the production `JAXMoistEulerPrimal` adapter on
   exactly the state passed to it.  It packs `(h,S,Qv,Qc,B)` in that order and
   evaluates analytical A and R at the same state.
4. `objectives` maps historical stage `M1` to `OperatorObjective` using
   `x_features`, `x_A`, and `x_R`.  No prefix is invoked in this M1 path.
5. `train` passes that fixed array objective to
   `CompactCheckpointObjective` and PyROL.  Hence historical M1 is
   unambiguously M1-X: both input and target use `X_n*`.

The separate H1 preparation path in `prepare_data` calls

`case.helper.take_forward_step_cached(X_n*, n*dt, dt).boundary_states[-2]`

for starts 0..79, then evaluates features and analytical rates on the
returned state.  The six children are dry RK4 stage 0, dry RK4 stage 1,
hyperviscosity Euler, DG SSPRK43 stage 0, DG SSPRK43 stage 1, and moist Euler;
boundary `-2` is therefore the state after the complete five-child prefix and
before moist Euler.

## M1-Y support and cache parity

The new preparation replays the same prefix for all historical M1 support
indices 0..80 inclusive.  It retains state 80 even though the historical H1
cache ended at rollout start 79.  This produces 81 x 65,536 = 5,308,416 fixed
offline samples.

For states 0..79, regenerated values were compared with the frozen H1 cache:

| Array | Maximum absolute difference | Maximum relative difference | Bitwise equal |
|---|---:|---:|---|
| normalized `(h,S,Qv,Qc,B)` | 0 | 0 | yes |
| analytical A | 0 | 0 | yes |
| analytical R | 0 | 0 | yes |

The flat-case B column is exactly zero throughout.  Qr is retained only as an
evaluation diagnostic for rain-activity tolerances and is not a network
input.

## Frozen normalization and objectives

The input normalization is loaded verbatim from
`fixed_learning_data.npz/.json`; no statistic is computed from Y.  Its
provenance fingerprint is
`794e074b2d3149f58025a7e6a74856374d86adab1e3ee518a64fe6f30ff0dd79`.
Historical A, active-R, and four-source output scales are also reused without
refitting.

The resulting M1-Y normalized-target denominators are:

| Representation | Denominator |
|---|---:|
| A | 2.026212938570984e15 |
| B | 2.0412968811274818e15 |
| C | 6.093310863157332e15 |

At seed zero, A/B/C objectives and gradients were finite.  Deterministic
directional finite-difference relative discrepancies were respectively
`5.64e-9`, `5.81e-8`, and `2.66e-9`.

## Independent numerical spot checks

Four stored samples were independently reconstructed from their restart
state, prefix replay, production feature interpolation, and a separate NumPy
transcription of the analytical moist law:

| Regime | step | sample | A*(Y) | R*(Y) | max feature discrepancy | target discrepancy |
|---|---:|---:|---:|---:|---:|---:|
| condensation-active | 0 | 25092 | -8.528669963488885e-7 | 0 | 0 | 0 |
| evaporation-active | 27 | 58559 | 3.244394353519655e-8 | 0 | 0 | 0 |
| near-inactive | 71 | 59478 | -2.484059191415694e-19 | 0 | 0 | 0 |
| active rain | 80 | 45011 | -2.7982333103579403e-9 | 5.0102933632508554e-11 | 0 | 0 |

The independently assembled Representation-C source target also agreed
exactly in production order `(S,Qv,Qc,Qr)` for every spot check.

## Offline/nonrecursive gate

M1-Y preparation has no neural-parameter argument.  Its materialized
`m1y_features`, `m1y_A`, `m1y_R`, and carrier weights are the only arrays
captured by the training objective.  There is no neural rollout and no
differentiation through P.  All 13 state, target, normalization, support,
initialization, and finiteness gates in
`external-results/m1y-test2b-20260828/preparation/pretraining_validation.json`
passed.
