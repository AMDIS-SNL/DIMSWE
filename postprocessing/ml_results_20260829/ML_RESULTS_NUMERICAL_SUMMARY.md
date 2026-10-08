# Machine-Learning Results numerical summary

This summary states what the frozen artifacts and evaluation-only calculations
show. It does not attempt a Discussion, causal attribution, or novelty claim.

## Validation basis

- Test 2B held-out X extraction covers exactly states 81--160, 80 states with
  65,536 samples per state. The normalized feature array has shape
  `(80, 65536, 5)` in production order `(h,S,Qv,Qc,B)`; analytical source
  targets have shape `(80, 65536, 4)` in order `(S,Qv,Qc,Qr)`. All 80 restart
  hashes matched the immutable truth manifest.
- Historical X-fitted input normalization and output scales were retained; no
  held-out or Y-dependent refit was performed. On states 0--80, the extracted
  normalized X features are bitwise identical to the accepted cache. Analytical
  A and R agree to maximum absolute differences `6.83e-20` and `6.46e-27`.
- The checkpoint catalog contains 174 references spanning all 21 main Test 2B
  runs and 156 unique parameter arrays. Across 960 accepted final direct-metric
  comparisons, every parity check passed; the maximum absolute metric
  difference was `9.68e-14`. All six M1-X/M1-Y final fitted objectives match
  their accepted values to approximately `1e-19` absolute.
- The nine reconstructed Test 2B M2-X/H1 final objectives agree exactly with
  the accepted endpoints at the reported precision. The global trajectory
  cache contains 3,381 rows (`21 models x 161 states`); all model time grids
  match truth, and the embedded truth kinetic-energy and projected-vorticity
  arrays agree bitwise across the 21 source records.

## Optimization progress under frozen budgets

- Test 2B M1-X/M1-Y objectives fell by factors from about `2.31e4` to `4.65e4`
  over 10,000 accepted iterations. Independent M2-X fell by `5.02e4`--`7.96e4`.
  The warm M2-X continuation changed by only `1.23`--`2.07`, while fixed-map H1
  changed by `5.90`--`9.78` over 5,000 iterations.
- H2/H5 are represented only by stored initial/final objective endpoints. In
  A/B, their 20-iteration endpoint changes are below `0.31%`; C changes by a
  factor `1.67` for H2 and `2.17` for H5. No recursive checkpoint history was
  reconstructed, so these endpoints do not establish the shape of the missing
  curves.
- Every accepted run terminates at its frozen maximum-iteration budget. The
  histories therefore document numerical progress but do not establish
  convergence to a common optimum.

## M1-X versus M1-Y cross-state comparison

- On Y training support, M1-Y reduces the direct normalized objective relative
  to the M1-X model evaluated diagnostically at Y by factors `9.47` (A), `8.75`
  (B), and `7.49` (C). Conversely, evaluating M1-Y on X raises J_M1-X relative
  to the fitted M1-X model by factors `3.31`, `2.85`, and `2.73`. This is a clear
  X/Y location specialization in the matched direct-regression comparison.
- For A on its nominal training state, relative A RMS is `0.00547` for M1-X at
  X and `0.00494` for M1-Y at Y. On nominal held-out support those values are
  `2.233` and `2.236`; the improvement at the fitted support does not become a
  lower held-out relative A error.
- For B, nominal training relative A RMS changes from `0.00629` (M1-X at X) to
  `0.00534` (M1-Y at Y), but active-R relative RMS changes from `0.00959` to
  `0.0163`. On nominal held-out support, A is `2.896` versus `2.927`, while
  active-R is `0.0189` versus `0.0261`. The Y-location change helps the fitted
  A mapping but does not improve direct learned-R accuracy under these metrics.
- For C, nominal training normalized source-vector relative RMS is nearly
  unchanged (`0.00622` at X versus `0.00617` at Y). On nominal held-out support
  it is `0.194` versus `0.268`. Effective-A held-out relative RMS is lower for
  M1-Y at Y (`1.57`) than M1-X at X (`2.39`), whereas effective-R is higher
  (`0.0824` versus `0.0588`). The component-level result is therefore mixed.
- All of these held-out values are post-hoc evaluations on temporally adjacent
  states 81--160 and were not optimization or selection signals.

## Frozen objective matrix

- J_M1-Y is now populated for all 21 main Test 2B rows, including the 15
  historical non-M1 models. The largest new diagnostic values are `0.147` for
  B independent M2-X and `0.576` for C independent M2-X. A historical H1/H2/H5
  models remain near `4.57e-5`; B near `2.38e-3`; and C rises from `7.52e-3`
  (H1) to `8.77e-3` (H5).
- Low values away from a row's fitted cell are diagnostic only. In particular,
  M1-Y was not trained recursively even when its J_H1/J_H2/J_H5 diagnostics are
  small.

## Deployment-only findings

- A/B structurally enforce the moist-source water and thermodynamic identities.
  Their maximum relative total-water drift after deployment remains about
  `1.71e-14`--`2.01e-14`; this is not learned conservation. C does not enforce
  those identities: maximum relative water drift ranges from `1.28e-4` to
  `3.38e-2`, alongside nonzero learned source defects.
- Representation A M1-Y has a slightly larger final mixed-state error than
  M1-X (`9.06e-6` versus `8.33e-6`) but a smaller absolute final rain-mass error
  (`1.84e6` versus `6.37e6`). Both reproduce the stored analytical-R onset at
  5100 s. A learned-rate FP/FN statistic is not defined for this representation.
- Representation B is the strongest deployment contrast. B M1-X has final
  mixed-state error `7.02e-3`, whereas B M1-Y has `5.10e-6`, a factor `1.38e3`
  reduction. Absolute final cloud- and rain-mass errors fall by factors about
  `1.21e3` and `17.8`. Nevertheless both learned-R models activate at time zero
  (`-5100 s` onset error); the pre-onset false-positive fraction changes from
  `0.217` to `0.274`, and the direct active-R errors above are worse for M1-Y.
- Representation C M1-Y lowers final mixed-state error by `23.4%` relative to
  C M1-X (`1.21e-5` versus `1.58e-5`) and lowers pre-onset effective-rain false
  positives (`0.634` versus `0.778`). Its maximum relative water drift is
  `4.45e-4` versus `1.28e-4`, and its absolute final cloud/rain errors are
  respectively `2.14` and `1.60` times larger. Thus M1-Y is not a uniform
  improvement across deployed diagnostics.
- In the full ladders, the smallest final mixed-state errors occur at H1/H2/H5
  for A (about `5e-6`) and B (about `3e-6`), but those models are sequential
  warm starts with unequal budgets. C instead degrades along the recursive
  ladder, reaching `1.39e-3` at H5. These are descriptive endpoint comparisons,
  not isolated causal effects of horizon length.

## Stored trajectory scope

Cloud mass, rain mass, total-water drift, kinetic energy, and projected
relative-vorticity-squared trajectories are immediately plot-ready for all 21
Test 2B models. The accepted Test 2B records do not contain a mixed-state-error
time series; only final/maximum/regime summaries are available. No Hamiltonian
total-energy or potential-enstrophy trajectory was reconstructed.
