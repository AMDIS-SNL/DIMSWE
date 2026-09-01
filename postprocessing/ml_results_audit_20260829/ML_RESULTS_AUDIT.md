# DIMSWE Machine-Learning Results audit

Audit date: 2026-08-29  
Scope: completed Test-2A/Test-2B neural fits and their already existing
evaluation artifacts. This was a read-only scientific inventory: no optimizer,
truth generator, prefix integration, hybrid rollout, or expensive Firedrake
simulation was launched.

## Executive result

The repository supports a strong Results section now, but it does not support
every originally proposed curve without post-hoc work.

- The exhaustive artifact census found **61 completed neural optimizer
  endpoints**: 38 for Test 2A and 23 for Test 2B. This intentionally includes
  a superseded 100-iteration precursor and 14 artifacts explicitly labeled
  non-scientific smoke/preflight runs; those 14 are inventory-only and must not
  enter scientific comparisons. Test 2A contains Representation A and the
  historical “Problem B”
  four-source model, which is Representation C in the later nomenclature; it
  has no two-rate Representation B campaign. Test 2B contains the full A/B/C
  ladders, two rain-output-transform ablations, and the three matched M1-Y
  fits.
- Test 2A has no historically defined held-out set. Test 2B has a legitimate
  temporal **held-out truth support** (states 81--160), but not a permanently
  untouched final test set.
- **No run recorded held-out loss during training.** Six Test-2A optimizer
  study/continuation runs have full accepted-iterate training histories; 12
  accepted Test-2A A/C runs have sparse checkpoint objective histories; most
  other curves would be post-hoc checkpoint reconstructions.
- Test-2B final direct metrics are complete for the 18 accepted historical
  A/B/C models and the three M1-Y models. The matched M1-X/M1-Y artifacts also
  contain the complete trained-at-X/Y by evaluated-at-X/Y cross-state matrix.
- Test-2B accepted deployments already contain rich conservation, rain-event,
  water-partition, state-error, kinetic-energy, and projected-vorticity-squared
  trajectories for all 18 core models and the three M1-Y models. No accepted
  deployed spatial state sequence was found.
- A thermal-shallow-water Hamiltonian energy is defined in the production
  code, but it was not stored for the accepted truth or ML rollouts. The saved
  `kinetic_energy` and `projected_enstrophy` must not be presented as total
  energy or potential enstrophy.

## 1. Scientifically correct support terminology

### Test 2A

Use **training support** for states 0--80. Use **training-support deployment**
or **post-hoc autonomous evaluation on the training interval** for model
rollouts over that range. States 81--160 exist but were outside all recorded
learning campaigns; call them the **unused-by-recorded-learning temporal
extension**, not a test set. There is no historical held-out or final-test
protocol.

### Test 2B

Use **training truth support** for states 0--80 and **held-out truth support**
for states 81--160. Static campaign safeguards excluded 81--160 from fitting,
model selection, and stopping, but historical and M1-Y postprocessors have
subsequently inspected them. Therefore “test set” or “untouched test set” is
too strong.

Both splits consist of temporally adjacent states from one deterministic
trajectory per physical case. The millions of spatial samples are correlated
within and across states; report the state count as well as the packed sample
count.

Full partition evidence is in `DATA_PARTITION_AUDIT.md`.

## 2. Completed-run inventory

`ML_RUN_INVENTORY.csv` contains one row per completed run, including exact
objective/state/target, architecture, parameter count, seed, dtype, optimizer,
L-BFGS memory, budgets/evaluation counts, termination, initialization,
checkpoint/history paths, and comparability notes.

The census was reconciled against every non-intermediate neural parameter NPZ
under the Test-2A, Test-2B-learning, and isolated M1-Y result trees: all 61
terminal candidates map to exactly one inventory row, and every row's terminal
checkpoint exists. `fit_progress.json` duplicates the same optimizer endpoint
as `fit_result.json` and is not counted as a second run. Intermediate
checkpoints are likewise not separate runs. Five FIML `FI_*_smoke_controls.npz`
files are optimized field-control vectors rather than neural-network
parameters and are explicitly outside the neural run count.

### Test-2A inventory (38)

- Representation A: the superseded canonical 100-iteration M1-X fit; three
  early optimizer trials; three continuations; an accepted independent M1-X;
  accepted independent and warm M2-X branches; an auxiliary shorter M2-X
  result; H1/H2/H5; and four FIML direct/pseudo-label H2/H5 models.
- Representation A also has 14 explicitly **NONSCIENTIFIC** neural optimizer
  endpoints: one warm-M2 smoke, three horizon prelaunch smokes, and ten
  direct/stage-2 FIML timing smokes (including superseded/aborted benchmark
  directories). They are retained in the CSV for completeness and excluded
  from every recommended scientific table/figure. Five field-inversion
  control-vector smokes were inspected but are not neural-network runs and are
  therefore not rows in `ML_RUN_INVENTORY.csv`.
- Representation C: M1-X, independent M2-X, warm M2-X, H1, H2, H5.
- Representation B: no completed run exists for this physical case.

### Test-2B inventory (23)

- For each A/B/C: independent M1-X, independent M2-X, warm M1-X-to-M2-X,
  H1 from M1-X, H2 from H1, and H5 from H2: 18 historical core models.
- BTP and BTPL M1-X rain-output-transform ablations: two models.
- Matched M1-Y A/B/C: three models.

The H1/H2/H5 ladder is not a clean horizon-only experiment: H1 starts from
M1-X, H2 starts from H1, and H5 starts from H2; each launches a fresh
optimizer but inherits parameters. The objective, initialization history, and
budget therefore change together. Independent and warm M2-X branches are
listed separately for the same reason.

M1-X versus M1-Y is the special matched comparison: the support indices,
architecture, seed-zero initialization, X-fitted normalization/output scales,
carrier-mass weighting, optimizer, L-BFGS memory, and 10,000-iteration budget
match. X versus post-prefix Y evaluation state is the intended change.

## 3. Optimization and held-out history

The four requested held-out-history classifications are applied in
`TRAINING_HISTORY_AVAILABILITY.csv`.

### What is actually recorded

- **Full accepted-iterate training objective and gradient histories:**
  `t2a-a-m1x-lbfgs-m10-500`, `t2a-a-m1x-lbfgs-m20-500`,
  `t2a-a-m1x-trust-tcg-100`, and the three Test-2A M1-X continuation runs to
  cumulative 2k, 7k, and 52k iterations. L-BFGS trial objective values exist,
  but accepted step lengths do not; the trust-region radius/step history was
  not recorded.
- **Evaluation/callback sequences without an accepted-iterate mapping:** the
  superseded canonical Test-2A M1-X 100-iteration fit.
- **Sparse training objective values at checkpoint iterations:** 12 accepted
  Test-2A A/C core fits.
- **Intermediate checkpoints but objective values not recorded:** the four
  Test-2A FIML fits and all 23 Test-2B fits.
- **Initial/final only:** 14 explicitly non-scientific Test-2A neural smokes.
  The auxiliary Test-2A independent M2-X 50k result is final-only.

### Held-out histories

**No run has both a recorded training curve and a held-out curve recorded
during training.** All core held-out-history classifications are therefore
`RECONSTRUCTABLE_FROM_INTERMEDIATE_CHECKPOINTS`, `FINAL_ONLY`, or
`NOT_AVAILABLE`; none is `RECORDED_DURING_TRAINING`.

- Test 2A: `NOT_AVAILABLE` because no historical held-out support was defined.
  Evaluating 81--160 now would create a new temporal-extension protocol, not
  reconstruct an old validation curve.
- All 23 Test-2B runs: a **held-out direct-law** curve can be reconstructed at
  their saved checkpoint iterations without retraining. Y features/targets
  are already cached; X held-out features/targets need one read-only
  truth-array extraction. Network evaluation is then cheap.
- Held-out values of the **fitted objective** are technically reconstructable
  from the saved checkpoints for all 23 Test-2B runs, but no such curve is a
  historically recorded validation signal. M1-X/M1-Y and BTP/BTPL are fixed
  array inference. H1 uses fixed Y arrays and the source/state mapping after
  freezing a held-out denominator convention. M2-X needs a held-out X mapping
  cache. H2/H5 require a newly specified held-out window schedule and a
  recursive prefix/rollout evaluation per checkpoint, so they are expensive.

For A/B/C M1-X and M1-Y, the checkpoint grid is sufficient for sparse
post-hoc plots of training and held-out direct loss. It is not sufficient for
a per-accepted-iteration curve, and the figure must say it was reconstructed
after training.

## 4. Final metrics for explicitly learned quantities

### Complete

- Test-2B A, all six historical models: final training and held-out A metrics.
- Test-2B B, all six: A plus all-sample and truth-active R metrics, bias,
  correlation, FP/FN and onset diagnostics.
- Test-2B C, all six: S/Qv/Qc/Qr source metrics, effective A/R projections,
  activation and structural/off-manifold diagnostics.
- Test-2B M1-X/M1-Y A/B/C: complete direct metrics on training and held-out
  X/Y supports, including all four cross-state cells.
- Test-2A A accepted/core models: complete final direct A metrics on training
  support only.

### Partial or absent

- Test-2A C: component physical RMS and structural/vector diagnostics exist,
  but component relative RMS, max, bias and scalar correlations are not all
  stored.
- BTP/BTPL: final fitted objective exists; the standard direct metric suite was
  not run.
- There is no Test-2A held-out direct metric suite under the historical
  protocol.

Exact field availability and the 156-row cross-state flattening are in
`METRIC_AVAILABILITY_MATRIX.md` and `M1_CROSS_STATE_DIRECT_METRICS.csv`.

## 5. Objective-ladder matrix

Existing artifacts support `J_M1-X`, `J_M2-X`, `J_H1`, `J_H2`, and `J_H5`
for the standard accepted ladders. The M1-Y study additionally evaluates
`J_M1-Y` and all five historical objectives on frozen M1-X and M1-Y models for
A/B/C. These H-objective values are diagnostic only; M1-Y was not recursively
trained.

`FROZEN_MODEL_OBJECTIVE_MATRIX.csv` preserves that distinction with exactly
one row per one of the 61 inventoried endpoints. It includes complete
five-objective Test-2A C records, full five-objective records for Test-2A A
M1/H1/H2/H5, partial A M2 cross-objective records, the historical Test-2B
matrices, and the complete matched M1-X/M1-Y rows. A complete six-column
Test-2B matrix requires only a cheap fixed-array `J_M1-Y` evaluation for the 15
non-M1 historical A/B/C models. Cross-case and cross-representation objective
magnitudes must not be treated as one common scale because physical case,
output normalization, source dimension, support, and state denominator differ.

## 6. Existing deployed physical diagnostics

For the 18 historical Test-2B A/B/C models and three M1-Y models, existing
standard hybrid output includes:

- total-water mass trajectory, final/max drift, and local source residual;
- the `source_S-beta2*source_Qv` residual;
- Qv/Qc/Qr minima and finite-state status;
- rain onset, learned/effective R false positives and false negatives where
  applicable;
- Qv/Qc/Qr mass trajectories, final/max Qr error, and integrated rain source;
- componentwise and mixed state errors;
- kinetic energy, projected relative-vorticity squared, and high-wavenumber
  velocity-energy fraction.

A/B impose total-water and thermodynamic source identities algebraically. C
learns four independent sources, so its residuals are genuine learned
off-manifold diagnostics. Positivity is not enforced in any representation;
stored minima measure deployed admissibility. Representation B's learned R is
linear-output and sign-unconstrained, so its active-R errors and FP/FN/onset
records are especially important.

Test-2A accepted A/C deployments have a useful but less uniform subset over
states 0--80 only. Early optimizer models have no standard deployment output;
Test-2B BTP/BTPL have no accepted deployment comparison.

See `DEPLOYED_DIAGNOSTIC_AUDIT.md`.

## 7. Immediately plottable global trajectories

For Test 2B, all 18 historical standard models and all three M1-Y models can
be plotted against truth immediately for:

- integrated cloud and rain water;
- total-water drift;
- componentwise/mixed state errors;
- kinetic energy;
- projected relative-vorticity squared;
- rain production/onset and water partition.

Test-2A truth scalar trajectories are cached for all 161 states, but accepted
model outputs are limited to 0--80 and differ by campaign. A/C state-error,
kinetic-energy and projected-vorticity-squared histories are available; C also
has total-water and signed Qr-mass/budget data. A uniform Test-2A cloud/rain
mass comparison would need additional rollout-time postprocessing.

The code-level Hamiltonian density is
`h|v|^2/2 + hS/2 + hB`, and accepted statistics code can assemble it. No
accepted truth or deployed ML artifact stores it. No accepted potential
enstrophy diagnostic was found; stored `projected_enstrophy` is
`0.5 integral zeta_h^2 dA` for CG(3)-projected relative vorticity.

See `GLOBAL_TRAJECTORY_AUDIT.md`.

## 8. Spatial availability

Truth caches already contain all 161 maps for saturation departure, q_c, q_r,
exact post-prefix A/R, height anomaly, and relative vorticity for both physical
cases. No accepted deployed model has saved HDF5/VTK/XDMF/NetCDF/Zarr state
fields. Therefore:

- truth-only contours are immediate;
- direct network-on-truth X/Y prediction/error maps are cheap where arrays are
  cached, but are not autonomous deployments;
- matched truth/deployed contours at pre-rain, onset, peak rain, and final
  times require selected frozen-model rollouts with state saving.

See `SPATIAL_OUTPUT_AUDIT.md`.

## 9. Major comparability cautions

1. “Test 2A/Test 2B” are physical cases; “Representation A/B/C” are model
   parameterizations.
2. Historical Test-2A “Problem B” is Representation C, not Representation B.
3. Test-2A accuracy/deployment metrics are training-support results, not
   held-out generalization.
4. Test-2B held-out metrics are final/post-hoc. They were not validation
   stopping signals.
5. H1-to-H2-to-H5 comparisons mix objective horizon, warm-started parameters,
   and budget.
6. Structural A/B conservation identities and learned C residuals have
   different meanings.
7. M1-X/M1-Y is matched in training contract, but different optimization
   outcomes must not by themselves be called causal mechanisms.
8. Report checkpoint-reconstructed curves as post-hoc sparse curves.
9. Do not compare normalized objective magnitudes across physical cases or
   representations without their denominator conventions.
10. Do not call kinetic energy “energy” generically or projected
    vorticity-squared “potential enstrophy.”

## 10. Minimal additional postprocessing campaign

The full Machine-Learning Results section can be completed without training:

1. Parse and plot the existing final direct metrics, M1 cross-state matrix,
   deployment diagnostics, global trajectories and rain-event table
   (**CHEAP**).
2. Evaluate `J_M1-Y` on the 15 non-M1 standard Test-2B checkpoints to complete
   the objective matrix (**CHEAP**, cached Y arrays).
3. Extract held-out X features/targets once and evaluate the saved M1-X/M1-Y
   checkpoint grids on training/held-out X and Y supports (**MODERATE** one-time
   extraction, then **CHEAP** inference).
4. Evaluate training objectives at sparse checkpoints only for the runs to be
   plotted. Prioritize M1 and H1; omit H2/H5 held-out recursive curves unless
   the added scientific question justifies their **EXPENSIVE** cost.
5. Optionally complete Test-2A C component metrics on training support
   (**CHEAP**).

No spatial model rerun, Hamiltonian-energy reconstruction, truth regeneration,
or checkpoint mutation is needed before writing the ML Results section. A
small selected rollout campaign is needed later only for the separate spatial
dynamics section.

## Safety and provenance conclusion

All new files are confined to
`ml_results_audit_20260829/`. The authoritative repository was inspected
read-only. No training, retraining, truth regeneration, accepted-checkpoint or
historical-result mutation, repository cleanup/reset/stash/checkout, new
prefix construction, hybrid rollout, or expensive simulation was performed.
`AUDIT_PROVENANCE.json` records the initial and final repository fingerprints
and hashes of the principal evidence files.
