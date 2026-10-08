# DIMSWE ML-results minimal postprocessing report

Date: 2026-08-29

Status: **COMPLETE WITH EXPLICIT COST-GATE DEFERMENTS**

The requested evaluation-only quantitative package is complete. All outputs
were written under `ml_results_postprocess_20260829/`; accepted checkpoints,
truth data, historical results, and the authoritative repository were treated
as read-only.

## 1. Numerical additions completed

1. **Completed J_M1-Y diagnostics.** Fixed-array inference populated J_M1-Y
   for the 15 historical non-M1 Test 2B models where it was missing. Together
   with the six accepted M1-X/M1-Y values, all 21 main A/B/C rows now have
   J_M1-Y in `data/completed_objective_matrix.csv`. Fitted and diagnostic cells
   are separately labeled.
2. **Extracted held-out X once.** States 81--160 were read from immutable truth
   restarts and packed into `data/heldout_x_test2b.npz`. It contains normalized
   X features `(80,65536,5)`, analytical A/R and h/Qr arrays `(80,65536)`,
   analytical C sources `(80,65536,4)`, and 65,536 carrier weights. The sidecar
   records every restart hash, array hash, normalization value, target order,
   and source path.
3. **Reconstructed all cheap sparse direct histories.** Every saved checkpoint
   for all 21 main Test 2B models was evaluated on X training support 0--80 and
   X held-out support 81--160. The resulting tidy artifact has 23,989 rows and
   retains A, R, activation, four C components, effective A/R, and structural
   diagnostics.
4. **Reconstructed the matched M1 cross-state histories.** All saved M1-X and
   M1-Y checkpoints for A/B/C were evaluated separately on X training, Y
   training, X held-out, and Y held-out support. No support was pooled.
5. **Reconstructed cheap fitted-objective histories.** Six M1 histories use
   fixed arrays (60 objective points). Nine M2-X/H1 histories use fixed prepared
   maps (84 objective points). Test 2A uses only 105 historically recorded
   sparse points. Test 2B H2/H5 retain their 12 stored initial/final endpoints.
6. **Parsed final deployment diagnostics without execution.** Existing accepted
   JSON/CSV arrays supplied all 21 Test 2B model trajectories for cloud mass,
   rain mass, total-water drift, kinetic energy, and projected relative-
   vorticity squared, plus physical/conservation/rain endpoint diagnostics.
   Test 2A A/C training-interval deployment records were parsed for the
   supplementary figure without accessing states after 80.

## 2. Validation gates

- Source reference is `d2f5d66ecb5500aad24eca37280f8a52e22a250f` on
  `dev/dimswe-learned-physics-framework`.
- All 80 held-out truth restart hashes matched the immutable truth manifest.
- Production feature order is `(h,S,Qv,Qc,B)`; C source order is
  `(S,Qv,Qc,Qr)`. The historical X-fitted input offsets/scales and output
  scales were reused and were not refitted.
- Training-cache parity is exact for normalized X features; maximum absolute
  A/R differences are `6.83e-20` and `6.46e-27`.
- The checkpoint manifest validates 174 checkpoint references across 21 runs
  and 156 unique parameter arrays against accepted paths/hashes.
- All 960 accepted final direct-metric parity comparisons passed; maximum
  absolute metric difference is `9.68e-14`. All six final M1 fitted objectives
  agree within approximately `1e-19` absolute.
- All nine reconstructed M2-X/H1 final objectives reproduce accepted values
  exactly at reported precision.
- The 21 Test 2B deployment time grids match truth at all 161 states. Embedded
  truth kinetic-energy and projected-vorticity arrays are bitwise identical
  across all source records.
- Held-out states occur only in post-hoc evaluation artifacts. No held-out
  quantity was passed to an optimizer, stopping rule, or model-selection step.

## 3. Deliberate cost-gate exclusions

- H2/H5 recursive fitted-objective checkpoint histories were **not**
  reconstructed. Doing so requires recursive model evaluation; ML-1 shows
  stored initial/final endpoints only.
- No hybrid or autonomous rollout was launched. The accepted Test 2B records
  do not contain a mixed-state-error time series, despite containing final,
  maximum, accumulated, and regime summaries. ML-6 therefore uses endpoint
  bars in that panel.
- No spatial state rerun, contour campaign, Hamiltonian-energy reconstruction,
  Test 2A temporal-extension evaluation, new objective, new seed, new model,
  optimization, or truth generation was performed.

## 4. Publication tables

Each table has a machine-readable CSV and Markdown/LaTeX draft:

- Table 1: data supports and evaluation protocol;
- Table 2: 33 accepted scientific training-run contracts, plus a detailed
  provenance CSV;
- Table 3: Test 2B final direct prediction accuracy on separate training and
  held-out support;
- Table S3: Test 2A training-support-only direct accuracy;
- Table 4: Test 2A/Test 2B frozen-model objective matrices, fitted cells marked;
- Table 5: Test 2B rain-event and water-partition diagnostics.

## 5. Publication figure candidates

Every figure has vector PDF, 300-dpi PNG, exact plotted CSV, JSON provenance,
metric definitions/units, source paths, hashes, and a draft caption.

### Recommended main-text variants after visual inspection

1. `ML1_optimization_progress_test2b`: the 3x3 family layout is readable after
   using compact checkpoint labels. H2/H5 endpoints are visibly unconnected.
2. `ML2_posthoc_direct_history_test2b`: the selected five-model version is the
   clearest main direct-history figure; it includes M1-X, M1-Y, warm M2-X, H1,
   and H5 and exposes rather than suppresses late-history failures. The all-
   seven-model version should remain supplementary.
3. `ML3_m1x_m1y_cross_state_final`: recommended as the main controlled X/Y
   comparison. The checkpoint-resolved 4x2 history is readable but better
   suited to supplement.
4. `ML4_frozen_model_objective_matrix_test2b`: use the vertically stacked A/B/C
   version. Independent colorbars and fitted-cell outlines are legible at
   report width.
5. `ML5_deployed_physical_diagnostics_test2b`: use as the compact all-21-model
   endpoint comparison. The C-only learned source-structure panel belongs in
   supplement because A/B are structurally enforced.
6. `ML6_global_trajectories_representation_A/B/C`: retain separate
   representation figures. A combined 21-model axis is not recommended. The
   main subset (truth, M1-X, M1-Y, H1, H5) is readable and retains the major B/C
   failures; complete seven-model variants remain supplementary.

The Test 2A optimization/deployment figures are appropriately supplementary:
they use training support only and do not imply a new validation protocol.

## 6. Strongest quantitative observations

The concise numerical record is in `ML_RESULTS_NUMERICAL_SUMMARY.md`. Most
notably:

- matched M1-Y lowers its nominal Y objective relative to M1-X evaluated at Y
  by factors `9.47`, `8.75`, and `7.49` for A/B/C, while its diagnostic X
  objective is `2.73`--`3.31` times higher than fitted M1-X;
- held-out direct errors remain much larger than training errors, and M1-Y does
  not uniformly lower them;
- B M1-Y reduces deployed final mixed-state error from `7.02e-3` to `5.10e-6`
  relative to B M1-X, despite worse active-R direct error and a larger pre-onset
  false-positive fraction;
- C M1-Y lowers final mixed-state error but increases total-water drift and
  final partition errors relative to C M1-X;
- A/B conservation identities are structural, while C exhibits genuine
  learned source-manifold defects and maximum relative deployed water drift up
  to `3.38e-2`.

These are descriptive frozen-model comparisons. Sequential H1/H2/H5 warm
starts and unequal budgets prevent attribution to horizon length alone.

## 7. What remains before drafting the Machine-Learning Results section

No additional numerical campaign is required to draft a defensible Results
section if H2/H5 are represented by endpoints and mixed-state deployment error
by final/maximum summaries. The remaining work is editorial:

1. choose the final main-versus-supplement figure subset after review;
2. select manuscript rounding/significant figures and table width treatment;
3. integrate the reviewed captions/tables/figures into the report source;
4. decide whether the absent H2/H5 recursive histories or mixed-state-error
   time series are important enough to justify a separate, nonminimal replay.

The latter two histories are the only substantive quantitative omissions from
the originally sketched figure suite, and both require work explicitly barred
from this pass.

## 8. What remains only for the later spatial/deployed-dynamics section

Truth map caches already contain all 161 states and the proposed fields.
Accepted deployed artifacts contain scalar diagnostics but no reusable spatial
state sequences. Actual truth-versus-deployed contours of saturation departure,
q_c, q_r, A/R, state error, or vorticity therefore require a selected rollout
that saves fields at steps 0, 50, 51, 89, 120, and 160. Cheap pointwise network-
on-truth maps are possible now but answer a different a-priori question. No
spatial rollout was begun.

## 9. Safety declaration

No training or retraining, truth regeneration, accepted-checkpoint/result
mutation, repository cleanup/reset/stash/checkout, expensive recursive-history
campaign, or spatial/autonomous rerollout occurred. The authoritative and M1-Y
repository status/diff fingerprints are rechecked in
`POSTPROCESS_PROVENANCE.json`.
