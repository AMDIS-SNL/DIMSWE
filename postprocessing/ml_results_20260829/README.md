# DIMSWE ML-results minimal postprocessing package

This directory is the isolated, evaluation-only quantitative package prepared
on 2026-08-29 for the Machine-Learning Results section. It contains fixed-array
and fixed-map checkpoint evaluations, publication table/figure candidates,
machine-readable sidecars, captions, and provenance. No model was trained, no
truth was regenerated, and no accepted artifact was modified.

## Frozen terminology

- **Test 2A:** states 0--80 are **TRAINING SUPPORT**. No historical held-out or
  test set exists. States 81--160 remain **UNUSED BY RECORDED LEARNING** and
  were not evaluated here.
- **Test 2B:** states 0--80 are **TRAINING TRUTH SUPPORT**. States 81--160 are
  **HELD-OUT TRUTH SUPPORT**, not an untouched test set. Every held-out value in
  this package is a post-hoc evaluation and was not used for stopping or model
  selection.
- Checkpoint direct-error curves are labeled **POST-HOC CHECKPOINT
  EVALUATION**, never validation loss during training.

## Package index

- `data/heldout_x_test2b.json`: metadata for the read-only X-state held-out
  extraction. The hashed NPZ itself is an external artifact; see
  `../../docs/provenance/EXTERNAL_ARTIFACTS.md`.
- `data/checkpoint_direct_histories.csv`: all saved-checkpoint direct metrics
  for the 21 main Test 2B models on X training/held-out support, plus the full
  X/Y cross-state histories for M1-X/M1-Y.
- `data/checkpoint_training_objectives.csv`: fixed-array M1, fixed-map M2-X/H1,
  stored H2/H5 endpoints, and historical Test 2A sparse objective histories.
- `data/completed_objective_matrix.csv`: the 33-row frozen-model matrix, with
  fitted and diagnostic cells explicitly marked.
- `data/deployed_diagnostics.csv`, `rain_event_diagnostics.csv`, and
  `global_trajectories/`: parsed accepted deployment-only diagnostics; no
  rollout was run.
- `tables/`: CSV plus Markdown/LaTeX drafts for Tables 1--5 and the Test 2A
  training-support supplement.
- `figures/main/`: the final seven-figure main-text set.
- `figures/supplement/`: nineteen accepted control, completeness, and Test 2A
  supplementary bundles retained after the final main-paper reset.
- `captions/ML_RESULTS_CAPTION_DRAFTS.md`: consolidated caption drafts.
- `ML_RESULTS_NUMERICAL_SUMMARY.md`: strongest source-grounded numerical
  observations, without Discussion or novelty claims.
- `ML_RESULTS_POSTPROCESS_REPORT.md`: validation, coverage, cost-gate, and
  remaining-work report.
- `POSTPROCESS_PROVENANCE.json`: hashes, commands, repository fingerprints,
  and the no-training/no-mutation declaration.
- `PUBLICATION_CLEANUP_REPORT.md`: exact rendering and terminology changes
  made after numerical acceptance.
- `PUBLICATION_CLEANUP_PROVENANCE.json`: final render hashes and proof that
  the frozen numerical artifacts remained byte-for-byte unchanged.

`POSTPROCESS_PROVENANCE.json` remains the baseline manifest for the accepted
numerical pass. The publication-cleanup provenance is the overlay governing
the revised PDFs, PNGs, figure metadata, inventory, and caption draft.

## Collaborator path contract

Source `../../scripts/reproduction_environment.sh` before running a script.
The postprocessors then use this repository for the historical and M1-Y code
and use sibling packages beneath `postprocessing/`. External mirrors can be
selected without editing source through `DIMSWE_REFERENCE_REPOSITORY`,
`DIMSWE_M1Y_REPOSITORY`, `DIMSWE_ML_RESULTS_AUDIT_ROOT`, and
`DIMSWE_GROUND_TRUTH_PACKAGE`.

Absolute paths in JSON sidecars are immutable records of the accepted 2026-08-29
run and have intentionally not been rewritten. Hashes, not those historical
locations, identify the frozen inputs.

The asset-writing scripts `generate_figures.py`,
`finalize_table_sidecars.py`, and `build_caption_inventory.py` require the
explicit `--overwrite-accepted-assets` flag. Their `--help` paths are read-only;
use the overwrite flag only in a disposable package copy after verifying the
frozen inputs.

## Important scope correction

The accepted Test 2B autonomous JSONs contain mixed-state final, maximum, and
regime summaries, but not a mixed-state-error time series. ML-6 therefore shows
stored final/maximum endpoints in that panel. Reconstructing the missing time
history would require rerunning deployments and is outside this pass.

The stored flow quantities are **kinetic energy**
`0.5 integral h |v|^2 dA` and **projected relative-vorticity squared**
`0.5 integral zeta_h^2 dA`. They are not total Hamiltonian energy or potential
enstrophy.
