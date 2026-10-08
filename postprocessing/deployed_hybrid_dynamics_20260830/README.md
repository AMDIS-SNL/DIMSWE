# Deployed Hybrid Dynamics

This isolated package contains the spatial replay products for the rain-active
64×64 DoubleVortex calculation.  It includes exactly twelve frozen deployed
models: M1-Y, H1/M2-Y, H2, and H5 for Representations A, B, and C.  No model was
trained or modified while creating this package.

## Scientific semantics

- Boundary maps—relative vorticity, saturation departure, specific cloud
  water, and specific rain water—are evaluated at the model-generated boundary
  state \(\hat X_n\).
- Rate maps are evaluated at the model-generated pre-moist state
  \(\hat Y_n=P(\hat X_n)\), exactly where the frozen moist parameterization is
  called.
- At \(n=160\), the fixed prefix and moist child are evaluated diagnostically;
  their update is not applied to a step-161 trajectory state.
- Representation A displays learned \(A\) and analytical \(R\).
- Representation B displays learned \(A\) and learned \(R\).
- Representation C displays effective \(A\) and \(R\) from the accepted
  scale-weighted physical two-rate projection.  The projection is diagnostic
  only; the four unconstrained predicted source components drive the replay.

All maps use the accepted 128×128 interior-GLL truth visualization grid.  All
figures and movies use the single cross-model normalization in
`COMMON_VISUAL_LIMITS.json`; negative values are retained and no model-specific
autoscaling is used.

## Contents

- `data/`: JSON provenance sidecars and aggregate replay validation. The
  twelve compressed 161-frame NPZ caches are external hashed artifacts.
- `figures/`: the common-scale truth gallery and twelve deployed event
  galleries as vector PDF, 300-dpi PNG, and JSON sidecars.
- `movies/`: JSON sidecars for twelve 161-frame, 10-fps GIFs. The GIFs are
  external hashed artifacts; `scripts/convert_gifs_to_mp4.sh` records the
  optional conversion.
- `captions/`: draft report captions.
- `REPLAY_PROVENANCE.json`: immutable-input hashes, repository fingerprints,
  checkpoint identities, commands, and output hashes.
- `REPLAY_VALIDATION.md`: parity gate and cache-validation record.
- `DEPLOYED_HYBRID_DYNAMICS_SUMMARY.md`: source-grounded visual summary and
  limited recommendations for later error maps.

The rejected visual-layout pilot, one-time continuation helpers, and original
closeout-only fingerprint scripts remain in the archaeological workspaces and
are deliberately absent here. Source
`../../scripts/reproduction_environment.sh` before replaying. The scripts use
the repository-local source and frozen checkpoints by default; optional
`DIMSWE_REFERENCE_REPOSITORY`, `DIMSWE_M1Y_REPOSITORY`, and
`DIMSWE_GROUND_TRUTH_PACKAGE` overrides select verified external mirrors.

## Safety result

Every replay reproduced its accepted scalar autonomous record bit-for-bit for
all compared trajectories and summaries.  Final provenance rechecked the
authoritative repository, the M1-Y workspace, all checkpoints, accepted scalar
records, and all 161 truth restart arrays.  Their fingerprints were unchanged.
