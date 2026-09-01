# Replay Validation

## Gate result

The required Representation-A/M1-Y pilot passed before any other model was
launched.  It reproduced the accepted autonomous record with exact array
equality for all 161 cloud-water, rain-water, total-water, kinetic-energy, and
projected-vorticity diagnostic values.  Final, maximum, and accumulated mixed
state errors were also exactly equal.

The remaining eleven replays were then executed serially with the same driver.
All passed the same comparison.  Although the fail-closed numerical tolerance
was `rtol=5e-13, atol=1e-12`, every reported maximum absolute and relative
difference was exactly zero.

| Representation | Method | Frames | Max. absolute parity difference | Max. relative parity difference | Result |
|---|---:|---:|---:|---:|---|
| A | M1-Y | 161 | 0 | 0 | PASS |
| A | H1 | 161 | 0 | 0 | PASS |
| A | H2 | 161 | 0 | 0 | PASS |
| A | H5 | 161 | 0 | 0 | PASS |
| B | M1-Y | 161 | 0 | 0 | PASS |
| B | H1 | 161 | 0 | 0 | PASS |
| B | H2 | 161 | 0 | 0 | PASS |
| B | H5 | 161 | 0 | 0 | PASS |
| C | M1-Y | 161 | 0 | 0 | PASS |
| C | H1 | 161 | 0 | 0 | PASS |
| C | H2 | 161 | 0 | 0 | PASS |
| C | H5 | 161 | 0 | 0 | PASS |

The machine-readable comparison is `data/REPLAY_VALIDATION.json`; each cache
sidecar contains its per-quantity comparison.

## Spatial and temporal validation

- Every NPZ contains steps 0–160 and times 0–16000 s at 100 s cadence.
- Every spatial array has shape `(161, 128, 128)` and finite `float32` values.
- Grid coordinates are exactly array-equal to the accepted truth-map cache.
- Boundary quantities use \(\hat X_n\); rates use
  \(\hat Y_n=P(\hat X_n)\).
- The final rate frame uses a non-advancing diagnostic prefix/moist call.
- Every final GIF contains exactly 161 frames with a 100 ms frame duration
  (10 fps), and its hash matches its sidecar.
- Contact sheets in `visual_audit/` were decoded from the final GIF files, not
  regenerated independently from NPZ data.

## Immutable inputs

The accepted truth-map cache SHA-256 is
`e9477a3ac4e54ebe4da73f9df5ffaafdf98c812afd7fe92b55820230db034de6`.
All checkpoint, configuration, preparation, truth, and accepted-result hashes
are recorded in `REPLAY_PROVENANCE.json` and were reverified after rendering.

Two environment preflights stopped before any timestep was advanced: first
because JAX x64 had not been explicit, then because compiler caches targeted a
restricted user directory.  The final driver freezes JAX x64 and redirects
compiler caches to `/private/tmp`; these changes affect execution plumbing, not
the numerical method.

