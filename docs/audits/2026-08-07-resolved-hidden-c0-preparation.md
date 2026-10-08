# 2026-08-07 J4B-PREP resolved hidden-c0 audit

## Separation from J4A

J4A/Test 1A was externally certified before these files were added.  J4B-PREP
does not alter `dimswe/hidden_c0.py`, its tiny configuration, learned-physics
core, or any J1/J2/J3 tracked source.  The separate manifests are
`docs/manifests/J4A_TEST1A_FILES.txt` and
`docs/manifests/J4B_PREP_FILES.txt`.

## Prepared scope

- Pure immutable pilot, inference-index, spectral, and objective-scan records.
- Opt-in production driver for native `doublevortex`, TC5, or TC2 initial
  conditions with configurable resolution and time settings.
- Paired configuration proof that non-c0 physics is identical.
- Firedrake HDF5 field checkpoints, atomic restart arrays, JSON metadata and
  diagnostics, and NPZ spectra at configurable cadence.
- Analysis-only paired mass separation, field blocks, energy, projected
  enstrophy, deployed hyperviscosity proxy, spectral content, finite-state
  checks, non-conclusive late-time-growth warnings,
  optional plots, and resolution summaries.
- Selection-neutral Test 1B train/held-out indexing, observation cadence,
  reset/rollout horizons, terminal/accumulated choice, four exact objective
  families, derivative landscape scans, common scalar optimizer, and common
  held-out evaluation.

No resolved production simulation, objective scan, or inference was executed
inside Codex.  After the external 32-by-32 pilot exposed suspicious growth,
the opt-in exact Euler-child spectral diagnostic was added without changing
the production hyperviscosity form.

The earlier preparation result was `15 passed in 0.34s`.  After the stability
audit and growth semantics were added, the focused stability plus preparation
result was `19 passed, 44 warnings in 4.29s` with x64 enabled.  The sandbox
emitted its known Open MPI TCP bind warning during Firedrake imports; no test
failed.

## Scientific boundary

`doublevortex` ranks first because it is the only nontrivial moist flow already
configured for production.  TC5 is a conditional secondary wave-generating
candidate.  TC2 is complete but likely too balanced; other evocatively named
cases are dry-incomplete or code stubs.  This ranking selects only an efficient
pilot sequence, not the Test 1B production flow.

Analysis flags never automatically select adequacy.  In particular, finite
coefficients and positive h are not relabelled as numerical stability, and a
high-k threshold crossing is not relabelled as physically populated modes.
External evidence must combine the exact Euler-child bound, non-conclusive
growth warnings, finite status, c0 separation beyond roundoff, a nonzero
deployed hyperviscosity proxy, sensible onset, and a credible resolution trend.

## Restart and diagnostic boundary

Saved NPY and Firedrake HDF5 state snapshots support experiment restart at the
output cadence and scientific inspection of all six fields.  They are not
adjoint checkpointing; no revolve implementation or claim is present.

Velocity spectra use physical uniform-grid samples, never arbitrary Firedrake
coefficient vectors.  Hyperviscosity tendency and child-update norms are
documented proxies, not claimed physical dissipation identities.

## Source checklist

1. J1/J2/J3 mathematics is unchanged.
2. J4A mathematics is unchanged.
3. UFL remains the default; JAX moist is opt-in.
4. Learned physics and every resolved driver are opt-in.
5. Test 1A remains the certified 2-by-2 plumbing case.
6. Test 1B-0 invokes the complete production six-field split.
7. Paired physics differs only in c0; output paths differ operationally.
8. No final case/resolution/duration is selected.
9. No neural architecture, Test-2 feature, or output choice is introduced.
10. No MPI, accelerator, or adjoint-checkpointing claim is made.
11. No production test is weakened.
12. No push, remote change, staging, or commit is authorized.
