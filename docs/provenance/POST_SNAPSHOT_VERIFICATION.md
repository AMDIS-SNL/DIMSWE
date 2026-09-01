# Post-snapshot integration verification

Date: 2026-08-31 (America/Denver)

## Scope and safety

The integration branch is `collaborator/track1-current-science-20260831`,
created from preserved commit
`24cc40a8bb4d23cde88844c401c850578fee5d09`. The preserved
`collaborator/track1-closeout` ref still points exactly to that commit.

Claude Code's review branch was not merged. This pass did not train, optimize,
regenerate truth, replay a model, render a replacement result, run B+, or
change a physical/numerical algorithm. The remote fetch URL remains GitHub and
the push URL is `DISABLE_PUSH`.

## Static and structural checks

Checks were run from the collaborator repository after all imports. They used
read-only parsing rather than Python bytecode compilation so no source-tree
cache was required.

| Check | Result |
|---|---|
| Python AST parse of every tracked `*.py` | 202 files, 0 errors |
| JSON parse of every tracked `*.json` | 319 files, 0 errors |
| `/bin/bash -n` on every tracked `*.sh` | 23 files, 0 errors |
| `git diff --check` | passed |
| Markdown local-link resolution, excluding fenced mathematical pseudocode | 41 links checked, 0 broken |
| explicit collaborator-facing `file.py:line[-line]` bounds | 94 citations checked, 0 errors |
| exact routine-at-line checks in `CODE_WALKTHROUGH.md` | 68 routines checked, 0 errors |
| active imported Python/shell machine-path scan | 0 `/Users/arjunsharma`, `/home`, or Windows-drive defaults |
| safe postprocessor `--help` smoke checks | 8 entry points passed |

The strict routine check independently caught and corrected five stale
pre-existing walkthrough citations: `DenseMLP.__call__`,
`FixedDiscreteCache`, `FastFixedDiscreteObjective`, `prepare_h1_cache`, and
`run_equivalence_audit`. It also reconfirmed the separately reviewed
`ProductionMTSWESplitHVP` line corrections and the `dimswe/output.py` module
name. No source was changed for a citation repair.

Historical absolute paths remain in immutable JSON/CSV/report provenance by
design. They are records of where accepted artifacts were produced, not active
runtime defaults.

## Import and artifact integrity

`POST_SNAPSHOT_IMPORT.tsv` contains 415 destination rows:

- 410 rows map to actual later-workspace source files;
- 395 of those are byte-identical source-to-destination;
- 15 have an explicit behavior-preserving portability or provenance-path edit;
- 4 are Codex-created integration files; and
- 1 is the prior collaborator `scripts/reproduction_environment.sh` extended
  to export its already validated repository root.

Every source and destination hash in all 415 rows was recomputed successfully.
The only source/destination differences are enumerated directly by the TSV; no
unexplained divergence exists.

The W2 M1-Y artifact tree contains 84 non-clutter files. Exactly 82 are
versioned and the only two exclusions are the externally contracted
`m1y_learning_data.npz` and `m1y_heldout_data.npz` caches. All three final
A/B/C parameter NPZs and their sidecars are present. The external-artifact
contract independently rehashed 32 source files by byte count and SHA-256 with
zero mismatches.

## Dependency-light numerical tests

The available interpreter was Python 3.12.12 with pytest 9.0.3 and JAX 0.8.1.
PyROL and Firedrake were not installed in this environment. With
`JAX_ENABLE_X64=True`, `JAX_PLATFORMS=cpu`, bytecode disabled, and pytest's
cache provider disabled, Codex ran:

```text
tests/test_jax_moist_local.py
tests/test_jax_moist_derivatives.py::TestPureJAXMoistDerivatives
tests/test_test2a_operator.py
tests/test_test2b_rain_learning.py
tests/test_learned_physics_framework.py

72 passed in 13.48s
```

The dedicated M1-Y command was attempted under the same CPU/x64 settings, but
collection stopped with `ModuleNotFoundError: No module named 'pyrol'` through
`dimswe/test2a_pyrol.py`. Therefore zero M1-Y tests executed in this
environment. PyROL/Firedrake were not installed or stubbed, and the
environment-pinned bit-exactness assertions were not weakened.

The accepted H1/M2-Y equivalence fact remains separate:

```text
python -m pytest -q tests/test_test2a_h1_m2_equivalence.py
5 passed in 6.62 s
```

Arjun Sharma ran that command manually in the recorded Firedrake environment
on 2026-08-28. Codex did not rerun it in this pass.

## Preserved-state verification

The archaeological checkout was checked read-only against the original frozen
manifest after integration:

- branch `dev/dimswe-learned-physics-framework`;
- HEAD `d2f5d66ecb5500aad24eca37280f8a52e22a250f`;
- `git status --short` SHA-256
  `f6aad727ab0001a0976827a41bce322cf325ae0757a50bd4153aa7461754e076`;
- porcelain-v1 `-uall` SHA-256
  `b4dbc351a8a4d4bcc18bb2b037888b69a38813a75fa7d85f1abb443cd1ebad22`;
- porcelain-v2 `-uall` SHA-256
  `174a8eb8e7b8fb068b0935f2bcfb0b94d2cc6c8e2e7d2acc5fc3e5c785c4489d`;
- tracked `git diff` SHA-256
  `ab006aabcaf02020390a5ecc16b1db557dcec1d52ff3b0680f2cfe9ca1e666fd`;
- all 6,948 frozen evidence files rehashed, totaling 7,004,023,494 bytes,
  with 0 missing files and 0 content/size mismatches.

These values match the frozen forensic record. No command in this pass wrote
to the archaeological checkout.

## Generated-clutter disposition

Ignored `.DS_Store`, pytest-cache, and `__pycache__`/bytecode files found in the
current collaborator working copy were removed after validation. A final scan
found none. These were never imported into Git; their historical counterparts
remain preserved and hashed in the archaeological checkout.
