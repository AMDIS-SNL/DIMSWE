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
| Markdown local-link resolution, excluding fenced mathematical pseudocode | 47 links checked, 0 broken |
| explicit collaborator-facing `file.py:line[-line]` bounds | 94 citations checked, 0 errors |
| exact routine-at-line checks in `CODE_WALKTHROUGH.md` | 68 routines checked, 0 errors |
| active imported Python/shell machine-path scan | 0 `/Users/arjunsharma`, `/home`, or Windows-drive defaults |
| safe postprocessor `--help` smoke checks | 8 entry points passed; the 3 accepted-output writers are explicitly gated |

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
- 392 of those are byte-identical source-to-destination;
- 18 have an explicit behavior-preserving portability, safety,
  test-collection, or provenance/documentation edit;
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
tests/test_test2b_m1y_campaign.py

72 passed, 1 skipped in 13.42s
```

PyROL was unavailable, so the M1-Y campaign module skipped explicitly at its
dependency gate and zero M1-Y scientific assertions executed in this
environment. PyROL/Firedrake were not installed or stubbed, and the
environment-pinned bit-exactness assertions were not weakened.

## Final pre-share hardening

The root README now exposes the principal science, ML hook, neural model,
derivative/optimizer, canonical-experiment, and reproduction reading paths
without replacing the deeper `docs/README.md` index. M2-X is explicitly marked
as the historical X-state deployed-discrete control, and the M1-Y training
routine citation was rechecked against its current source extent.

The three W4 scripts that write accepted figures, table sidecars, or caption
inventories now require `--overwrite-accepted-assets`. Each `--help` invocation
exited 0 and each no-argument invocation refused before its original write
body with exit 2. Before and after those checks, the 169 accepted W4
data/figure/table/caption files totaled 44,039,287 bytes and had aggregate
SHA-256
`097852912a5dccffa102db698338f461a1428e7d2d9196fc097e757286b62508`.
The numerical and formatting bodies were not changed.

The external-artifact documentation now describes the actual enforcement
boundary: explicit receipt-time size/SHA-256 verification plus frozen input
hash records, rather than automatic rehashing by every consumer. The complete
415-row import manifest and all 32 external-artifact contracts were rehashed
successfully after these edits.

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
