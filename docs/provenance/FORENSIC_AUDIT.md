# Track 1 forensic audit record

Date: 2026-08-28 (America/Denver)

This is a report of the frozen authoritative checkout before collaborator
cleanup. Commands were read-only. The complete per-file evidence is in the TSV
manifests beside this document.

## Frozen Git state

- repository/top level:
  `/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615`
- branch: `dev/dimswe-learned-physics-framework`
- HEAD: `d2f5d66ecb5500aad24eca37280f8a52e22a250f`
- origin fetch: `https://github.com/AMDIS-SNL/DIMSWE.git`
- origin push: `DISABLE_PUSH`
- staged diff: empty
- unstaged tracked diff: 10 files, 819 insertions, 58 deletions
- untracked inventory: 6,938 files, approximately 6.523 GiB
- total frozen evidence: 6,948 modified/untracked files
- SHA-256 of `git status --short`: `f6aad727ab0001a0976827a41bce322cf325ae0757a50bd4153aa7461754e076`
- SHA-256 of `git status --porcelain=v2 -uall`:
  `174a8eb8e7b8fb068b0935f2bcfb0b94d2cc6c8e2e7d2acc5fc3e5c785c4489d`

The ten modified tracked files were:

1. `dimswe/configs/resolved_hidden_c0_pilot.cfg`
2. `dimswe/initial_conditions.py`
3. `dimswe/jax_moist_adapter.py`
4. `dimswe/jax_moist_hvp.py`
5. `dimswe/moist_backend.py`
6. `dimswe/mtswe_split_hvp.py`
7. `dimswe/resolved_hidden_c0.py`
8. `dimswe/resolved_hidden_c0_driver.py`
9. `dimswe/timestepping.py`
10. `docs/LEARNED_PHYSICS_EXPERIMENTS.md`

## Untracked inventory summary

| Root | Files | Approximate role |
|---|---:|---|
| `external-results/test2b-rain-active-case-design` | 1,961 | preparatory/case-design archaeology |
| `external-results/test2b-rain-active-truth` | 1,294 | generated truth/reproduction data |
| `external-results/test2a` | 917 | Test 2A canonical plus earlier/superseded campaigns |
| `external-results/test2b-preparation` | 747 | preparatory truth/case archaeology |
| `external-results/test1b0` | 698 | Test 1B archaeology |
| `external-results/test1b-production` | 672 | Test 1B production evidence |
| `external-results/test2b-rain-active-learning` | 428 | final A/B/C, B+ preparation, and uncertain partial BTP/BTPL evidence |
| `dimswe/` | 119 | source and configs plus generated cache files |
| `docs/` | 43 | scientific syntheses/reports plus `.DS_Store` |
| `tests/` | 38 | scientific/regression tests |
| `scripts/` | 18 | drivers and postprocessors |
| other external roots/root clutter | 13 | earlier preparation and generated clutter |

By MIME/type, the largest scientific payloads were 1,323 HDF5 files
(approximately 4.636 GiB), 1,323 NumPy files (approximately 1.376 GiB), 1,799
ZIP archives (approximately 0.443 GiB), and 2,122 JSON/text-like records.
`external-results/` held 6,729 files and approximately 6.520 GiB.

## Hash manifests

The freeze manifest has 6,949 lines including its header and records every
file's status, bytes, SHA-256, MIME, and path. Its SHA-256 is
`010801095f7f4f93dbd54f6c3a70aa742acdbfde2c1fcb36b33923f8a0059a36`.

After the filesystem copy received independent Git metadata, every one of the
6,948 evidence paths was re-read and compared. Result: `MATCH 6948`. The
verification table's SHA-256 is
`75281692acf1d78927ce6ead7b3e4bc3a0906196d6e284c4dc8da7f885936649`.

## Baseline reconstruction

`BASE_CANDIDATE = d0eb61598a2cb1049628c3cc054ab9a1f3143bf6`

`CONFIDENCE = HIGH`

`d0eb615` is the last pre-Track-development commit on the supplied branch and
is ancestral to `d2f5d66`. The next commits begin the environment,
characterization, ROL, HVP, JAX-moist, and learned-physics sequence. GitHub
`main` (`149fe55`) is older and lacks later coefficient-scaling work already in
the supplied state. No competing later Chris tag/ref was found.

The baseline-to-frozen tracked diff contains 94 files, 42,906 insertions, and
44 deletions. Most additions are new derivative/JAX/learned-physics modules,
tests, and docs; original source modifications are concentrated in
`dimswe/initial_conditions.py`, `dimswe/timestepping.py`, the `ode_adjoint`
prototype, and a small set of legacy tests.

## Audit limits

- No dependency was installed.
- No experiment, training campaign, or B+ campaign was run.
- No source checkout file was modified, deleted, renamed, reset, stashed, or
  committed.
- Reports were treated as evidence and cross-checked against source, configs,
  result JSON, hashes, and test names; they were not assumed infallible.
- The manual H1/M2-Y regression result documented elsewhere was supplied by
  Arjun Sharma and was not produced by Codex.
