# Reproducing and validating results

This document separates inspection/verification from expensive scientific
campaigns. Track 1 did not install dependencies, launch training, regenerate
truth, or run B+.

## 1. Environment contract

The recorded serial macOS environment is described in
`docs/DEVELOPMENT_ENVIRONMENT.md`. Its validated versions were Python 3.12.13,
Firedrake 2026.4.1, PETSc/petsc4py 3.25.0, Open MPI 5.0.9, mpi4py 4.1.2, JAX
0.11.0, `rol-python` 2025.9.10.dev1712, and PyROL API 0.1.0.

Required runtime properties are:

- JAX x64 enabled (`JAX_ENABLE_X64=True`);
- serial execution (`COMM_SELF` where applicable); the learned JAX moist bridge
  is not MPI-certified;
- CPU JAX, quadrilateral mesh, spatial order 3;
- Firedrake/PETSc double precision; and
- PyROL importable from `pyrol` for training.

Do not install or upgrade packages merely to make this snapshot look clean.
Create and validate a separate environment if the recorded environment is not
available.

## 2. Artifact prerequisites

An ordinary Git clone contains source, configs, reports, compact final parameter
artifacts, and compact result summaries. Full reproduction also needs the
hash-addressed sidecar data retained locally in this snapshot.

Critical Test 2B files include:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `external-results/test2b-rain-active-learning/preparation/fixed_learning_data.npz` | 351,575,764 | `6e159015234fd94881b0b97888b7481eb049a02dcd96c571078920c0bedc901c` |
| `external-results/test2b-rain-active-learning/production/representation-A/representation_a_final_comparison.json` | compact | `75506a833862b69437a29a7a2b30e64c361d35b8560342bf8abe93302daf5b7f` |
| `.../representation-B/representation_b_final_comparison.json` | compact | `6044c0fbd42484e3bd6f0ec53bef91d9a871fa315fc83c193c10f1879813aadd` |
| `.../representation-C/representation_c_final_comparison.json` | compact | `8bc1d9fad90d1d5907c3ff8bc4a5e396ae09ce34f4d887d1f77ad429dfbba926` |

Critical Test 2A caches include
`external-results/test2a/deployed-discrete-offline/fixed_operator_cache.npz`
(11,246,260 bytes, SHA-256
`baee2dd3ae8a5e3f9ec16f6883e3583d4ac61281d777c3079b002e611504bacf`)
and the post-prefix H1 cache named in
`dimswe/configs/test2a_horizon_curriculum_h1_h2_h5.json`.

The complete transfer/check contract is
`docs/provenance/FROZEN_DIRTY_STATE_MANIFEST.tsv`. Compare received artifacts by
path, byte count, and SHA-256 before running any evaluation.

## 3. Cheap read-only verification

From the repository root, these commands inspect provenance without running the
model:

```sh
git rev-parse HEAD
git merge-base --is-ancestor d0eb61598a2cb1049628c3cc054ab9a1f3143bf6 HEAD
shasum -a 256 \
  external-results/test2b-rain-active-learning/production/representation-A/representation_a_final_comparison.json \
  external-results/test2b-rain-active-learning/production/representation-B/representation_b_final_comparison.json \
  external-results/test2b-rain-active-learning/production/representation-C/representation_c_final_comparison.json
```

All JSON reports/configs can be syntax-checked without Firedrake. The Track 1
verification record lists exactly which checks were actually run; do not infer
that a documented command was executed by Codex.

## 4. Test entry points

Tests under `tests/` fall into three groups:

- pure Python/JAX local algebra and codec tests;
- serial Firedrake/PETSc primal and derivative tests; and
- optional PyROL tests guarded by dependency availability.

The most direct regression for the corrected H1 semantics is
`tests/test_test2a_h1_m2_equivalence.py`. The accepted manual fact is:

```text
python -m pytest -q tests/test_test2a_h1_m2_equivalence.py
5 passed in 6.62 s
```

This was run manually by Arjun Sharma in the recorded Firedrake environment on
2026-08-28. Codex did not run or rerun it during the forensic audit.

Run tests only in a configured serial environment. Tests may create temporary
Firedrake compiler caches; `tests/conftest.py` redirects those caches to a
per-process temporary location.

## 5. Evaluation and postprocessing entry points

The canonical Test 2B CLI is:

```sh
python -m dimswe.test2b_rain_learning_campaign --help
```

Its `postprocess` command consumes existing artifacts and writes a new comparison
record; its `certify` and `certify-oracles` commands execute numerical derivative
checks. They are evaluations, not training, but still require the complete
Firedrake/JAX environment and sidecar cache.

Representation-specific final evaluation is implemented in:

- `dimswe/test2b_representation_a_postprocess.py`;
- `dimswe/test2b_representation_b_postprocess.py`; and
- `dimswe/test2b_representation_c_postprocess.py`.

The historical Test 2A entry points are:

- `python -m dimswe.test2a_pyrol --help` for M1;
- `python -m dimswe.test2a_discrete_training --help` for cached M2-X;
- `python -m dimswe.test2a_horizon_curriculum --help` for H1/H2/H5; and
- `python -m dimswe.test2a_problem_b_campaign --help` for historical Problem B
  (authoritative Representation C terminology).

## 6. Training campaigns: explicit authorization required

The `train` subcommands and `scripts/run_test2*.sh` files are preserved
reproduction entry points, not routine validation commands. They may consume
hours, overwrite a chosen output directory if a guard is bypassed, and create
new scientific results. Before any rerun:

1. verify every input hash and the clean collaborator commit;
2. select a new, empty output root so frozen evidence is never overwritten;
3. record environment/package versions, configuration hash, initial parameter
   hash, and sidecar-cache hash;
4. preserve independent-versus-warm-start semantics;
5. start each continuation stage with the recorded empty ROL L-BFGS history;
6. keep held-out 81--160 data forbidden for training selection; and
7. label the result as a reproduction, not the frozen accepted artifact.

Do not run B+ merely to populate this repository. Its current scientific status
is `PREPARED_NOT_RUN`, and changing that status requires a separately authorized
campaign and review.

## 7. Expected outputs

Each final fit directory contains:

- `final_parameters.npz`: numeric JAX pytree leaves;
- `final_parameters.json`: architecture, representation, stage, and pytree hash;
- `fit_result.json`: optimizer termination, counts, objective, and provenance;
- optional progress/checkpoint records retained locally.

The final comparison JSON must be created in evaluation-only mode and record
that no optimizer was instantiated and no truth was regenerated. Cross-check
all six final sidecars against the comparison record before updating any
synthesis.
