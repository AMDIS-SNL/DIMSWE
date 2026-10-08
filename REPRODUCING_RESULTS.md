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
- CPU backend selected explicitly (`JAX_PLATFORMS=cpu`);
- serial execution (`COMM_SELF` where applicable); the learned JAX moist bridge
  is not MPI-certified;
- CPU JAX, quadrilateral mesh, spatial order 3;
- Firedrake/PETSc double precision; and
- PyROL importable from `pyrol` for training.

Do not install or upgrade packages merely to make this snapshot look clean.
Create and validate a separate environment if the recorded environment is not
available.

The shell entry points discover the repository relative to `scripts/`; they no
longer contain machine-local checkout or environment paths. Normally activate
the environment before invoking a runner. Alternatively set one or more of:

```sh
export DIMSWE_REPOSITORY=/path/to/DIMSWE-collaborator
export DIMSWE_VIRTUAL_ENVIRONMENT=/path/to/dimswe-firedrake-environment
export DIMSWE_PYTHON=/path/to/dimswe-firedrake-environment/bin/python
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
```

The shared validation logic is in `scripts/reproduction_environment.sh`.

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

Post-snapshot direct-regression/evaluation caches include:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz` | 187,181,823 | `6f16e6db2c6ebdbd8c00a23cdae9b5318355384723a2f1276b2ea93d95145668` |
| `external-results/m1y-test2b-20260828/evaluation/m1y_heldout_data.npz` | 216,940,889 | `1ddfa2d2e28b6f8dc2a0fbe0a12d2fe7da42158745a70eb2e088706501c42d2f` |
| `postprocessing/ml_results_20260829/data/heldout_x_test2b.npz` | 332,789,878 | `fd55559e2eb3277228099106c8043d3c1d11848a83c77efb87a3f4373f03274f` |
| `postprocessing/ml_results_20260829/data/training_x_carriers_test2b.npz` | 29,434,314 | `0c6bf9378fd38eace300d0b2a6b8d6efdcfc865493a910b12e7b1a4a25a69bf4` |

Critical Test 2A caches include
`external-results/test2a/deployed-discrete-offline/fixed_operator_cache.npz`
(11,246,260 bytes, SHA-256
`baee2dd3ae8a5e3f9ec16f6883e3583d4ac61281d777c3079b002e611504bacf`)
and the post-prefix H1 cache named in
`dimswe/configs/test2a_horizon_curriculum_h1_h2_h5.json`.

The original transfer/check contract is
`docs/provenance/FROZEN_DIRTY_STATE_MANIFEST.tsv`. Post-snapshot M1-Y,
quantitative-postprocessing, replay, ground-truth-map, and movie requirements
are enumerated separately in
`docs/provenance/EXTERNAL_ARTIFACTS.md`. Compare received artifacts by path,
byte count, and SHA-256 before running any evaluation.

Some frozen JSON result records retain the absolute path at which an artifact
was originally produced. Those strings are immutable run provenance, not
portable defaults. Supply the current configuration/artifact paths through the
documented CLI when evaluating a transferred snapshot; do not edit an accepted
result record merely to relocate it.

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

The dedicated M1-Y regression is
`tests/test_test2b_m1y_campaign.py`. It checks the Y-state contract, exact
feature/target ordering, frozen architecture, and seed-zero parameter hashes.
Importing its campaign module requires PyROL in addition to JAX.

Bit-exact cache/equivalence assertions are environment-pinned certifications:
they depend on the recorded Firedrake/PETSc/JAX versions, float64 CPU execution,
mesh/layout, and deterministic ordering. Failure in a materially different
environment is not license to loosen a tolerance or rewrite the accepted
artifact; first reproduce the recorded environment.

Run tests only in a configured serial environment. Tests may create temporary
Firedrake compiler caches; `tests/conftest.py` redirects those caches to a
per-process temporary location.

The inherited GitHub workflow does not provision the complete recorded
Firedrake/JAX/PyROL stack. Its status is therefore not a substitute for these
environment-pinned regressions.

## 5. Evaluation and postprocessing entry points

The canonical Test 2B CLI is:

```sh
python -m dimswe.test2b_rain_learning_campaign --help
```

The current direct-regression baseline has separate preparation/training and
evaluation CLIs:

```sh
source scripts/reproduction_environment.sh
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu

"$PYTHON" -m dimswe.test2b_m1y_campaign --help
"$PYTHON" -m dimswe.test2b_m1y_evaluation --help
```

To evaluate a transferred frozen M1-Y model without training, first verify the
two cache hashes, then run `dimswe.test2b_m1y_evaluation evaluate` with the
configuration, learning cache, held-out cache, representation, and a **new**
output path. `prepare-heldout` replays the analytical prefix and should be
used only when the immutable held-out cache cannot be transferred.

Its `postprocess` command consumes existing artifacts and writes a new comparison
record; its `certify` and `certify-oracles` commands execute numerical derivative
checks. They are evaluations, not training, but still require the complete
Firedrake/JAX environment and sidecar cache.

Representation-specific final evaluation is implemented in:

- `dimswe/test2b_representation_a_postprocess.py`;
- `dimswe/test2b_representation_b_postprocess.py`; and
- `dimswe/test2b_representation_c_postprocess.py`.

The accepted manuscript-facing packages are:

- `postprocessing/ml_results_20260829/`: quantitative fixed-array/fixed-map
  postprocessing, tables, and main/supplement figures;
- `postprocessing/deployed_hybrid_dynamics_20260830/`: twelve frozen-model
  replay sidecars, replay/render scripts, and thirteen accepted galleries; and
- `postprocessing/ground_truth_figures_20260829/`: deterministic analytical
  truth diagnostics and Figures 1, 2, 3, and 5.

The path helpers use this repository by default. A transferred external mirror
can be selected with:

```sh
export DIMSWE_REFERENCE_REPOSITORY=/path/to/reference-artifact-tree
export DIMSWE_M1Y_REPOSITORY=/path/to/m1y-artifact-tree
export DIMSWE_ML_RESULTS_AUDIT_ROOT=/path/to/ml_results_audit_20260829
export DIMSWE_GROUND_TRUTH_PACKAGE=/path/to/ground_truth_figures_20260829
```

Canonical evaluation/render entry points are:

```sh
"$PYTHON" postprocessing/ml_results_20260829/scripts/complete_final_callsite_y_metrics.py
"$PYTHON" postprocessing/ml_results_20260829/scripts/generate_final_main_paper_assets.py
"$PYTHON" postprocessing/deployed_hybrid_dynamics_20260830/scripts/replay_spatial_maps.py --help
"$PYTHON" postprocessing/deployed_hybrid_dynamics_20260830/scripts/render_spatial_package.py --help
"$PYTHON" postprocessing/ground_truth_figures_20260829/scripts/ground_truth_figures_20260829/make_ground_truth_figures.py --help
```

The accepted directories already contain outputs. Several W4/W5 entry points
refuse collisions; the W6 generators write their selected output directory and
some final-asset tasks intentionally replace their own task outputs after
establishing a baseline. Run these commands only in a disposable package copy
or with a new explicit output path. Missing NPZ/movie inputs must be restored
by exact hash; the scripts never substitute another cache.

The historical Test 2A entry points are:

- `python -m dimswe.test2a_pyrol --help` for historical M1-X;
- `python -m dimswe.test2a_discrete_training --help` for cached M2-X;
- `python -m dimswe.test2a_horizon_curriculum --help` for H1/H2/H5; and
- `python -m dimswe.test2a_problem_b_campaign --help` for historical Problem B
  (authoritative Representation C terminology).

## 6. Training campaigns: explicit authorization required

The `train` subcommands and the retained `scripts/run_test2*.sh` files are
canonical or accepted-provenance reproduction entry points, not routine
validation commands. They may consume
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

For the historical recursive ladder, preserve the accepted parameter-only
genealogy `M1-X -> H1 -> H2 -> H5`. M1-Y is a separate seed-zero fit and
must not be used to restart H1/H2/H5 without explicit authorization for a new
scientific campaign.

Do not run B+ merely to populate this repository. Its current scientific status
is `PREPARED_NOT_RUN`, and changing that status requires a separately authorized
campaign and review. Historical BTP/BTPL/B+ launch sketches are intentionally
isolated under `archive/development-history/test2b_constrained_rain_variants/`.

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
