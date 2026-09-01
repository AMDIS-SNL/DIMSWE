# DIMSWE learned-physics collaborator snapshot

This repository is the Track 1 collaborator-facing snapshot of the AMDIS/DIMSWE
learned-physics work. It preserves the accepted scientific state, including work
that was not committed at the former development `HEAD`, while keeping its
relationship to Chris's DIMSWE code explicit.

The authoritative archaeological checkout is **not** this directory and must
remain untouched:

`/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615`

The collaborator snapshot was made from that checkout only after freezing and
hashing all 6,948 modified or untracked evidence files. See
[PROVENANCE_AND_DISPOSITION.md](PROVENANCE_AND_DISPOSITION.md) and
[`docs/provenance/DISPOSITION_MANIFEST.tsv`](docs/provenance/DISPOSITION_MANIFEST.tsv).

## Start here

| Goal | Read |
|---|---|
| Overview | this `README.md` |
| Science and objective definitions | [LEARNED_PHYSICS_EQUATIONS.md](LEARNED_PHYSICS_EQUATIONS.md) |
| Original DIMSWE → learned-physics hook | [`dimswe/physics.py`](dimswe/physics.py), [`dimswe/timestepping.py`](dimswe/timestepping.py), [`dimswe/moist_backend.py`](dimswe/moist_backend.py), [`dimswe/jax_moist_adapter.py`](dimswe/jax_moist_adapter.py) |
| Neural physics | [`dimswe/test2b_rain_learning.py`](dimswe/test2b_rain_learning.py), [`dimswe/test2b_m1y_campaign.py`](dimswe/test2b_m1y_campaign.py) |
| Tangent, adjoint, and optimization | [`dimswe/jax_moist_hvp.py`](dimswe/jax_moist_hvp.py), [`dimswe/mtswe_split_hvp.py`](dimswe/mtswe_split_hvp.py), [`dimswe/test2a_trajectory.py`](dimswe/test2a_trajectory.py), [`dimswe/test2a_pyrol.py`](dimswe/test2a_pyrol.py) |
| What was actually run | [CANONICAL_EXPERIMENTS.md](CANONICAL_EXPERIMENTS.md) |
| How to reproduce it | [REPRODUCING_RESULTS.md](REPRODUCING_RESULTS.md) |

For the full module map and routine-level guide, continue with
[CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) and
[CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md). Provenance relative to Chris's
baseline is in [CHANGES_FROM_UPSTREAM_DIMSWE.md](CHANGES_FROM_UPSTREAM_DIMSWE.md),
with the complete documentation index in [docs/README.md](docs/README.md).

## Scientific vocabulary

Physics representation and training objective are independent axes.

| Representation | Learned quantity | Structure retained |
|---|---|---|
| A | `A_theta` | analytical `R` and exact moist-source map |
| B | `(A_theta, R_theta)` | exact moist-source map |
| C | complete moist tendency vector | no analytical source constraint |
| B+ | constrained learned rain rate | prepared and derivative-certified; no completed full campaign established |

| Objective | Actual execution |
|---|---|
| M1-X | direct physical-law regression with features and targets at timestep-boundary truth `X*`; historical state-location control |
| M1-Y | direct physical-law regression with features and targets at `Y*=P(X*)`; current direct-regression baseline |
| M2-X | fixed/cacheable deployed-discrete loss at timestep-boundary truth `X*`; historical state-location control |
| M2-Y / H1 | fixed/cacheable one-step loss at post-prefix truth `Y*=P(X*)` |
| H2 | first recursive objective; step two sees a learned-model state |
| H5 | the same recursion over five steps |

M1-X, M1-Y, M2-X, and M2-Y/H1 are offline/cacheable in this implementation.
H2 is the first objective with genuine model-generated-state feedback. M1-Y
corrects the direct-regression sampling location; it does not initialize the
recursive ladder. The verified initialization genealogy is:

```text
M1-X -> H1 -> H2 -> H5

M1-Y (independent seed-zero fit; initializes none of the ladder)
```

Older Problem-A, Problem-B, and M1--M4 wording is retained only as historical
provenance.

## Repository orientation

Canonical configurations live in `dimswe/configs/`, final regression tests in
`tests/`, campaign wrappers in `scripts/`, and frozen result summaries under
`external-results/`. Manuscript quantitative results, deployed-hybrid
galleries, and deterministic ground-truth figures are under `postprocessing/`.
Superseded development material is isolated under `archive/` and is not part
of this reading path.

The root-level packaging, license, and legacy runner files are inherited DIMSWE
scaffolding. They are not the learned-physics front door; begin with the files
above. The inherited GitHub CI predates the complete Firedrake/JAX/PyROL
campaign environment and is not evidence that the environment-pinned numerical
regressions ran.

Integrity and cleanup records are indexed under `docs/provenance/`, including
`POST_SNAPSHOT_VERIFICATION.md`, `FINAL_VERIFICATION.md`, and
`SECOND_PASS_HYGIENE.md`.

## Repository policy

Source, tests, configurations, compact canonical parameter files, and compact
result summaries are suitable for Git. Multi-gigabyte truth trajectories,
fixed caches, raw VTK/checkpoint output, and archaeological campaigns remain
locally preserved under `external-results/`, ignored by default, and
hash-addressed in the disposition manifest. Post-snapshot cache and movie
requirements are listed in
[`docs/provenance/EXTERNAL_ARTIFACTS.md`](docs/provenance/EXTERNAL_ARTIFACTS.md).
They require an artifact store or release bundle for distribution to another
machine.

No training campaign or B+ experiment was launched during Track 1 cleanup. The
cleanup does not intentionally change scientific behavior.
