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
| M1 | direct/a-priori regression to known rates or tendencies |
| M2-X | fixed/cacheable deployed-discrete loss at timestep-boundary truth `X*` |
| H1 / M2-Y | fixed/cacheable one-step loss at post-prefix truth `Y*=P(X*)` |
| H2 | first recursive objective; step two sees a learned-model state |
| H5 | the same recursion over five steps |

M2-X and H1 are offline/cacheable in this implementation. H2 is the first
objective with genuine model-generated-state feedback. Older Problem-A,
Problem-B, and M1--M4 wording is retained only as historical provenance.

## Start here

- [LEARNED_PHYSICS_EQUATIONS.md](LEARNED_PHYSICS_EQUATIONS.md): equations and objective semantics.
- [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md): module map and call graph.
- [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md): important routines and technology boundaries.
- [CHANGES_FROM_UPSTREAM_DIMSWE.md](CHANGES_FROM_UPSTREAM_DIMSWE.md): provenance relative to the Chris baseline.
- [CANONICAL_EXPERIMENTS.md](CANONICAL_EXPERIMENTS.md): evidence matrix and accepted limitations.
- [REPRODUCING_RESULTS.md](REPRODUCING_RESULTS.md): environment and reproduction entry points.
- [`docs/provenance/FINAL_VERIFICATION.md`](docs/provenance/FINAL_VERIFICATION.md): final integrity and static-check record.
- [`docs/provenance/SECOND_PASS_HYGIENE.md`](docs/provenance/SECOND_PASS_HYGIENE.md)
  and [`SECOND_PASS_VERIFICATION.md`](docs/provenance/SECOND_PASS_VERIFICATION.md):
  collaborator-surface decisions and the checks run after cleanup.

## Collaborator source surface

For the shortest route through the implementation, read these files in order:

1. `dimswe/physics.py` and `dimswe/timestepping.py` — Chris's analytical moist
   source and the modified split/backend hook;
2. `dimswe/moist_backend.py` and `dimswe/jax_moist_adapter.py` — the
   Firedrake/PETSc ↔ JAX boundary;
3. `dimswe/test2a_operator.py` and `dimswe/test2b_rain_learning.py` — features,
   networks, normalization, and Representation A/B/C output maps;
4. `dimswe/jax_moist_hvp.py` and `dimswe/mtswe_split_hvp.py` — child and
   complete-split tangent/adjoint/HVP paths;
5. `dimswe/test2a_discrete_training.py` and `dimswe/test2a_trajectory.py` —
   M2-X and H1/H2/H5 objectives; and
6. `dimswe/test2a_pyrol.py` and `dimswe/test2b_rain_learning_campaign.py` —
   parameter-vector/PyROL integration and the canonical A/B/C driver.

Canonical configurations live in `dimswe/configs/`, final regression tests in
`tests/`, campaign wrappers in `scripts/`, and frozen result summaries under
`external-results/`. Superseded development material is isolated under
`archive/` and is not part of this reading path.

## Repository policy

Source, tests, configurations, compact canonical parameter files, and compact
result summaries are suitable for Git. Multi-gigabyte truth trajectories,
fixed caches, raw VTK/checkpoint output, and archaeological campaigns remain
locally preserved under `external-results/`, ignored by default, and
hash-addressed in the disposition manifest. They require an artifact store or
release bundle for distribution to another machine.

No training campaign or B+ experiment was launched during Track 1 cleanup. The
cleanup does not intentionally change scientific behavior.
