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

## Repository policy

Source, tests, configurations, compact canonical parameter files, and compact
result summaries are suitable for Git. Multi-gigabyte truth trajectories,
fixed caches, raw VTK/checkpoint output, and archaeological campaigns remain
locally preserved under `external-results/`, ignored by default, and
hash-addressed in the disposition manifest. They require an artifact store or
release bundle for distribution to another machine.

No training campaign or B+ experiment was launched during Track 1 cleanup. The
cleanup does not intentionally change scientific behavior.
