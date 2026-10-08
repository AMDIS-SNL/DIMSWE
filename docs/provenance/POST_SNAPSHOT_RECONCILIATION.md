# Post-snapshot reconciliation provenance

## Scope and review independence

This update began from the preserved Codex cleanup commit
`24cc40a8bb4d23cde88844c401c850578fee5d09` on a new branch,
`collaborator/track1-current-science-20260831`. The preserved
`collaborator/track1-closeout` branch was not modified.

Claude Code acted only as an independent reviewer. Codex read and hashed:

| External review | SHA-256 |
|---|---|
| `PRE_SHARE_REVIEW.md` | `d41fdec620e0e1bc34ae1dadec7390433fc6baa7b0eed918ca770060cd4264df` |
| `M1Y_RECONCILIATION_REPORT.md` | `77edc3a296bccda24fc5f72aa7e8def14374802ec397783e256638370ba46ea3` |
| `POST_SNAPSHOT_RECONCILIATION.md` | `c52ad451c482417bc41fea416c1c85b698eaecc64696670bbc0bc1327ec093e1` |

Claude's `review/claude-pre-share` branch was not merged. Codex independently
checked symbols and current line numbers before reproducing the verified
`dimswe/output.py` and `mtswe_split_hvp.py` citation corrections.

## Scientific verification

Source and frozen artifacts independently established:

- M1-X uses `x_features`, `x_A`, and `x_R` at boundary truth `X*`.
- M1-Y constructs `Y*=P(X*)` with
  `take_forward_step_cached(...).boundary_states[-2]`, then caches both
  features and analytical targets at Y.
- M1-Y is a fixed-array, nonrecursive `OperatorObjective`; its optimization
  does not execute or differentiate through the prefix.
- A/B/C M1-Y fits completed 10,000 accepted iterations with the frozen
  5-32-32-`d`, tanh, float64, seed-zero architecture and historical X-fitted
  normalization.
- The accepted parameter-only genealogy is
  `M1-X -> H1 -> H2 -> H5`; M1-Y is independent and initializes none of that
  ladder.
- H2 remains the first objective whose second learned-physics call receives a
  model-generated state.

No material scientific or provenance contradiction with the reviewer reports
was found. One stale W4 README count said eight main figures; the final reset
provenance and actual accepted directory both contain seven. The collaborator
documentation uses seven.

## Workstream disposition

| Workstream | Collaborator disposition | Rationale |
|---|---|---|
| W1 `feature_sufficiency_20260828` | Two state-location/stop records retained under `docs/provenance/m1_state_location_20260828/` | Canonical discovery provenance; the proposed feature study itself stopped |
| W2 `m1y_test2b_20260828_workspace` | M1-Y source, config, regression, compact caches/sidecars, checkpoints/final models, matched evaluations, and reports imported | Completed canonical A/B/C M1-Y campaign |
| W3 `ml_results_audit_20260829` | Run inventory, objective matrix, training-history availability, M1 cross-state metrics, audit synthesis/provenance imported | Required quantitative census and W4 inputs; one-time hard-coded builders excluded |
| W4 `ml_results_postprocess_20260829` | Canonical quantitative scripts, compact inputs, tables, final/supplement figures, and provenance imported | Accepted evaluation-only manuscript chain |
| W5 `deployed_hybrid_dynamics_20260830` | Replay/render source, hash sidecars, validation, accepted galleries, and movie sidecars imported | Exactly M1-Y/H1/H2/H5 × A/B/C; scalar replay parity passed |
| W6 `DIMSWE-groundtruth-figures-20260829` | Deterministic generators, case/physics docs, compact diagnostics, and accepted Figures 1/2/3/5 imported | Canonical non-ML ground-truth package |
| W7 `DIMSWE-track2-analysis-20260828` | Not imported | Superseded planning, failed attempts, caches, and temporary output; no unique current collaborator value established |

## Existing-file collision rule

The cleaned collaborator version won every collision. No later copy of an
existing scientific source file replaced the hygiene-cleaned version. M1-Y
modules/config/test were additive. The only pre-existing runtime file changed
for integration is `scripts/reproduction_environment.sh`, which now exports
its already validated repository root for child postprocessors.

Machine-specific paths were removed only from runnable imported source. The
portable helpers accept `DIMSWE_REPOSITORY`,
`DIMSWE_REFERENCE_REPOSITORY`, `DIMSWE_M1Y_REPOSITORY`,
`DIMSWE_ML_RESULTS_AUDIT_ROOT`, and `DIMSWE_GROUND_TRUTH_PACKAGE`.
Historical absolute paths inside immutable results/provenance remain unchanged.

## Deliberate exclusions

- M1-Y learning and held-out NPZ caches;
- W4 held-out-X and training-carrier NPZ caches;
- twelve W5 replay-map NPZ caches and twelve GIF movies;
- W6 truth-map NPZ caches, frames, and GIF movies;
- W4/W5 one-time archive movers, dirty-worktree fingerprint closers, and
  continuation helpers;
- W5 rejected visual-audit pilot;
- W6 retired figures;
- W3 hard-coded one-time audit builders;
- W7 in full;
- `.DS_Store`, bytecode, cache directories, and Matplotlib cache files.

Large required artifacts are hash-addressed in
`docs/provenance/EXTERNAL_ARTIFACTS.md`. The file-by-file copy/hash mapping is
`docs/provenance/POST_SNAPSHOT_IMPORT.tsv`.

The import table has 415 rows: 410 map to later-workspace files, of which 392
are byte-identical and 18 carry an explicitly recorded behavior-preserving
portability, safety, test-collection, or provenance/documentation edit. Four
rows are Codex-created integration files and one is the pre-existing
collaborator environment helper extended to export its validated repository
root. All source and destination hashes were recomputed successfully after
the final pre-share hardening pass.

## Safety statement

This pass copied, checked, documented, and rendered no new scientific result.
It did not train, optimize, regenerate truth, replay a model, run B+, or change
physics/objective/architecture/normalization/optimizer semantics. The
authoritative archaeological checkout remained read-only, and nothing was
pushed.
