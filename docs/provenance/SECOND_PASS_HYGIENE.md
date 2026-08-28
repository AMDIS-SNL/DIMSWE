# Track 1 second-pass collaborator code hygiene

Date: 2026-08-28 (America/Denver)

This pass began from collaborator commit
`10a0482a16cb190c2012550a40a72af53893abfe`. It audited the committed
post-Chris source surface, not the scientific validity of the already frozen
results. The authoritative archaeological checkout was treated as read-only.

## Decision rule

A candidate stayed in the normal tree when it is needed for canonical
reproduction, scientific validation, regression protection, equivalence
certification, accepted postprocessing, result provenance, or an implemented
representation/objective. A closed precursor or incomplete launcher moved to
`archive/` only after import, CLI, config, manifest, and report references were
traced. No numerical algorithm, network, feature normalization, output map,
objective, optimizer setting, or accepted configuration was changed.

The item-level archive map, including source hashes, is
`docs/provenance/SECOND_PASS_DISPOSITION.tsv`.

## Candidate classifications

| Candidate | Classification | Basis |
|---|---|---|
| `dimswe/test2a_operator.py`, `test2b_rain_learning.py`, deployed bridge/derivative modules, objective modules, campaign drivers and A/B/C postprocessors | `KEEP_CANONICAL` | actual canonical call graph and accepted evidence matrix |
| `tests/test_test2a_h1_m2_equivalence.py`, JAX/Firedrake derivative tests, complete-split tangent/adjoint tests, backend-offset and B+ preparation tests | `KEEP_REGRESSION` | mathematical equivalence, backend parity, derivative duality, and source-invariant protection |
| Test 2A optimizer/continuation source, configs, tests, and reports | `KEEP_REPRODUCTION` | generated the frozen practical M1 artifact still named by embedding, trajectory, and evaluation configs |
| the three deployed-discrete configs (`offline`, `direct50k`, cached `50k`) and Test 2A-3A/3B/3C reports | `KEEP_REPRODUCTION` / `KEEP_PROVENANCE` | distinct preparation, accepted direct-fit, and certified cached-operator contracts; not duplicate configs |
| `dimswe/test2_moist_activity.py` and its test/report | `KEEP_USEFUL_DIAGNOSTIC` | established the A-active/R-weak support and the Representation A boundary |
| `dimswe/test2a_backend_offset_audit.py` and its test/audit | `KEEP_USEFUL_DIAGNOSTIC` | corrected a material backend-routing interpretation and checks UFL/JAX parity |
| `dimswe/test2a_problem_b_signed_water_budget.py` and its test | `KEEP_USEFUL_DIAGNOSTIC` | accepted source-manifold/signed-water evidence for unconstrained tendencies |
| Test 1A/1B, hidden-c0, hyperviscosity stability, `ode_adjoint/hvp.py`, and their tests/audits | `KEEP_PROVENANCE` / `KEEP_REGRESSION` | trace the selected truth and production derivative construction; still provide independent safety checks |
| `dimswe/test2b_rain_truth.py`, its driver/config/test, and the segment-1 runner | `KEEP_REPRODUCTION` / `KEEP_PROVENANCE` | reproduces the completed dry-refinement result and supplies rain/source audit utilities used by the accepted truth driver |
| generic horizon stage plus H1/H2/H5 wrappers | `KEEP_REPRODUCTION` | wrappers encode stage-specific input hashes, predecessor validation, and parameter-only restart semantics; not redundant aliases |
| `dimswe/test2b_representation_bplus_prepare.py`, B+ output-map preparation, and tests | `KEEP_REGRESSION` / `KEEP_PROVENANCE` | B+ remains prepared and derivative-certified, not run; provider safety checks are scientific assets |
| isolated Firedrake HVP prototype package/test/design note | `ARCHIVE_HISTORICAL` | no production caller; superseded by DIMSWE child/split derivative implementations and regression tests |
| Test 2A residual-structure source/test/report | `ARCHIVE_HISTORICAL` | one-off endpoint investigation with no canonical caller; conclusion frozen elsewhere |
| BTP/BTPL and B+ launch scripts plus BTP/BTPL preparation note | `ARCHIVE_HISTORICAL` | not canonical accepted campaigns; retaining them in `scripts/` suggested an unsupported reproduction status |
| Representation A interim M1/M2-X report | `ARCHIVE_HISTORICAL` | explicitly superseded by the final six-fit synthesis |
| first-pass disposition generator | `KEEP_PROVENANCE` (archived) | useful chain-of-custody tool, but not a collaborator scientific/reproduction entry point |
| local-only BTP/BTPL result trees | `UNCERTAIN` | relationship to the frozen B+ claim remains incomplete; retained untouched and unpromoted under `external-results/` |

No candidate was permanently discarded from the collaborator snapshot. The
closed groups above were moved into the explicit archive, and their original
layout remains available in the authoritative checkout.

## Source and runner cleanup

- Added `scripts/reproduction_environment.sh`. Retained campaign scripts now
  discover the repository from their own location and accept optional
  `DIMSWE_REPOSITORY`, `DIMSWE_VIRTUAL_ENVIRONMENT`, and `DIMSWE_PYTHON`
  settings.
- Replaced the sole machine-specific path in an active config,
  `dimswe/configs/test2a_m20_plus45000_continuation.json`, with the equivalent
  repository-relative artifact path.
- Replaced machine-specific paths in active reproduction documentation with
  placeholders. Active source, configurations, and runners contain no literal
  `/Users` or `/home` paths. Absolute paths remain where they are deliberate
  chain-of-custody/audit evidence, in byte-preserved archive files, and in 84
  frozen result JSON files whose recorded output paths are scientific
  provenance rather than executable defaults.
- Eight case-building/postprocessing locations use a `/tmp/...-no-output`
  value to satisfy the nonempty `ResolvedPilotConfiguration.output_directory`
  contract without invoking a result writer. Ten campaign runners use
  `mktemp -d /tmp/...` for disposable runtime/compiler caches. These paths are
  operational scratch sentinels, not scientific artifact inputs or outputs;
  they were retained rather than introducing unvalidated runtime plumbing.
- Examined progress prints and internal `*_debug` records. Retained prints are
  CLI progress/reporting or test diagnostics; retained debug records expose
  zero-derivative safety checks. No dead debug branch was removed from a
  canonical numerical module.
- Examined commented code/TODO/HACK matches. Matches in original upstream files
  predate the Chris baseline or are descriptive constraints; no safe
  post-Chris numerical deletion was identified.
- Removed 24 statically verified unused import lines from 14 post-Chris Python
  modules. This includes unused convenience imports and type names only; no
  executable statement, branch, function, class, numerical expression, or
  public interface was changed.

## Duplicate and unused-code findings

The apparently duplicate M2-X configs encode different scientific stages and
iteration/certification contracts. The horizon wrappers encode real dependency
and restart checks. The early optimizer and continuation programs remain the
reproduction chain for a still-referenced frozen model. Removing or merging any
of these would either break provenance or change behavior, so they were kept.

No obsolete CLI option or unused private helper inside a canonical numerical
module was removed: static reference tracing did not establish a safe candidate
whose deletion was independent of Firedrake/JAX/PyROL runtime behavior.

A repository-local import scan found one unresolved legacy import,
`dimswe/run_model_set.py -> dimswe.model`. Both the file and import predate the
Chris baseline `d0eb61598a2cb1049628c3cc054ab9a1f3143bf6`; this pass did not
rewrite upstream code without evidence that doing so would preserve behavior.
The remaining imports reported as unused by the conservative AST scan are in
the original `dimswe/timestepping.py` import block and likewise predate the
Chris baseline. They were retained.
