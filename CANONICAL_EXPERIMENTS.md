# Canonical experiments and evidence matrix

## Status vocabulary

- `CANONICAL_ACCEPTED`: completed evidence supports the stated, bounded claim.
- `CANONICAL_WITH_LIMITATION`: canonical endpoint, but optimization budget,
  support, generalization, or interpretation is materially limited.
- `PREPARED_NOT_RUN`: implementation/certification exists without a completed
  scientific campaign.
- `SUPERSEDED`: explicitly replaced evidence retained for provenance.
- `ARCHAEOLOGY`: useful precursor, smoke, or earlier campaign, not a final claim.
- `UNKNOWN`: provenance or scientific role is not resolved enough to promote.

## Evidence matrix

| Scientific question | Representation / objective | Driver and config | Frozen model/result evidence | Postprocessor and synthesis | Validation evidence | Status / confidence |
|---|---|---|---|---|---|---|
| Is direct physical-law regression aligned with the deployed moist call site? | A/B/C; M1-Y | `dimswe/test2b_m1y_campaign.py`; `dimswe/configs/test2b_m1y_20260828.json` | `external-results/m1y-test2b-20260828/production/representation-{A,B,C}/m1y-seed0-m20-10k/`; final NPZ SHA-256 A `a7f98ffe...`, B `3a829fff...`, C `169eadab...` | `dimswe/test2b_m1y_evaluation.py`, `dimswe/test2b_m1y_report.py`; `docs/m1y_test2b_20260828/M1Y_RESULTS.md` | immutable-input, independent-law, cache-parity, objective-gradient, and seed-hash gates all passed; `tests/test_test2b_m1y_campaign.py` | CANONICAL_ACCEPTED / HIGH: completed A/B/C seed-zero fits, 10,000 accepted iterations each; fully offline and nonrecursive |
| How do the four current methods compare for structured learned condensation? | A; M1-Y vs M2-Y/H1 vs H2 vs H5 | M1-Y driver above; historical ladder driver `test2b_rain_learning_campaign.py` | frozen M1-Y model plus `h1-from-m1-m20-5k`, `h2-from-h1-m20-20`, `h5-from-h2-m20-20` | W4 tables/main figures and W5 common-scale galleries; `FINAL_MAIN_PAPER_RESET_REPORT.md` | final Y* law metrics, accepted scalar deployment parity, 161-state galleries | CANONICAL_ACCEPTED / HIGH: all four work very well; quantitative differences are small relative to Representation A robustness |
| Does learning rain preserve rain physics and water evolution? | B; M1-Y vs M2-Y/H1 vs H2 vs H5 | same matched sources/configs | same four frozen B models; matched M1-Y evaluation SHA `4d11a81f...` | W4 Table 3/5 and Figures ML3--ML5B; W5 Representation-B galleries | M1-Y has much smaller R errors and rain-water evolution closest to truth; all replay scalar parity checks pass | CANONICAL_ACCEPTED / HIGH for accepted artifacts; H1/H2/H5 can retain similarly small aggregate state errors despite less faithful R |
| Does an unconstrained four-source model remain physically faithful? | C; M1-Y vs M2-Y/H1 vs H2 vs H5 | same matched sources/configs | same four frozen C models; matched M1-Y evaluation SHA `7bd7adfc...` | W4 Table 3/5 and Figures ML3--ML5C; W5 Representation-C galleries | direct source/invariant, water-partition, autonomous, and spatial checks | CANONICAL_WITH_LIMITATION / HIGH: M1-Y is substantially more faithful/accurate; H1/H2/H5 show increasingly unphysical conservation/partition behavior, but their short recursive budgets do not prove an architecture-wide impossibility |
| What is the effect of sampling direct regression at X* versus Y*=P(X*)? | A/B/C; M1-X vs M1-Y matched control | historical M1-X in `test2b_rain_learning_campaign.py`; M1-Y driver/config | matched JSON SHA-256 A `34db97a0...`, B `4d11a81f...`, C `7bd7adfc...`; `M1_CROSS_STATE_DIRECT_METRICS.csv` | W4 supplementary ML3 cross-state figures; `docs/m1y_test2b_20260828/M1Y_RESULTS.md` | same architecture, initialization, X-fitted normalization, optimizer, support, and iteration budget; only state location changes | CANONICAL_ACCEPTED / HIGH as a state-location control; not every individual metric universally improves |
| Can manuscript quantitative claims be traced from checkpoints to tables/figures? | A/B/C; M1-Y, H1, H2, H5 primary set; controls in supplement | `postprocessing/ml_results_20260829/scripts/` | compact data, main/supplement tables, seven main and nineteen supplement bundles | `FINAL_MAIN_PAPER_RESET_PROVENANCE.json`, `ML_RESULTS_NUMERICAL_SUMMARY.md` | final reset status passed; frozen checkpoint hashes and rendered asset hashes recorded | CANONICAL_ACCEPTED / HIGH; evaluation and rendering only, no training or truth generation |
| Do frozen deployed models reproduce accepted dynamics and spatial structure? | A/B/C; exactly M1-Y, H1, H2, H5 | `postprocessing/deployed_hybrid_dynamics_20260830/scripts/replay_spatial_maps.py` and `render_spatial_package.py` | twelve map sidecars, thirteen accepted galleries, movie sidecars; large caches/movies external | `REPLAY_PROVENANCE.json`, `REPLAY_VALIDATION.md`, deployed-hybrid summary | all 12 replay scalar arrays reproduced exactly (maximum difference zero) | CANONICAL_ACCEPTED / HIGH; rejected visual pilot deliberately excluded |
| Can deterministic analytical ground truth be regenerated and interpreted independently of ML? | analytical Test 2A/2B | five scripts under `postprocessing/ground_truth_figures_20260829/scripts/` | compact temporal/vortex data and accepted Figures 1,2,3,5; map/movie data external | exact case specification, physical analysis, captions, vortex audit | source hashes and figure sidecars; no ML model involved | CANONICAL_ACCEPTED / HIGH |
| Can learned `A` replace the phase-change law while retaining analytical rain and exact source structure? | A; historical M1-X, independent M2-X, M1-X→M2-X, H1, H2, H5 | `dimswe/test2b_rain_learning_campaign.py`; `dimswe/configs/test2b_rain_active_learning.json` | six final-parameter/fit directories under `external-results/test2b-rain-active-learning/production/representation-A/`; comparison SHA `75506a833862b69437a29a7a2b30e64c361d35b8560342bf8abe93302daf5b7f` | `dimswe/test2b_representation_a_postprocess.py`; `docs/TEST2B_REPRESENTATION_A_FINAL_SYNTHESIS.md`; combined A/B/C synthesis | final comparison says six complete; common objective matrix and 160-step rollout; source invariants structural | CANONICAL_WITH_LIMITATION / HIGH: historical state-location ladder; all fits ended at MAXITER; H2/H5 only 20 accepted iterations |
| What changes when rain is also learned but the source map is retained? | B; same six-fit ladder | same campaign driver/config, representation `B` | six final-parameter/fit directories under `.../production/representation-B/`; comparison SHA `6044c0fbd42484e3bd6f0ec53bef91d9a871fa315fc83c193c10f1879813aadd` | `dimswe/test2b_representation_b_postprocess.py`; `docs/TEST2B_REPRESENTATION_B_FINAL_SYNTHESIS.md`; combined synthesis | six complete; rate/onset/invariant diagnostics and matched rollout | CANONICAL_WITH_LIMITATION / HIGH: unconstrained `R_theta`; MAXITER; H2/H5 only 20 iterations |
| What changes when all four moist tendencies are unconstrained? | C; same six-fit ladder | same campaign driver/config, representation `C` | six final-parameter/fit directories under `.../production/representation-C/`; comparison SHA `8bc1d9fad90d1d5907c3ff8bc4a5e396ae09ce34f4d887d1f77ad429dfbba926` | `dimswe/test2b_representation_c_postprocess.py`; `docs/TEST2B_REPRESENTATION_C_FINAL_SYNTHESIS.md`; combined synthesis | six complete; signed water/source-manifold and 160-step rollout audits | CANONICAL_WITH_LIMITATION / HIGH: observed recursive/autonomous degradation is canonical, but short H2/H5 budgets do not prove an architecture-wide impossibility |
| Are A/B/C comparable under identical rain-active support and deployed metrics? | A/B/C; historical cross-objective/cross-rollout evaluation | three representation postprocessors; shared `test2b_rain_active_learning.json` | three comparison JSON files above; fixed cache `preparation/fixed_learning_data.npz`, SHA `6e159015234fd94881b0b97888b7481eb049a02dcd96c571078920c0bedc901c` | `docs/TEST2B_REPRESENTATIONS_A_B_C_FINAL_SYNTHESIS.md` | same truth, 0--80 training and 81--160 held-out, architecture family, solver, metric, schedules, and optimizer budgets | CANONICAL_ACCEPTED / HIGH for the matched completed historical endpoints; M1-X scalar losses are intentionally not cross-representation comparable |
| Does fixed post-prefix H1 equal an offline deployed-discrete M2-Y objective? | A; H1/M2-Y | `dimswe/test2a_h1_m2_equivalence.py`; `dimswe/test2a_horizon_curriculum.py`; `test2a_horizon_curriculum_h1_h2_h5.json` | `external-results/test2a/equivalence-audit/`; `.../horizon-curriculum-h1-h2-h5/h1_postprefix_cache.{npz,json}` | `docs/TEST2A_HORIZON_CURRICULUM_H1_H2_H5.md`; `docs/TEST2A_PROBLEM_A_FINAL_SYNTHESIS.md` | dedicated five-test regression plus cached/literal value-gradient certification | CANONICAL_ACCEPTED / HIGH. Manual regression fact is attributed below; H1 is not recursive |
| How do M1-X, M2-X, H1, H2, and H5 compare on the original no-rain Test 2A support? | A; complete historical objective ladder | `test2a_pyrol.py`, `test2a_discrete_training.py`, `test2a_horizon_curriculum.py`; fair-longfit/M1→M2/curriculum configs | fair comparison SHA `c6ff386c987f0c6755c2a1fc14fb1323e2214f73ec8f5348a3efe8344ebe3d8c`; final parameter files under fair-longfit, M1-to-M2, and horizon-curriculum | `docs/TEST2A_PROBLEM_A_FINAL_SYNTHESIS.md`; supporting M3/M4, H1/H2/H5, and M1→M2 reports | exact caches, derivative certificates, autonomous diagnostics on states 0--80 | CANONICAL_WITH_LIMITATION / HIGH: this is training-support deployment, not held-out rain-active evidence |
| Does the unconstrained four-tendency output change Test 2A behavior? | C (historical “Problem B”); six-fit ladder | `dimswe/test2a_problem_b_campaign.py`; `dimswe/configs/test2a_problem_b.json` | `external-results/test2a/problem-b/production/`; comparison SHA `c9bba696f957b34e40a1f95d29e645463bbdd4456144abd536b2fed4d24561d2` | `docs/TEST2A_PROBLEM_B_FINAL_SYNTHESIS.md` | six final pytree hashes cross-checked; source-manifold/autonomous comparison | CANONICAL_WITH_LIMITATION / HIGH: no-rain training support only; all endpoints MAXITER |
| Can sparse endpoint field inversion improve H2/H5? | A; FIML sparse endpoint H2/H5 | `dimswe/test2a_fiml.py`; `dimswe/configs/test2a_fiml_sparse_endpoint_h2_h5.json` | `external-results/test2a/fiml-sparse-endpoint-h2-h5/` | `docs/TEST2A_FIML_SPARSE_ENDPOINT_H2_H5.md`; included in Problem-A synthesis | direct endpoints, pseudo-labels, Stage 2 and postprocessed dense/sparse/autonomous comparisons | CANONICAL_WITH_LIMITATION / MEDIUM-HIGH: incremental study, not the primary closure-discovery result |
| Does thresholded/nonnegative learned rain repair B? | B+ / historical B_TPL; prepared M1/M2-X/H1/H2/H5 ladder | `dimswe/test2b_rain_learning_campaign.py`, `dimswe/test2b_representation_bplus_prepare.py`; shared Test2B config | `preparation/representation_bplus_output_map.json` SHA `2e898bb4ff130b89039fee2c39ccea89a6085dcf4ab6af6609ce67a4f867acfa`; certificate SHA `341bcbd4313cab4e7fc275815cd5ed0437b0b8e11875fb0e397263045fc45ee2` | `docs/TEST2B_REPRESENTATION_BPLUS_PREPARATION.md` | local, fixed-objective, H1/H2/H5 gradient and invariant certificates; certificate says optimizer not instantiated and production not launched | PREPARED_NOT_RUN / HIGH |
| What are the BTP/BTPL M1-X-only directories? | constrained rain variants; historical M1-X only | historical launcher and preparation under `archive/development-history/test2b_constrained_rain_variants/` | `external-results/test2b-rain-active-learning/production/representation-BTP/` and `representation-BTPL/` | no final accepted synthesis establishing a completed B+ ladder | fit files exist, but relationship to the frozen B+ claim is incomplete | UNKNOWN / MEDIUM; archived and not promoted |
| Do the component tangent/adjoint/HVP implementations agree? | analytical/JAX moist and complete split; derivative certification | production derivative test suites and Test 1A/1B drivers/configs | Test 1A/1B outputs under `external-results/`; isolated prototype under `archive/development-history/firedrake_hvp_prototype/` | derivative design/audit docs under `docs/DIMSWE_*` and `docs/audits/` | dot-product, directional-FD, incremental-adjoint, full-split parity tests | ARCHAEOLOGY for final science, CANONICAL_TEST infrastructure / HIGH |

## Current comparison and initialization genealogy

The primary manuscript comparison is **M1-Y versus M2-Y/H1 versus H2 versus
H5**, separately within each representation. The **M1-X versus M1-Y** study is
a separate matched state-location control; it is not another panel in which
every metric is expected to improve monotonically.

The accepted recursive models predate M1-Y and have the verified parameter-only
genealogy:

```text
M1-X -> H1 -> H2 -> H5
M1-Y  independent seed-zero fit; initializes none of H1/H2/H5
```

This is established by
`scripts/run_test2b_rain_learning_representation.sh:50-61` and by the
`initial_parameters`/fit metadata in
`external-results/m1y-test2b-20260828/`. Rerunning the ladder from M1-Y would
be new science.

## Shared Test 2B contract

The final Test 2B campaign uses:

- truth trajectory `external-results/test2b-rain-active-truth/production-n64-zeta-m0p06-dt100-t16000`;
- fixed training support 0--80 and held-out support 81--160;
- fixed cache `external-results/test2b-rain-active-learning/preparation/fixed_learning_data.npz`
  (351,575,764 bytes; SHA-256 above) for the historical M1-X/M2-X/H ladder;
- M1-Y Y-state learning/held-out caches with SHA-256
  `6f16e6db...` and `1ddfa2d...`, transferred externally;
- inputs `(h,S,Qv,Qc,B)`; exact 5-32-32-`d` MLP with `d=1,2,4` for
  A/B/C; tanh hidden activations, linear output, float64, seed-zero
  Glorot-uniform weights and zero biases;
- exact six-child deployed solver and common mixed-state metric for M2-X/H1/H2/H5;
- non-overlapping dense H2/H5 schedules; and
- PyROL/ROL line-search L-BFGS with secant storage 20.

The completed common state-objective endpoints reported in the combined
synthesis are:

| Stage evaluated under its deployed objective | A | B | C |
|---|---:|---:|---:|
| M1-X under M2-X | `1.83511e-5` | `2.69831e-5` | `2.91056e-5` |
| warm M2-X | `1.48758e-5` | `1.48142e-5` | `1.40395e-5` |
| H1 | `3.00182e-5` | `2.28200e-5` | `3.42671e-5` |
| H2 | `5.51872e-5` | `3.99059e-5` | `1.40039e-4` |
| H5 | `1.10199e-4` | `7.39776e-5` | `7.03413e-4` |

These are attained, budget-limited endpoints. All A/B/C fits report MAXITER;
H2 and H5 have only 20 accepted iterations. The evidence supports comparison of
these artifacts, not a claim that each objective reached its mathematical
minimum.

## Parameter-artifact layout

Each accepted representation directory has six model directories:

- `m1-seed0-m20-10k/`;
- `m2x-seed0-m20-10k/`;
- `m1-to-m2x-m20-5k/`;
- `h1-from-m1-m20-5k/`;
- `h2-from-h1-m20-20/`; and
- `h5-from-h2-m20-20/`.

Each contains `final_parameters.npz`, `final_parameters.json`, and
`fit_result.json`. Pytree fingerprints and stage/representation metadata live
in the sidecars and final comparison JSON. Compact final artifacts are selected
for Git; raw progress/checkpoint data remain locally hash-addressed.

The independent M1-Y fits live under
`external-results/m1y-test2b-20260828/production/representation-{A,B,C}/m1y-seed0-m20-10k/`.
Their final models and saved checkpoint histories are compact and versioned.

## Validation fact carried into this snapshot

Arjun Sharma manually ran, in the recorded Firedrake environment on 2026-08-28:

```text
python -m pytest -q tests/test_test2a_h1_m2_equivalence.py
5 passed in 6.62 s
```

Codex did not run or rerun this test during the forensic audit/cleanup.
