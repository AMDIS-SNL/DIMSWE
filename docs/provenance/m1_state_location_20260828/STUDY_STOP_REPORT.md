# Feature-sufficiency study stop report

**Namespace:** `feature_sufficiency_20260828`

**Disposition:** stopped after implementation and analytical audits, before
feature implementation or training.

## Requested final record

1. **Copied starting revision.** Branch
   `dev/dimswe-learned-physics-framework`, HEAD
   `d2f5d66ecb5500aad24eca37280f8a52e22a250f`.  The complete starting status
   and diff-stat transcript is in `STARTING_COPY_PROVENANCE.md`.
2. **Claimed F0 verification.** False.  Production Test-2A and Test-2B use
   `(h,S,Qv,Qc,B)`, not `(S,Qv,Qc,Qr)`.  `h` is explicit; `Qr` is absent.
3. **Existing feature/normalization path.** Exact conservative arrays
   `h,S,Qv,Qc` plus prescribed `B` are packed at deployed 4x4 GLL points and
   affinely normalized.  Test-2A uses an unweighted mean/population standard
   deviation over 331,776 boundary-state samples.  Test-2B uses a
   carrier-mass-weighted mean/population RMS deviation over 81 x 65,536
   boundary-state samples.  Exact offsets/scales and source paths are in
   `FEATURE_MAP_AUDIT.md` and `FEATURE_MAP_AUDIT.csv`.
4. **Minimal sufficient local state.** In the flat accepted cases, `A^*` is a
   deterministic function of `(h,S,Qv,Qc)`, equivalently `(h,b,q_v,q_c)`.
   General nonflat production also requires local prescribed `B` because
   `q_sat` contains `h+B`.  `R^*` requires `q_c` and fixed constants; its
   conservative source requires `h`.
5. **Qr redundancy.** `Qr/qr` enters neither `A^*` nor `R^*`, directly or
   indirectly.  An executable perturbation of `Qr` changed both rates by
   exactly zero (`CONTRACT_AUDIT.json`).
6. **Modular-feature changes.** None.  The material baseline discrepancy
   triggered the user's mandatory stop condition before Phase 3.  Production
   equations, feature code, configs, and checkpoint formats are unchanged.
   Only namespaced audit documents and two read-only diagnostic drivers were
   added.
7. **Checks.** Frozen code/config/preparation/checkpoint feature contracts and
   the A-M1 architecture passed; analytical Qr-independence passed; Test-2A
   and Test-2B stored-sample X-versus-Y reconstructions passed; all audit JSON
   parsed; the CSV has a consistent 14-column schema.  No training occurred.
8. **F0 backward parity.** Not applicable: the requested F0 was never the
   historical feature map.  For the actual historical map, reconstructed
   frozen Test-2A features matched exactly and Test-2B features matched to
   `4.44e-16` maximum absolute error on the audited row.  Network-output
   parity was not claimed because no feature machinery was changed.
9. **Matched A-M1 F0/F1/F2 settings.** No matched feature pilot was launched.
   The frozen accepted Test-2B A-M1 baseline was
   `(h,S,Qv,Qc,B)`, `5->32->32->1`, tanh/linear, float64, seed-0 per-layer
   Glorot uniform with zero biases, carrier-mass-weighted full batch,
   line-search L-BFGS (memory 20), and a 10,000-accepted-iteration cap.
10. **Quantitative feature comparison.** Not produced because the comparison
    premise was false.  For context only, the existing accepted actual-baseline
    A-M1 artifact has training-overall relative/normalized RMS A error
    `0.005467845789312924` (physical RMS
    `4.949635437515388e-10`) and held-out mature-rain normalized RMS
    `0.03211509544887551`, relative RMS `2.233067604462163`, physical RMS
    `2.9071415076049104e-9`.  These are historical single-map metrics, not an
    F0/F1/F2 comparison.
11. **Dependence of error on h.** Not assessed.  An h-stratified comparison
    cannot test a missing-h hypothesis when the accepted model already receives
    `h`; no causal inference was made.
12. **Decision recommendation.** A/B/C/D is not applicable because the
    feature pilot was correctly not entered.  The evidence supports a new,
    separately approved question: compare conditioning of the actual baseline
    `(h,S,Qv,Qc,B)` with `(h,b,q_v,q_c,B)`, or deliberately ablate `h`.
13. **New configs/checkpoints/metrics.** None.  New artifacts are audit-only:
    `STARTING_COPY_PROVENANCE.md`, `FEATURE_MAP_AUDIT.md`,
    `FEATURE_MAP_AUDIT.csv`, `M1_STATE_LOCATION_AUDIT.md`,
    `M1_STATE_LOCATION_TEST2A.json`, `M1_STATE_LOCATION_TEST2B.json`,
    `M1_STATE_LOCATION_TEST2B_RAIN_TARGET.json`, `CONTRACT_AUDIT.json`, and
    the two namespaced drivers under `scripts/`.  `STUDY_MANIFEST.json` records
    their hashes and all immutable inputs.
14. **Authoritative repository.** It was read only.  No file there was
    modified, created, deleted, restored, or cleaned.  Its final branch, HEAD,
    and short status match the captured starting authoritative state.

## M1 state-location correction

For both frozen Test-2A and Test-2B, the production chain is

```text
config -> immutable x/boundary dataset -> normalized feature row -> M1 objective
                     |-> A^*(X_n^*) / R^*(X_n^*)
```

Production M1 performs no prefix operation.  The exact M1 equation is

```text
z_n = Normalize([h,S,Qv,Qc,B](X_n^*))
A_target,n = A^*(X_n^*)
R_target,n = R^*(X_n^*)  # only for representations that learn/use R targets
```

The separate `y_*` cache uses `Y_n^*=P(X_n^*)` for H1/M2-Y, and live
deployment evaluates the network at the post-prefix state.  Full source trace
and numerical evidence are in `M1_STATE_LOCATION_AUDIT.md`.

## Decision table

| feature set | A direct error | held-out error | threshold/sign behavior | training status | interpretation |
|---|---|---|---|---|---|
| Claimed F0 `(S,Qv,Qc,Qr)` | n/a | n/a | n/a | not production; not run | false description of the completed campaign |
| Verified P0 `(h,S,Qv,Qc,B)` | historical training relative RMS `5.468e-3` | historical mature-rain relative RMS `2.233` | historical artifact only; no new comparison | frozen accepted baseline | already supplies `h`; omits `Qr`; flat `B=0` |
| Proposed F1 `(h,S,Qv,Qc)` | not run | not run | not run | stopped | identical varying information to P0 in this flat case |
| Proposed F2 `(h,b,q_v,q_c)` | not run | not run | not run | stopped | conditioning experiment, not a missing-h correction |

The copied workspace also lacks the Test-2B frozen result/truth file payloads
and all tracked test payloads present in the authoritative tree.  This
starting-copy discrepancy was preserved and is independently documented in
`STARTING_COPY_PROVENANCE.md`.
