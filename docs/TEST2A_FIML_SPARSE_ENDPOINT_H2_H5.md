# Test 2A sparse-endpoint direct training and two-step FIML

This campaign is an incremental comparison conditioned on the completed H1 model. It does not claim closure recovery from state observations without prior closure information. Both direct sparse-endpoint training and FIML begin from H1-final pytree SHA256 `ebc49083bda299d91e614adeaeefdda0400ca1e8cfccc95a3b4ba953044f963c`.

## Sparse information contract

For horizon (H\in\{2,5\}), each non-overlapping window starts from stored (X_k^*), recursively advances the exact six-child split, and observes only (X_{k+H}^*). H2 uses starts `0,2,...,78` and endpoints `2,4,...,80`; H5 uses starts `0,5,...,75` and endpoints `5,10,...,80`. No intermediate truth state enters either optimization, regularization, initialization, stopping, checkpoint selection, or pseudo-label construction.

The direct objective is

\[
J_H^{\rm endpoint}(\theta)=
\frac{\sum_{k\in S_H}\|\widehat X_{k+H}(\theta)-X_{k+H}^*\|_M^2}{D},
\qquad D=4.0901719676623027\times10^{12}.
\]

The denominator fingerprint is `10bda77bf2e003802c560ef1218fe28b17531da6b30e3f97cf22fa04a62d4753`. Target weights are one; there is no per-target normalization.

## Field inversion

At each internal child-6 call, the dimensionless control is cell-local on the exact deployed `256 x 16` GLL representation:

\[
A_{FI}(Y_{w,j})=A_{H1}(Y_{w,j})+A_{scale}c_{w,j},
\quad A_{scale}=9.354880031073948\times10^{-9}.
\]

The H1 network is evaluated on the current field-inversion state. The original analytical, state-dependent rain law remains active. The known source structure is unchanged, so water and (S-\beta_2Q_v) source invariants hold structurally.

For each independent window,

\[
J_{FI,w}(c_w)=
\frac{\|\widehat X_{k+H}(c_w)-X_{k+H}^*\|_M^2}{D}
+\lambda\,\frac{1}{H\,4096}\sum_{j,p}c_{w,j,p}^2.
\]

There is no truth-A, spatial-smoothness, temporal-smoothness, or intermediate-state regularizer. Every window starts from zero control, uses a fresh PyROL/ROL line-search L-BFGS optimizer, and shares no secant pairs with any other window. The serial window loop is the certification and production reference. Process-level parallelism is deliberately not required; separate processes with independent Firedrake ownership are a future optional accelerator, not a certified MPI claim.

Lambda is selected independently for H2 and H5 from deterministic early/quarter/middle/late windows. The rule is the maximum interior curvature of median `log10(endpoint data misfit)` versus `log10(control RMS)` over a broad positive-lambda sweep. Analytical A is evaluated only after selection as a synthetic post-hoc check.

The preparation sweep used candidates `0, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4`. Near-coincident adjacent log-log points shorter than `1e-3` of total curve arc length are excluded so a numerically duplicated nearly-unregularized solution cannot create a false curvature spike. This selects `lambda_H2 = 1` and `lambda_H5 = 1e-2` without true-A information.

## Pseudo-labels and Stage 2

Each optimized internal model step contributes features `[h,S,Qv,Qc,B]` from its actual FI post-prefix state and the physical pseudo-target (A_{FI}). H2 and H5 each contribute 80 fields, or `80 * 256 * 16 = 327,680` samples. Cell-local duplicates are retained. Intermediate truth is not used as a feature or target.

Stage 2 trains the unchanged float64 `5 -> 32 -> 32 -> 1` tanh MLP from H1-final with frozen Test-2A feature normalization and output scale. Its loss is full-batch normalized pseudo-label MSE. Stage-2 objective and gradient calls contain exactly zero DIMSWE/Firedrake/PETSc solver calls. Analytical A on FI states is stored separately and cannot enter the Stage-2 objective.

## Derivatives and controls

The implementation reuses the certified trajectory tape, exact complete-step state tangent/adjoint, and child-6 parameter JVP/VJP. Each time-local control receives its own exact reverse contribution. The zero-control path is identical to the H1 baseline, and the H1 control diagnostic collapses to the fixed post-prefix `dt G(Y)c` map. Truth-control tests are post-hoc implementation/controllability oracles only.

## Operational contract

Production direct fits, every FI window, and both Stage-2 fits use new optimizer processes and empty L-BFGS history. Parameter-only recovery does not restore process-local secant pairs. Completed FI windows carry configuration, baseline, lambda, and control fingerprints and are reused only when all agree.

The production runner is intentionally not launched by Codex. The guarded stage script is `scripts/run_test2a_fiml_stage.sh`; the sequential wrapper is `scripts/run_test2a_fiml_campaign.sh`; postprocessing is `scripts/run_test2a_fiml_postprocess.sh`. The wrapper stops after any failed or incomplete predecessor.

Preparation recommends caps of 100 accepted iterations for each direct endpoint branch, 25 accepted iterations per H2 FI window, 50 per H5 FI window, and 50,000 for each pure-JAX Stage-2 fit. These are caps; canonical convergence may stop earlier. H2's stronger selected regularization reached very small gradients within the five-step sweep; H5 was still improving and receives the larger FI cap. Fresh H2 full value/value+gradient timings were `3.443/34.862 s`; H5 timings were `4.456/57.238 s`. Representative FI value+gradient timings were `0.829 s` (H2) and `3.367 s` (H5). The production-size 327,680-sample Stage-2 timing was `0.0131/0.0275 s` for value/value+gradient and contains zero solver calls.

The short-smoke extrapolation projects about `15,080 s` for both direct branches and `10,570 s` for both FIML branches (serial FI plus Stage 2). These are non-authoritative engineering estimates. The resulting combined-branch architecture break-even estimate is about `0.513`; production accounting, not this smoke extrapolation, is authoritative.

Normal-Terminal launch after review:

```bash
cd /path/to/DIMSWE-collaborator
nohup caffeinate -i bash scripts/run_test2a_fiml_campaign.sh \
  > external-results/test2a/fiml-sparse-endpoint-h2-h5/campaign.log 2>&1 &
```

Each stage may instead be launched separately by passing one of `direct-h2`, `direct-h5`, `fi-h2`, `fi-h5`, `pseudo-h2`, `pseudo-h5`, `stage2-h2`, or `stage2-h5` to `scripts/run_test2a_fiml_stage.sh`.

Postprocessing retains H1, direct H2/H5, raw FI, and FIML H2/H5. It reports sparse endpoint losses, dense H1/H2/H5 references, M2-X, operator metrics, the common autonomous 0..80 diagnostic, raw-FI versus amortized-NN endpoint error, and separate cost accounting. Autonomous metrics and analytical-A diagnostics are post-hoc only.

## Cost accounting

The reported comparison uses

\[
C_{FIML}(N)=C_{FI}+N C_{offline\,ML},\qquad
C_{direct}(N)=N C_{solver\mbox{-}in\mbox{-}loop}.
\]

A break-even architecture count is reported only when the measured direct per-architecture cost exceeds measured Stage-2 cost. Short smoke extrapolations are engineering estimates, not runtime promises or scientific results.
