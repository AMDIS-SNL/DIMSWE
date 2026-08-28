# Test 2A Problem A final scientific synthesis and freeze audit

Date: 2026-08-10
Repository HEAD audited: `d2f5d66ecb5500aad24eca37280f8a52e22a250f`
Branch audited: `dev/dimswe-learned-physics-framework`

## 1. Scope and frozen experimental contract

Problem A replaces only the phase-change rate `A` by a float64 neural model

\[
(h,S,Q_v,Q_c,B)\longmapsto A_\theta,
\]

with architecture `5 -> 32 -> 32 -> 1`, tanh activations, and 1,281 parameters. The model is evaluated on the exact cell-local 4-by-4 GLL representation used by the deployed JAX moist child; shared physical CG points remain repeated. The original analytical, state-dependent rain law `R` is always retained. The learned source is structurally constrained to

\[
S_t=h\beta_2A_\theta,\qquad
Q_{v,t}=hA_\theta,\qquad
Q_{c,t}=-h(A_\theta+R),\qquad
Q_{r,t}=hR.
\]

All fitting and all reported autonomous diagnostics use truth states 0 through 80. The autonomous integrations are therefore **training-support deployment diagnostics**, not held-out generalization tests. No result in this synthesis establishes behavior on states 81 through 160.

The main machine-readable sources used below are:

- **FL** — `external-results/test2a/fair-longfit/comparison/fair_longfit_comparison.json`;
- **FT** — `external-results/test2a/m1-to-m2-finetune/postprocess/m1_to_m2_finetune_report.json` and the corresponding `fit_result.json`;
- **HC** — `external-results/test2a/horizon-curriculum-h1-h2-h5/postprocess/horizon_curriculum_report.json`;
- **HR** — `external-results/test2a/horizon-curriculum-h1-h2-h5/benchmarks/recursion_certification.json`;
- **H1C** — `external-results/test2a/horizon-curriculum-h1-h2-h5/h1_postprefix_cache.json`;
- **FE** — `external-results/test2a/fiml-sparse-endpoint-h2-h5/postprocess/fiml_sparse_endpoint_report.json`;
- **FI2/FI5** — `external-results/test2a/fiml-sparse-endpoint-h2-h5/pseudo-labels/h2/fiml_pseudo_labels.json` and its H5 counterpart;
- **S2-2/S2-5** — the H2/H5 `stage2/*/fit_result.json` files under the FIML root;
- **ACT** — `external-results/test2-prep/doublevortex_training_moist_activity.json`; and
- **BO** — `external-results/test2a/backend-offset-audit/backend_offset_audit.json` and `docs/audits/2026-08-09-test2a-ufl-jax-backend-offset.md`.

Every central number in this document is attributed to one of these artifacts or to a named production fit artifact. The audit found no disagreement among the final JSON reports, fit records, parameter sidecars, and generated Markdown summaries.

## 2. Six-child deployed timestep and learned-source location

The production step is

\[
F_\theta=C_{6,\theta}\circ C_5\circ C_4\circ C_3\circ C_2\circ C_1,
\]

where the children are, in forward order:

1. dry RK4 half-step at `t_n`;
2. dry RK4 half-step at `t_n + dt/2`;
3. hyperviscosity Euler full-step;
4. DG SSPRK43 half-step at `t_n`;
5. DG SSPRK43 half-step at `t_n + dt/2`; and
6. moist Euler full-step at `t_n`.

Only child 6 contains `A_theta`. Define

\[
P=C_5\circ C_4\circ C_3\circ C_2\circ C_1,
\qquad Y_k=P(X_k^*).
\]

The physical closure is deployed at `Y_k`, not at the stored boundary state `X_k^*`. This split location is the essential distinction between M2-X and H1/M2-Y. The implementation statement is enforced in `dimswe/mtswe_split_hvp.py` by the fixed six-child order and by `take_forward_step_from_prefix`; `dimswe/test2a_horizon_curriculum.py` constructs and certifies the analytical post-prefix target with `local_physics=None`.

The backend-offset audit established that a freshly reconstructed prefix plus a fresh analytical UFL moist child reproduces every stored next boundary exactly. A genuinely analytical JAX moist child differs from UFL by at most `5.6722e-17` in relative full mixed-state mass norm and `1.0815e-11` relative to the moist increment. The earlier apparent `1.2453e-6` backend offset was a frozen-neural-provider routing error in a diagnostic helper, not a production UFL/JAX discrepancy [BO].

## 3. Objective taxonomy

### 3.1 M1: physical operator regression, non-discretizer-aware

Let `z_i = (h,S,Qv,Qc,B)_i` be the deployed GLL input and let

\[
a_{\rm scale}=\operatorname{RMS}_{0:80}(A^*).
\]

The M1 objective is

\[
J_{\rm op}(\theta)
=\frac1N\sum_{i=1}^N
\left(\frac{A_\theta(z_i)-A_i^*}{a_{\rm scale}}\right)^2,
\qquad N=331{,}776.
\]

Because the output scale is the uncentered training RMS of `A`, this is also the global pointwise squared-error ratio

\[
J_{\rm op}=\frac{\sum_i|A_\theta-A^*|^2}{\sum_i|A^*|^2}.
\]

M1 is called **non-discretizer-aware** here because its training loss does not pass an `A` error through source injection, weak assembly, the mixed mass inverse, the complete state metric, or any timestep. It learns the analytical scalar law directly. This definition does not mean M1 uses different GLL samples: its inputs are already in the exact deployed local representation. It means the *loss* is independent of the numerical source-to-state map. The objective implementation is `normalized_operator_objective` in `dimswe/test2a_operator.py`; its frozen data and normalization are described in `docs/TEST2A_OPERATOR_LEARNING.md`.

### 3.2 M2-X: discretizer-aware fixed-boundary-state regression

At a fixed stored truth state `X_k^*`, define

\[
G_k^X=M^{-1}WH(X_k^*),
\]

where `H` injects a scalar `A` into the structural four-field source using `h` and `beta2`, `W` is exact production weak assembly, and `M^-1` is the exact mixed mass solve. Then

\[
J_{\rm M2-X}(\theta)=
\frac{\sum_{k=0}^{80}
\left\|G_k^X\left[A_\theta(X_k^*)-A^*(X_k^*)\right]\right\|_M^2}
{\sum_{k=0}^{80}\left\|G_k^XA^*(X_k^*)\right\|_M^2}.
\]

This is discretizer-aware because it weights local rate error by

\[
K_k=(G_k^X)^*MG_k^X,
\]

thereby encoding structural source coupling, `h`-dependence, weak projection, the mixed mass solve, and the mixed state metric. Projection null spaces and nonidentity weighting can make the finite-network optimum differ from the M1 optimum even though an exact representation of `A^*` is a zero-loss solution for both. It remains fully offline: every input is a fixed trusted state, and no prediction becomes a later input. The original `R` is evaluated on both sides and cancels at the common state; it is not set to zero. See `dimswe/test2a_discrete_offline.py`, `dimswe/test2a_discrete_training.py`, and `docs/TEST2A_DEPLOYED_DISCRETE_OFFLINE.md`.

### 3.3 H1/M2-Y: deployment-consistent one-step regression

For `k = 0,...,79`, first form the parameter-independent prefix state

\[
Y_k=P(X_k^*).
\]

The neural and analytical moist children see the same `Y_k`, so the analytical `R(Y_k)` cancels exactly. Their one-step state defect is

\[
\widehat X_{k+1}-X_{k+1}^*
=\Delta t\,G(Y_k)
\left[A_\theta(Y_k)-A^*(Y_k)\right]

\]

up to the certified UFL/JAX operation-order roundoff. The frozen H1 objective is

\[
J_{H1}(\theta)=
\frac{\sum_{k=0}^{79}
\left\|\Delta t\,G(Y_k)
[A_\theta(Y_k)-A^*(Y_k)]\right\|_M^2}{D},
\]

with

\[
D=\sum_{k=0}^{79}\|\Delta t\,G(Y_k)A^*(Y_k)\|_M^2
=4.0901719676623027\times10^{12}.
\]

The denominator fingerprint is `10bda77bf2e003802c560ef1218fe28b17531da6b30e3f97cf22fa04a62d4753` [H1C]. The common `dt^2` cancels between numerator and denominator.

H1 differs from M2-X in the fixed evaluation state, support, and corresponding `G`: M2-X uses 81 boundary states `X_0^*,...,X_80^*`, while H1 uses the 80 post-prefix deployment states `Y_0,...,Y_79`. H1 is nevertheless fully offline and cacheable. Children 1–5 of each first step are independent of `theta`, every `Y_k` is precomputed once, and all later algebra in this one-step objective is fixed except the batched network evaluation. The accepted cache performs zero Firedrake/PETSc actions in its optimization hot loop [H1C]. H1 is therefore not a genuinely recursive solver-in-loop method.

### 3.4 Dense H2 and H5: genuine recursion

For each nonoverlapping window beginning at an exact truth state,

\[
\widehat X_k=X_k^*,\qquad
\widehat X_{k+j}=F_\theta(\widehat X_{k+j-1}),
\quad j=1,\ldots,H,
\]

and

\[
J_H(\theta)=\frac{
\sum_{k\in\mathcal S_H}\sum_{j=1}^H
\|\widehat X_{k+j}-X_{k+j}^*\|_M^2}{D}.
\]

The schedules were `S_1 = {0,1,...,79}`, `S_2 = {0,2,...,78}`, and `S_5 = {0,5,...,75}`. Thus every target boundary 1 through 80 appears exactly once for every horizon, all weights are one, and the same `D` is used.

After the first moist update, `Xhat_{k+1}` depends on `theta`. At the second step this dependence passes through both dry children, hyperviscosity, both DG children, the next neural `A`, and the analytical `R` through its state dependence. Consequently H2 is the first genuinely recursive objective; H5 lengthens that feedback path.

### 3.5 Sparse-endpoint direct training and two-stage FIML

This campaign is distinct from the dense sequential curriculum. Direct sparse H2 and H5 each start independently from H1-final and minimize only

\[
J_H^{\rm endpoint}(\theta)=
\frac{\sum_{k\in\mathcal S_H}
\|\widehat X_{k+H}(\theta)-X_{k+H}^*\|_M^2}{D},
\qquad H\in\{2,5\}.
\]

H2 observes only origins `0,2,...,78` and endpoints `2,4,...,80`; H5 observes only origins `0,5,...,75` and endpoints `5,10,...,80`. Intermediate truth states do not enter optimization.

FIML replaces shared NN weights during Stage 1 by one free cell-local control field per internal moist call:

\[
A_{\rm FI}(Y_{w,j})
=A_{H1}(Y_{w,j})+A_{\rm scale}c_{w,j},
\qquad A_{\rm scale}=9.354880031073948\times10^{-9}.
\]

Each independent window minimizes

\[
J_{{\rm FI},w}(c_w)=
\frac{\|\widehat X_{k+H}(c_w)-X_{k+H}^*\|_M^2}{D}
+\lambda\frac{1}{4096H}\sum_{j,p}c_{w,j,p}^2.
\]

The selected regularization is `lambda_H2 = 1` and `lambda_H5 = 1e-2`. Stage 2 then fits the same shared 1,281-parameter MLP, starting from H1-final, to the 327,680 inferred `A_FI` pseudo-labels. Its full-batch JAX objective uses zero solver calls. The comparison is therefore between repeated direct differentiation of shared network parameters through sparse endpoint trajectories and one-time flexible field inversion followed by offline amortization. Both inherit the same strong H1 prior; this is not closure discovery from sparse states alone.

## 4. Exact derivative and certification status

The following are certified facts, not interpretations:

| component | certificate |
|---|---|
| M1 | Exact JAX all-parameter gradient; directional finite-difference and tiny HVP checks; arbitrary-pytree PyROL adapter. |
| M2-X production | Complete deployed objective, all-parameter gradient, and HVP certified through JAX/Firedrake source assembly and mass solve. |
| M2-X fast cache | Objective and all-parameter gradients agree with the immutable production oracle to float64 roundoff at seed 0, trained artifacts, and perturbations; hot loop has zero Firedrake/PETSc solves. |
| Neural moist child | Primal, state JVP/VJP, parameter JVP/VJP, joint differentiated VJP, weak assembly, Euler update, and one complete six-child step externally certified. |
| H1 cache | Across five probes, relative value discrepancy at most `1.09e-11`, relative gradient discrepancy at most `1.54e-9`, and gradient cosine 1 against the literal complete-step H1 oracle [H1C]. |
| Recursive trajectory | H2 state tangent/adjoint relative discrepancy `1.74e-16`; scale-aware H1/H2 directional-gradient discrepancies about `1.59e-7` and `2.33e-7`; exact reverse traverses timesteps and children in reverse order [HR and trajectory certification]. |
| FIML controls | Zero-control primal equals H1; tangent/adjoint relative discrepancies `6.93e-16` (H2) and `3.35e-15` (H5); centered directional differences `8.74e-5` and `1.45e-4`; source invariants retained. |
| Analytical backend | Fresh UFL prefix/child reproduces stored truth exactly; analytical JAX/UFL difference is ordinary float64 operation-order error and requires no prior certificate revision [BO]. |

At M1-200k, the recursive H2 gradient has cosine 0.964983 and nonproportional residual 0.262313 against 80 independent H1 resets. H5 has cosine 0.905246 and residual 0.424889 [HR]. These values prove that the recursive gradients contain new sensitivity information; they do not prove that a short optimization budget will extract a large practical benefit.

## 5. M1 -> M2-X -> H1 mechanism decomposition

### 5.1 Authoritative parameter artifacts

| label | role | parameter pytree SHA256 |
|---|---|---|
| historical practical M1 | earlier operator reference; iteration-limited | `f7d2fd9577ba2a824d3df6c1f8c90a425e6964e1fb5962508315509b247bed56` |
| historical practical M2-X | original 50k direct-production discrete fit; iteration-limited | `4db13a84f52d6fcf66f0345ce5c677aff2b46a84350228a78bc05b073ff6b12a` |
| matched M1-200k | seed-matched operator reference and H1 start | `f86ee79be3086028f21de10b947c0089147234f494c066f8bbbb2fffb3f8bef8` |
| matched M2-X-200k | independent seed-matched M2-X branch | `94bb112961bc2f2e05cbca459bc50d64513110a077e2b15cded39fe8427de6f8` |
| M1->M2-X-50k | secondary warm-start discretization diagnostic | `e68110b18ea29748830b70683da321bb8e670aa69ddc94598692d72a6f278fc3` |
| H1/M2-Y-final | post-prefix one-step fit starting from M1-200k | `ebc49083bda299d91e614adeaeefdda0400ca1e8cfccc95a3b4ba953044f963c` |
| dense H2-final | recursive H2 continuation from H1 | `92241e66a93e063af2bfc56a22b5cdb7e154e21b936ad2d80d89dce115bf1fab` |
| dense H5-final | recursive H5 continuation from H2 | `fe22c92e56b5b9421835a4ff9a973250c6a186e541356c033b7810521fa3c566` |

Sources: FL, FT, HC, and H1C.

For historical context, the earlier practical M1 artifact had `J_op = 0.004285912836972889` and `J_M2-X = 0.00794193542678781`; the original direct-production 50k M2-X artifact had `J_op = 0.0020819762080123453` and `J_M2-X = 0.0017427829635521567` [FL `historical_practical_fits` and `deployed-discrete-offline/direct-fit-50k-rerun/fit_result.json`]. They remain provenance evidence, but the matched 200k and warm-start rows below are the more relevant mechanism diagnostics.

### 5.2 Quantitative comparison

| artifact | `J_op` | `J_M2-X` | `J_H1` | relative RMS(A) | autonomous final | autonomous max | autonomous accumulated |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1-200k | 3.730061e-4 | 8.346864e-4 | 7.048714e-4 | 0.0193134 | 3.763657e-7 | 6.215496e-7 | 4.692863e-7 |
| matched M2-X-200k | 2.489118e-3 | 1.721967e-3 | 1.592553e-3 | 0.0498911 | 7.406769e-7 | 9.289391e-7 | 8.223406e-7 |
| M1->M2-X-50k | 6.177669e-4 | 5.167360e-4 | 4.946440e-4 | 0.0248549 | 4.207029e-7 | 4.922823e-7 | 4.257358e-7 |
| H1/M2-Y-final | 5.960029e-4 | 5.628143e-4 | 4.510806e-4 | 0.0244132 | 3.883592e-7 | 5.022919e-7 | 4.229449e-7 |

Sources: FL supplies the matched M1/M2-X rows; FT supplies the warm-start row; HC supplies M1 and H1 common diagnostics; H1C supplies `J_H1` for matched M2-X and the warm-start M2-X artifact.

The matched 200k experiment is decisive about optimization confounding: direct seed-zero M2-X ended in a worse basin than M1 even under its own objective,

\[
J_{M2-X}(\theta_{M1})=8.3469\times10^{-4}
<1.7220\times10^{-3}
=J_{M2-X}(\theta_{M2-X}).
\]

Both reached `MAXITER`. Therefore this branch cannot measure the causal benefit of the M2-X objective.

The separate M1->M2-X warm start establishes that discretizer weighting contains useful, nonproportional information in the good M1 basin: 50,000 accepted iterations lowered `J_M2-X` by 38.09%, from `8.3469e-4` to `5.1674e-4`. It simultaneously raised `J_op` by 65.62%, raised relative RMS(A) from 0.01931 to 0.02485, improved autonomous maximum error by 20.80% and accumulated error by 9.28%, but worsened final error by 11.78% [FT]. Thus M2-X creates a real tradeoff rather than a uniformly superior physical-law fit.

Starting independently from the same M1 artifact, H1 lowered its own objective by 36.01%, the common dense H2 and H5 objectives by 34.50% and 32.86%, autonomous maximum error by 19.19%, and accumulated error by 9.87%. Its autonomous final error was 3.19% higher than M1. It also traded pointwise accuracy: relative RMS(A) rose from 0.01931 to 0.02441 [HC].

### 5.3 What can and cannot be attributed

The statement “most gain comes from making training discretizer-aware” is too broad for the completed evidence. M2-X and H1 are both discretizer-aware, but they optimize at different fixed states. The largest change in the tested sequential curriculum occurs at H1, which jointly supplies:

1. the deployed weak/mass/mixed-metric weighting; and
2. the correct post-prefix state `Y_k` at which the learned source is actually called.

The stronger supported statement is:

> Most of the practical improvement in the tested curriculum arrived when the one-step target became consistent with both the deployed discretization and the post-prefix deployment state.

Discretizer awareness alone is useful in the M1 basin, as the warm-start M2-X diagnostic shows. Deployment-state consistency is also consequential: H1 obtains a lower `J_H1`, lower final and accumulated autonomous error, and slightly better physical-A error than the M1->M2-X endpoint, whereas the M2-X endpoint obtains a slightly lower `J_M2-X` and autonomous maximum error. But there is no matched sequential M1->M2-X->H1 experiment with comparable optimization budgets and basin control. A percentage split between “discretizer weighting” and the `X -> Y` correction would therefore be manufactured and is not reported.

## 6. Dense H1 -> H2 -> H5 results

The dense curriculum is sequential: M1-200k -> H1 (50,000 accepted iterations) -> H2 (100) -> H5 (100). Every stage used a new L-BFGS process with empty secant history. All three stages terminated at `MAXITER`; H2 and H5 are explicitly limited-budget tests [HC and stage `fit_result.json` files].

| stage-boundary artifact | `J_H1` | `J_H2` | `J_H5` | `J_M2-X` | `J_op` | rel RMS(A) |
|---|---:|---:|---:|---:|---:|---:|
| M1-200k | 7.048714e-4 | 1.120554e-3 | 2.250633e-3 | 8.346864e-4 | 3.730061e-4 | 0.0193134 |
| H1-final | 4.510806e-4 | 7.339235e-4 | 1.511100e-3 | 5.628143e-4 | 5.960029e-4 | 0.0244132 |
| H2-final | 4.524130e-4 | 7.314762e-4 | 1.498248e-3 | 5.656311e-4 | 6.021623e-4 | 0.0245390 |
| H5-final | 4.606034e-4 | 7.356325e-4 | 1.484799e-3 | 5.792879e-4 | 6.148734e-4 | 0.0247966 |

| artifact | autonomous final | maximum | accumulated | KE final / max mismatch | enstrophy final / max mismatch |
|---|---:|---:|---:|---:|---:|
| M1-200k | 3.763657e-7 | 6.215496e-7 | 4.692863e-7 | 3.674e-8 / 3.674e-8 | 4.368e-8 / 4.945e-8 |
| H1-final | 3.883592e-7 | 5.022919e-7 | 4.229449e-7 | 1.341e-8 / 1.341e-8 | 4.287e-8 / 5.603e-8 |
| H2-final | 3.861652e-7 | 5.022232e-7 | 4.210796e-7 | 1.024e-8 / 1.251e-8 | 4.221e-8 / 4.813e-8 |
| H5-final | 3.800073e-7 | 4.962602e-7 | 4.147441e-7 | 6.277e-9 / 1.706e-8 | 3.275e-8 / 4.572e-8 |

Source for both tables: HC.

The incremental recursive effect under the tested budgets was small:

- H1 -> H2 reduced `J_H2` by 0.333%, `J_H5` by 0.851%, autonomous final error by 0.565%, and accumulated error by 0.441%; maximum error changed by only 0.014%.
- H2 -> H5 reduced `J_H5` by 0.898%, autonomous final error by 1.595%, maximum error by 1.187%, and accumulated error by 1.505%. It worsened `J_H1` by 1.810% and `J_H2` by 0.568%, showing the expected horizon tradeoff.

The physical-law error drifted upward from H1 to H5 while the longer-horizon and autonomous metrics improved modestly. This is evidence that recursive objectives can favor trajectory-compensating directions. It is not proof of an incorrect law, and the changes are small.

The certified gradient comparison makes the conceptual result stronger than the performance result: genuine recursive information first appears at H2 and becomes less aligned with independent one-step regression by H5, but 100 accepted iterations at each recursive stage produced only modest practical gains. Since both ended at `MAXITER`, the study does not establish the fully optimized H2/H5 outcome.

## 7. Sparse direct versus FIML results

### 7.1 Matched network comparison

The following branches all start independently from H1-final. Direct H2/H5 use only sparse endpoints and 100 accepted L-BFGS iterations. FIML uses the same sparse endpoints for field inversion, then 50,000 iterations of pure-JAX offline pseudo-label regression. Intermediate truth is post-hoc only [FE].

| network | sparse H2 | sparse H5 | dense H1 | dense H2 | dense H5 | `J_M2-X` | `J_op` |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1 baseline | 4.982220e-4 | 4.749011e-4 | 4.510806e-4 | 7.339235e-4 | 1.511100e-3 | 5.628143e-4 | 5.960029e-4 |
| direct H2 | 4.939369e-4 | 4.657671e-4 | 4.567549e-4 | 7.333003e-4 | 1.493791e-3 | 5.738573e-4 | 6.140898e-4 |
| FIML H2 | 4.934001e-4 | 4.692344e-4 | 4.479612e-4 | 7.275639e-4 | 1.494970e-3 | 5.589210e-4 | 5.943360e-4 |
| direct H5 | 5.006763e-4 | 4.599906e-4 | 4.784992e-4 | 7.519166e-4 | 1.500104e-3 | 6.063118e-4 | 6.404887e-4 |
| FIML H5 | 4.733065e-4 | 4.366084e-4 | 4.452388e-4 | 7.061274e-4 | 1.410575e-3 | 5.603588e-4 | 5.582155e-4 |

| network | autonomous final | maximum | accumulated | off-manifold A rel RMS | KE final / max | enstrophy final / max |
|---|---:|---:|---:|---:|---:|---:|
| H1 baseline | 3.883592e-7 | 5.022919e-7 | 4.229449e-7 | 0.025008 | 1.341e-8 / 1.341e-8 | 4.287e-8 / 5.603e-8 |
| direct H2 | 3.834731e-7 | 5.011219e-7 | 4.199125e-7 | 0.025320 | 8.735e-9 / 1.074e-8 | 3.577e-8 / 5.385e-8 |
| FIML H2 | 3.850143e-7 | 4.989979e-7 | 4.201173e-7 | 0.024963 | 1.210e-8 / 1.210e-8 | 4.314e-8 / 5.672e-8 |
| direct H5 | 3.786186e-7 | 4.895150e-7 | 4.124262e-7 | 0.025699 | 7.464e-9 / 3.448e-8 | 3.331e-8 / 3.841e-8 |
| FIML H5 | 3.560783e-7 | 4.919251e-7 | 4.066517e-7 | 0.024064 | 2.661e-9 / 1.966e-8 | 5.105e-8 / 5.151e-8 |

Source for both tables: FE. These are training-support diagnostics and were not used to select controls, pseudo-label fits, or checkpoints.

The final network parameter fingerprints recorded by FE are:

| network | parameter SHA256 |
|---|---|
| H1 baseline | `ebc49083bda299d91e614adeaeefdda0400ca1e8cfccc95a3b4ba953044f963c` |
| direct H2 | `d2e4dd4532b55ae99f9d6f8edb4e0564ef5743f9d65cced377857fc07751d5f6` |
| FIML H2 | `46cc90d55f31ddaee93e231223f519ada6a4f438a217bb469c724c85bd7ea093` |
| direct H5 | `6efc83ce1f0da2b308fccfe028addc2ecbc9a046a86db80cfa6f8ccc39a46dd9` |
| FIML H5 | `de5d02ce84100ad6bdc90db54699a490242b7ae5bb1c9424e65414818fcc52a3` |

Direct H2 and H5 both improved their own sparse endpoint objectives relative to H1, but both stopped at the prescribed 100-iteration `MAXITER` caps. Hence the comparison is accuracy and cost under the actual matched production budgets, not FIML versus the mathematical optimum of direct recursive training.

FIML H2 and direct H2 are close. FIML H2 is slightly better on sparse H2, dense H1/H2, `J_M2-X`, `J_op`, physical A, and autonomous maximum error; direct H2 is slightly better on sparse H5, dense H5, autonomous final/accumulated error, KE, and enstrophy. There is no universal H2 winner.

FIML H5 is broader. It is best among these five networks on both sparse endpoint losses, all three dense horizon losses, `J_op`, autonomous final and accumulated error, and off-manifold A error. Direct H5 retains a slightly lower autonomous maximum error and better enstrophy mismatch. This is strong training-support evidence that the FIML-H5 network carries a more reusable approximation than the 100-step direct-H5 endpoint fit, not a held-out generalization theorem.

### 7.2 Field inversion and compression loss

Stage 1 obtained

| horizon | raw FI endpoint data misfit | Stage-2 NN endpoint objective | Stage2/FI ratio |
|---|---:|---:|---:|
| H2 | 3.910721e-4 | 4.934001e-4 | 1.26166 |
| H5 | 1.299611e-6 | 4.366084e-4 | 335.953 |

Source: FE `amortization_compression_loss`.

Raw FI has `H x 4096` independently adjustable controls in every window and no requirement that corrections from different windows be generated by one shared constitutive function. It can exploit nonuniqueness of a sparse endpoint inverse problem and solver-compensating directions. Stage 2 compresses all 327,680 pseudo-label samples into a shared 1,281-parameter map of five local features. The gap is therefore an **amortization/compression loss**, not a mismatch in endpoint data or solver semantics.

The H5 ratio is large largely because the raw FI denominator is almost zero. Its absolute compression loss is `4.35309e-4`. It does not by itself imply FIML failure: the compressed FIML-H5 network still improves on H1 and direct H5 across the broad post-hoc table. Stage 2 fit its H2 pseudo-labels to relative RMS 0.000983 and its H5 pseudo-labels to 0.008558; the corresponding normalized supervised objectives were `1.16834e-6` and `8.85489e-5`, with exactly zero solver calls [S2-2/S2-5]. The remaining H5 endpoint gap therefore also reflects how small shared local-law errors propagate through an ill-conditioned, nonunique long-window inverse problem.

Post-hoc analytical comparisons on the FI states support the same interpretation. Raw inferred `A_FI` has relative RMS error 0.02256 (H2) and 0.02094 (H5) against genuine `A*` [FI2/FI5]. The Stage-2 networks have 0.02290 and 0.02216 on those same FI states [S2-2/S2-5]. FI controls were not selected with true A.

## 8. Physical-law recovery versus trajectory optimization

| network | relative RMS(A) on fixed truth support | correlation | active (10^{-3}) rel RMS | active (10^{-3}) sign accuracy |
|---|---:|---:|---:|---:|
| H1 baseline | 0.024413 | 0.999701 | 0.023882 | 0.990098 |
| direct H2 | 0.024781 | 0.999692 | 0.024255 | 0.990051 |
| FIML H2 | 0.024379 | 0.999702 | 0.023850 | 0.990135 |
| direct H5 | 0.025308 | 0.999679 | 0.024756 | 0.989386 |
| FIML H5 | 0.023627 | 0.999720 | 0.023078 | 0.988039 |

Source: FE `dense_and_fixed_support.direct_A_metrics`.

The direct recursive fits reduce their selected sparse endpoint losses while slightly degrading `J_op` and physical-A error relative to H1. Direct H5 also worsens all three dense horizon objectives relative to H1 while improving sparse H5 and autonomous trajectory metrics. This is evidence of horizon-specific compensating physics under a sparse endpoint objective.

FIML H2 essentially preserves the H1 constitutive accuracy. FIML H5 improves fixed-support physical-A relative RMS, `J_op`, both sparse losses, all dense losses, and autonomous final/accumulated error. This pattern is consistent with field inversion extracting useful recursive information and supervised amortization acting as a constitutive regularizer. It remains an inference: all diagnostics are on training support, the architectures and budgets are single choices, and the free-field inverse problem is nonunique.

The best physical-law fit remains M1-200k at relative RMS(A) 0.01931. Solver-aware objectives consistently trade some pointwise physical-law accuracy for better trajectory-weighted behavior. Problem A therefore demonstrates a Pareto tradeoff, not a single scalar ranking of learning methods.

## 9. Cost accounting

Production costs reported by FE are:

| branch | solver-in-loop / FI Stage 1 (s) | offline Stage 2 (s) | total (s) | accepted iterations and evaluations |
|---|---:|---:|---:|---|
| direct sparse H2 | 3,848.443 | — | 3,848.443 | 100 accepted; 213 objective; 101 gradient |
| FIML H2 | 168.151 serial-equivalent | 3,163.402 | 3,331.553 | FI: 150 accepted across 40 windows, 340/190; Stage 2: 50,000 accepted, 104,564/50,001 |
| direct sparse H5 | 6,056.825 | — | 6,056.825 | 100 accepted; 213 objective; 101 gradient |
| FIML H5 | 1,219.596 serial-equivalent | 3,119.096 | 4,338.692 | FI: 329 accepted across 16 windows, 696/345; Stage 2: 50,000 accepted, 104,508/50,001 |

Under these actual budgets, FIML cost 13.43% less than direct H2 and 28.37% less than direct H5. Combined, direct cost was 9,905.268 s and FIML cost was 7,670.245 s, a 22.56% reduction. These timings are serial and architecture-specific.

The amortization model is

\[
C_{FIML}(N)=C_{FI}+N C_{offline\,ML},\qquad
C_{direct}(N)=N C_{solver\mbox{-}in\mbox{-}loop}.
\]

Using measured times, the formal break-even architecture counts are 0.245 (H2) and 0.415 (H5), so FIML was already cheaper for one architecture under these budgets. The scientific amortization claim is narrower: Stage-1 inversion is a reusable one-time cost and Stage 2 contains zero solver calls. This single small serial experiment does not establish scaling to other architectures, meshes, parallel environments, or tighter direct-optimization tolerances. Direct and Stage 2 both ended at their iteration caps, so runtime is inseparable from the selected practical budgets.

For context, dense H1 cost 3,859.4 s for 50,000 accepted cached iterations; dense H2 cost 4,017.9 s for 100 accepted iterations; and dense H5 cost 6,271.4 s for 100 accepted iterations. These stage fit records reinforce why long recursive optimization is expensive even though H1 is cacheable.

## 10. Structural invariants and no-rain checks

The Test-2 activity audit found analytical `R = 0` at every one of 331,776 truth-support samples: minimum, maximum, RMS, and all activity fractions were exactly zero [ACT]. Importantly, Problem A never encodes `R = 0`. Every neural child evaluates the original analytical `R` at its current state.

Across all dense-curriculum and sparse/FIML autonomous rollouts reported in HC and FE:

- maximum `|R| = 0`;
- exact-nonzero-R timestep count `= 0`;
- physically meaningful-R timestep count `= 0`;
- maximum water-source residual `Qv_t + Qc_t + Qr_t = 0` exactly; and
- maximum `S_t - beta2 Qv_t` residual `= 1.734723475976807e-18`.

Thus the structured source identities remained intact to exact/roundoff precision, and none of the learned trajectories spuriously activated rain on this support. This does not test learned behavior in an active-rain regime.

## 11. What is established, suggestive, and not established

### Established by certification and completed artifacts

- M1, M2-X, H1/M2-Y, dense recursive H2/H5, sparse endpoint direct training, and two-stage FIML are mathematically distinct as described above.
- M2-X is a fixed-state discretization-weighted regression at `X_k^*`; H1 is a fixed-state post-prefix regression at `Y_k`, remains offline/cacheable, and contains no recursive feedback.
- H2 is the first horizon with recursive solver-mediated sensitivity. H5 contains a longer and more nonproportional feedback path.
- The largest change in the dense curriculum occurred at H1. H2 and H5 supplied new gradients but modest additional practical gains under 100 accepted iterations each.
- In the good M1 basin, M2-X fine-tuning can materially reduce the deployed-discrete objective, but it trades against direct physical-A accuracy and does not improve every autonomous statistic.
- Under the actual sparse production budgets, FIML H5 produced the broadest network-level improvement, despite large compression loss relative to its highly flexible raw field inversion.
- Structural moist-source invariants and the analytical-R contract were preserved.

### Suggestive, not proved

- Direct sparse recursive training appears to learn horizon-specific compensating physics, whereas FIML—especially H5—appears to retain a more reusable local constitutive law.
- Offline regression of FI pseudo-labels may regularize a nonunique sparse endpoint inversion in a scientifically useful way.
- H1's practical value likely contains contributions from both discrete weighting and correcting the `X_k^* -> Y_k` evaluation-state mismatch.

### Not established

- No percentage decomposition between M1->M2-X discretizer weighting and M2-X->H1 deployment-state consistency is identifiable from the available unmatched basins/budgets.
- No recursive fit reached demonstrated mathematical stationarity; H2/H5 and direct sparse runs ended at `MAXITER`.
- No result is held-out generalization: all truth comparisons are on states 0 through 80.
- The experiment does not cover active rain, alternative meshes, architectures, seeds, regularization choices, or MPI/parallel scaling.
- FIML was not tested from seed zero without the strong H1 prior.
- Because Problem A hard-wires the exact four-field source structure, it cannot reveal how much performance depends on that structural prior.

## 12. Frozen Problem-A conclusions

1. **The most accurate description of the main practical gain is deployment-consistent one-step training, not merely “offline versus online” or “discretizer-aware versus non-discretizer-aware.”** H1 makes the target consistent with both the deployed source-to-state discretization and the post-prefix state where the closure is called. It is still an offline fixed-state objective.

2. **Discretizer awareness alone is real but not cleanly quantifiable as a share of the H1 gain.** The M1->M2-X warm-start diagnostic materially lowered `J_M2-X`, while the independent seed-matched M2-X run exposed severe optimizer-basin confounding. A numerical percentage split is therefore not scientifically defensible.

3. **Recursive sensitivity is mathematically new at H2, but its measured practical benefit was modest at the tested budgets.** H2 and H5 improved longer-horizon/autonomous measures by small increments and shifted the tradeoff away from pointwise (A) accuracy. New gradient information should not be conflated with a large realized gain.

4. **Sparse direct and FIML answer a different question from the dense curriculum.** Under sparse endpoints and common H1 initialization, both methods made modest direct improvements. FIML H5 gave the strongest broad post-hoc network result and lower measured cost, but direct runs were capped and no mathematical-optimum claim is warranted.

5. **Problem A is frozen as a structured, no-rain, training-support benchmark.** Its strongest result is that correct deployment location and numerical weighting matter substantially even before recursion, while longer recursive information and FIML can refine the trajectory/constitutive tradeoff.

## 13. Implications and exact handoff to Problem B

Problem B replaces the structured scalar `A_theta` by a black-box four-component moist tendency

\[
N_\theta(X)=(S_t,Q_{v,t},Q_{c,t},Q_{r,t})_\theta.
\]

In Problem A's no-rain truth, the exact tendency lies on the one-dimensional manifold

\[
hA^*(\beta_2,1,-1,0)^T.
\]

Problem A imposed this manifold exactly; the network could neither violate water conservation nor invent an inconsistent thermal/moisture coupling. Problem B is therefore the necessary next test of how much of Problem A's accuracy, stability, optimizer behavior, and FIML performance depended on the correct source structure rather than on the learning strategy alone. It should separately measure source-invariant violation, spontaneous rain-component tendency, physical-law identifiability, and autonomous stability. No Problem-B implementation or result is part of this freeze.

**Freeze decision:** Problem A is complete for its stated structured-A, analytical-R, no-rain, training-support scope. The limitations above are part of the frozen conclusion, not unresolved artifact conflicts.
