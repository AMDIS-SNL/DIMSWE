# AMDIS Learned-Physics Research Plan

**Project:** AMDIS / DIMSWE learned-physics framework  
**Purpose:** Frozen high-level research plan and terminology for the controlled moist-shallow-water learning studies, followed by a DNS-to-LES closure-learning study.

## 1. Overall scientific objective

The long-term goal is to understand how embedded physical parameterizations should be learned inside numerical PDE solvers, separating the effects of:

1. representation of the learned physics;
2. the numerical discretization in which that physics is deployed;
3. short-time solver feedback; and
4. recursive cross-time feedback.

The current DIMSWE moist-shallow-water problem is a controlled proving ground because the hidden physical law is known exactly. This permits interpretation before moving to genuine unresolved physics such as moist DNS-to-LES subgrid modeling.

The program is:

\[
\text{known hidden moist physics}
\longrightarrow
\text{unknown subgrid physics from DNS-to-LES}.
\]

The first stage is intended to be publishable as a methodological/computational physics study. The later DNS-to-LES study is expected to be the more ambitious fluid-mechanics application.

## 2. Two physics-learning problems

### Problem A: structured learning of the physical rate

The current problem learns the signed moist phase-change rate

\[
A_\theta=A_\theta(h,S,Q_v,Q_c,B),
\]

while retaining the known physical source structure,

\[
Q_{v,t}=hA_\theta,\qquad
Q_{c,t}=-h(A_\theta+R),\qquad
Q_{r,t}=hR,\qquad
S_t=h\beta_2A_\theta.
\]

The original analytical rain law \(R\) remains embedded.

This preserves structural identities such as

\[
Q_{v,t}+Q_{c,t}+Q_{r,t}=0,
\]

and

\[
S_t-\beta_2Q_{v,t}=0.
\]

Thus the neural network learns a physically meaningful latent rate while the exact source structure is imposed analytically.

### Problem B: black-box learning of the complete physics tendencies

The alternative is to learn the complete moist-physics tendency vector directly,

\[
\mathcal N_\theta(X)
=
\begin{pmatrix}
S_t^{\rm phys}\\
Q_{v,t}^{\rm phys}\\
Q_{c,t}^{\rm phys}\\
Q_{r,t}^{\rm phys}
\end{pmatrix}_\theta.
\]

The network is not explicitly told that the tendencies arise from lower-dimensional rates \(A\) and \(R\), and conservation/thermodynamic identities need not be imposed.

For the current no-rain regime, the true tendency vector lies on the one-dimensional manifold

\[
\mathbf s_{\rm phys}^\star
=
hA
\begin{pmatrix}
\beta_2\\
1\\
-1\\
0
\end{pmatrix}.
\]

Problem A imposes that structure. Problem B asks the network to discover it.

A central diagnostic for B is whether the learned tendencies recover approximately

\[
Q_{v,t}+Q_{c,t}+Q_{r,t}=0,
\]

and

\[
S_t-\beta_2Q_{v,t}=0,
\]

without those constraints being imposed.

### Rain-active structured diagnostic

Once \(R\neq0\),

\[
\mathbf s_{\rm phys}^\star
=
h
\begin{pmatrix}
\beta_2A\\
A\\
-(A+R)\\
R
\end{pmatrix}.
\]

At that point Problem A still learns only \(A\) while retaining exact \(R\), whereas Problem B implicitly has to learn both effects. Therefore the rain-active study should include a structured diagnostic in which both

\[
(A_\theta,R_\theta)
\]

are learned and then mapped through the exact source structure. This is an interpretability/fairness diagnostic rather than a separate headline problem.

## 3. Four learning strategies

### Method 1: direct operator / a-priori learning

Learn the physical mapping directly from trusted states:

\[
A_\theta(X_k^\star)\approx A^\star(X_k^\star),
\]

or for Problem B,

\[
\mathbf s_\theta(X_k^\star)\approx\mathbf s^\star(X_k^\star).
\]

No information about the deployed discretization or time evolution enters the objective.

### Method 2: deployed-discrete offline learning

Still use only trusted states, but measure error after the exact numerical source-to-state mapping.

For Problem A,

\[
\delta A_k(\theta)=A_\theta(X_k^\star)-A_k^\star,
\]

and

\[
J_{\rm disc}(\theta)
=
\frac{
\sum_k\|\mathcal G_k\delta A_k\|_M^2
}{
\sum_k\|\mathcal G_kA_k^\star\|_M^2
},
\qquad
\mathcal G_k=M^{-1}WH_k.
\]

This is still offline: there is no recursive model-state propagation.

For Problem B, the same principle applies to the complete learned tendency vector through the corresponding exact discrete source-to-state map.

### Method 3: one-step truth-reset solver-in-the-loop learning

At each training time,

\[
\widehat X_k=X_k^\star,
\]

run one complete deployed timestep,

\[
\widehat X_{k+1}=F_\theta(X_k^\star),
\]

compare with \(X_{k+1}^\star\), then discard the prediction and reset to the next truth state.

Training therefore consists of independent one-step windows.

Method 3 optimizes the error of the complete deployed timestep, but does not expose the network to recursively generated model states beyond one step.

Independent truth-reset windows are naturally parallelizable.

### Method 4: finite-horizon rollout solver-in-the-loop learning

Start each window from truth,

\[
\widehat X_k=X_k^\star,
\]

then propagate autonomously for a finite horizon \(H\),

\[
\widehat X_{k+j+1}=F_\theta(\widehat X_{k+j}),
\qquad j=0,\ldots,H-1.
\]

After the window, reset to another trusted state.

Initial practical horizons include

\[
H=5,\qquad H=10,
\]

rather than differentiating through an entire simulation.

Eventually, horizon should be expressed relative to a physical timescale,

\[
\frac{H\Delta t}{T_{\rm flow}},
\]

rather than as an arbitrary number of timesteps.

Method 4 introduces genuine recursive cross-time feedback.

## 4. Interpretation of the hierarchy

The progression is

\[
M1\rightarrow M2\rightarrow M3\rightarrow M4.
\]

- M1: local physical/operator information.
- M2: deployment-aware discretization information.
- M3: one complete deployed timestep.
- M4: recursive finite-horizon state feedback.

Key comparisons:

- **M1 vs M2:** does discretization-aware offline training improve deployment without solver-in-the-loop training?
- **M2 vs M3:** does differentiating through one complete deployed timestep add information beyond the exact discrete offline source objective?
- **M3 vs M4:** what is gained specifically from recursive cross-time feedback?

## 5. Full autonomous simulation is primarily evaluation

Training horizon need not equal deployment horizon.

Models trained with \(H=1,5,10\) should be evaluated on a much longer autonomous trajectory with no truth resets.

Evaluation should include:

- mixed-state trajectory error;
- fieldwise errors;
- kinetic energy;
- projected enstrophy;
- moist-process activity;
- conservation/source residuals;
- stability and boundedness.

This avoids making entire-simulation adjoints a prerequisite for evaluating long-time deployment.

## 6. Method-4 computational strategy

Exact long-horizon rollout training may be prohibitively expensive.

Exact M4 should therefore remain an expensive reference capability while later work investigates cheaper approximations such as:

- shorter horizons;
- stochastic/minibatched reset windows;
- progressive horizon training;
- frozen/truth-tangent approximations;
- field-inversion-like approaches;
- other approximations to the dominant cross-time sensitivity.

## 7. Shared computational architecture

The four methods should share one learned-physics framework:

\[
\text{FeatureMap}
\rightarrow
\text{ParameterizedModel}
\rightarrow
\text{OutputMap}
\rightarrow
\text{host solver}
\rightarrow
\text{training objective}.
\]

The learned physics, host problem, and training objective should remain modular.

Exact computational reuse should be exploited where valid, including fixed discretization objects and same-\(\theta\) forward-tape reuse. For the current A problem, children 1–5 of the first timestep of a truth-reset window are independent of \(\theta\) and may be precomputed.

Method-3 windows admit exact map/reduce parallelism:

\[
J_3(\theta)=\sum_wJ_w(\theta),
\qquad
\nabla_\theta J_3(\theta)=\sum_w\nabla_\theta J_w(\theta).
\]

Method 4 remains sequential in ordinary forward and reverse time.

## 8. Simulation hierarchy for Problems A and B

### Level I: current short double-vortex problem

Run the full primary matrix

\[
2\text{ physics representations}\times4\text{ training strategies}=8
\]

experiments:

| Physics problem | M1 | M2 | M3 | M4 |
|---|---:|---:|---:|---:|
| A: structured rate learning | ✓ | ✓ | ✓ | ✓ |
| B: black-box tendency learning | ✓ | ✓ | ✓ | ✓ |

Use practical finite horizons for M4 and preserve the held-out trajectory until primary training strategies are frozen.

### Level II: longer, higher-resolution double vortex

Increase resolution, initially targeting approximately

\[
64\times64,
\]

and extend the simulation until rain activates.

First generate a trusted truth trajectory and diagnose:

- condensation/evaporation activity;
- cloud-water development;
- first \(R\neq0\);
- substantial rain onset;
- later rain-active evolution;
- numerical stability.

Use this physical sequence to define training and evaluation intervals.

Repeat M1–M4 for Problems A and B, with the structured \((A,R)\) diagnostic.

### Level III: additional existing DIMSWE cases

Apply the same framework to selected DIMSWE flows that expose qualitatively different dynamics. Do not mechanically run every possible combination if it does not answer a scientific question.

### Level IV: moist mountain / Hartney-type cases

Add moist flow over isolated topography and selected additional Hartney/Bendall/Shipton test cases.

Before ML, independently certify the underlying topographic/Hamiltonian formulation against the correct quasi-Hamiltonian thermal shallow-water formulation, including the earlier paper **A Quasi-Hamiltonian Discretization of the Thermal Shallow Water Equations**.

Do not allow ML to compensate for a deterministic solver/topography error.

## 9. First paper

The first paper uses known hidden moist physics to answer:

> How do physical representation, discretization consistency, and solver-in-the-loop time horizon affect learning and autonomous deployment of embedded moist physics?

Primary axes:

\[
\text{representation}
=
\{\text{structured physical rate},\text{black-box tendencies}\},
\]

and

\[
\text{training}
=
\{M1,M2,M3,M4\}.
\]

Increasing dynamical complexity provides robustness tests:

\[
\text{short no-rain vortex}
\rightarrow
\text{long rain-active vortex}
\rightarrow
\text{additional/topographic moist flows}.
\]

Main scientific questions:

1. When is differentiating through the solver actually necessary?
2. How much can a better-posed offline objective recover?
3. How much additional benefit comes from recursive cross-time feedback?
4. How much does imposed physical structure reduce learning difficulty and improve physical consistency?

JCP is a natural possible venue if the main contribution is numerical/methodological.

## 10. Second major application: moist thermal DNS-to-LES subgrid learning

After validating the methodology on known hidden physics, move to a genuine closure problem.

Let

\[
X^{DNS}
\stackrel{\mathcal C}{\longrightarrow}
\overline X
\]

be the DNS-to-LES coarse-graining/filtering operation.

The coarse-grained evolution is schematically

\[
\frac{d\overline X}{dt}
=
F_{\rm resolved}(\overline X)
+
\mathcal S^\star,
\]

where \(\mathcal S^\star\) contains unresolved/subgrid effects.

Learn

\[
\mathcal S_\theta
=
\mathcal N_\theta(\overline X,\nabla\overline X,\ldots).
\]

The moist SGS output may contain multiple coupled momentum, thermal, vapor, cloud-water, or related unresolved flux/source contributions. The exact SGS representation must first be derived carefully from the filtered equations.

## 11. Four strategies for DNS-to-LES

The same hierarchy carries over:

- **LES-M1:** direct filtered-DNS SGS regression.
- **LES-M2:** discretization-consistent offline SGS learning.
- **LES-M3:** one-step truth-reset LES training from coarse-grained DNS.
- **LES-M4:** finite-horizon LES rollout training from coarse-grained DNS.

Again, exact long-horizon rollout is an expensive reference capability rather than a requirement to backpropagate through the whole simulation.

## 12. Structured versus black-box SGS learning

The LES study should preserve the representation comparison:

- a structured SGS closure with known conservation, flux-divergence form, tensor symmetries, thermodynamic constraints, or other physics imposed;
- a black-box resolved-tendency correction.

This transfers the methodological matrix from the known-physics paper to a genuine unresolved-physics problem.

## 13. DNS-to-LES filtering is itself part of the science

The DNS-to-LES truth pipeline must treat carefully:

- continuous/filter-defined SGS terms;
- discrete LES residuals;
- commutation errors;
- spatial filtering;
- temporal sampling;
- DNS/LES timestep mismatch;
- conservation under coarse graining;
- consistency between training data and deployed LES discretization.

The study should be positioned explicitly relative to modern ML-LES work on a-priori versus a-posteriori learning, discretization-consistent closures, differentiable LES, rollout horizons, and stable deployment.

## 14. Expected publication sequence

### Paper I — known moist physics

Controlled DIMSWE study:

- Problem A versus Problem B;
- M1–M4;
- short/no-rain double vortex;
- longer/rain-active double vortex;
- selected additional/topographic cases;
- exact known hidden physics for mechanistic interpretation;
- autonomous a-posteriori evaluation.

### Paper II — moist DNS-to-LES subgrid learning

Apply the validated framework to unresolved SGS physics:

- carefully derived moist LES filtering/closure problem;
- DNS truth generation;
- structured versus black-box SGS representation;
- M1–M4;
- stable autonomous LES deployment;
- connection to current ML-LES literature.

Depending on the final emphasis, JFM or JCP may be appropriate.

## 15. Immediate roadmap

1. Finish the current fair-convergence comparison of M1 and M2 for Problem A.
2. Freeze the current Problem-A M1/M2 baselines.
3. Make M3 computationally efficient, including safe independent-window parallelism where needed.
4. Run exact one-step truth-reset M3 for Problem A.
5. Use exact M4 only at practical finite horizons, initially \(H=5\), \(H=10\), or horizons selected relative to a physical timescale.
6. Use the full autonomous trajectory primarily for evaluation.
7. Implement Problem B within the same framework.
8. Complete the small double-vortex \(A/B\times M1\text{--}M4\) benchmark.
9. Generate and diagnose the longer \(64\times64\) rain-active double-vortex truth case.
10. Repeat the structured/black-box and four-strategy comparison in the rain-active regime, including the structured \((A,R)\) diagnostic.
11. Audit/correct topographic Hamiltonian physics before mountain ML.
12. Add selected additional DIMSWE/Hartney-type cases.
13. Write Paper I.
14. Derive the moist DNS-to-LES filtering/closure problem in the context of current ML-LES literature.
15. Apply the same framework to the moist-thermal LES study.

## 16. Guiding principles

- Keep the learned-physics representation, production solver, and training objective modular.
- Preserve exact physical structure for the structured problem; remove it deliberately only for the black-box comparison.
- Do not confuse optimizer nonconvergence with differences between training strategies.
- Do not use held-out trajectories to make training-method choices.
- Treat the numerical discretization as part of the deployed model.
- Exploit exact computational reuse aggressively, but never reuse parameter-dependent trajectories after \(\theta\) changes.
- Use independent-window parallelism for truth-reset training.
- Do not require entire-simulation adjoints when finite physical horizons answer the scientific question.
- Retain exact long-horizon rollout training as a reference capability and investigate cheaper surrogates if needed.
- Correct and certify deterministic solver physics before introducing ML into a new test case.
- Use the known-physics study to understand mechanisms before moving to genuine unresolved SGS learning.

## Frozen terminology

**Problem A:** structured learning of the physical moist rate \(A\), with known source structure and analytical \(R\) retained.

**Problem B:** black-box learning of the complete moist-physics tendency vector.

**Method 1 (M1):** direct operator / a-priori learning.

**Method 2 (M2):** deployed-discrete offline learning.

**Method 3 (M3):** one-step truth-reset solver-in-the-loop learning.

**Method 4 (M4):** finite-horizon recursive rollout solver-in-the-loop learning, with horizons eventually expressed in physical/nondimensional time.

**Full autonomous rollout:** primarily an a-posteriori evaluation diagnostic, not automatically the production M4 training horizon.
