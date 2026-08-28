# Test 2A Method-3/Method-4 preparation and fair offline long fits

## Scope

This stage prepares, but does not scientifically select or run, neural truth-reset (Method 3) and autonomous rollout (Method 4) optimization. It also defines a matched, sequential Method-1/Method-2 overnight comparison. Truth access is restricted to states 0 through 80; states 81 through 160 remain locked.

No MPI, accelerator, parallel-in-time, or held-out claim is made.

## Shared exact trajectory formulation

For a configured window starting at trusted state \(X_k^*\), the shared implementation constructs

\[
\widehat X_k=X_k^*,\qquad
\widehat X_{k+j+1}=F_\theta(\widehat X_{k+j}),\quad j=0,\ldots,H-1,
\]

where \(F_\theta\) is the complete deployed six-child step in the unchanged order:

1. dry RK4 half-step at \(t_n\);
2. dry RK4 half-step at \(t_n+\Delta t/2\);
3. hyperviscosity Euler full-step;
4. DG SSPRK43 half-step;
5. DG SSPRK43 half-step;
6. neural-A/original-analytical-R moist Euler full-step.

Method 3 is a deterministic map/reduce over independently truth-initialized windows. A later reset window never receives the preceding predicted endpoint. Method 4 is one window beginning at \(X_0^*\), with every predicted endpoint reused recursively and no intermediate reset.

Endpoint and accumulated losses are distinct explicit configurations:

\[
L_{k,H}^{\mathrm{end}}=\ell(\widehat X_{k+H},X_{k+H}^*)
\]

and

\[
L_{k,H}^{\mathrm{acc}}=\sum_{j=1}^{H}w_j\ell(\widehat X_{k+j},X_{k+j}^*).
\]

All weights are explicit. The certification-only metric is

\[
\ell(\widehat X_n,X_n^*)=
\frac{1}{2}\frac{\|\widehat X_n-X_n^*\|_M^2}{\|X_n^*\|_M^2}.
\]

It reuses the production mixed mass metric. It is not the frozen scientific Method-3/4 metric. The final metric, field weighting, reset schedule, horizon, endpoint/accumulated choice, accumulated weights, and Method-4 horizon remain unfrozen.

## Exact reverse differentiation

Each trajectory stores owned complete-step and child-stage primal caches. Reverse accumulation traverses complete timesteps in reverse and, within each non-prefix step, children 6 through 1. The moist reverse returns both the state adjoint and the all-1281-parameter neural VJP. This retains state-mediated feedback into all subsequent dry, hyperviscosity, DG, neural-A, and original-R evaluations. R is never frozen or hard-coded to zero.

## Exact reuse audit

| Reuse class | Quantities | Treatment |
|---|---|---|
| Fixed for all \(\theta\) | mesh, spaces, quadrature, topography, assembled fixed operators/solvers, truth states, time metadata, normalization, configured loss weights | constructed or retained once |
| Fixed for all \(\theta\) at each reset origin | children 1 through 5 of the first complete step, including exact time arguments | precomputed once as \(Y_k^*\) |
| Reusable only at bitwise-identical \(\theta\) | timestep states, all split/RK/DG/moist primal caches, and local loss values | one bounded owned forward tape keyed by exact float64 pytree SHA256 and complete problem fingerprint |
| Must be recomputed after \(\theta\) changes | child-6 neural result, every model-generated state after it, later split stages, reverse data | never reused across changed parameters |

At the first fixed truth state, \(\theta\) enters only child 6. Children 1 through 5 are therefore exactly independent of \(\theta\). Their reverse traversal is also unnecessary for a parameter gradient because the reset state is fixed and there is no earlier parameter dependence. Later steps are fully recomputed and fully reversed.

The owned tape retains timestep boundaries and the complete existing child caches, including RK and SSPRK stages. No replay is used during a same-parameter value-to-gradient request. An H=10 production 16x16 tape was approximately 68.38 MB. No revolve/checkpoint algorithm is introduced.

## Production 16x16 certification results

The short certification used the accepted Test-1B 16x16 configuration, c0=0.14, the frozen neural-A/original-R embedding, and truth states no later than 10 (the training-only truth loader exposes 0 through 80 and no later state).

- H=2 fixed-prefix primal states matched ordinary complete-step calls fieldwise to float64 tolerance.
- Fixed-prefix and ordinary objective/parameter gradient agreed to roundoff.
- H=1 all-1281 gradient: directional absolute discrepancy \(3.78\times10^{-19}\), scale-aware relative discrepancy \(1.59\times10^{-7}\).
- H=2 all-1281 gradient: directional absolute discrepancy \(6.68\times10^{-19}\), scale-aware relative discrepancy \(2.33\times10^{-7}\).
- H=2 state tangent/adjoint pairing: relative discrepancy \(1.74\times10^{-16}\).
- Cached same-parameter gradient versus a clean evaluation: zero relative difference, cosine 1 to roundoff.
- A deterministic parameter change invalidated the one-entry tape and forced a new forward trajectory.
- The recursive H=2 gradient differed from two independently truth-reset H=1 gradients by relative norm 0.362 (cosine 0.9924), confirming nontrivial cross-time state feedback.

Measured production 16x16 timings (single serial run; engineering measurements, not performance guarantees):

| Horizon | fresh value (s) | gradient after same-theta value (s) | fresh value+gradient (s) | tape bytes |
|---:|---:|---:|---:|---:|
| 1 | 0.0291 | 0.0316 | 0.0566 | 6,926,520 |
| 2 | 0.0938 | 0.8006 | 0.8871 | 13,754,736 |
| 5 | 0.2639 | 3.5376 | 3.7801 | 34,239,384 |
| 10 | 0.7689 | 7.4541 | 7.7737 | 68,380,464 |

For H=1, reusing the fixed children-1-through-5 prefix reduced the measured value time from 0.0617 s to 0.0294 s, a 2.10x speedup. Same-theta value-to-gradient tape reuse removes the second forward traversal; the measured combined speedups ranged from 1.04x at H=10 to 1.41x at H=1 because reverse work dominates at longer horizons.

The existing child solvers remain exact Firedrake/PETSc operations. No new per-child solve counter was added in this stage.

## Window-level parallelism

The serial reference exposes each Method-3 window as an independent value/gradient contribution followed by a deterministic tree reduction. The safest future parallel route is process-level window batches with one serial Firedrake case per worker and deterministic parent reduction. Startup, duplicated mesh/operator/tape memory, CPU contention, and result reduction must be measured.

No parallel prototype was introduced tonight because mutable Firedrake operations are not certified thread-safe and process/MPI initialization deserves a separate gate. No MPI scalability claim is made.

Method 4 retains sequential forward and reverse time dependence. Ordinary timesteps are not independently parallelizable. Parareal and MGRIT are out of scope.

## Optimizer plumbing and nonscientific smokes

The shared trajectory objective provides PyROL-compatible exact values and gradients, exact SHA-keyed same-parameter tape reuse, accepted-update callbacks, and parameter artifact support. Production L-BFGS requests no HVP. Parameter artifacts can restart exactly, but ROL L-BFGS secant history is process-local and is not restored by parameter-only restart.

Two-accepted-iteration implementation smokes were run for two H=1 reset windows and one H=2 continuous rollout. Both reduced their objectives and used zero HVPs. The smoke loss was uniformly multiplied by \(10^{12}\) because the certification-only unscaled gradient was already below ROL's absolute 1e-8 tolerance. This positive scaling was used only to exercise optimizer callbacks and is not a scientific loss choice.

## Fair Method-1/Method-2 long-fit contract

Both primary fits use:

- the same canonical seed-0 pytree SHA256 `6d4ac7bafe775b90a70e2a199ef3305308c3d02333f7daeac1c870b6f18e0975`;
- the frozen 5-32-32-1 tanh float64 architecture and training-only normalization;
- truth support 0 through 80 only;
- PyROL/ROL line-search L-BFGS, secant storage 20;
- exact gradients, no production HVP;
- gradient tolerance 1e-8 and step tolerance 1e-12;
- a common cap of 200,000 accepted iterations;
- checkpoints at 25k, 50k, 75k, 100k, 150k, and 200k.

The accepted Method-2 fixed cache SHA256 is `baee2dd3ae8a5e3f9ec16f6883e3583d4ac61281d777c3079b002e611504bacf`. Its production-oracle certification remains mandatory.

A matched 20-iteration NONSCIENTIFIC smoke reproduced:

- Method 1: 0.9135568694 to 0.7419320232 in 1.16 s, 43 value and 21 gradient evaluations, zero HVP;
- Method 2: 1.2027413730 to 0.9849326308 in 1.57 s, 43 value and 21 gradient evaluations, zero HVP.

Linear smoke timing suggests roughly 3.2 hours for Method 1 and 4.4 hours for Method 2 at 200k if evaluation behavior remains similar, before autonomous postprocessing. These are non-authoritative estimates.

The unattended runner executes Method 1, Method 2, cross-objective/direct-A postprocessing, two autonomous 0-through-80 evaluations, and final JSON/Markdown reporting sequentially. It stops on the first failure and never automatically resumes a primary fit, because a new process would lose L-BFGS secant history.

