# HVP performance plan (TODO-hvp.md step 20)

The forward-over-adjoint HVP is correct (steps 10-18), but slow. This plan lists where the time
goes, as far as the code structure shows it, and proposes changes in order of expected payoff per
effort. Every change must keep the step 10-12 Taylor, symmetry and repeatability tests passing.
Performance-only changes must also leave the step 19 baseline counts unchanged
(`bash workflows/optimizer_baseline.sh <ranks> --check`). Speedups are then measured with the
recorded `optimize_seconds`, and a new baseline is saved once a change is accepted.

## 1. Cost model of the current implementation

Let S be the number of forward stage solves per time step. For a `LieSplit` step this is the sum
over sub-integrators of stages × subcycles. With `mtswe.cfg` (RK4×2, Euler, SSPRK43×2, Euler):
S = 8 + 1 + 8 + 1 = 18, plus the aux solves. Counting solves per time step:

| Evaluation | forward stages | tangent stages | adjoint stages | SOA stages | increment assemblies (per substep) | total |
|---|---|---|---|---|---|---|
| `obj` | S | | | | | S |
| `jac` | 3S: the forward pass, then the `LieSplittingIntegrator.take_adjoint_step` substep recompute, then the `GeneralRK.take_adjoint_step` stage recompute | | S | | 2 (`L_delta_lambda`, `L_delta_grad`) | ≈ 4S |
| `hessp` | 3S: the `compute_states_and_tangents` pass, the `_compute_substep_tangents` recompute, and the `GeneralRK` stage recompute | 3S (same three places) | S | S | 4 (adds `L_delta_lambda_dot`, `L_delta_grad_dot`) | ≈ 8S |

Further costs on top of the solve counts:

- **Direction-independent work is redone on every HVP.** `_hvp_sweep` recomputes the forward trajectory and the whole first-order adjoint (λ_n, μ_i, the gradient) for every HVP. Newton-CG and the trust-region methods evaluate several HVPs at the same point, one per inner CG/Krylov iteration, right after `jac` was evaluated there. For a fixed point only the tangent and second-order adjoint actually change.
- **The second-order forms are bigger.** The tangent-linear and second-order-adjoint residuals are directional derivatives of forms that already contain `adjoint(derivative(...))`. Their kernels are larger and slower to assemble than the first-order ones, especially through the `threewayphysics` conditionals and `dg1limiter`.
- **There is a one-off setup cost per job.** `_setup_second_order` builds up to 4 × nstages + 1 new `LinearVariationalSolver`s per sub-integrator (tangent, tangent-aux, SOA and SOA-aux per stage, plus Δλ̇), on top of the 4 × nstages + 1 first-order ones. Each has its own LU factorization of the same mass matrix, and compiles their forms (TSFC). The workflow builds a new model for every job, so this is paid per job. The disk cache only removes the compilation part on later jobs.
- **There is Python and UFL overhead per substep.** `assign(wn + dt * sum(...))` builds a new UFL expression every call. `assemble(L)` allocates a new Cofunction every call, and `create_flattened_numpy_arr_from_mixed_function` copies every increment to numpy (see the "VERY SLOW" comments in `take_adjoint_step`).

## 2. Measure first (do this before any change)

1. **Separate setup from steady state.** In a small driver on the workflow problem (25×25 `mtswe.cfg`, the 16-step `ic-params` window), time:
   - `obj`;
   - `jac`;
   - a first `hessp`, which includes `_setup_second_order` and compilation;
   - a second `hessp` at the same point;
   - a third `hessp` at the same point with another direction.

   This gives the setup cost and the steady-state cost ratio `hessp / jac`, which should be about 2 by the model above.
2. **Find where the time goes.** Profile the steady-state `hessp` with PETSc events, using the existing flamegraph tooling (`-log_view :flamegraph.txt:ascii_flamegraph` and `flamegraph.pl`). Add `PETSc.Log.Event` regions for:
   - the forward/tangent pass;
   - the substep recompute;
   - the forward, tangent, adjoint and SOA stage solves;
   - the increment assemblies;
   - the observation sources.

   This shows whether the time is in the solves (assembling stage right-hand sides), in the increment assemblies, or in Python overhead.
3. **Split the time per sub-integrator.** Record which sub-integrator dominates: the dynamics (RK4), `threewayphysics` or `dg1limiter`. The second-order forms of the forcing terms may be disproportionately expensive.
4. **Count the optimizer-level work.** Use the step 19 baseline: `nhvp / nit` and `njev / nit` show how many HVPs each Newton iteration buys.

Record the results in this file. They decide which of the changes below are worth doing.

## 3. Changes, in proposed order

### A. Reuse direction-independent work at a fixed point (largest expected gain)

**A1. Cache the forward trajectory and first-order adjoint per point.** Key the cache on the exact point, as the `ReducedProblem` fun/jac memo already does. For coefficient directions, also key it on the coefficient values.
- **Store:** `jac` already computes and stores x_n. It would also store λ_n for every step, at the cost of one more state per step.
- **Use:** an HVP at a cached point skips the forward pass and gets λ_{n+1} from the cache instead of propagating it. The first-order stage adjoints μ_i are still needed as inputs to the SOA stages, so without A2 they are re-solved: S per step.
- **Saves:** the forward pass (S) and the first-order increment assemblies, and the gradient is no longer recomputed.
- **Requires:** an HVP entry point that takes "states, adjoints from cache" and returns only the Hessian part (`_hvp_sweep(..., need_gradient=False)`).

**A2. Store stage values for the whole trajectory (checkpoint level 2).** This is the "enriching checkpointing of stage values" item of TODO-hvp.md step 20. Store the forward stages F_i, W_i of every substep of every step during the forward pass, and the adjoint stages μ_i, ν_i during `jac`.
- **Steady-state HVP cost** at a cached point becomes only the tangent stages (S) plus the SOA stages (S): about 2S instead of 8S, a ~4× reduction in solves.
- **`jac`** drops from 4S to 2S if `obj` or the first-order forward stores stages too.
- **Memory:** nsteps × nsubsteps × nstages states per stored quantity. That's small for the workflow windows, but it must be optional (for example `checkpoint: states | stages`), with the current recompute path as the default until it is validated.

**A3. Without full stage storage, keep the stages of one step.** In `LieSplittingIntegrator`, the substep states are recomputed forward and then traversed in reverse within the same step. Keeping each substep's stage values (only nsubsteps × nstages states, for the current step) lets the `GeneralRK` SOA and adjoint steps skip re-solving their forward (and tangent) stages.
- **Saves:** S per step for `jac` (4S → 3S) and 2S for `hessp` (8S → 6S).
- **Memory:** negligible, and it is independent of A1/A2. This is a good first change: it is local to `timestepping.py` and can be checked with the existing step-level Taylor tests.

### B. Cheaper per-substep work

**B1. Skip unneeded first-order increments.** In an HVP-only sweep, `L_delta_grad` and `L_delta_lambda` are only needed for the gradient output, not for propagating λ, which uses `deltalambdasolver`. Skip their assembly when `need_gradient=False`. This saves 2 of the 4 assemblies per substep.

**B2. Preallocate assembly targets and accumulate in Firedrake.**
- Use `assemble(form, tensor=buffer)`, or a stored assembler, for `L_delta_lambda`, `L_delta_grad`, `L_delta_lambda_dot` and `L_delta_grad_dot`.
- Accumulate the dual increments in Cofunctions, and convert to numpy once per sweep instead of once per substep.
- This also applies to `jac`.

**B3. Build update expressions once.** Construct the update expressions (`wn + dt * sum_i b_i Fdot_i` and the forward update) once, in `_setup_second_order`, and only `assign` them per step.

**B4. Simplify the stage right-hand sides.** `create_linear_solver_from_residual` uses `L = action(a, var) - residual`, which assembles M·var only for it to cancel. For residuals that are linear in `var` (every stage residual here), use `L = -replace(residual, {var: 0})`. This removes one mass-matrix action per stage solve, in the first-order solvers too.

### C. Setup cost

**C1. Share one factorization per space.** Every stage solver (forward, aux, adjoint, tangent, SOA, Δλ, Δλ̇) solves with the same mass matrix of its space. Use one `LinearSolver` per space (x and aux) with a preassembled matrix and its factorization, and solve with different right-hand sides. That gives 2 factorizations per sub-integrator instead of up to 8 × nstages + 2. With `lump_mass: true` the x-space solve is a pointwise division and needs no factorization at all.

**C2. Reuse the model across jobs.** Build the model and timestepper once per (truth run, spec) in `optimize_params`, instead of once per job, so `_setup_second_order` and the solvers are built once per process. Jobs of the same truth run differ only in the objective.

**C3. Check form complexity.** Inspect the expanded SOA forms of `threewayphysics` and `dg1limiter`: number of integrals, and kernel compile and run time. If the conditionals produce large kernels, assemble the forcing-term SOA contributions with simpler forms, or exclude terms whose second derivatives vanish (see `directional_derivative`, which already drops forms that don't depend on a pair).

### D. Optimizer-level

**D1. Fewer HVPs per iteration.** Compare Newton-CG with `trust-krylov` and `trust-ncg` for the unbounded targets, using the step 19 tooling. Their inner iterations can stop earlier. Expose Newton-CG's inner CG tolerance if scipy's default is too tight.

**D2. Gauss-Newton is not worth it here.** A Gauss-Newton HVP drops the second-order adjoint terms but still needs a tangent-linear and an adjoint-of-tangent sweep, so its cost structure is the same. It is not proposed.

### E. Broader timestepper support

DIRK and implicit RK are not implemented in DIMSWE even for first-order adjoints, so there is nothing to extend yet. When they are added, their HVP support should follow the same directional-derivative approach (`_setup_second_order`).

## 4. Suggested sequence

1. Measurements (section 2).
2. A3 + B1 + B3 + B4: local, low-risk, no memory cost. Re-run the Taylor and repeatability tests and the baseline.
3. A1 (cache per point), then A2 (optional stage checkpointing): the largest gains, behind a configuration option.
4. B2 and C1/C2, if the measurements show assembly or setup dominating.
5. D1 with the comparison tooling; save a new baseline with the improved times.

For each step:
- the step 10-12 tests must pass;
- `optimizer_baseline.sh --check` must pass with unchanged counts;
- `optimize_seconds` must improve; record it with `--time-factor` and a new `--save`.
