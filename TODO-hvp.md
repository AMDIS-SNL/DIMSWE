# Hessian-Vector Product Plan

1. Read and map the existing first-order optimization path.
   Identify the exact data flow in `dimswe/optimize.py` from `ReducedProblem.fun` and `ReducedProblem.jac` down to `Lagrangian_ODEConstrainedOptimization.jac`, including forward state storage, observation-source assembly, reverse sweeps through `timestepper.take_adjoint_step`, and reduced-space scaling and call counting. This pins down where Hessian-vector products should be inserted without creating a second optimization stack.

2. Translate PETSc equation `2.11` into DIMSWE's discrete objects and naming.
   Write down the concrete DIMSWE equivalents of the optimizer-space direction `v`, the full-space perturbation for coefficients and-or initial conditions, the tangent-linear state perturbations `w_n`, the first-order adjoint `lambda_n`, the second-order adjoint directional variables, and the observation-level second derivatives of the loss. The goal is a one-to-one map from the notation in `petsc-opt.pdf` to the variables already used in `dimswe/optimize.py` and `dimswe/timestepping.py`.

3. Extend the timestepper API from first-order adjoint support to the primitives required by forward-over-adjoint HVPs.
   The current interface exposes `take_forward_step` and `take_adjoint_step`. Add the lower-level capabilities needed by equation `2.11`: tangent-linear propagation through one step, second-order adjoint directional propagation through one step, and matrix-free vector-Hessian contractions of the step residual or right-hand side with tangent and adjoint directions. Keep these as additive API extensions alongside the existing first-order path.

4. Decide the minimal storage and recomputation strategy for HVP evaluation.
   Determine exactly which quantities must be available at each step: primal state `x_n`, tangent state `w_n`, first-order adjoint `lambda_n`, and observation source terms plus their directional derivatives at observation steps. Keep the first implementation aligned with the current checkpointing style by reusing full forward state storage and recomputing timestepper internals as needed instead of introducing a new checkpointing system.

5. Implement full-space Hessian-vector products in `Lagrangian_ODEConstrainedOptimization`.
   Replace the `hessp` placeholder in `dimswe/optimize.py` with a full-space HVP routine that accepts a point and a direction, runs a tangent-linear forward sweep for that direction, runs the first-order adjoint sweep, runs the forward-over-adjoint second-order sweep from equation `2.11`, and returns the Hessian action in full parameter space. Cover the supported optimization targets `coeffs`, `ics`, `coeffs+ics`, and `ic-params`.

6. Extend `ReducedProblem` so SciPy can use Hessian-vector products in reduced, scaled coordinates.
   Add reduced-space HVP support that mirrors the existing reduced `fun` and `jac` behavior: map the reduced direction to full space with the scaling and `free_mask`, call the full-space `hessp`, map the result back into reduced coordinates with the correct chain rule, and count HVP evaluations separately from `nfev` and `njev`.

7. Update `optimize()` to expose HVP-enabled second-order methods cleanly.
   Extend the optimizer driver in `dimswe/optimize.py` so it passes `hessp` to `scipy.optimize.minimize` when the chosen method supports it. Preserve existing first-order methods unchanged while enabling practical HVP-consuming methods such as `Newton-CG` and compatible trust-region methods, with explicit handling of any restrictions caused by bounds.

8. Update the workflow spec and validation logic to recognize HVP-enabled optimizers.
   In `workflows/optimize_params.py`, extend optimizer-spec handling so users can select second-order methods in YAML, validate method-specific restrictions, especially around bounds and target type, and preserve current defaults so existing workflow configs continue to run unchanged.

9. Record HVP metrics throughout the workflow outputs.
   Extend optimization bookkeeping so each job records optimizer method, optimization iterations, `nfev`, `njev`, `nhvp` or equivalent HVP call count, final loss, and convergence status. Update `history.csv` and `result.yaml` only where necessary so existing consumers remain intact.

10. Add unit tests for the reduced and full-space HVP plumbing.
    Add focused tests under `tests/unit/` for `ReducedProblem` HVP scaling and `free_mask` handling, HVP call counting, and optimizer-method selection logic for supported and unsupported second-order methods. Keep these tests lightweight and mostly pure Python.

11. Add Firedrake-side unit tests for Hessian-vector-product correctness.
    Following the existing gradient-test pattern in `tests/gradient_helpers.py` and `tests/unit/test_optimize_gradients.py`, add HVP Taylor tests based on the gradient expansion `g(x + eps d) = g(x) + eps H(x)d + O(eps^2)`. Cover at least coefficient optimization, full initial-condition optimization, and analytic initial-condition parameters via `ParametricInitialCondition`.

12. Add repeatability tests for HVP evaluation.
    Since the current code already has repeatability concerns around timestepper internal-state resets, add explicit repeatability checks that running the same HVP Taylor test twice gives identical remainders and rates.

13. Add workflow-level tests for second-order optimizer configuration and reporting.
    Extend `tests/integration/test_optimization_workflows.py` with pure-Python tests that validate second-order methods are accepted in optimization specs, unsupported method and bounds combinations are rejected with clear errors, and the new result fields for HVP counts and method metadata are present.

14. Update workflow examples to demonstrate first-order versus second-order optimizer choices.
    Add or extend example configs in `workflows/examples/` so there are clear example runs for an existing first-order method such as `L-BFGS-B` and an HVP-enabled second-order method such as `Newton-CG`. Keep the example problem small and aligned with the existing optimization examples.

15. Add comparison tooling for optimizer performance across methods.
    Add a workflow script, or extend the existing workflow scripts, to compare optimization jobs across methods using recorded metrics. Summarize at minimum optimizer method, optimization steps, function calls, Jacobian calls, HVP calls, final loss, and optimizer status in a machine-readable format such as CSV or YAML.

16. Add a simple user-facing script path to run optimizer comparisons.
    Provide a script entry point in `workflows/` and a matching shell wrapper that can scan one optimization root or a list of job directories, aggregate the metrics above, and print and save a comparison table.

17. Document the new second-order optimization capability.
    Update `README.md` to remove the statement that only Jacobians are available. Update `workflows/README.md` to describe supported second-order optimizers, any restrictions, the new result fields, and the optimizer-comparison script. Explain briefly that DIMSWE uses a forward-over-adjoint Hessian-vector product following `petsc-opt.pdf`, equation `2.11`.

18. Verify the feature through staged correctness checks before considering performance.
    Use the following verification sequence: unit tests for reduced and full-space HVP plumbing, Firedrake-side HVP Taylor tests, workflow config and reporting tests, one small end-to-end optimization with a first-order method, one with an HVP-enabled second-order method, and a comparison-script run on those results to confirm the reported counts and summaries are correct.

19. Compare optimizer behavior on a representative small problem.
    Run the same optimization problem with at least two methods, ideally `L-BFGS-B` and one HVP-enabled second-order method, and use the comparison tooling to assess optimization steps, `nfev`, `njev`, `nhvp`, final loss, and convergence status. Treat this as a concrete regression baseline for future optimizer work.

20. Defer performance tuning to a follow-up pass if needed.
    After correctness is established, consider follow-up optimizations such as reducing duplicate recomputation between gradient and HVP paths, caching tangent and adjoint observation terms, enriching checkpointing of stage values, and broadening support across more timestepper variants if some methods need to be excluded initially.
    Plan: `docs/hvp_performance_plan.md`. It gives a cost model (a steady-state HVP is about 8S stage solves per step, against about 4S for a gradient), measurements to do first, and changes in order: per-step stage reuse, caching the forward and adjoint trajectories per point, optional stage checkpointing, assembly and setup reductions, and optimizer-level choices. Every change is checked against the step 19 baseline.
    Measurement tooling (plan section 2) is done: profiling regions in `dimswe/profiling.py`, instrumented in `dimswe/optimize.py` and `dimswe/timestepping.py`, and the driver `workflows/profile_hvp.py` / `profile_hvp.sh`. The step 19 baseline already shows an HVP costing about 3.8 gradients (model: 2), and HVPs taking 82-91% of the Newton-CG time. Profiler results (plan section 2.3): stage passes match the cost model and second-order stage passes cost about the same as forward ones. The bottleneck is the bare `assemble()` of the increment forms in the RK adjoint steps (52% of an HVP, 31% of a gradient), so the next change is plan item B0, persistent increment assemblers. B0 is implemented (`dimswe/assembly.py`, `GeneralRK._assemble_increment`) and re-profiled: an HVP went from 46 s to 22 s and a gradient from 15.4 s to 10.4 s, so hessp/jac is now 2.1 (model: 2). A3 (stage reuse within a LieSplit step, `timestepping.reuse_stages`, default true) is implemented and re-profiled: stage passes per substep went from 3 to 2, and an A/B run in one session (`profile_hvp.py --recompute-stages`) shows the gradient 21% faster (11.7 → 9.2 s) and the HVP 20% faster (23.8 → 19.1 s). The initial-condition map now caches its forward state, tangents and second tangents per parameter point (`ParametricInitialCondition(cache=True)`), so HVPs at the point of the last gradient project nothing. Re-profiled: the steady HVP went from 19.1 s to 16.8 s. Stage solves are now 76% of an HVP, so the next large gains are A1/A2. A1/A2 are implemented as the optional `optimizer.checkpoint: stages` (default `states`): `jac` stores every step's stage values and stage adjoints, and HVPs at its point only solve the tangent and second-order adjoint stages (2 instead of 6 passes per substep). Profiled: the steady HVP went from 16.8 s to 7.5 s and the gradient from 9.8 s to 7.3-7.9 s; an HVP now costs about one gradient. The end-to-end and baseline checks passed, and `stages` is now the default wherever the timestepper supports it (`optimizer.checkpoint: null`; `states` remains selectable). Next: save a new optimizer baseline. The λ/λ̇ increment solves (13%) now assemble their right-hand side once and solve with a shared factored mass matrix (`IncrementSolver`), and the stage stores and restores (9%) copy values directly (`copy_function_values`). Profiled: the steady HVP went from 7.5 s to 6.2 s and the gradient from 7.3 s to 6.7 s (the copies gave the expected 0.6 / 0.4 s, the increments only 0.3 / 0.05 s). The one-off setup (about 60 s per process, half of a Newton-CG run) was split with the old flamegraph: about 16 s of LU factorizations (one per stage solver, all of the same mass matrix), about 22 s of kernel compilation and loading, and about 8 s of UFL `action` expansions and solver construction. Plan item C1 is implemented: every stage and increment equation is an `AffineSolver` (one residual assembly, a mass matrix factored once per model and shared by all sub-integrators, no `action` expansions), checked by its residual on the first solves, with the old solver as fallback. Profiled (second run): the setup went from 58.6 s to 27.7 s per process, but the steady stage solves got about 25% slower through Firedrake's `LinearSolver` and numpy copies. `MassSolver` now uses a bare PETSc KSP, and the steady `AffineSolver` path solves straight from the assembled tensor. Re-profiled: the steady obj / gradient / HVP are 2.47 / 6.24 / 5.65 s (pre-C1 2.75 / 6.86 / 6.40 s) with the setup still 28 s per process; the estimated Newton-CG job is 91.7 s (was 112.6 s). A flamegraph of the first calls shows that, besides one-off compilation (about 8 s) and Python setup (about 5 s), the initial-condition map re-factored a mass matrix in every projection, about 3.5 s at every new point. Its projections are now `PersistentProjector`s (shared factored mass matrices, persistent right-hand sides). The flamegraph also shows that the MUMPS triangular solves of the (lumped, diagonal or cellwise) mass matrices are about half of every steady evaluation; `MassSolver` now divides by diagonal masses, applies the cellwise inverse for DG fields and works field by field for mixed spaces (`mass_inverse`, default `auto`; `factor` keeps LU). Re-profiled (all mass matrices diagonal): steady obj / gradient / HVP 0.85 / 2.12 / 2.81 s (were 2.47 / 6.24 / 5.65 s), the estimated Newton-CG job 49.2 s (was 91.7 s), of which the one-off setup (29 s per process) is now about 60%. Next: save a new optimizer baseline, then the setup (compilation, C2) and the λ̇ increment.

## Design Constraints

- Reuse the existing reduced-problem abstraction instead of creating a separate second-order optimizer path.
- Implement HVPs matrix-free, consistent with `petsc-opt.pdf` and the current code structure.
- Keep the first implementation compatible with the current forward-and-recompute style.
- Use Taylor tests of `H d` via gradient differences as the main correctness criterion.
- Base workflow comparisons on recorded counters rather than parsing optimizer logs.
