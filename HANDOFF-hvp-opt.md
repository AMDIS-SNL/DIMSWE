# Handoff: HVP and gradient performance optimization

Status of the HVP and gradient performance work (TODO-hvp.md step 20, `docs/hvp_performance_plan.md`),
and what is left to do. Measurements are from
`bash workflows/profile_hvp.sh 4 workflows/examples/optimize_ic_params_newton_cg.cfg --job ox-0p086034__final --repeats 2`
(4 MPI ranks, `tswe`-type `advdens-cf-h1` model, Q3 with `lump_mass: true`, LieSplit of RK4 / Euler / SSPRK43 / Euler, 16 steps).

Working rules (AGENTS.md): only `py_compile` can be run by the agent; the user runs tests and profiles.
Do not commit, pull or push. Every method needs a docstring and comments. Tests are `test_<feature>.py`
with explicit seeds; new CLI or pipeline paths need an integration test.

## 1. Where things stand

| per evaluation (steady) | original | now |
|---|---|---|
| obj | ~3.3 s | **0.85 s** |
| jac (gradient) | 15.4 s | **2.12 s** |
| hessp (HVP) | 46 s | **2.81 s** |
| one-off setup per process (first obj + jac + hessp) | 58.6 s | **29 s** (3.8 + 7.6 + 17.6) |
| estimated Newton-CG job (5 its, 7 HVPs) | 127 s recorded | **49.2 s**, ~60% of it setup |

Steady ratios: jac/obj 2.50 (model 2), hessp/jac 1.32 (model 1). Stage passes per substep match the
cost model (obj 1, jac 1+1, hessp tangent 1 + soa 1).

### Implemented (see the docs for details)

| item | where | doc |
|---|---|---|
| B0 persistent increment assemblers | `dimswe/assembly.py` `PersistentAssembler` | plan §3 B0 |
| A3 stage reuse within a step | `timestepping.reuse_stages` | plan §3 A3 |
| IC-map point cache | `dimswe/optimize.py` `ParametricInitialCondition(cache=True)` | plan §2.3 |
| A1/A2 stage checkpoint (default) | `optimizer.checkpoint: stages` | HVP doc §24 |
| increment solves, direct stage copies | `copy_function_values` | HVP doc §25 |
| C1 one shared mass inverse per operator, `AffineSolver` (no `action()` expansions, probe check, reference fallback) | `dimswe/assembly.py` `AffineSolver`, `MassSolver`, `shared_mass_solver`; `GeneralRK._affine_solver` | HVP doc §26 |
| bare PETSc KSP without OptionsManager (not in `firedrake.petsc` of the installed Firedrake) | `MassSolver._setup_ksp` | HVP doc §26 |
| persistent IC-map projections (no LU per `project`) | `PersistentProjector`, `ParametricInitialCondition._project` | HVP doc §27 |
| cheap mass inverses: diagonal / cellwise (Slate) / per field; option `mass_inverse` (`optimizer.mass_inverse`, default `auto`) | `MassSolver._try_diagonal`, `_try_cellwise`, `_try_fieldsplit` | HVP doc §28 |
| collective fallbacks (all ranks switch together; original error surfaced) | `MassSolver._all_ranks`, `PersistentProjector.project` | HVP doc §27 |

The profiler (`workflows/profile_hvp.py`) prints per sub-integrator the increment and stage solve
strategies, `why <solve>` for fallbacks, and `mass solver <method>`. Currently every sub-integrator
reports `mass solver diagonal` and all solves `mass`.

## 2. Open verification items (do first)

1. **Save a new optimizer baseline.** Nothing since the A1/A2 work has been baselined. All solves are
   still exact, so the optimizer path should match to round-off:
   ```bash
   rm -rf runs/optimizations/doublevortex-params runs/optimizations/doublevortex-params-newton-cg
   bash workflows/optimizer_baseline.sh 4 --save
   ```
   (Run `--check` against the old baseline first if one exists.)
2. **Tests not yet confirmed by the user after the last changes:**
   ```bash
   pytest tests/unit/test_mass_inverse.py tests/unit/test_affine_solver.py tests/unit/test_persistent_projection.py \
          tests/unit/test_ic_map_cache.py tests/unit/test_persistent_assembly.py tests/unit/test_stage_checkpoint.py \
          tests/unit/test_workflow_optimizer_tables.py tests/integration/test_optimization_workflows.py \
          tests/integration/test_profile_hvp.py -s
   mpiexec -n 4 python -m pytest tests/unit/test_persistent_projection.py -s
   ```
   In `test_mass_inverse.py`, the mixed CG×CG×DG case expects `fieldsplit(diagonal, diagonal, cellwise)`.
   If it reports `..., ksp)`, the `split_form` lookup of the per-field forms failed in the installed
   Firedrake (see `inverse_notes`). That only affects non-diagonal DG fields, not the current job.
3. **Unexplained earlier failure.** Before the fallbacks were made collective, the first profile run with
   `PersistentProjector` failed on 4 ranks: the persistent projection raised (error swallowed), and the
   `Function.project` fallback then failed with `DIVERGED_LINEAR_SOLVE`. After the collective-fallback
   change the run worked with no fallback warning, so the likely cause was a rank-local error
   desynchronizing a collective call. The original error was never seen. If
   `PersistentProjector falls back to Function.project: ...` ever appears, investigate that reason.
4. **`ic map second tangents` at a new point.** All HVPs in the profile are at one point, so the
   projection only appears in hessp-1 (1.65 s, mostly compilation). Check in an optimizer run (or a
   profile at two points) that it is tens of milliseconds at later points.

## 3. Remaining optimization tasks, by expected payoff

### 3.1 First-call setup (29 s per process, ~60% of a Newton-CG job)

Flamegraph of hessp-1 against a steady HVP (before the diagonal-mass change; under PETSc logging):

| cost | s | notes |
|---|---|---|
| TSFC `compile_form` (cache lookups + compilation) | 5.4 | kernels are on disk after the first run; the cost is largely form processing / signatures |
| `setup second order` Python self time (UFL `derivative`, `replace`, `expand_derivatives`, `directional_derivative`) | 3.0 | `dimswe/timestepping.py` `_setup_second_order` / `_build_second_order_solvers` |
| first `assemble` per form, Python self time | 2.4 | one-off form preprocessing |
| `pyop2.compilation.load` | 2.5 | loading compiled kernels |
| observation-source solvers | ~1.0 | `lambda_np1_projector`, `lambda_dot_projector` in `dimswe/optimize.py` (LVS with LU) |
| first stage-store allocations (`Function.copy`) | ~1.0 | stage checkpoint storage |

Tasks:
- **C2: reuse the model across jobs** (plan §3 C2). Build the model and timestepper once per (truth
  run, spec) in `workflows/optimize_params.py` instead of once per job, and change only the objective.
  This removes the whole setup for every job after the first in a process. It is the largest remaining
  gain for multi-job runs. Watch out for per-job state: the stage checkpoint (`clear_checkpoint`), the
  IC-map point cache, `_shared_mass_solvers` (valid across jobs on the same mesh), and data blocks.
- **Duplicate symbolic work in `setup second order`** (3.6 s of hessp setup is form and solver
  construction). Profile `_setup_second_order` with cProfile, look for derivatives or `replace`s
  computed more than once per stage pair, and cache them. C3 below also shrinks it.
- **Observation-source solvers.** Replace the two `LinearVariationalSolver`s with `AffineSolver` or
  `PersistentProjector`-style solves using the shared x-space `MassSolver` (diagonal here). This saves
  about 1 s of one-off setup plus a little per call, and `assemble(self.source_form)` then comes free
  from the same assembly (it is currently assembled twice per observation).
- **Stage-store allocation** (~1 s, once): could be preallocated during setup. That moves rather than
  removes the cost, so it is low priority.
- **Compilation and loading** (~8 s): mostly intrinsic. Check that `FIREDRAKE_TSFC_KERNEL_CACHE_DIR` /
  PyOP2 cache are on a fast local disk and shared between runs. Fewer and smaller forms (C3) also help.

### 3.2 Steady HVP (2.81 s; hessp/jac 1.32, model 1)

Steady hessp by region: soa stages 0.80 s, tangent stages 0.66 s, other 0.53 s, λ̇ increment 0.50 s,
update 0.19 s, restore stages 0.08 s.

- **λ̇ increment of RK4** (0.29 s per HVP, against 0.09 s for the λ increment): its residual is the
  large second-order form. Inspect `L_delta_lambda_dot` / the λ̇ `AffineSolver` residual in
  `GeneralRK._build_second_order_solvers`, drop terms that vanish for the job (the coefficient
  increments are already skipped when there are no coefficients), and check the kernel size.
- **C3: second-order form complexity** (plan §3 C3). SOA stages cost 1.53 (RK4) and 1.61 (SSPRK43)
  forward passes, tangent stages 1.14 / 1.33. Inspect the expanded SOA and tangent forms of
  `dg1limiter` and `threewayphysics` (number of integrals, kernel run time) and remove terms whose
  second derivatives vanish (`directional_derivative` already drops forms independent of a pair).
- **Expression assigns ("update" and part of "other", ~0.7 s per HVP, ~0.5 s per gradient).** Updates
  such as `lambda_np1.assign(lambda_np1 + delta_lambda)` in `dimswe/optimize.py` (adjoint and SOA
  sweeps) and the RK update combinations in `dimswe/timestepping.py` generate and run an expression
  kernel each. Replace them with PETSc Vec operations (`axpy`, `maxpy` on `dat.vec`) or plan item B3
  (build update expressions once). Expected 5-10%.
- **`other` time** (0.4-0.5 s per evaluation): Python overhead between regions. A flamegraph or cProfile of
  one steady HVP would show what it is (likely assigns, `reset_internal_vars`, sub-function lookups).

### 3.3 Steady gradient (2.12 s) and objective (0.85 s)

The forward stages (0.51-0.53 s) and adjoint stages (0.62 s) are now mostly assembly kernels. The
remaining items are the same as for the HVP: expression assigns (`update` 0.11 s), `other` (0.16-0.40 s),
and form complexity (C3).

### 3.4 Optimizer level

- **D1: fewer HVPs per iteration** (plan §3 D1). HVPs are 76% of the steady job time. Compare
  Newton-CG with `trust-ncg` and `trust-krylov` (unbounded targets), and expose Newton-CG's inner CG
  tolerance if scipy's default is too tight. This changes the optimization path, so compare final
  losses and iterations, not only time.

### 3.5 Generality (not needed for the current job)

- **Non-diagonal masses.** Without `lump_mass` (or with non-collocated DG), `mass_inverse: auto` uses the
  cellwise inverse for DG fields and LU for continuous ones, the latter back at ~4.5 ms per MUMPS solve
  on 4 ranks. Options: MUMPS Cholesky, distributed right-hand sides (`mat_mumps_icntl_20`), or a few
  CG iterations with a tight tolerance (would need `is_direct_solver` / the probe check relaxed).
- **Aux spaces** of forcing terms use `mass_aux[i]` (derivative of the aux left-hand side), which may
  not be a plain mass. They are currently diagonal for this job; check `mass solver` for other configs.

## 4. Reference

- Plan and measurements: `docs/hvp_performance_plan.md` (§2.3 profiler measurements, §3 items, §4 sequence).
- Design notes: `docs/hessian_vector_products.md` §24-§28.
- Step status: `TODO-hvp.md` step 20.
- Profiling: `workflows/profile_hvp.sh` (add `--flamegraph` for PETSc's nested log; render with
  `perl flamegraph.pl --countname us flamegraph.txt > flamegraph.svg`; use runs without it for timings).
  Output in `runs/profiles/doublevortex-params-newton-cg/ox-0p086034__final/`.
- Baseline: `workflows/optimizer_baseline.sh 4 [--check|--save]`.
