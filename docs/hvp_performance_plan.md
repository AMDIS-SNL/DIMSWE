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

### 2.1 Tooling

- **Profiling regions.** `dimswe/profiling.py` provides `region(name)` and `@timed(name)`. These are nested timing regions that also open a PETSc log event of the same name once Firedrake has loaded petsc4py. They are disabled by default, so the instrumented code only pays for a PETSc event begin/end. The instrumented regions are:
  - `obj`, `jac` and `hessp`, the entry points of `Lagrangian_ODEConstrainedOptimization`;
  - inside them: `forward pass`, `forward+tangent pass`, `observation sources`, `adjoint sweep`, `second-order adjoint sweep`, `loss` and `setup hvp`;
  - inside a LieSplit step: `substep recompute` and `substep adjoints`, with one region per sub-integrator, named like `0:RK4(model)` (`LieSplittingIntegrator.profile_labels`);
  - inside a GeneralRK step:
    - the stage passes `forward stages`, `tangent stages`, `adjoint stages` and `soa stages`;
    - the increment assemblies `gradient increments` and `hessian increments`;
    - the increment solves `lambda increment solve` and `lambda_dot increment solve`;
    - `update` (the `assign(xn + dt * sum(...))` updates);
    - `setup second order` (form and solver construction only, not compilation);
  - the initial-condition map: `ic map forward`, `ic map tangents` and `ic map second tangents`.
- **Driver.** `workflows/profile_hvp.py` (wrapper `workflows/profile_hvp.sh`) builds one job of an optimization spec exactly as `optimize_params.py` does. At the starting point it then evaluates `obj` ×(1+R), `jac` ×(1+R), `hessp` in direction p1 ×(1+R), and `hessp` once in a second direction p2 (R = `--repeats`, default 1). It calls the full-space functions directly, so the `ReducedProblem` memo does not hide repeats. It reports:
  - first-call, steady-state and setup seconds per evaluation type;
  - the measured `jac/obj` and `hessp/jac` ratios against the cost model;
  - the stage passes per time step and substep, checked against the section 1 table;
  - the time per region, and `other`, the time outside every instrumented solve or assembly (Python, UFL expression building, numpy copies);
  - the cost of one stage pass per sub-integrator and stage kind, relative to a forward pass;
  - an estimate of the optimizer's run time from the counts in its `result.yaml`, next to the recorded `optimize_seconds`.

  The output is `runs/profiles/<optimization>/<job>/profile.yaml` and `profile-tree.txt`. `--flamegraph` also writes the PETSc nested log (`flamegraph.txt`, plus `flamegraph.svg` via `./flamegraph.pl`).
- **Tests.** `tests/unit/test_profiling.py` covers the regions. `tests/integration/test_profile_hvp.py` covers the driver arithmetic and the CLI. Its Firedrake test checks that the instrumented `tests/tswe_small.cfg` gradient and HVP do exactly the stage passes of the section 1 cost model.

Measurement runs:

```bash
# steady-state costs of the baseline problem (truth runs from optimizer_baseline.sh), no flamegraph overhead
bash workflows/profile_hvp.sh 4 workflows/examples/optimize_ic_params_newton_cg.cfg --job ox-0p086034__final --repeats 2
# the same in serial, to see the per-rank work without communication
bash workflows/profile_hvp.sh 1 workflows/examples/optimize_ic_params_newton_cg.cfg --job ox-0p086034__final --repeats 2
# PETSc view of the same: which solves, assemblies and kernels sit under each region
bash workflows/profile_hvp.sh 4 workflows/examples/optimize_ic_params_newton_cg.cfg --job ox-0p086034__final --flamegraph
```

The Newton-CG spec is used so that the job's own `result.yaml` (`nhvp` > 0) feeds the optimizer estimate. Pass the L-BFGS-B job's `result.yaml` with `--results` to estimate both runs.

### 2.2 What the step 19 baseline already shows

`workflows/baselines/doublevortex-params.yaml` records `optimize_seconds` and the counts for the three `__final` jobs with both methods (4 ranks). `optimize_seconds` excludes the first `obj` + `jac`, which `run_job` evaluates before it starts the clock.

- **L-BFGS-B.** 131.5 s for 7 `obj` + 7 `jac`, 135.6 s for 7 + 7, and 113.8 s for 6 + 6. That is a consistent 19.0 ± 0.3 s per `obj` + `jac` pair. Splitting the pair with the cost model (`jac` ≈ 4 `obj`) gives `obj` ≈ 3.8 s and `jac` ≈ 15.2 s.
- **Newton-CG.** 4 `obj` + 4 `jac` ≈ 76 s, which leaves the HVPs:

  | job | nhvp | seconds in HVPs (incl. setup) | per HVP | per HVP / jac | share of the run |
  |---|---|---|---|---|---|
  | ox-0p086034__final | 7 | 404 | 57.8 | 3.8 | 84% |
  | ox-0p0929533__final | 7 | 730 | 104.2 | 6.8 | 91% |
  | ox-0p106037__final | 6 | 340 | 56.7 | 3.7 | 82% |

Conclusions so far:

- **An HVP costs about 3.7-3.8 gradients, not the 2 of the solve-count model.** The extra 1.7-1.8 gradients per HVP come from the one-off setup (amortized over 6-7 HVPs), from more expensive second-order forms, or from overhead outside the solves. Telling these apart is the first job of the profiler: `hessp` setup, the `vs_forward` stage costs, and `other`.
- **The middle job's HVPs took twice as long** as the others' for the same counts. This is not explained by the counts, so it is probably machine load or a cold cache in that job. Reprofile that job to rule out a real effect.
- **HVPs are 82-91% of the Newton-CG time.** Newton-CG uses only 1.2-1.4 HVPs per iteration, because there are only 2 parameters, so plan item D1 (fewer HVPs per iteration) cannot help here. The per-HVP cost is what matters.
- **Break-even.** Newton-CG saves 3 `obj` + `jac` pairs (about 57 s) against L-BFGS-B, but spends 7 HVPs. It only wins on time for this problem if an HVP costs less than about 8 s, about half a gradient. Even with full stage storage (A2, about 2S against 4S for today's gradient), that is out of reach. On small, well-conditioned problems L-BFGS-B will stay faster. The HVP path pays off where it saves many iterations, for example `ic-field` or stiff coefficient problems, and A1-A3 decide how many.

### 2.3 Profiler measurements

**Run:** `profile_hvp.sh 4 optimize_ic_params_newton_cg.cfg --job ox-0p086034__final --repeats 2`. That is `mtswe.cfg` on 25×25, 16 steps, LieSplit RK4×2 / Euler / SSPRK43×2 / Euler, on 4 ranks, without the flamegraph. The full output is in `runs/profiles/doublevortex-params-newton-cg/ox-0p086034__final/`.

| quantity | value |
|---|---|
| obj / jac / hessp steady seconds | 2.60 / 15.43 / 46.26 |
| obj / jac / hessp setup seconds (first - steady) | 6.9 / 12.3 / 32.2 |
| of the hessp setup: construction (`setup hvp` + `setup second order`) | 10.7 s (the remaining 21.5 s is compilation) |
| jac/obj, hessp/jac (model 4, 2) | 5.93, 3.00 |
| stage passes per substep | exactly the model: jac 3 forward + 1 adjoint; hessp 3 forward + 3 tangent + 1 adjoint + 1 soa |
| hessp breakdown | hessian increments 40.3%, tangent stages 13.1%, forward stages 12.4%, gradient increments 12.3%, soa stages 4.6%, adjoint stages 4.3%, ic map 7.0%, increment solves 3.6%, other 1.4%, update 0.9% |
| jac breakdown | forward stages 37.5%, gradient increments 31.5%, adjoint stages 13.1%, ic map 8.1%, lambda increment solve 4.6%, other 3.5% |
| tangent / adjoint / soa pass vs a forward pass | RK4(model) 1.02 / 0.98 / 1.06; hyperviscosity 0.99 / 0.99 / 1.01; SSPRK43(dg1limiter) 1.09 / 1.10 / 1.20; threewayphysics 1.02 / 1.03 / 1.01 |
| estimated vs recorded `optimize_seconds` | 428 s vs 480.5 s (89% explained; 83% of the estimate is HVPs) |

**Findings:**

1. **The increment assemblies are the bottleneck, not the solves.**
   - In steady state, `hessian increments` (18.5 s) and `gradient increments` (5.6 s) take 52% of an HVP. `gradient increments` also takes 31.5% of a gradient.
   - Per call in the RK4 dynamics substep, `assemble(L_delta_lambda_dot)` takes 0.46 s and `assemble(L_delta_lambda)` 0.11 s. A whole RK4 stage pass, four stage solves including their right-hand-side assemblies, takes 0.025 s.
   - `deltalambdasolver` already assembles the same form `L_delta_lambda` as its right-hand side, in 7.5 ms (`lambda increment solve`). The difference is that the solver keeps a persistent assembler, while the bare `assemble(L)` in `take_adjoint_step` / `take_second_order_adjoint_step` redoes the per-call symbolic work. That work scales with the size of the form, which is why the second-derivative form is 4× slower than the first-derivative one.
   - This is the "THIS IS VERY SLOW, LIKELY DUE TO SET UP COSTS" comment in `take_adjoint_step`, now quantified.
2. **The second-order forms are not more expensive to solve.** A tangent, adjoint or SOA stage pass costs within 0-20% of a forward pass in every sub-integrator, so C3 is not needed.
3. **Python/UFL overhead between solves is small:** `other` + `update` is about 2%, so B3 has low priority.
4. **The cost model's solve counts are right.** Once item 1 is fixed, hessp/jac should come close to the model's 2.
5. **Setup is large but paid once per job.** It is about 51 s in total (obj 6.9 + jac 12.3 + hessp 32.2), of which 10.7 s is form and solver construction and the rest compilation. That is about 10% of the Newton-CG run, and only the hessp part (32 s) falls inside `optimize_seconds`. C2 (reuse the model across jobs) would remove the construction part after the first job.
6. **The initial-condition map costs 7% of an HVP (3.3 s).** `ic map tangents` runs twice per HVP (in `tangent_action` and `backward`), `ic map second tangents` once, and the tangents do not depend on the direction. Caching the tangents per parameter point saves about 2.4 s per HVP and 0.8 s per gradient.

**Expected effect of fixing item 1.** If the increment assemblies drop to the cost of the solver's own right-hand-side assembly:
- hessp goes from 46 s to about 23 s, and jac from 15.4 s to about 11 s (hessp/jac ≈ 2.1);
- the Newton-CG estimate goes from 428 s to about 250 s;
- L-BFGS-B also gains about 30% per gradient.

**After B0** (persistent increment assemblers; same run settings, same job):

| quantity | before B0 | after B0 |
|---|---|---|
| obj / jac / hessp steady seconds | 2.60 / 15.43 / 46.26 | 2.61 / 10.45 / 22.12 |
| jac/obj, hessp/jac (model 4, 2) | 5.93, 3.00 | 4.01, 2.12 |
| increment assemblies in hessp (gradient + hessian) | 24.1 s (52%) | 0.70 s (3%) |
| increment assemblies in jac | 4.77 s (31%) | 0.22 s (2%) |
| hessp setup (first - steady) | 32.2 s | 34.2 s (includes the validation assemblies) |
| Newton-CG run estimate | 428 s | 241 s |

- **The gain matches the prediction:** HVPs are 2.1× faster and gradients 1.5× faster. The measured ratios now agree with the solve-count model, and the profile is dominated by stage solves.
- **New hessp profile:**
  - stage passes 70%: forward 24.9%, tangent 26.4%, soa 9.6%, adjoint 8.7%;
  - the initial-condition map 14.9% (3.3 s);
  - increment solves 7.3%, other + update 5.0%, increments 3.2%.
- **The step 19 baseline's `optimize_seconds` predates B0.** Rerun `optimizer_baseline.sh --check`, then `--save`.
- **L-BFGS-B is still faster on this problem:** about 7 × (2.6 + 10.4) ≈ 91 s, against about 241 s for Newton-CG. See section 2.2.

**After A3** (stage reuse within a LieSplit step; same run settings and job, `--repeats 3`):

| quantity | after B0 | after A3 |
|---|---|---|
| obj / jac / hessp steady seconds | 2.61 / 10.45 / 22.12 | 2.79 / 9.29 / 20.45 |
| jac/obj, hessp/jac (model 3, 2 with reuse) | 4.01, 2.12 | 3.34, 2.20 |
| stage passes per substep | jac 3 forward; hessp 3 forward + 3 tangent | jac 2 forward; hessp 2 forward + 2 tangent (exactly the model) |
| hessp forward / tangent stage seconds | 5.51 / 5.84 | 4.21 / 4.35 |
| hessp seconds per pass: forward / tangent / adjoint / soa | 1.84 / 1.95 / 1.92 / 2.12 | 2.11 / 2.17 / 2.21 / 2.34 |
| `restore stages` (jac, hessp) | - | 0.11 s, 0.22 s (about 1%) |
| hessp setup (first - steady) | 34.2 s | 35.5 s |

- **The passes are gone as intended,** and restoring costs about 1%. The measured stage-pass counts match the new model exactly.
- **Compared with the previous run, the gain looked smaller than predicted:** jac -11% and hessp -7.5%, against about 18% and 17%. In that run every kind of stage pass, and `obj`, was 7-15% slower, including code A3 does not touch.
- **A/B in one session** (`--repeats 2`, the default run and then `--recompute-stages`) confirms the full gain:

| quantity | recompute (`reuse_stages: false`) | reuse (default) | change |
|---|---|---|---|
| obj steady seconds | 2.79 | 2.71 | (no A3 change) |
| jac steady seconds | 11.74 | 9.24 | -21% |
| hessp steady seconds | 23.80 | 19.12 | -20% |
| jac/obj, hessp/jac | 4.20, 2.03 (model 4, 2) | 3.41, 2.07 (model 3, 2) | |
| jac forward stages | 6.48 s | 3.94 s | -2.53 s |
| hessp forward / tangent stages | 6.06 / 6.38 s | 3.79 / 3.94 s | -4.71 s |
| hessp adjoint / soa stages | 2.09 / 2.27 s | 1.99 / 2.12 s | no cache penalty |
| `restore stages` (jac, hessp) | - | 0.11 / 0.22 s | |

  The saving is exactly one forward pass per gradient, and one forward plus one tangent pass per HVP, minus the restores. The adjoint and SOA solves do not get slower when they no longer follow a stage solve, so there is no cache effect. Against the after-B0 run the totals are HVP 22.1 → 19.1 s and gradient 10.45 → 9.24 s.
- **The initial-condition map is now the largest single item outside the stage solves.** It takes 3.4 s (18%) of an HVP and 1.35 s (15%) of a gradient, including `ic map forward`.

**Next items, by size (after A3):**
1. **Caching the initial-condition map** (forward state, tangents and second tangents, per parameter point). **Implemented:**
   - `ParametricInitialCondition` keeps the three for the most recent point and drops them when the parameters change (`cache=True` by default; `clear_cache()`).
   - Expected: an HVP at the point of the last `jac` projects nothing, about 3.4 s (18%) of a 19.1 s HVP. A gradient only saves the forward projection when `obj` ran at the same point (0.4 s); its tangents are new at every point.
   - The profiler clears the cache before every `obj` and `jac` call, so those are timed as at a new point. The HVPs reuse what `jac` computed, as in Newton-CG; the first HVP projects the second tangents.
   - Tests: `tests/unit/test_ic_map_cache.py`.
   - **Re-profiled** (`--repeats 2`): steady hessp 19.12 → 16.84 s (-12%). The steady HVPs no longer show any `ic map` region. `hessp-1` only projects the second tangents (1.77 s), because `jac` left the state and tangents in the cache.
   - The cache removed the full 3.4 s. Without the ic map, the previous run's HVP was 15.72 s. The other 1.1 s is a slower machine state: every kind of stage pass was 5-10% slower, as was `obj` (2.71 → 2.81 s) and `jac` (9.24 → 9.79 s), whose ic map work is unchanged by design.
   - hessp/jac is now 1.72, below the model's 2, because `jac` still pays the projections (1.3 s) and the HVPs do not.
   - New hessp profile: stage solves 76% (tangent 25.7%, forward 23.9%, soa 13.8%, adjoint 12.8%), increment solves 10.4%, other + update 7.8%, increments 4.1%, restores 1.3%.
2. **A1/A2:** the remaining gains, behind a configuration option.

**After A1/A2** (`profile_hvp.sh 4 ... --repeats 2 --checkpoint stages`, steady = mean of the calls after the first):

| | after ic cache (`states`) | `stages` | change |
|---|---|---|---|
| steady obj | 2.81 s | 2.63 s | machine state |
| steady jac | 9.79 s | 7.90 s (jac-3 7.31 s) | -19% (-25%) |
| steady hessp | 16.84 s | 7.50 s | -55% |
| stage passes jac / hessp | 2F+1A / 2F+2T+1A+1SOA | 1F+1A / 1T+1SOA | match the model |

- **An HVP at the gradient point now costs about one gradient** (hessp/jac 0.95). Since the original 46 s it is 6.1× faster; the gradient is 1.9-2.1× faster than the original 15.4 s.
- **jac/obj is 3.0 against the model's 2.0**, because the model only counts stage passes. On top of 1.9 s of forward and 2.0 s of adjoint stages, `jac` pays the ic map tangents (0.96 s), the λ increment solves (0.71 s), stores and restores (0.47 s) and the gradient increments (0.23 s).
- **The second-order forms cost 1.0-1.3 forward passes.** Per pass, against the steady obj: tangent RK4 1.04, hyperviscosity 1.22, dg1limiter 1.16, threewayphysics 1.13; SOA 1.19 / 1.16 / 1.33 / 1.23. The form size is not a problem (C3 stays unneeded).
- **jac-2 is 1.2 s slower than jac-3.** All of it is in the RK4 forward stages (1.42 vs 0.80 s), so it is a one-off, probably the first allocation of the stage store. 7.3 s is the representative gradient.
- **The first calls are now a large share of a short run.** One-off setup: hessp 36.7 s (11.8 s of it form and solver construction), jac 13.2 s, obj 7.5 s. The profiler's estimate for the Newton-CG job (nit 5, nhvp 7) is 131 s, and 37 s of it is the HVP setup. The recorded 253.8 s is the old `states` run.
- Steady hessp by region: SOA stages 2.37 s (32%), tangent stages 2.09 s (28%), λ̇ increment solve 0.99 s (13%), other 0.57 s (8%), restores 0.53 s (7%), Hessian increments 0.49 s (7%), update 0.20 s, stores 0.16 s.
- The profiler's per-stage-pass cost table printed "-": a checkpointed HVP has no forward stages to compare with. It now falls back to the steady obj's forward pass of the same sub-integrator, marked `*`.

**Next items, by size (after A1/A2):**
1. **The one-off setup** (37 s per process for the HVP, 13 s for the gradient), now about 30% of a Newton-CG job:
   - Check first whether the compilation part (about 25 s) is a disk-cache miss: profile twice in a row and compare `hessp-1`.
   - Then consider C2 (reuse the model across jobs), and cheaper construction in `_setup_second_order` (11.8 s).
2. **The λ and λ̇ increment solves.** These are 0.99 s (13%) of an HVP and 0.71 s (10%) of a gradient. One RK4 λ̇ solve takes about 14 ms, while one RK4 tangent stage solve takes about 6.5 ms. **Implemented:**
   - The cause: each substep assembled the large adjoint right-hand side twice. The `LinearVariationalSolver` assembled it inside its residual, and `_assemble_increment` assembled it again for the returned dual increment.
   - `dimswe/assembly.py` `IncrementSolver` now zeroes the unknown, so the cancelling M·δ terms of `L = action(M, δ) - residual` are exactly zero. It assembles `L` once with the persistent assembler; that result is also the returned increment. It then solves M δ = r with a `MassSolver`, whose mass matrix is assembled and factored once and shared by λ and λ̇ (same expanded form signature).
   - As with B0, the first two uses of each increment solver are checked against the old solver (to `VALIDATION_RTOL`, on all ranks); on a mismatch or error it falls back to the old solver for good.
   - `GeneralRK._solve_increment`. The `lambda increment solve` / `lambda_dot increment solve` regions now include the single right-hand side assembly, which used to be under `gradient increments` / `hessian increments`.
   - Tests: `tests/unit/test_increment_solver.py` covers the strategy logic with fakes, and on `tswe_small` checks that each sub-integrator shares one mass solver and matches the old solver to 1e-10.
   - Expected: about half of the solve regions, roughly -0.4 to -0.5 s per HVP and -0.3 s per gradient. Measured: -0.3 s per HVP and -0.05 s per gradient (section 2.3, after the increment solves).
3. **Stores and restores.** These are 0.69 s (9%) of an HVP and 0.47 s (6%) of a gradient, for about 3 ms per call. **Implemented:**
   - They copied with `Function.assign`, about 0.3 ms per Function, mostly Python-side expression machinery.
   - `dimswe/numpy_helpers.py` `copy_function_values` copies each component's owned values with numpy (`data_ro` → `data_wo`, which marks the halos for update, as `assign` does). It is used by the `GeneralRK` stage store and restore and by the LieSplit substep store and restore.
   - Covered by the existing checkpoint and stage-reuse equality tests. Measured: 0.69 → 0.10 s per HVP and 0.47 → 0.05 s per gradient.
4. **Hessian increments, update and other** (1.3 s together): small individually; revisit after 1-3.

**After the increment solves and stage copies** (same job, `--repeats 2`, default `stages`):

| | after A1/A2 | now | change |
|---|---|---|---|
| steady obj | 2.63 s | 2.81 s | machine state (`ic map forward` 0.37 → 0.53 s) |
| steady jac | 7.31 s (jac-3) | 6.64-6.78 s | -9% |
| steady hessp | 7.50 s | 6.19-6.36 s | -17% |
| stores + restores, jac / hessp | 0.47 / 0.69 s | 0.05 / 0.10 s | as expected |
| λ increment (solve + its assembly), jac | 0.94 s | 0.90 s | -0.05 s, less than expected |
| λ̇ increment (solve + its assembly), hessp | 1.44 s | 1.15 s | -0.3 s |

- **Since the original code:** HVP 46 → 6.2 s (7.4×), gradient 15.4 → 6.7 s (2.3×). The recorded Newton-CG run took 127.0 s, against 253.8 s in the old baseline; the profiler's estimate is 118.9 s.
- **The stage copies (item 3) did what was expected.** They removed 0.6 s per HVP and 0.4 s per gradient.
- **The increment solves (item 2) saved less than expected,** most clearly for the gradient.
  - One increment still costs about one stage solve. RK4: 9.6 ms (λ) and 15.8 ms (λ̇) per call. Euler sub-integrators: 8-9 ms, against 6-10 ms for their single SOA stage solve.
  - So the remaining cost is the one assembly of Σ_i J_iᵀ μ_i (and its second-order counterpart) plus the factored mass solve; the second assembly the old solver did was cheap.
  - The profiler now prints the increment assembler and solver strategies (`solver_strategies`), to confirm that no sub-integrator fell back to the old solver. If one did, its solve region is the old cost plus the new assembly.
- **Steady HVP by region:** SOA stages 2.18 s (35%), tangent stages 1.99 s (32%), λ̇ increment 1.15 s (19%), other 0.53 s (9%), update 0.20 s, restores 0.10 s. Steady gradient: adjoint stages 1.97 s (30%), forward stages 1.92 s (29%), λ increment 0.90 s (13%), ic map tangents 0.84 s (13%), ic map forward 0.40 s, other 0.41 s.
- **The first calls are now about 60 s per process:** hessp 37.3 s, jac 16.2 s, obj 7.1 s. That is about half of the 119-127 s Newton-CG run.

**Next items, by size (after the increment solves and stage copies):**
1. **The one-off setup** (about 60 s per process). First check whether the roughly 25 s of compilation is a disk-cache miss: profile twice in a row and compare the first calls. Then C2, and the 11.3 s of form construction in `_setup_second_order`. **Analysed and partly implemented (C1), see "First-call setup" below.**
2. **The stage solves themselves** (67% of an HVP, 59% of a gradient). The pass counts are at the minimum for this formulation, and the second-order passes cost 0.94-1.20 forward passes. Further gains need cheaper kernels: C3, the quadrature degree of the forcing terms, or `lump_mass` for the x-space solves.
3. **The λ and λ̇ increments** (13-19%). Removing the remaining assembly needs a reformulation:
   - Solve for g_j = M⁻¹ J_jᵀ μ_j instead of μ_j. Then μ_i = dt b_i λ + dt Σ_j A_ji g_j, and δλ = Σ_j g_j is a vector sum with no extra form.
   - This is a large change to the first-order adjoint (aux terms included) and to the directionally differentiated SOA forms. Only worth it after items 1 and 2.

**First-call setup** (the 60 s above: obj 6.2 s, jac 15.7 s, hessp 36.7 s, first call minus steady):

- **Where it goes, by region.**
  - hessp: `setup second order` 10.9 s; first tangent stages +6.6 s, first SOA stages +7.6 s and first λ̇ increments +9.1 s over steady.
  - jac: first adjoint stages +6.7 s and first λ increments +6.9 s.
  - obj: first forward stages +6.0 s.
  - Per solver that is about 0.5-0.75 s on first use. RK4's single λ̇ increment alone adds 5.1 s: the old solver it was validated against built its own right-hand side kernel, matrix and factorization.
- **Where it goes, by PETSc event.** The `--flamegraph` log of the Nov 6 run (before B0, same solver structure) splits the first calls (obj + jac + hessp, first minus steady):
  - LU factorizations (`MatLUFactorSym` + `MatLUFactorNum`, MUMPS) about 16 s. Each of the ~50 stage and increment `LinearVariationalSolver`s factored its own copy of the same mass matrix.
  - Kernel compilation and loading about 22 s (`compile_form` 11.6 s, `pyop2.compilation.load` 6.7 s, wrapper code generation 3.7 s). Only this part can be a disk-cache question.
  - UFL construction about 8 s in hessp (`action` 3.1 s, `adjoint` 1.3 s, `NonlinearVariationalSolver.__init__` 2.2 s, plus `setup second order` itself). `create_linear_solver_from_residual` builds `L = action(a, var) - residual`, and `LinearVariationalProblem` builds `action(a, var) - L` again. Each `action` expands the whole form, which for the directional-derivative forms is very large.
- **Implemented: C1 with cheap construction** (`dimswe/assembly.py` `AffineSolver`, `MassSolver`, `shared_mass_solver`; `GeneralRK._affine_solver`; docs/hessian_vector_products.md section 26):
  - every stage, adjoint, tangent, SOA and increment equation is solved as an affine residual: zero the unknown, assemble the residual once, solve with a mass matrix factored once per operator and shared by all sub-integrators of the model (one x-space factorization for the whole LieSplit);
  - no `action` expansions and no `LinearVariationalSolver` construction; `L_delta_grad` / `L_delta_grad_dot` are `-residual`;
  - `directional_derivative` collects the form's coefficients once instead of once per pair;
  - checked on the first two informative solves by re-assembling the residual at the solution (one extra assembly, no extra kernels or factorizations); a failure falls back to the old solver, built only then. The old λ/λ̇ validation needed the old solver up front.
  - Iterative solver parameters keep the old solvers.
  - Expected: most of the ~16 s of factorizations, the ~8 s of UFL construction, and the old λ/λ̇ validation solvers, so roughly 25-30 s of the 60 s. A small steady-state gain is also expected from dropping the SNES residual evaluation per stage solve.
  - The new residual forms are new kernels. **Profile twice:** the first run after the change compiles them, and the second shows the steady setup and answers the disk-cache question.
- **Left:** compilation (one kernel per residual form), and C2 for jobs after the first in one process.

**Next items, by size (after B0, kept for reference):**
1. **A3** (keep the stage values within a step): removes one of the three forward and one of the three tangent passes, about 3.8 s (17%) of an HVP and 1.8 s (18%) of a gradient.
2. **Caching the initial-condition map** (tangents and second tangents, per parameter point): about 2.8-3.3 s (13-15%) of an HVP and 0.85 s (8%) of a gradient.
3. **A1/A2:** the remaining gains, behind a configuration option.

## 3. Changes, in proposed order

### A. Reuse direction-independent work at a fixed point (largest expected gain)

**A1 + A2. Implemented together as `optimizer.checkpoint: stages`** (`Lagrangian_ODEConstrainedOptimization(..., checkpoint='stages')`; the default `states` is the previous path):
- **`jac` stores.** `compute_states(store_prefix=block)` stores every step's forward stages and substep states. The adjoint sweep passes `checkpoint_slot=(block, n)`, so it restores them instead of recomputing the step (a gradient drops from 3S to 2S), and stores the stage adjoints μ_i, ν_i and each substep's λ.
- **Key.** `jac` remembers the point (coefficients and initial states, compared exactly and reduced over the ranks) and its gradient.
- **`hessp` at that point.**
  - The tangent pass restores the forward stages and only solves the tangent stages, storing those too.
  - Each reverse step restores the states, tangents, forward and tangent stages, stage adjoints and λ. It only solves the SOA stages and the λ̇ increment, so an HVP is 2S instead of 6S.
  - The first-order increments are skipped and the gradient comes from `jac`.
  - At any other point the HVP recomputes as before.
- **Timestepper API.** `GeneralRK` takes `store_slot` / `stages_slot` / `adjoint_slot` / `checkpoint_slot`, and `LieSplittingIntegrator` takes `store_slot` / `checkpoint_slot`. Stores are allocated once per slot and overwritten by every later `jac`.
- **Memory.** Per step and substep: forward stages, stage adjoints and tangent stages (3 × nstages × (x + aux)), plus the substep state and tangent.
- **Cost model.** jac 1 forward + 1 adjoint pass, hessp 1 tangent + 1 SOA (`COST_MODEL_STAGE_PASSES_CHECKPOINT`). `profile_hvp.py --checkpoint stages` profiles it.
- **Tests.** `tests/unit/test_stage_checkpoint.py` checks that gradients and HVPs (LieSplit and single RK4, coefficient + initial-state directions, and `ic-params`) match `states` to 1e-10. It also checks the fallback away from the `jac` point, that `obj` elsewhere is harmless, and the stage-pass counts.
- **Profiled** (section 2.3, after A1/A2): steady hessp 16.84 → 7.50 s (-55%), jac 9.79 → 7.90 s (7.31 s without a one-off). The stage passes match the model.
- **Default.** The tests, the `states` and `stages` baseline checks, the end-to-end comparison and the memory check all passed. `stages` is now the default wherever the timestepper supports it: `checkpoint=None`, resolved by `dimswe.optimize.resolve_checkpoint`; `optimizer.checkpoint: null` in the workflows. `states` stays available for memory-bound windows.
- **Still to do:** save a new baseline with the faster times (`optimizer_baseline.sh 4 --save`).

The original descriptions follow.

**A1. Cache the forward trajectory and first-order adjoint per point.** Key the cache on the exact point, as the `ReducedProblem` fun/jac memo already does. For coefficient directions, also key it on the coefficient values.
- **Store:** `jac` already computes and stores x_n. It would also store λ_n for every step, at the cost of one more state per step.
- **Use:** an HVP at a cached point skips the forward pass and gets λ_{n+1} from the cache instead of propagating it. The first-order stage adjoints μ_i are still needed as inputs to the SOA stages, so without A2 they are re-solved: S per step.
- **Saves:** the forward pass (S) and the first-order increment assemblies, and the gradient is no longer recomputed.
- **Requires:** an HVP entry point that takes "states, adjoints from cache" and returns only the Hessian part (`_hvp_sweep(..., need_gradient=False)`).

**A2. Store stage values for the whole trajectory (checkpoint level 2).** This is the "enriching checkpointing of stage values" item of TODO-hvp.md step 20. Store the forward stages F_i, W_i of every substep of every step during the forward pass, and the adjoint stages μ_i, ν_i during `jac`.
- **Steady-state HVP cost** at a cached point becomes only the tangent stages (S) plus the SOA stages (S): about 2S instead of 8S, a ~4× reduction in solves.
- **`jac`** drops from 4S to 2S if `obj` or the first-order forward stores stages too.
- **Memory:** nsteps × nsubsteps × nstages states per stored quantity. That's small for the workflow windows, but it must be optional (for example `checkpoint: states | stages`), with the current recompute path as the default until it is validated.

**A3. Without full stage storage, keep the stages of one step. Implemented:**
- `GeneralRK.store_stages(slot, tangents)` and `stages_slot` arguments to `take_adjoint_step` / `take_second_order_adjoint_step`.
- `LieSplittingIntegrator` stores each substep's stages (and tangent stages) while recomputing the substep states, and restores them in the substep adjoint steps. This is on by default; `timestepping.reuse_stages: false` restores the old recompute path.
- The cost model is now jac 2 forward + 1 adjoint pass per substep, and hessp 2 forward + 2 tangent + 1 adjoint + 1 SOA (`COST_MODEL_STAGE_PASSES` in `workflows/profile_hvp.py`).
- Tests: `tests/unit/test_stage_reuse.py` checks that reuse matches recompute to 1e-10, and the stage-pass counts both ways.
- Re-profiled (section 2.3): the stage-pass counts match the model, and an A/B run in one session (`profile_hvp.py --recompute-stages`) gives jac 11.74 → 9.24 s (-21%) and hessp 23.80 → 19.12 s (-20%). Still to do: the baseline check.

The original description follows.

 In `LieSplittingIntegrator`, the substep states are recomputed forward and then traversed in reverse within the same step. Keeping each substep's stage values (only nsubsteps × nstages states, for the current step) lets the `GeneralRK` SOA and adjoint steps skip re-solving their forward (and tangent) stages.
- **Saves:** S per step for `jac` (4S → 3S) and 2S for `hessp` (8S → 6S).
- **Memory:** negligible, and it is independent of A1/A2. This is a good first change: it is local to `timestepping.py` and can be checked with the existing step-level Taylor tests.

### B. Cheaper per-substep work

**B0. Assemble the increments with persistent assemblers (first change, from section 2.3). Implemented:** `dimswe/assembly.py` `PersistentAssembler`, used by `GeneralRK._assemble_increment` for all four increment forms. It tries `get_assembler`, then `create_assembly_callable`, then `assemble(tensor=)`, validates each against a plain `assemble()` on its first two uses, and falls back to plain assembly if none agrees. Tests: `tests/unit/test_persistent_assembly.py`. Re-profiled: HVP 46 → 22 s, gradient 15.4 → 10.4 s (section 2.3). Still to do: `optimizer_baseline.sh --check`, then `--save`. The original description follows.

 Replace the bare `assemble(self.L_delta_lambda)`, `assemble(self.L_delta_grad)`, `assemble(self.L_delta_lambda_dot)` and `assemble(self.L_delta_grad_dot)` calls in `GeneralRK.take_adjoint_step` and `take_second_order_adjoint_step`. Two options:
- **Persistent assemblers.** Build one assembler per form once, the way the stage solvers hold theirs, with preallocated Cofunction targets (`get_assembler(form).assemble(tensor=buffer)`, or the equivalent for the installed Firedrake version).
- **Skip the reassembly for the λ increments.** The right-hand side the solver assembled equals M δλ. With the direct solve this holds to round-off, so the dual increment can be taken as the action of a preassembled mass matrix on `delta_lambda_var`. `L_delta_grad` and `L_delta_grad_dot` have no solver and need the persistent assembler.

Check with the step 10-12 Taylor and repeatability tests, `optimizer_baseline.sh --check` (unchanged counts), and a new `profile_hvp.sh` run. Expected: hessp about 46 s → 23 s, jac about 15 s → 11 s. This subsumes most of B2. B1 does not apply to `ic-params`, whose HVP needs the gradient for the second-tangent term.

**B1. Skip unneeded first-order increments.** In an HVP-only sweep, `L_delta_grad` and `L_delta_lambda` are only needed for the gradient output, not for propagating λ, which uses `deltalambdasolver`. Skip their assembly when `need_gradient=False`. This saves 2 of the 4 assemblies per substep.

**B2. Preallocate assembly targets and accumulate in Firedrake.**
- Use `assemble(form, tensor=buffer)`, or a stored assembler, for `L_delta_lambda`, `L_delta_grad`, `L_delta_lambda_dot` and `L_delta_grad_dot`.
- Accumulate the dual increments in Cofunctions, and convert to numpy once per sweep instead of once per substep.
- This also applies to `jac`.

**B3. Build update expressions once.** Construct the update expressions (`wn + dt * sum_i b_i Fdot_i` and the forward update) once, in `_setup_second_order`, and only `assign` them per step.

**B4. Simplify the stage right-hand sides.** `create_linear_solver_from_residual` uses `L = action(a, var) - residual`, which assembles M·var only for it to cancel. For residuals that are linear in `var` (every stage residual here), use `L = -replace(residual, {var: 0})`. This removes one mass-matrix action per stage solve, in the first-order solvers too.

### C. Setup cost

**C1. Share one factorization per space.** **Implemented** (section 2.3, first-call setup): `AffineSolver` and `shared_mass_solver` in `dimswe/assembly.py`, one factorization per operator for the whole model. Every stage solver (forward, aux, adjoint, tangent, SOA, Δλ, Δλ̇) solves with the same mass matrix of its space. Use one `LinearSolver` per space (x and aux) with a preassembled matrix and its factorization, and solve with different right-hand sides. That gives 2 factorizations per sub-integrator instead of up to 8 × nstages + 2. With `lump_mass: true` the x-space solve is a pointwise division and needs no factorization at all.

**C2. Reuse the model across jobs.** Build the model and timestepper once per (truth run, spec) in `optimize_params`, instead of once per job, so `_setup_second_order` and the solvers are built once per process. Jobs of the same truth run differ only in the objective.

**C3. Check form complexity.** Inspect the expanded SOA forms of `threewayphysics` and `dg1limiter`: number of integrals, and kernel compile and run time. If the conditionals produce large kernels, assemble the forcing-term SOA contributions with simpler forms, or exclude terms whose second derivatives vanish (see `directional_derivative`, which already drops forms that don't depend on a pair).

### D. Optimizer-level

**D1. Fewer HVPs per iteration.** Compare Newton-CG with `trust-krylov` and `trust-ncg` for the unbounded targets, using the step 19 tooling. Their inner iterations can stop earlier. Expose Newton-CG's inner CG tolerance if scipy's default is too tight.

**D2. Gauss-Newton is not worth it here.** A Gauss-Newton HVP drops the second-order adjoint terms but still needs a tangent-linear and an adjoint-of-tangent sweep, so its cost structure is the same. It is not proposed.

### E. Broader timestepper support

DIRK and implicit RK are not implemented in DIMSWE even for first-order adjoints, so there is nothing to extend yet. When they are added, their HVP support should follow the same directional-derivative approach (`_setup_second_order`).

## 4. Suggested sequence

Revised after the measurements in section 2.3:

1. Measurements (section 2): done; see section 2.3.
2. **B0** (persistent increment assemblers): done. HVP 46 → 22 s, gradient 15.4 → 10.4 s.
3. **A3** (stage reuse within a step): done. Stage passes per substep 3 → 2; A/B in one session: jac -21%, hessp -20% (section 2.3).
4. **Cache the initial-condition map per parameter point:** done. HVP 19.1 → 16.8 s; the ic map is gone from the steady HVPs (section 2.3, next items).
5. **A1/A2** (cache per point, optional stage checkpointing): implemented as `optimizer.checkpoint: stages` (section 3, A1 + A2). Expected: HVP about 16.8 → 7 s and gradient about 9.8 → 7.5 s. Measured: HVP 16.8 → 7.5 s and gradient 9.8 → 7.3-7.9 s (section 2.3, after A1/A2). Validated end to end and made the default where supported.
6. **Setup and the remaining per-substep work** (section 2.3, next items after A1/A2): C1 is implemented, since the flamegraph showed about 16 s of factorizations in the first calls, together with cheaper solver construction (section 2.3, first-call setup). Next: profile twice for the disk-cache check, then C2 (reuse the model across jobs) for the per-job setup. C3 and B3 are not needed (section 2.3, findings 2 and 3); B4 is small.
7. **D1** with the comparison tooling; save a new baseline with the improved times.

For each step:
- the step 10-12 tests must pass;
- `optimizer_baseline.sh --check` must pass with unchanged counts;
- `optimize_seconds` must improve; record it with `--time-factor` and a new `--save`.
