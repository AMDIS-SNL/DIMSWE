# Baseline/ROL overnight audit

Original date: 2026-08-03 (America/Denver)

Pre-commit correction audit: 2026-08-04 (America/Denver)

Status: **PRE-COMMIT CORRECTIONS VERIFIED IN THE VALIDATED LOCAL MACOS
ENVIRONMENT**

## 0. Pre-commit correction audit

The narrow correction pass made no production-model, physics, HVP, JAX
physics, neural-network, or MPI changes.  It:

- split the environment smoke into core, optional JAX, and optional PyROL
  tests; registered `jax` and `rol`; and placed precise optional-import guards
  before their guarded imports;
- enabled and strictly checked JAX x64 before importing the Firedrake bridge;
- checked the installed `rol-python` wheel with
  `importlib.metadata.version("rol-python")`, independently of the PyROL API
  string;
- made configured and applied moist timesteps independent in the non-gating
  characterization matrix `(100,100)`, `(100,50)`, and `(50,100)`;
- selected the unique hyperviscosity `GeneralRK` child by its configured
  `terms == ["hyperviscosity"]` property and explicitly asserted uniqueness;
- kept the core module-import smoke and added a separately guarded Matplotlib
  plotting import;
- replaced the shared cache default with a unique `mkdtemp` root per pytest
  process, with `DIMSWE_TEST_CACHE_DIR` as the explicit override;
- rewrote the validated local macOS setup with `$HOME`/configurable roots and
  the complete PETSc, HDF5, JAX, and OpenMP environment contract; and
- moved this dated report to `docs/audits/2026-08-03-baseline-rol.md`, leaving
  `docs/DIMSWE_DEVELOPMENT_NOTES.md` as permanent living documentation.

On the fixed moist branch, each active conversion tendency has a configured
timescale denominator and Euler contributes the independently applied
timestep.  In symbols,

\[
\Delta Q_v = \frac{dt_{\mathrm{applied}}}{dt_{\mathrm{configured}}}
\,G(Q_v,Q_c,Q_r,S,h),
\]

The corrected outside-sandbox characterization measured:

| Configured `dt` | Applied `dt` | Integrated vapour increment | Ratio to `(100,100)` |
| ---: | ---: | ---: | ---: |
| `100` | `100` | `-1.339285714285711e13` | `1.0` |
| `100` | `50` | `-6.696428571428555e12` | `0.5` |
| `50` | `100` | `-2.678571428571422e13` | `2.0` |

These measurements confirm the implemented
`applied_dt/configured_dt` scaling on the fixed active branch.  They remain
non-gating characterization, not a physics specification.

### 2026-08-04 outside-sandbox rerun

The corrected numerical tests ran in the validated local environment without
optional JAX or PyROL skips and without assertion failures:

| Command scope | Result |
| --- | --- |
| Environment smoke | 3 passed, 0 skipped; 1.88 s |
| Permanent MTSWE baseline | 2 passed; 59.82 s |
| ROL adapter | 6 passed, 0 skipped; 71.19 s |
| MTSWE characterization | 5 passed; 68.67 s |
| Coefficient + IC gradient files | 4 passed; 158.63 s |
| Full suite | 31 passed, 1 optional plotting skip, 1 pre-existing xfail, 12200 warnings; 469.66 s |

The warnings were the already documented NumPy shape deprecation, DQ-element,
quadrature-metadata, high estimated quadrature degree, and unused PETSc option
messages.  They did not change pass/fail results.

The non-PETSc checks also completed successfully:

- `py_compile` for every corrected Python file;
- legacy NumPy/SciPy ODE suite: nine passed and one pre-existing xfail;
- isolated optional PyROL construction smoke: one passed, no skip;
- distribution/API versions: `2025.9.10.dev1712` / `0.1.0`;
- AST guard ordering: guard line 10 precedes PyROL line 15 and
  `dimswe.rol_adapter` line 26;
- simulated missing JAX and PyROL imports produced the three precise requested
  skip reasons before guarded imports; no package was uninstalled;
- two independent cache probes received different process-local roots, and an
  explicit `DIMSWE_TEST_CACHE_DIR` controlled both child paths;
- the naturally absent Matplotlib produced only the plotting test's precise
  optional skip; and
- `git diff --check` passed.

The full-suite result comes from the authoritative
`/tmp/dimswe-final-verification/full-suite-final.log`.  Its one skip is the
separately guarded optional Matplotlib plotting import; JAX and PyROL ran
without skipping.  Its one xfail remains the pre-existing
`ode_adjoint/test_optimize.py::test_optimize_params_plus_ic` case.  The
2026-08-03 results below are retained as historical baseline evidence; the
table above is the authoritative post-correction verification.

## 1. Repository state

The write gate was checked before any edit:

- branch: `dev/baseline-and-rol` (matched request);
- HEAD: `d0eb61598a2cb1049628c3cc054ab9a1f3143bf6` (matched request);
- initial working tree: clean.

Final state:

- branch and HEAD are unchanged;
- the working tree contains only the files listed below;
- no commit, tag, branch operation, reset, rebase, merge, pull, fetch, clean,
  installation, download, build, network access, MPI launch, or multi-rank run
  was performed;
- production `mtswe.cfg`, existing `tests/mtswe.cfg`, and existing
  `tests/tswe.cfg` are unchanged.

Final whitespace audits were clean: `git diff --check` produced no output, and
an explicit trailing-whitespace search across every tracked/untracked changed
file found no matches.  The requested `git diff --stat` reports the five
tracked test-file edits as 15 insertions and 12 deletions; as expected, Git
does not include untracked new files in that statistic.

## 2. Files created or modified

Created:

- `dimswe/rol_adapter.py` — serial one-element PyROL Objective, normalized
  bounds, and bounded L-BFGS ParameterList;
- `docs/DEVELOPMENT_ENVIRONMENT.md` — activation, versions, imports, serial
  contract, caches, and MUMPS caveat;
- `docs/DIMSWE_CHARACTERIZATION.md` — five non-gating observations and code
  evidence;
- `docs/DIMSWE_DEVELOPMENT_NOTES.md` — living mathematics/call-path/results
  note;
- `docs/audits/2026-08-03-baseline-rol.md` — this dated audit;
- `pytest.ini` — registers the `characterization`, `jax`, and `rol` markers;
- `tests/conftest.py` — assigns unique process-local Firedrake/PyOP2 cache
  defaults with a `DIMSWE_TEST_CACHE_DIR` override;
- `tests/mtswe_small.cfg` — 2x2, order-3, one-step MTSWE test configuration;
- `tests/tswe_rol_small.cfg` — 2x2, order-3, one-step dry ROL configuration;
- `tests/test_environment.py` — import/construction smoke test without solve;
- `tests/test_mtswe_baseline.py` — permanent moist invariants, finiteness, and
  repeatability specifications;
- `tests/test_mtswe_characterization.py` — marked non-gating probes;
- `tests/test_rol_adapter.py` — packing, bounds, direct comparison, FD,
  determinism, profile, and bounded ROL solve tests.

Modified, test-only:

- `tests/test_timestepping_coeff_gradients.py` — converts physical coefficient
  points/directions to the normalized coordinates required by the existing
  reduced objective;
- `tests/test_dynamics_gradients.py` — directs the existing RK-stage probe to
  the configured hyperviscosity `GeneralRK` child rather than its Lie wrapper;
- `tests/test_import.py` — keeps the core-import smoke test to core modules;
  the optional plotting frontend genuinely requires absent matplotlib;
- `ode_adjoint/test_gradients.py` and `ode_adjoint/test_optimize.py` — remove
  unused matplotlib imports (all plotting calls were already comments).

No existing production implementation was modified.  The SciPy optimization
path and discrete adjoint remain unchanged.

## 3. Mathematics and predictions made before tests

### Moist conversion

Writing condensation, evaporation, and rain conversion as `C`, `E`, and `R`,
the signs in `ThreeWayPhysics.rhs` and `GeneralRK` imply

\[
Q_{v,t}=h(E-C),\quad Q_{c,t}=h(C-E-R),\quad Q_{r,t}=hR,
\quad S_t=h\beta_2(E-C).
\]

Predictions:

\[
\frac{d}{dt}\int(Q_v+Q_c+Q_r)\,dx=0,
\qquad
\frac{d}{dt}\int(S-\beta_2Q_v)\,dx=0.
\]

Because the moist residual contains no `h` or `v` test term, isolated physics
must leave those fields unchanged.  Tests used integrated quantities because
`S` is CG3 while the water fields are DG1, and used `qv=0.003 > qsat=0.002`
with `qc=0.001 > qprecip=0.0001` to stay off switch surfaces.

### Normalized scalar ROL coordinate

For `c0=d_c0*z`,

\[
\frac{dJ}{dz}=d_{c0}\frac{dJ}{dc_0},\qquad
z_l=c_{0,l}/d_{c0},\qquad z_u=c_{0,u}/d_{c0}.
\]

The existing reduced objective consumes normalized `[s,c0]` and its adjoint
already returns the chain-rule-scaled normalized gradient.  Therefore the
adapter prediction was: pack `[fixed_s_normalized,z]`, delegate value and
gradient, and return adjoint entry 1 unchanged.  Direct and adapter results
should agree to roundoff, and a centered finite difference should confirm the
chain rule.

### Characterization predictions

Code inspection predicted, without asserting scientific correctness:

- no limiter `post_step` call in the active Lie step;
- zero Hamiltonian-owned topography because its initializer is not called;
- cancellation when configured and applied `dt` were varied together in the
  moist Euler update;
- a nontrivial DG `Qr` flux for a resolved probe;
- finite modal damping from positive hyperviscosity on the chosen case.

## 4. Permanent MTSWE results

The isolated conversion was nontrivial (`Qv` change L2 norm
`2678571.428571422`) and produced:

| Quantity | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Integrated total water | `7.874999999999995e13` | `7.874999999999995e13` | `0.0` |
| Integrated `S-beta2*Qv` | `1.7834953499999994e17` | `1.7834953499999994e17` | `0.0` |
| `h` L2 change | — | — | `0.0` |
| `v` L2 change | — | — | `0.0` |

The complete one-global-step MTSWE map produced finite fields.  Repeating it
from the same stored input produced maximum coefficient difference `0.0`.  The
documented per-field tolerance is `64*eps*max(1,max(abs(field)))`; the largest
absolute tolerance in this case was `1.0629237529771114e-10`.

## 5. Objective profile before ROL

The profile was evaluated with the existing direct DIMSWE objective before the
adapter constructed or ran a ROL solver:

| Physical `c0` | Objective |
| ---: | ---: |
| `0.01` | `6.11181830616236e13` |
| `0.03` | `4.375917248790977e13` |
| `0.05` | `2.9293330343144117e13` |
| `0.08` | `1.301925793028656e13` |
| `0.14` | `0.0` |
| `0.20` | `1.3019257930291863e13` |
| `0.30` | `9.258138972649708e13` |

The sampled minimum is unique at the synthetic generating value `c0=0.14`,
and the adjacent points are higher.  This supports local one-dimensional
identifiability for this test; exact recovery is not made a general contract.

## 6. Direct versus adapter and finite difference

At physical `c0=0.05` (`z=0.7142857142857143`) with fixed normalized `s=1`:

| Check | Direct DIMSWE | Adapter | Difference |
| --- | ---: | ---: | ---: |
| Value | `2.9293330343144117e13` | `2.9293330343144117e13` | `0.0` |
| Normalized `c0` gradient | `-4.556740275600571e13` | `-4.556740275600571e13` | `0.0` |

Centered finite difference with normalized step `1e-3` gave
`-4.556740275508789e13`; the relative error against the discrete adjoint was
`2.0142037004271277e-11`.  Repeated adapter values and gradients were
deterministic within the 64-epsilon envelope (measured differences were zero).

## 7. ROL convergence and counts

Physical bounds `[0.01,2.0]` became normalized bounds
`[0.14285714285714285,28.57142857142857]`.  The returned normalized coordinate
was `1.9999999999999938`, so recovered physical `c0` was
`0.13999999999999957`, inside the bounds.

Recorded value history (line-search trials included):

| Evaluation | Normalized `z` | Physical `c0` | Objective |
| ---: | ---: | ---: | ---: |
| 1 | `0.2857142857142857` | `0.02` | `5.207703172114747e13` |
| 2 | `28.57142857142857` | `2.0` | `1.251150687100722e16` |
| 3 | `14.428571428571429` | `1.01` | `2.7372989798431965e15` |
| 4 | `7.357142857142857` | `0.515` | `5.085647629019031e14` |
| 5 | `3.821428571428571` | `0.2675` | `5.879008659146498e13` |
| 6 | `2.0535714285714284` | `0.14375` | `5.0856476290422424e10` |
| 7 | `0.1428571428571428` | `0.01` | `6.11181830616236e13` |
| 8 | `1.0982142857142856` | `0.076875` | `1.4410747628784803e13` |
| 9 | `1.575892857142857` | `0.1103125` | `3.1873590174900703e12` |
| 10 | `1.8147321428571428` | `0.12703125` | `6.082469881294724e11` |
| 11 | `1.9341517857142856` | `0.135390625` | `7.683654251969943e10` |
| 12 | `1.993861607142857` | `0.1395703125` | `6.677119825350659e8` |
| 13 | `1.999999999999956` | approximately `0.14` | `6.285852503145395e-13` |
| 14 | `2.0000000000000115` | approximately `0.14` | `4.558076350367286e-13` |
| 15–17 | `1.9999999999999938` | approximately `0.14` | `9.028607444196265e-18` |

The solve used six ROL iterations, 17 recorded value calls (including the
final report check), and seven gradient calls.  Gradient history ended at
normalized gradient `-0.0017989442587644103`; ROL reported
`EXITSTATUS_STEPTOL`.  Objective decrease and movement toward truth were both
strict, and all recorded trial coordinates respected the normalized bounds to
a `64*eps*max(1,|bound|)` boundary-roundoff allowance.

## 8. Original characterization results (non-gating)

All five marked procedures executed on 2026-08-03 and produced finite
observations before the independent-`dt` correction:

1. **Limiter:** active global step called `DG1LimiterTransport.post_step` zero
   times (`not-called`).
2. **Hamiltonian topography:** dynamics-owned topography L2 norm
   `2.589916345847779e-07`; Hamiltonian-owned norm `0.0`.
3. **Moist `dt`:** integrated vapour increment was
   `-1.339285714285711e13` for both coupled cases `(50,50)` and `(100,100)`;
   difference `0.0`.  The corrected characterization now varies configured
   and applied values independently as documented in Section 0.
4. **DG rain transport:** cosine `Qr` initial norm `1.0152840849994279e6`;
   isolated transport change norm `243.18594664569247` (`modified`).
5. **Hyperviscosity:** measured amplitude changed from
   `9737.450795091365` to `9737.348077384857`, amplification
   `0.9999894512733702` (`damped`).

These are observations, not production specifications.  Code evidence,
possible intended behavior, and the author decision required for each are in
`docs/DIMSWE_CHARACTERIZATION.md`.

## 9. Original 2026-08-03 verification results

The original milestone's final authoritative runs were:

| Scope | Result |
| --- | --- |
| `tests/test_environment.py` | 1 passed, 0 failed, 0 skipped |
| `tests/test_mtswe_baseline.py` | 2 passed, 0 failed, 0 skipped |
| `tests/test_rol_adapter.py` | 6 passed, 0 failed, 0 skipped |
| Characterization (`-m characterization`) | 5 passed, 0 failed, 0 skipped |
| Existing dry coefficient + IC derivative files | 4 passed, 0 failed, 0 skipped |
| Final full suite | 29 passed, 0 failed, 0 skipped, 1 pre-existing xfail; 120.37 s |

The full-suite xfail is the pre-existing
`ode_adjoint/test_optimize.py::test_optimize_params_plus_ic`.  It was not
introduced or changed into an xfail by this milestone.

Intermediate failures and their diagnoses were kept visible:

- environment smoke: the initial assertion incorrectly assumed PyROL would
  leave its ParameterList empty; Solver construction populates defaults;
- first MTSWE run: Firedrake tried to create a cache inside the read-only venv;
  test cache defaults were redirected to a permitted temporary root;
- first ROL run: the returned point and objective succeeded, but the test used
  bit-exact comparisons for boundary line-search trials; a 64-epsilon bound
  allowance was derived and used;
- first dry coefficient-derivative run: physical coefficients were passed to
  an API that expects normalized coordinates, producing the observed false
  `[10.24,0.0049]` point; the test-only coordinate map was corrected;
- first full run: two collection errors from unused matplotlib imports in
  legacy ODE tests;
- second full run: the legacy dynamics test addressed a Lie wrapper with a
  child-only API, and the core import smoke test imported an optional plotting
  frontend; both were corrected test-only;
- final focused, derivative, narrow, and full reruns all passed as reported.

## 10. Environment caveats

- Serial-only scope remains mandatory; no distributed/MPI behavior was tested.
- PETSc advertises MUMPS and mixed-precision MUMPS, but distributed MUMPS is
  unvalidated.
- Open MPI emits a TCP-listener permission warning in this restricted macOS
  runner even for the one-process Python invocations; `COMM_SELF` tests pass.
- Pytest now assigns unique process-local Firedrake/PyOP2 cache defaults; set
  `DIMSWE_TEST_CACHE_DIR` only when an explicit cache root is required.
- Matplotlib is absent.  Its plotting import is a separate precise optional
  skip and is not part of the core import contract.
- Current Firedrake/UFL emits many NumPy shape deprecation, DQ-element, and
  quadrature metadata warnings.  PETSc also reports pytest CLI flags as unused
  options.  These warnings did not change pass/fail results.
- Legacy ODE SciPy tests warn that L-BFGS-B ignores their pre-existing `hessp`.
  The new DIMSWE ROL path neither defines nor calls any Hessian method.

## 11. Unresolved scientific behavior

Author decisions remain required for:

- where, if anywhere, the DG limiter hook belongs in the active split method;
- whether the Hamiltonian topography should be initialized, aliased to the
  dynamics Function, or intentionally zero;
- whether moist relaxation time is a physical parameter or intentionally tied
  to global `dt`;
- the intended rain transport/boundary/limiter semantics;
- the intended hyperviscosity spectrum and mesh/timestep scaling.

No topography, limiter, rain transport, moist timestep, hyperviscosity,
dynamics, or timestepper behavior was fixed or altered.

## 12. Command ledger

Every terminal command invocation is recorded below.  Multi-line inspection
commands are shown compactly with the files/ranges they read; Python heredocs
are identified by their measured purpose and output.  File changes themselves
were made with `apply_patch`, not shell redirection.

### Gate and inventory

1. `git branch --show-current && git rev-parse HEAD && git status --porcelain=v1`
2. `pwd && rg --files -g 'AGENTS.md' ... && rg --files | sed -n '1,240p'`
   (stopped after no `AGENTS.md` match).
3. `rg --files | sed -n '1,280p'`
4. `sed` inspection of `dimswe/physics.py`, `timestepping.py`, `models.py`,
   `optimize.py`, `coeff_optimize.py`, and `adjoint_optimize.py`.
5. `python - <<'PY' ...` environment/API introspection: versions, PETSc/MPI,
   tiny mesh, Firedrake JAX imports, PyROL imports/signatures (ended on the
   built-in ParameterList signature query after recording requested data).
6. `sed` inspection of existing derivative/timestep tests, both existing test
   configs, and `dimswe/parameters.py`.
7. `rg -n "^(class|def) |forcing_terms|limiter|topograph|hypervisc|threeway|take_forward|initialize|rhs\\(" ...`
8. `nl -ba ... | sed ...` for models, dynamics, variables, dissipation,
   transport, and timestepping call paths.
9. `nl -ba ... | sed ...` for variables, dissipation, Hamiltonians, physics,
   and meshes.
10. `nl` inspection of initial conditions plus PyROL `help`/constructor probe.
11. First one-dimensional PyROL quadratic probe (failed because nested
    `ParameterList.sublist` returned `"Invalid sublist name"`).
12. PyROL `help(ParameterList.sublist/set/get)` and overload probe.
13. Successful one-dimensional bounded PyROL L-BFGS quadratic probe.
14. Environment variable print plus activation/PETSc/JAX/MUMPS repository
    search.
15. PETSc configuration search for MPI/MUMPS plus PyROL/JAX/PETSc runtime
    attributes (PETSc `ARCH` attribute query ended the latter probe).
16. `mkdir -p docs`

### Focused implementation and measurements

17. Initial `python -m pytest -q tests/test_environment.py` (one failed test
    assertion), with new-file inspection.
18. `python -m pytest -q tests/test_environment.py` (pass), followed by `sed`
    audit of environment doc/test.
19. `sed`/`rg`/`nl` inspections of split, initialization, variables, logger,
    coefficients, and PyROL API paths used to derive tests.
20. Initial `python -m pytest -q tests/test_mtswe_baseline.py` (two cache-path
    errors).
21. `rg -n "CACHE_DIR|cachedir|cache_dir"` across Firedrake/PyOP2/TSFC.
22. Two runs of the baseline test with explicit PyOP2 and TSFC directories
    below the former shared temporary cache root (pass).
23. Bare baseline run after `tests/conftest.py`, plus the isolated invariant
    test run (pass).
24. Characterization run with temporary JUnit XML, then `sed` of that XML.
25. Narrow cosine DG-transport characterization with temporary JUnit XML,
    then `sed` of that XML.
26. Adapter forbidden-callback `rg`, adapter `sed`, and first focused ROL run.
27. Logged ROL rerun and log inspection (five passed, one overly exact bound
    assertion failed).
28. ROL rerun with temporary JUnit XML (six passed), then XML inspection.
29. `nl` inspection of UFL helpers, optimizer, adapter, and new tests.
30. Baseline rerun with temporary JUnit XML (two passed), then XML inspection.
31. README and exact core line-range inspections used for the learning note.

### Prescribed verification and diagnosis

32. `python -m pytest -q tests/test_environment.py` — pass.
33. `python -m pytest -q tests/test_mtswe_baseline.py` — pass.
34. `python -m pytest -q tests/test_rol_adapter.py` — pass.
35. `python -m pytest -q tests/test_mtswe_characterization.py -m characterization`
    — pass.
36. Initial requested two-file dry derivative command — two coefficient
    failures, two IC passes; output was captured below the former shared
    temporary cache root and polled with `sed`, `wc`, and 15/25-second waits.
37. `ps ... | rg ...` diagnostic attempt (runner denied process listing).
38. Narrow corrected coefficient derivative run, captured/polled in a
    temporary log — two passed.
39. Final requested two-file derivative run, captured/polled in a temporary
    log — four passed.
40. First `python -m pytest -q` — two matplotlib collection errors.
41. `sed`/`rg` inspection of both legacy ODE test files confirmed matplotlib
    was unused.
42. Second full-suite run, captured/polled in a temporary log — 27 passed, one
    xfailed, two legacy test-interface failures.
43. `sed`/`rg` inspection of `tests/test_dynamics_gradients.py`,
    `tests/test_import.py`, and all `get_rhs_expr` uses.
44. Narrow dynamics/import reruns.  Because the command wrapper yielded while
    leaving the first process active, a second logged copy was inadvertently
    started; no result was accepted from overlapping execution.
45. Targeted `pkill -TERM -f 'python -m pytest -q tests/test_dynamics_gradients.py tests/test_import.py'`
    attempt (runner denied process-list access; nothing was killed).
46. Read-only `git status`, tracked diff, `git diff --check`, and
    `git diff --stat` while waiting.
47. Logged narrow process polling with `sed`, `wc`, and 20/30/45-second waits;
    authoritative completed result: two passed.
48. Final tracked `python -m pytest -q`, polled through terminal session 35883
    without overlap — 29 passed, one xfailed.
49. Final branch/HEAD/status, line counts, config line-number inspection, and
    forbidden new HVP/JAX reference search.
50. Prescribed `git diff --check`, `git status --short`, and
    `git diff --stat` final audit.
51. Explicit `rg -n '[ \\t]+$' ...` trailing-whitespace audit across every
    changed file (no matches).

Tool-level session polls (`functions.wait` / terminal `write_stdin`) only
retrieved output from already-running commands; they did not start additional
programs.  Temporary logs/XML remained below the then-permitted shared cache
root; current tests instead use unique process-local roots.

## 13. Recommended next step

The exact next action should be an author decision review of the five
characterization findings, producing an explicit statement for limiter
placement, Hamiltonian topography ownership, and moist-`dt` semantics before
any of those operators is changed.  Once those scientific/interface decisions
are recorded, begin the next planned technical milestone with the isolated
NumPy ODE HVP prototype.

No HVP callback or implementation was added, the existing empty DIMSWE
`hessp` was not exposed or called, and no JAX-physics or neural-network work
was started.
