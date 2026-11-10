# TODO: `EllipticProblem` — Hodge Laplacian for arbitrary k-forms

Goal: add a new **time-invariant** model type, `EllipticProblem`, that solves the Hodge
Laplacian `(dδ + δd) u = f` for a k-form `u` using the FEEC spaces DIMSWE already has
(`DeRhamComplex`, `get_diff_form_space`). It must cover:

| dim | complex                         | k            | spaces (k-form proxy)               |
|-----|---------------------------------|--------------|-------------------------------------|
| 1   | `CG -∂x-> DG`                   | 0, 1         | CG, DG                              |
| 2   | grad-curl `CG -grad-> RTCE -rot-> DG` | 0, 1, 2 | CG, Hcurl (RTCE), DG                |
| 2   | curl-div `CG -skewgrad-> RTCF -div-> DG` | 0, 1, 2 | CG, Hdiv (RTCF), DG             |
| 3   | `CG -grad-> NCE -curl-> NCF -div-> DG` | 0, 1, 2, 3 | CG, Hcurl (NCE), Hdiv (NCF), DG |

i.e. 12 (dim, complex, k) combinations, each with **periodic**, **essential**, **natural**
boundary conditions, or a **per-face mix** of them (including partial periodicity, e.g. periodic
in x and essential/natural in y).

---

## 1. Mathematical formulation (reference for implementation and tests)

### 1.1 Mixed weak form (Arnold–Falk–Winther)
With `V^j` the discrete space of j-forms and `𝔥^k` the (discrete) harmonic k-forms, find
`σ ∈ V^{k-1}`, `u ∈ V^k`, `p ∈ 𝔥^k` such that

```
(σ, τ) − (u, dτ)                 = −∫_{Γn} B_{k-1}(τ, g_u)        ∀ τ ∈ V^{k-1}
(dσ, v) + (du, dv) + (p, v)      =  (f, v) + ∫_{Γn} B_k(v, g_du)  ∀ v ∈ V^k
(u, q)                           =  0                              ∀ q ∈ 𝔥^k
```

- `σ = δu` is the auxiliary variable; it is absent for `k = 0` (primal problem
  `(du, dv) + (p, v) = (f, v) + boundary terms`).
- For `k = n` the `du` term vanishes (`d` of a top form is 0).
- `p` is the projection of `f` onto harmonic forms; it is absent when `𝔥^k = {0}`.
- **Essential BCs** (on `Γe`): `tr u = g_e` and `tr σ = g_σ`, enforced through `DirichletBC`
  on the corresponding sub-spaces (tangential trace for Hcurl, normal trace for Hdiv, value for
  CG, nothing for DG).
- **Natural BCs** (on `Γn`): `tr ⋆u = g_u` and `tr ⋆du = g_du`, enforced weakly through the
  boundary integrals above (homogeneous ⇒ boundary terms drop out).
- **Periodic**: handled by the mesh topology, no boundary terms or `DirichletBC`s.

### 1.2 Proxy operator table
Each derivative satisfies `(d w, z) = (w, δ z) + ∫_∂Ω B(w, z) ds`. This table defines the
`d`, `δ` (formal adjoint; used for manufactured solutions) and the boundary pairing `B`:

| operator `d` (on `w`)                  | `δ z`                       | `B(w, z)`                   |
|----------------------------------------|-----------------------------|-----------------------------|
| 1D `∂x w`                              | `−∂x z`                     | `w z n`                     |
| `grad w` (2D/3D)                       | `−div z`                    | `w (z·n)`                   |
| 2D `skewgrad w = (−w_y, w_x)`          | `−curl2D z`                 | `w (n × z)` (2D cross)      |
| 2D `curl2D w = w1_x − w0_y`            | `−skewgrad z = (z_y, −z_x)` | `(n × w) z`                 |
| 3D `curl w`                            | `curl z`                    | `(n × w)·z`                 |
| `div w` (2D/3D)                        | `−grad z`                   | `(w·n) z`                   |

(`skewgrad`/`curl2D` match the existing helpers in `dimswe/ufl_helpers.py`.)
Every sign here **must be verified by a unit test** (see §6) before use in forms.

### 1.3 What the BC types mean per case (physical interpretation, for docs and tests)
| case                         | essential (`Γe`)                 | natural (`Γn`)                       |
|------------------------------|----------------------------------|--------------------------------------|
| k = 0                        | `u = g` (Dirichlet)              | `∇u·n = g` (Neumann)                 |
| k = n, σ ∈ Hdiv / CG (1D)    | `σ·n = g` (flux)                 | `u = g` (weak Dirichlet)             |
| k = n = 2, grad-curl (σ ∈ Hcurl) | `n × σ = g`                  | `u = g` (weak)                       |
| 2D grad-curl k=1 (u ∈ Hcurl) | `n × u`, `σ = −div u`            | `u·n`, `curl2D u`                    |
| 2D curl-div k=1 (u ∈ Hdiv)   | `u·n`, `σ = −curl2D u`           | `n × u`, `div u`                     |
| 3D k=1 (u ∈ Hcurl)           | `n × u`, `σ = −div u`            | `u·n`, `n × curl u`                  |
| 3D k=2 (u ∈ Hdiv)            | `u·n`, `n × σ` (`σ = curl u`)    | `n × u`, `div u`                     |

### 1.4 Harmonic forms
For the mesh families DIMSWE builds (intervals/rectangles/boxes, each direction either periodic
or bounded with one BC type per face), Künneth gives a closed-form basis of **constant forms**
`dx^I`, `|I| = k`, where with `P` = periodic directions, `E` = directions whose two faces are
both essential, `N` = directions whose two faces are both natural:

- if any bounded direction has *different* BC types on its two faces ⇒ `𝔥^k = {0}`;
- otherwise `𝔥^k = span{ dx^I : E ⊆ I ⊆ E ∪ P, |I| = k }`, dimension `C(|P|, k − |E|)`.

Examples: k=0 all-natural ⇒ constants; k=n all-essential ⇒ constants; fully periodic ⇒
`C(n, k)` forms; mixed Dirichlet/Neumann ⇒ none.

Proxy map for `dx^I` (needed to turn index sets into UFL constants):
- k=0, k=n: `1`
- Hcurl 1-forms (2D grad-curl, 3D): `dx_i ↦ e_i`
- 2D curl-div 1-forms (Hdiv proxy, matches `skewgrad`): `dx ↦ (0, 1)`, `dy ↦ (−1, 0)`
- 3D 2-forms (Hdiv): `dy∧dz ↦ e_x`, `dz∧dx ↦ e_y`, `dx∧dy ↦ e_z`

Constant forms lie in every discrete space of the Q family and are discretely harmonic under the
above rule, so no eigen-solve is needed for these meshes. For `mesh.type: file` (general
topology) see §8 (numerical harmonic forms — deferred).

---

## 2. Design overview

`EllipticProblem` is **not** a subclass of the time-dependent `Model` (which is built around
`dynamics`/`timestepper`). Instead it implements the subset of the `Model` interface that
`run_model.py`/`Output`/the workflows actually touch, plus a `solve()` method:

```
EllipticProblem
  .mesh, .spaces                 # via get_mesh_and_spaces (reused)
  .problem                       # elliptic problem data (source f, exact solution, domain sizes)
  .config                        # parsed EllipticConfig dataclass (k, complex, BCs, solver)
  .solve()                       # assemble + solve once, fill solution Function
  get_x_var_list() -> ['sigma', 'u', 'p'] (only those present)
  get_aux_var_list()/get_coeff_list() -> []
  get_coeff_var() -> (None, {}, {}), set_coeffs() no-op   # hooks for future coefficients
  create/compute_statistics/diagnostics, get_statistics(_list), get_diagnostics(_list)
  is_time_invariant = True
```

New modules (small, composable, each function documented per AGENTS.md):

| module                               | responsibility                                                      |
|--------------------------------------|---------------------------------------------------------------------|
| `dimswe/differential_forms.py`       | k-form → space, `d`, `δ`, boundary pairing `B`, trace helpers, constant-form proxies |
| `dimswe/boundary_conditions.py`      | parse BC config, logical face registry per mesh, build `DirichletBC`s and natural boundary terms |
| `dimswe/harmonic_forms.py`           | Künneth selection rule (pure Python), UFL basis construction        |
| `dimswe/elliptic_problems.py`        | library of problem data (manufactured solutions, sources), like `initial_conditions.py` |
| `dimswe/elliptic.py`                 | `EllipticConfig`, `EllipticProblem`, form assembly, solve, errors   |
| `dimswe/plotting/plot_elliptic.py`   | plots of solution, errors, convergence                              |

---

## 3. Configuration (new cfg section)

```yaml
model:
  type: elliptic

mesh:
  type: rectangle            # line | line-periodic | rectangle | rectangle-periodic | box | box-periodic | file
  periodic_directions: [x]   # NEW, optional: partial periodicity for rectangle/box (overrides *-periodic)
  nx: 32
  ny: 32
  simplicial_cells: false

spatial-discretization:
  family: Q
  order: 2
  lump_mass: false           # elliptic: see §4.6

elliptic:
  form_degree: 1             # k
  complex: curl-div          # 2D only: grad-curl | curl-div (ignored in 1D/3D)
  boundary_conditions:       # per logical face; omitted faces of periodic directions are periodic
    left:   essential        # essential | natural
    right:  essential
    # bottom/top/front/back ...
  boundary_data: exact       # exact (derive g's from the problem's exact solution) | homogeneous
  harmonic_forms: auto       # auto (Künneth rule) | none | numerical (future)
  solver: direct             # key into overall_solver_parameters

elliptic-problem:
  name: sine                 # entry in elliptic_problems.get_elliptic_problem
  Lx: 1.0
  Ly: 1.0
  modes: [1, 2]

output:
  outfile_name: runs/elliptic-test/sim
  output_aux_vars: false
```

Logical face names: `left/right` (x), `bottom/top` (y), `back/front` (z) — mapped to Firedrake
subdomain ids by the registry in §4.2. Validation (`EllipticConfig.from_parameters`) must reject:
k out of range, BCs given on periodic faces, missing BCs on bounded faces, `complex` other than
the two allowed values in 2D, `lump_mass: true` with mixed problems (unless §4.6 decides otherwise).

---

## 4. Implementation tasks

### 4.1 Mesh / space prerequisites (`dimswe/meshes.py`)
- [ ] Add optional `mesh.periodic_directions` handling:
  - 2D: `PeriodicRectangleMesh(..., direction='x'|'y'|'both')`.
  - 3D quads: `ExtrudedMesh(base, nz, periodic=<z periodic>)` with base `RectangleMesh` or
    `PeriodicRectangleMesh(direction=...)` — covers all 8 periodicity combinations.
  - 1D: only fully periodic or fully bounded (Firedrake limitation is fine).
- [ ] Non-periodic `box` with hex cells: currently `BoxMesh(hexahedral=True)` (comment says it
  "will choke" for NCE/NCF). Build it as `ExtrudedMesh(RectangleMesh(...quadrilateral=True), nz)`
  instead, like the periodic box. Keep `simplicial_cells: true` path unchanged.
- [ ] Make the mesh builder accept the elliptic problem object in place of `initcond` (it only
  needs `Lx, Ly, Lz, xc, yc, zc`); document this duck-typed contract in the docstring.
- [ ] Record on the mesh/spaces which directions are periodic (needed by BC registry and harmonic
  forms), e.g. `spaces.periodic_directions`.
- [ ] Use the existing `get_diff_form_space(spaces, 'S', k, dim, nm1)` for k-form spaces
  (`nm1=True` ⇔ curl-div, `False` ⇔ grad-curl). In 1D it already returns CG/DG correctly; do
  **not** touch `DeRhamComplex.Hdiv = CG` in 1D (used by existing models).
- [ ] Fix pre-existing bugs we will hit: `mesh_size` uses unimported `Function`/`CellDiameter`.

### 4.2 Boundary handling (`dimswe/boundary_conditions.py`)
- [ ] `BoundaryFace` dataclass: logical name, Firedrake subdomain id(s), measure kind
  (`ds`, `ds_v`, `ds_t`, `ds_b`), DirichletBC subdomain token (`int`, `"top"`, `"bottom"`).
- [ ] `get_boundary_registry(mesh, mesh_type, periodic_directions)` returning faces for:
  `IntervalMesh` (1,2), `RectangleMesh` (1–4), `PeriodicRectangleMesh(direction=…)` (1,2 renumbered —
  **verify numbering against installed Firedrake**), `ExtrudedMesh` (base ids via `ds_v` +
  `top`/`bottom` via `ds_t`/`ds_b`), simplicial `BoxMesh` (1–6).
- [ ] `parse_boundary_conditions(cfg, registry)` → `{face: 'essential'|'natural'}` with validation.
- [ ] `build_essential_bcs(W, sub_index_sigma, sub_index_u, faces, g_sigma, g_u)` → list of
  `DirichletBC` (skip `u` for DG / k=n, skip σ for k=0).
- [ ] `build_natural_terms(tau, v, faces, g_u, g_du, k, dim, complex, n)` → UFL boundary forms
  using `B_{k-1}` and `B_k` from `differential_forms`; returns 0 for homogeneous data.
- [ ] Helper `measure_for(face)` so the forms work on both extruded and non-extruded meshes.

### 4.3 Differential-form utilities (`dimswe/differential_forms.py`)
- [ ] `form_space(spaces, k, dim, complex)` (thin wrapper over `get_diff_form_space`).
- [ ] `exterior_derivative(w, k, dim, complex)` per the §1.2 table (returns `None`/0 for k=n).
- [ ] `codifferential(z, k, dim, complex)` (formal adjoint of `d_{k-1}`; for MMS only).
- [ ] `boundary_pairing(w, z, k, dim, complex, n)` = `B_k(w, z)`.
- [ ] `hodge_laplacian_expr(u, k, dim, complex)` = `d δ u + δ d u` symbolic (for computing `f`
  from an exact solution).
- [ ] `constant_form_proxy(I, k, dim, complex)` per §1.4 proxy map.
- [ ] Possibly move `skewgrad/curl2D` usage here; keep `ufl_helpers` API unchanged.

### 4.4 Harmonic forms (`dimswe/harmonic_forms.py`)
- [ ] `harmonic_index_sets(k, dim, periodic_dirs, face_bcs)` — pure Python, implements §1.4
  (unit-testable without Firedrake).
- [ ] `harmonic_basis(k, dim, complex, index_sets, mesh)` → list of UFL constant proxies.
- [ ] Mixed-space treatment: one `R`-space dof per basis form
  (`VectorFunctionSpace(mesh, "R", 0, dim=m)` or `FunctionSpace(mesh,"R",0)` when m=1),
  giving `p = Σ p_i h_i` and the constraint rows `(u, h_i) = 0`.
- [ ] Option `harmonic_forms: none` for users who know `𝔥 = 0`; error out on `file` meshes with
  `auto` until §8 numerical harmonic forms exist.

### 4.5 Problem data library (`dimswe/elliptic_problems.py`)
Pattern mirrors `initial_conditions.py` (`get_elliptic_problem(parameters)` dispatcher, classes
with `Lx, Ly, Lz, xc, yc, zc`).
- [ ] Base class `EllipticProblemData` with `get_source(mesh, k, dim, complex)` and optional
  `get_exact(mesh, k, dim, complex)` returning dict `{u, sigma, du, f}` (σ = δu, f = Δu via §4.3).
- [ ] `SineProblem` (manufactured, products of sin/cos with configurable modes) valid for every
  (dim, k, complex); chosen so that `u ⊥ 𝔥^k` (zero mean for each constant-form component) so
  `p = 0` and errors are measured directly against `u_exact`.
- [ ] `PolynomialProblem` (low-degree, tests exactness/consistency on coarse meshes, e.g. `u`
  in the discrete space ⇒ error ≈ round-off for k=0).
- [ ] `GaussianSourceProblem` (no exact solution; `f` given, homogeneous BCs; for demos/plots).
- [ ] When harmonic forms are present and an exact solution is used, compare against
  `u_exact − P_𝔥 u_exact` (document that inhomogeneous BC data + harmonic forms require compatible
  data; MMS problems avoid this by construction).

### 4.6 Core model (`dimswe/elliptic.py`)
- [ ] `EllipticConfig` dataclass + `from_parameters(parameters, spaces)` with validation (§3).
- [ ] `EllipticProblem.__init__(parameters, logger, mesh=None)`:
  problem data → mesh/spaces → config → spaces `V^{k-1}, V^k, R^m` → `MixedFunctionSpace`
  (skip absent blocks) → var list → BC data → forms → `LinearVariationalSolver`.
- [ ] `_build_forms()`: split into `_mass_sigma`, `_coupling`, `_stiffness`, `_harmonic`,
  `_rhs`, `_natural_terms` small functions assembled per §1.1.
- [ ] `solve()`: run solver, log KSP convergence, store `solution` and `sub` dict
  (`sigma`, `u`, `p`).
- [ ] Error statistics (if exact solution available): `L2(u)`, `L2(σ)`, `L2(du)`, `L2(dσ)`,
  `|p|`, plus orthogonality check `max_i |(u_h, h_i)|`; stored in the statistics dict shape
  `Output` expects (arrays of length 1).
- [ ] Diagnostics fields: `u_exact`, `u_error`, `sigma_error`, `du` (projected/interpolated
  to the appropriate spaces) for VTK.
- [ ] Quadrature: default to exact `dx` (not GLL). Decide whether `lump_mass: true` is
  rejected or only applied to CG/DG blocks — recommend **reject** for mixed problems initially.
- [ ] Solver parameters: add `overall_solver_parameters['elliptic-direct']` to
  `parameters.py` (`preonly` + `lu` + `mumps`, `mat_mumps_icntl_14` bump). Iterative
  (Hypre AMS/ADS block preconditioners) deferred to §8.

### 4.7 Integration with the existing driver and workflows
- [ ] `models.get_model`: add `elif parameters['model']['type'] == 'elliptic': return EllipticProblem(...)`
  (while there: `AdvectionModel` and `CGTransport` are referenced but undefined — leave or flag,
  do not silently fix unrelated code).
- [ ] `run_model.py`: if `getattr(model, 'is_time_invariant', False)`, branch to a new
  `run_elliptic(model, parameters, logger)` that calls `solve()`, computes statistics/diagnostics,
  and writes one output step (`t = 0`, step 0) via the existing `Output` — so `run_batch.py`,
  `generate_cfgs.py` sweeps and checkpoint layout (`runs/<name>/sim.h5`) work unchanged.
  Skip timestep/CFL logic entirely in that branch.
- [ ] Ensure `Output` works with `coeff_sub = {}` and with the `R`-space `p` (VTK cannot write
  `R` functions — exclude `p` from `vtk_vars`, still checkpoint it / store as statistic).
- [ ] Example configs: `elliptic.cfg` (repo root) and `tests/elliptic.cfg`.
- [ ] `workflows/elliptic_convergence.py` (+ `.sh` wrapper): run an `nx` sweep for a given
  (dim, complex, k, BC) case, compute observed rates from the statistics in each `sim.h5`, write a
  YAML/CSV summary; reuse `workflows/common.py` helpers.

### 4.8 Plotting (`dimswe/plotting/plot_elliptic.py`)
- [ ] 1D: line plots of `u`, `u_exact`, `σ`.
- [ ] 2D: scalar plots (k=0,2), quiver/magnitude (k=1) via existing `plotting.py` helpers.
- [ ] 3D: no static plots (VTK only).
- [ ] Convergence plot (log-log error vs h with reference slopes) for the workflow output.
- [ ] Hook into `plot_model.py` dispatch on `model.type == 'elliptic'`.

---

## 5. Implementation order (each step py_compile-clean and independently reviewable)
1. `differential_forms.py` + unit tests for the §1.2 table.
2. `harmonic_forms.py` index-set rule + unit tests (pure Python).
3. Mesh prerequisites (§4.1) + boundary registry (§4.2).
4. `elliptic_problems.py` (`SineProblem` first).
5. `elliptic.py`: k = 0 (primal) on all dims/BCs → then k = n → then middle k.
6. Driver/output integration (§4.7) + example cfgs.
7. Integration/convergence tests (§6).
8. Plotting, convergence workflow, README/workflows README updates.

---

## 6. Testing plan
(Per AGENTS.md: only `py_compile` can be run in this environment; tests are written for the
user to run in a Firedrake environment.)

Unit tests (`tests/unit/`):
- [ ] `test_harmonic_forms.py::test_harmonic_index_sets_*` — table-driven: fully periodic gives
  `C(n,k)`; all-natural k=0 gives `{∅}`; all-essential k=n gives `{all}`; mismatched faces give
  none; partial periodic examples (cylinder: k=1 natural ⇒ `{x}`, essential ⇒ `{y}`).
- [ ] `test_differential_forms.py::test_d_squared_is_zero` — `d(d w)` assembles to ~0 for smooth
  `w` for all (dim, complex, k).
- [ ] `test_differential_forms.py::test_integration_by_parts_identity` — for random smooth
  `w, z`, `∫ d w·z − ∫ w·δz − ∫_∂ B(w,z)` ≈ 0 (verifies every sign in §1.2).
- [ ] `test_harmonic_forms.py::test_harmonic_basis_is_discretely_harmonic` — for each basis form
  `h`: `d h = 0` and `(h, dτ) = 0` for all `τ` in `V^{k-1}` with the essential BCs applied.
- [ ] `test_boundary_conditions.py` — config parsing/validation errors; registry ids for each
  mesh type (including partial periodic and extruded).
- [ ] `test_elliptic_config.py` — `EllipticConfig` validation (bad k, bad complex, BC on periodic
  face, lump_mass rejection).

Integration tests (`tests/integration/test_elliptic.py`):
- [ ] Parametrized MMS convergence over the full matrix (12 (dim, complex, k) combos ×
  {periodic, essential, natural, mixed}) at two resolutions; assert observed L2 rate for `u`,
  `σ`, `du` ≥ expected − 0.3 (expected: `order + 1` for k=0 `u`, `order` otherwise — confirm
  empirically and record). Keep 3D cases at `order: 1`, small `nx` and mark `@pytest.mark.slow`.
- [ ] Harmonic constraint: fully periodic case with `f` having a constant-form component ⇒
  `p_h` matches the projection of `f`, and `(u_h, h_i) ≈ 0`.
- [ ] Inhomogeneous essential and natural data (`boundary_data: exact`) reproduce MMS rates.
- [ ] Driver test: `run_model(parameters)` with `tests/elliptic.cfg` writes `sim.h5` containing
  `u`, `sigma` and the error statistics (CLI/pipeline path requirement).
- [ ] Shared fixtures (`elliptic_parameters(dim, complex, k, bcs)` builder) in `tests/conftest.py`.
- Deterministic: problems are analytic; no randomness except seeded test vectors in the
  integration-by-parts test.

---

## 7. Open decisions (recommended default in **bold**)
1. Inhomogeneous BC data: support **yes, via `boundary_data: exact` from MMS** plus explicit
   per-face expressions later; or homogeneous only.
2. Harmonic constraint: **Lagrange multipliers in `R` space** (exact AFW formulation, works with
   LU) vs PETSc nullspace (better for iterative solvers, only works when `f ⊥ 𝔥`).
3. `lump_mass` for mixed problems: **reject** vs apply only to CG/DG blocks.
4. Simplicial `P^-` family (CG / N1curl / RT / DG) in `DeRhamComplex`: **out of scope for now**;
   the plan is family-agnostic so it can be added by extending `DeRhamComplex`.
5. `EllipticProblem` lives in **`dimswe/elliptic.py`** (dispatched from `models.get_model`) rather
   than inside `models.py`, to keep `models.py` time-dependent only.

---

## 8. Deferred / future work
- Numerical harmonic forms for general meshes (`mesh.type: file`): compute the kernel of the
  discrete Hodge Laplacian with SLEPc (generalized eigenproblem, smallest eigenvalues), then
  orthonormalize.
- Scalable iterative solvers: block-diagonal preconditioners with Hypre AMS (Hcurl) / ADS (Hdiv),
  `fieldsplit` setups, nullspace-based harmonic handling.
- Variable coefficients / weighted inner products (e.g. `(a σ, τ)`) exposed as model coefficients,
  enabling reuse of the adjoint/`optimize.py` machinery for elliptic PDE-constrained optimization.
- Hodge Laplacian eigenproblems (spectral correctness check: no spurious modes).
- Mixed BCs on sub-portions of a face (needs mesh markers from `file` meshes; harmonic forms then
  require the numerical path).
