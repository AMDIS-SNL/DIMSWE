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
- k=0, k=n: `1`, except 2D curl-div k=2: `dx∧dy ↦ −1` (forced by `d = div` and the 1-form map below:
  `d(x dy) = div(x·(−1, 0)) = −1`; the sign does not affect the harmonic spans)
- Hcurl 1-forms (2D grad-curl, 3D): `dx_i ↦ e_i`
- 2D curl-div 1-forms (Hdiv proxy, matches `skewgrad`): `dx ↦ (0, 1)`, `dy ↦ (−1, 0)`
- 3D 2-forms (Hdiv): `dy∧dz ↦ e_x`, `dz∧dx ↦ e_y`, `dx∧dy ↦ e_z`

Constant forms lie in every discrete space of every family in §4.9 (all contain constant
proxies) and are discretely harmonic under the above rule, so no eigen-solve is needed for these
meshes. For `mesh.type: file` (general topology) see §8 (numerical harmonic forms — deferred).

### 1.5 Two ways of enforcing the harmonic constraint
Both are supported (`elliptic.harmonic_constraint`):

- **`lagrange`** — the AFW formulation of §1.1: `p ∈ 𝔥^k` is an unknown in an `R` space,
  the system is non-singular, and `p = P_𝔥(rhs)` comes out of the solve. The installed Firedrake
  cannot assemble monolithic matrices with `R` blocks, so it is solved **matrix-free with GMRES**,
  preconditioned by an assembled `aP = a_0 + ε (u, v) + (p, q)` (`a_0` = form without the harmonic
  coupling): additive fieldsplit with `firedrake.AssembledPC` + MUMPS LU on the non-`R` blocks and
  no preconditioner on the `R` blocks, right-preconditioned GMRES so the stopping test uses the true residual (`harmonic_forms.lagrange_preconditioner_form`,
  `lagrange_solver_parameters`). Few iterations: eigenvalues `λ/(λ+ε)` plus `2m` values
  `±(ε|Ω|)^(-1/2)` from the harmonic coupling.
- **`nullspace`** — drop `p` and the third equation. The system matrix is then singular with
  kernel `{(σ, u) = (0, h) : h ∈ 𝔥^k}` (check: `(h, dτ) = 0` and `dh = 0`). Attach this as a
  PETSc `MixedVectorSpaceBasis` (nullspace and transpose nullspace; the operator is symmetric),
  make the right-hand side compatible with the **rhs projection helper** (§4.4), solve, then
  remove the harmonic part of `u_h` so that `(u_h, h_i) = 0`. `p` is reported as the projection
  computed by the helper, so both approaches produce the same output fields.

### 1.6 Inhomogeneous boundary data
Inhomogeneous data are supported for both BC types:
- essential: `DirichletBC` values `g_u` (for `tr u`) and `g_σ` (for `tr σ`);
- natural: the boundary functionals `−∫ B_{k-1}(τ, g_u)` and `+∫ B_k(v, g_du)` in §1.1.

Problem-data classes provide these values per face (§4.5); by default they come from the exact
solution. When `𝔥^k ≠ {0}` the data may not be compatible: with `lagrange` the incompatible part
goes into `p`, and with `nullspace` the rhs projection helper removes it (and logs its size).

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
  get_aux_var_list() -> []
  get_coeff_* / set_coeffs         # empty until §4.10, then exposes mu/kappa/c
  create/compute_statistics/diagnostics, get_statistics(_list), get_diagnostics(_list)
  is_time_invariant = True
```

New modules (small, composable, each function documented per AGENTS.md):

| module                               | responsibility                                                      |
|--------------------------------------|---------------------------------------------------------------------|
| `dimswe/differential_forms.py`       | k-form → space, `d`, `δ`, boundary pairing `B`, trace helpers, constant-form proxies |
| `dimswe/boundary_conditions.py`      | parse BC config, logical face registry per mesh, build `DirichletBC`s and natural boundary terms |
| `dimswe/harmonic_forms.py`           | Künneth selection rule (pure Python), UFL basis construction, L2-orthonormalization, rhs/solution projection helpers, nullspace builder |
| `dimswe/elliptic_problems.py`        | library of problem data (manufactured solutions, sources), like `initial_conditions.py` |
| `dimswe/elliptic.py`                 | `EllipticConfig`, `EllipticCoefficient`, `EllipticProblem`, shared space/BC/form builders, solve, adjoint, errors |
| `dimswe/elliptic_eigen.py`           | `HodgeLaplacianEigenproblem` (SLEPc), exact box spectra (§4.11)     |
| `dimswe/elliptic_solvers.py`         | solver-parameter builders, Riesz-map preconditioner forms (§4.12)   |
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
  family: Q                  # Q (existing) | P- | P+ | S- | S   (see §4.9)
  order: 2
  lump_mass: false           # must be false for elliptic (rejected otherwise)

elliptic:
  form_degree: 1             # k
  complex: curl-div          # 2D only: grad-curl | curl-div (ignored in 1D/3D)
  boundary_conditions:       # per logical face; omitted faces of periodic directions are periodic
    left:   essential        # essential | natural
    right:  essential
    # bottom/top/front/back ...
  boundary_data: exact       # exact (from the problem's exact solution) | problem (problem class
                             # supplies per-face data) | homogeneous
  harmonic_forms: auto       # auto (Künneth rule) | none | numerical (future)
  harmonic_constraint: lagrange   # lagrange | nullspace   (§1.5)
  solver: elliptic-direct    # elliptic-direct | elliptic-direct-singular | riesz-amg | riesz-gmg | riesz-pmg | schur-mass (§4.12)
  coefficients: {}           # mu / kappa / c (§4.10); defaults 1, 1, 0
  eigen:                     # only for model.type: elliptic-eigen (§4.11)
    nev: 20
    target: 0.0
    tolerance: 1.0e-10

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
the two allowed values in 2D, `lump_mass: true` (always rejected for elliptic problems),
`harmonic_constraint` not in `{lagrange, nullspace}`, and family/cell/order combinations that
are not supported (§4.9). BC data are never given as strings in the cfg (no `eval`); custom
data come from problem classes.

---

## 4. Implementation tasks

### 4.1 Mesh / space prerequisites (`dimswe/meshes.py`)
- [x] Add optional `mesh.periodic_directions` handling:
  - 2D: `PeriodicRectangleMesh(..., direction='x'|'y'|'both')`.
  - 3D quads: `ExtrudedMesh(base, nz, periodic=<z periodic>)` with base `RectangleMesh` or
    `PeriodicRectangleMesh(direction=...)` — covers all 8 periodicity combinations.
  - 1D: only fully periodic or fully bounded (Firedrake limitation is fine).
- [x] Non-periodic `box` with hex cells: currently `BoxMesh(hexahedral=True)` (comment says it
  "will choke" for NCE/NCF). Build it as `ExtrudedMesh(RectangleMesh(...quadrilateral=True), nz)`
  instead, like the periodic box, for the `Q` family. Serendipity families need genuine
  (non-extruded) hexahedral meshes, so keep a `BoxMesh(hexahedral=True)` path selected by
  family (§4.9). Keep `simplicial_cells: true` path unchanged; the simplicial families (§4.9)
  use it.
- [x] Make the mesh builder accept the elliptic problem object in place of `initcond` (it only
  needs `Lx, Ly, Lz, xc, yc, zc`); document this duck-typed contract in the docstring.
- [x] Record on the mesh/spaces which directions are periodic (needed by BC registry and harmonic
  forms), e.g. `spaces.periodic_directions`.
- [x] Use the existing `get_diff_form_space(spaces, 'S', k, dim, nm1)` for k-form spaces
  (`nm1=True` ⇔ curl-div, `False` ⇔ grad-curl). In 1D it already returns CG/DG correctly; do
  **not** touch `DeRhamComplex.Hdiv = CG` in 1D (used by existing models).
- [x] Fix pre-existing bugs we will hit: `mesh_size` uses unimported `Function`/`CellDiameter`.
- Notes (done): partial periodicity works for every cell type: quads and triangles via `PeriodicRectangleMesh(direction=...)`, Q-family hexes via `ExtrudedMesh`, serendipity (genuine) hexes via `PeriodicBoxMesh(directions=..., hexahedral=True)` (>= 3 cells per direction); tets only bounded or fully periodic (Firedrake's `PeriodicBoxMesh` supports `directions` only for hexahedra, so partial periodicity raises `NotImplementedError`); periodic directions and domain bounds are on `mesh.periodic_directions`/`mesh.domain_bounds` and `spaces.periodic_directions`/`spaces.domain_bounds`; meshes with any periodic direction, and all boxes, occupy `[0, L]` per direction (bounded line/rectangle stay centered). Unit tests: `tests/unit/test_meshes.py`.

### 4.2 Boundary handling (`dimswe/boundary_conditions.py`)
- [x] `BoundaryFace` dataclass: logical name, Firedrake subdomain id(s), measure kind
  (`ds`, `ds_v`, `ds_t`, `ds_b`), DirichletBC subdomain token (`int`, `"top"`, `"bottom"`).
- [x] `get_boundary_registry(mesh, mesh_type, periodic_directions)` returning faces for:
  `IntervalMesh` (1,2), `RectangleMesh` (1–4), `PeriodicRectangleMesh(direction=…)` (1,2 renumbered —
  **verify numbering against installed Firedrake**), `ExtrudedMesh` (base ids via `ds_v` +
  `top`/`bottom` via `ds_t`/`ds_b`), simplicial `BoxMesh` (1–6).
- [x] `parse_boundary_conditions(cfg, registry)` → `{face: 'essential'|'natural'}` with validation.
- [x] `build_essential_bcs(W, sub_index_sigma, sub_index_u, faces, g_sigma, g_u)` → list of
  `DirichletBC` (skip `u` for DG / k=n, skip σ for k=0).
- [x] `build_natural_terms(tau, v, faces, g_u, g_du, k, dim, complex, n)` → UFL boundary forms
  using `B_{k-1}` and `B_k` from `differential_forms`; returns 0 for homogeneous data.
- [x] Helper `measure_for(face)` so the forms work on both extruded and non-extruded meshes.
- Notes (done): ids are numbered consecutively over the bounded directions (x, y, z order, low side first); extruded z faces are `back`=`ds_b`/"bottom", `front`=`ds_t`/"top". `verify_boundary_registry` checks the numbering geometrically (run by the unit tests for every mesh type). Data (`g_u`, `g_sigma`, `g_du`) may be `None`, one expression, or a per-face dict. Extra helpers: `build_face_registry` (pure Python), `faces_of_type`, `face_bcs_by_direction` (input for §1.4). `boundary_pairing` and `normalize_complex` were added to `dimswe/differential_forms.py` ahead of §4.3. Unit tests: `tests/unit/test_boundary_conditions.py`.
- Caveats found while testing: (a) Q-family Hcurl/Hdiv elements (RTCE/RTCF/NCE/NCF) cannot be interpolated into, so inhomogeneous essential data on them go through Firedrake's DirichletBC fallback (global L2 projection, exact only for data in the discrete space); homogeneous data use UFL `zero()` (assigned directly). If the §6 convergence tests show reduced σ/u rates from this, replace it by a face-wise trace projection (e.g. `EquationBC`). (b) Firedrake's `PeriodicBoxMesh` needs >= 3 cells per direction and keeps the fixed ids 1-6 even for periodic directions (`directions=` argument, unlike `PeriodicRectangleMesh`, which renumbers); partial periodicity is built this way for serendipity hexes (tets: Firedrake supports `directions` only with `hexahedral=True`), and `build_face_registry` uses the fixed ids 1-6 for all non-extruded 3D meshes.

### 4.3 Differential-form utilities (`dimswe/differential_forms.py`)
- [x] `form_space(spaces, k, dim, complex)` (thin wrapper over `get_diff_form_space`).
- [x] `exterior_derivative(w, k, dim, complex)` per the §1.2 table (returns `None`/0 for k=n).
- [x] `codifferential(z, k, dim, complex)` (formal adjoint of `d_{k-1}`; for MMS only).
- [x] `boundary_pairing(w, z, k, dim, complex, n)` = `B_k(w, z)`.
- [x] `hodge_laplacian_expr(u, k, dim, complex)` = `d δ u + δ d u` symbolic (for computing `f`
  from an exact solution).
- [x] `constant_form_proxy(I, k, dim, complex)` per §1.4 proxy map.
- [x] Possibly move `skewgrad/curl2D` usage here; keep `ufl_helpers` API unchanged.
- Notes (done): absent operators return `None` (`exterior_derivative` for k=n, `codifferential` for k=0); `boundary_pairing` returns `0` for k=n. Index sets `I` are normalized to increasing x, y, z order (`normalize_index_set`, accepts names or axis numbers), so in 3D `{x, z}` = dx∧dz ↦ −e_y; in 2D curl-div `dx∧dy ↦ −1` (found by the proxy-vs-d test). `constant_form_values` is the pure-Python proxy map, `constant_form_proxy` its UFL version (no domain: project/interpolate it, or combine it with mesh quantities). Extra helper `is_scalar_form(k, dim)`. `skewgrad`/`curl2D` are imported from `ufl_helpers` (API unchanged). Unit tests: `tests/unit/test_differential_forms.py` (d²=0 symbolic and discrete, δ²=0, integration by parts for all 8 cases with d, Hodge Laplacian = −componentwise Laplacian for all 12 cases, proxy map vs. d(x_{i1} dx^{I'})).

### 4.4 Harmonic forms (`dimswe/harmonic_forms.py`)
- [x] `harmonic_index_sets(k, dim, periodic_dirs, face_bcs)` — pure Python, implements §1.4
  (unit-testable without Firedrake).
- [x] `harmonic_basis(k, dim, complex, index_sets, mesh)` → list of UFL constant proxies.
- [x] `orthonormalize_harmonic_basis(basis, V, dx)` — interpolate each constant proxy into
  `V^k`, then do Gram–Schmidt in the L2 inner product (constant forms are already orthogonal on
  boxes, but this also normalizes them and keeps the code general for numerical bases in §8).
- [x] **`project_rhs_orthogonal_to_harmonic(rhs, basis)`** (the rhs projection helper):
  - for a UFL source `f`: return `f − Σ_i (f, h_i) h_i` plus the coefficients `(f, h_i)`;
  - for an assembled rhs `Cofunction` (which includes the natural boundary functionals):
    subtract `Σ_i b(h_i) M h_i` (with `M` the `V^k` mass action, applied only to the `u` block)
    so that `b(h_i) = 0` for every `i`;
  - log the size of the removed component (it should be ~0 for compatible data).
- [x] `remove_harmonic_component(u_h, basis)` — in-place `u_h ← u_h − Σ (u_h, h_i) h_i`, used
  after a `nullspace` solve.
- [x] Lagrange-multiplier treatment: one `R`-space dof per basis form
  (m scalar `FunctionSpace(mesh, "R", 0)` blocks; vector-valued `R` arguments are not supported by Firedrake),
  giving `p = Σ p_i h_i` and the constraint rows `(u, h_i) = 0`.
- [x] Nullspace treatment: `build_harmonic_nullspace(W, u_index, basis)` returning a
  `MixedVectorSpaceBasis` with `VectorSpaceBasis(vecs=…)` on the `u` block (orthonormalized in
  the PETSc l2 sense, which PETSc requires) and the other blocks as their own space; passed as
  both `nullspace` and `transpose_nullspace` to the solver.
- [x] Option `harmonic_forms: none` for users who know `𝔥 = 0`; error out on `file` meshes with
  `auto` until §8 numerical harmonic forms exist.
- Notes (done): `harmonic_index_sets` accepts per-face (`parse_boundary_conditions`) or per-direction (`face_bcs_by_direction`) BC dicts and returns sorted index-set tuples; `get_harmonic_index_sets(mode, ...)` implements `auto`/`none`/`numerical`. `harmonic_basis(k, dim, complex, index_sets)` has no `mesh` argument (the proxies are domain-free). Constant proxies are interpolated, or L2-projected with LU for RTCE/RTCF/NCE/NCF (no dual basis). The rhs helper works in place on Cofunctions (plain or mixed via `u_index`), zeroes the correction `M h_i` on the nodes of the passed u-block `bcs` so lifted BC rows stay intact (h_i vanishes there), and returns `(rhs, coefficients)`. Extra helpers: `combine_harmonic_forms` (p from coefficients), `harmonic_constraint_form` (the `Σ p_i (h_i, v) + q_i (u, h_i)` terms), `multiplier_coefficients`, `l2_inner`. `lagrange` uses one scalar `R` block per harmonic form (Firedrake has no arguments on vector-valued `R` spaces), and cannot use monolithic LU (Firedrake rejects `R` blocks in monolithic matrices), so `lagrange_preconditioner_form` + `lagrange_solver_parameters` give a matrix-free GMRES solve (§1.5). Coefficients are Python floats (`numpy.float64 * Function` would build an object array). The rule assumes essential ⇒ zero tangential trace of u (I ∋ x_i) and natural ⇒ zero trace of ⋆u (x_i ∉ I). Unit tests: `tests/unit/test_harmonic_forms.py`, including lagrange-vs-nullspace solves for periodic k=0 (CG + jacobi with the nullspace) and 2D k=2 mixed (GMRES + MUMPS with `mat_mumps_icntl_24`).

### 4.5 Problem data library (`dimswe/elliptic_problems.py`)
Pattern mirrors `initial_conditions.py` (`get_elliptic_problem(parameters)` dispatcher, classes
with `Lx, Ly, Lz, xc, yc, zc`).
- [x] Base class `EllipticProblemData` with `get_source(mesh, k, dim, complex)` and optional
  `get_exact(mesh, k, dim, complex)` returning dict `{u, sigma, du, f}` (σ = δu, f = Δu via §4.3).
- [x] Inhomogeneous boundary data API on the base class:
  `get_essential_data(mesh, face, ...) -> (g_u, g_sigma)` and
  `get_natural_data(mesh, face, ...) -> (g_u, g_du)`. The default implementation derives them
  from `get_exact` (`g_u = u`, `g_σ = δu`, `g_du = du`; the traces/pairings are applied by
  `boundary_conditions`, so the data are full fields). Subclasses override them for problems
  without an exact solution; `boundary_data: homogeneous` short-circuits to zero.
- [x] `SineProblem` (manufactured, products of sin/cos with configurable modes) valid for every
  (dim, k, complex); chosen so that `u ⊥ 𝔥^k` (zero mean for each constant-form component) so
  `p = 0` and errors are measured directly against `u_exact`.
- [x] `PolynomialProblem` (low-degree, tests exactness/consistency on coarse meshes, e.g. `u`
  in the discrete space ⇒ error ≈ round-off for k=0).
- [x] `GaussianSourceProblem` (no exact solution; `f` given, homogeneous BCs; for demos/plots).
- [x] `BoundaryDrivenProblem` (`f = 0`, nonzero data on one face given by overriding the
  boundary-data methods; tests the inhomogeneous path without MMS).
- [x] When harmonic forms are present and an exact solution is used, compare against
  `u_exact − P_𝔥 u_exact`. MMS problems are built to be compatible; a dedicated
  `IncompatibleSourceProblem` (`f` with a constant-form component) tests §1.5/§1.6.
- Notes (done): problems are written in normalized coordinates `s_i ∈ [0, 1]` from `mesh.domain_bounds`; `Lx, Ly, Lz` default to 1 and `xc, yc, zc` to `L/2` (so bounded meshes also cover `[0, L]`). `get_exact(...)` returns `{u, sigma, dsigma, du, f, p}` (`None` for absent fields; `p` is the expected harmonic part of `f`, `None` = 0); `get_source`/`get_exact` take an optional `harmonic_index_sets` (only used by `IncompatibleSourceProblem`, since only harmonic constants can be added without changing `u`). With exact boundary data the rhs is always compatible (natural-face terms cancel; harmonic forms have zero trace on essential faces), so MMS problems only need `u ⊥ 𝔥`: `SineProblem` uses full-period factors `sin(2π m_i s_i + φ)` (periodic and zero-mean, valid for every BC/periodicity), `PolynomialProblem` zero-mean factors `s^d − 1/(d+1) + c (s − 1/2)` (bounded meshes only). `collect_boundary_data(problem, mode, mesh, registry, face_bcs, k, dim, complex)` implements `boundary_data` and returns per-face dicts for `build_essential_bcs`/`build_natural_terms` (`exact` bypasses subclass overrides). Zero sources are `firedrake.Constant`s (`zero_form`), because literal zeros fold into domain-less `Zero`s. Cfg entries: `modes` (sine), `degree` (polynomial), `amplitude` (incompatible, boundary-driven), `width`/`center` (gaussian), `face` (boundary-driven). Unit tests: `tests/unit/test_elliptic_problems.py`.

### 4.6 Core model (`dimswe/elliptic.py`)
- [x] `EllipticConfig` dataclass + `from_parameters(parameters, spaces)` with validation (§3).
- [x] `EllipticProblem.__init__(parameters, logger, mesh=None)`:
  problem data → mesh/spaces → config → spaces `V^{k-1}, V^k` (+ `R^m` only for `lagrange`) →
  `MixedFunctionSpace` (skip absent blocks) → var list → BC data → forms → solver.
- [x] `_build_forms()`: split into `_mass_sigma`, `_coupling`, `_stiffness`, `_harmonic`
  (lagrange only), `_rhs`, `_natural_terms` small functions assembled per §1.1.
- [x] `solve()` dispatches on `harmonic_constraint`:
  - `lagrange`: `LinearVariationalSolver`, read off `p`.
  - `nullspace`: assemble the system with BCs, call the rhs projection helper on the assembled
    rhs, solve with the nullspace attached, call `remove_harmonic_component`, and fill `p`
    from the helper's coefficients.
  - Either way, log KSP convergence and store `solution` plus the `sub` dict (`sigma`, `u`, `p`).
- [x] Inhomogeneous essential BCs and `nullspace`: lift the BCs first (the `DirichletBC`s change
  the rhs), and only then project, so the projection sees the final rhs.
- [x] `lump_mass: true` → `ValueError` in `EllipticConfig` validation; all forms use exact
  (non-GLL) `dx`/`ds`, with `elliptic.quadrature_degree` (default `2·order+2`, always explicit).
- [x] Error statistics (if exact solution available): `L2(u)`, `L2(σ)`, `L2(du)`, `L2(dσ)`,
  `|p|`, plus orthogonality check `max_i |(u_h, h_i)|`; stored in the statistics dict shape
  `Output` expects (arrays of length 1).
- [x] Diagnostics fields: `u_exact`, `u_error`, `sigma_error`, `du` (projected/interpolated
  to the appropriate spaces) for VTK.
- [x] Solver parameters in `parameters.py`:
  - `elliptic-direct`: `preonly` + `lu` + `mumps` (`mat_mumps_icntl_14` bump), for problems
    without harmonic forms; for `lagrange` with `𝔥^k ≠ 0` use `harmonic_forms.lagrange_solver_parameters`
    (matrix-free GMRES + assembled LU preconditioner, see §1.5; monolithic `R` blocks are not
    supported by Firedrake);
  - `elliptic-direct-singular`: as above plus MUMPS null-pivot detection
    (`mat_mumps_icntl_24: 1`), for `nullspace` with LU;
  - `elliptic-minres`: `minres` + a simple `fieldsplit`/Jacobi preconditioner, for `nullspace`
    (verifies the nullspace path independently of MUMPS null-pivot handling).
  Scalable block preconditioners (Hypre AMS/ADS) stay in §8.
- Notes (done): the σ equation is multiplied by −1 (`−(σ,τ) + (u,dτ) + (dσ,v) + (du,dv)`), so the operator is symmetric (nullspace = transpose nullspace; MINRES applicable); natural σ-terms change sign accordingly. `nullspace` path: explicit lifting `w = w_g + w_0` (Firedrake's internal lifting would run after the projection and, with nonzero `g_σ`, reintroduce a harmonic component since `(h, dτ_j) ≠ 0` for boundary basis functions), rhs zeroed on constrained dofs, projected with the u-block BCs, solved with homogeneous BCs and the nullspace. `lagrange` with harmonic forms: matrix-free GMRES (§1.5); `nullspace` + `elliptic-direct` is upgraded to `elliptic-direct-singular` (right-preconditioned GMRES with the nullspace, preconditioned by the LU of the regularized `a + ε(u, v)`; MUMPS null-pivot detection was not robust in 1D, giving ~1e-5 errors); `elliptic-minres` = MINRES with the SPD Riesz preconditioner `(σ,τ)+(dσ,dτ)+(u,v)+(du,dv)` (LU per block). `quadrature_degree` is always explicit (default `2·order+2`): with UFL's per-form estimates the lagrange (one combined mixed residual) and nullspace (separate rhs) paths integrated the source with different rules and differed by ~1e-5 in 1D. Extra cfg entries: `solver_options` (merged PETSc options), `error_quadrature_degree` (default `2·order+4`), `lagrange_regularization`. Only family `Q` on quads/hexes is accepted until §4.9. `p` is always reported as a Function in `V^k` (`sub['p']`; R functions cannot go to VTK). Statistics: `u_norm, p_norm, harmonic_orthogonality, num_harmonic_forms, ksp_iterations` (+ `u_error, sigma_error, dsigma_error, du_error, p_error` with an exact solution; `u_error` is against `u_exact − P_𝔥 u_exact`). Diagnostics: `u_exact, u_error, sigma_error, du`. Not yet hooked into `models.get_model`/`run_model` (§4.7). Tests: `tests/unit/test_elliptic_config.py`, `tests/integration/test_elliptic.py` (MMS matrix over 12 cases × 6 BC sets, lagrange vs nullspace with incompatible sources, MINRES, boundary-driven/Gaussian problems; 3D marked `slow`); shared builder `make_elliptic_parameters` / fixture `elliptic_parameters` in `tests/conftest.py`.

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

### 4.9 New FEEC families in `DeRhamComplex` (separate task, `dimswe/meshes.py`)
Restructure `DeRhamComplex.__init__` into one small builder per family (`_build_Q`, `_build_Pminus`,
`_build_P`, `_build_Sminus`, `_build_S`) that all set the same attributes (`CG`, `Hcurl`,
`Hdiv`, `DG`, `CGV`, `DGV`, measures), so `get_diff_form_space` and the existing models keep
working. Each family must form a subcomplex (`d V^k ⊂ V^{k+1}`) for a given `order = r`:

| family | cells | 0-forms | 1-forms (Hcurl) | (n−1)-forms (Hdiv) | n-forms | notes |
|--------|-------|---------|-----------------|--------------------|---------|-------|
| `Q` (existing, `Q_r^-`) | quad/hex/extruded | `CG r` | `RTCE r` / `NCE r` | `RTCF r` / `NCF r` | `DG r−1` | unchanged |
| `P-` (`P_r^-`) | interval/triangle/tet | `CG r` | `N1curl r` | `RT r` | `DG r−1` | same degree throughout |
| `P+` (full `P_r`) | interval/triangle/tet | `CG r` | `N2curl r−1` | `BDM r−(n−1)` | `DG r−n` | degree drops by one per form degree; requires `r ≥ n` |
| `S-` (trimmed serendipity) | quad/hex | `S r` | `SminusCurl r` (2D) / `SminusE r` (3D) | `SminusDiv r` (2D) / `SminusF r` (3D) | `DPC r−1` | |
| `S` (full serendipity `S_r`) | quad/hex | `S r` | `BDMCE r−1` (2D) / 3D element: check availability | `BDMCF r−(n−1)` (2D) / 3D: check | `DPC r−n` | 3D full serendipity H(curl)/H(div) may not exist in Firedrake/FIAT |

- [ ] **Verify every element name, degree convention, and supported cell against the installed
  Firedrake/FIAT** before coding the table (`r` conventions differ between elements); record the
  verified table in the docstring.
- [ ] 1D: `P-` and `P+` both reduce to `CG r → DG r−1` (`P+` in 1D: `DG r−1` too, since
  `n = 1`); serendipity families on intervals are the same as `Q`, so map to the `Q` builder.
- [ ] 2D: both complexes (grad-curl uses the Hcurl element, curl-div the Hdiv element) for every
  family.
- [ ] Mesh constraints enforced in validation: `P±` need `simplicial_cells: true`; `S±` need
  non-extruded quad/hex meshes (so `box`/`box-periodic` with `S±` uses
  `BoxMesh/PeriodicBoxMesh(hexahedral=True)`, and needs a check that the installed Firedrake
  supports this); `Q` keeps extruded meshes in 3D.
- [ ] Vector spaces `CGV`/`DGV` (used by existing models) defined for every family with the
  matching scalar element; `HdivV`/`HcurlV` left unset as today.
- [ ] `lump_mass` GLL rules exist only for `Q`; raise for other families if `lump_mass: true`
  (independent of the elliptic rejection).
- [ ] Expected convergence rates per family/k go in a table in the tests (§6), since `P+`/`S`
  give different rates for `u` vs `σ` than `P-`/`Q`.
- [ ] The existing time-dependent models are not required to support the new families, but the
  space construction must not break them (`Q` path identical to current behaviour).

### 4.10 Variable coefficients, optionally exposed as model coefficients (separate task)
**Weighted problem.** Coefficients go only on the `σ` mass, the `du` stiffness and an optional
reaction term. The `u` inner product stays unweighted:

```
(μ σ, τ) − (u, dτ)                     = −∫_{Γn} B_{k-1}(τ, g_u)
(dσ, v) + (κ du, dv) + (c u, v) + (p, v) = (f, v) + ∫_{Γn} B_k(v, g_du)
```
i.e. `L u = d(μ⁻¹ δu) + δ(κ du) + c u`. Here `μ` (on `Λ^{k-1}`, e.g. inverse permeability for
mixed Darcy at k=n), `κ` (on `Λ^{k+1}`, e.g. conductivity for k=0, magnetic reluctivity for 3D
k=1) and `c ≥ 0` (reaction) are scalar or SPD-tensor fields. Leaving the `u` inner product
unweighted keeps the kernel `{(0, h)}` and the harmonic-form rule of §1.4 unchanged. If `c > 0`
everywhere, the problem is non-singular and the harmonic constraint is switched off
automatically. (A weighted `u` inner product changes the harmonic space and needs the numerical
path, so it is in §8.)

- [ ] `EllipticCoefficient` dataclass: `name ∈ {mu, kappa, c}`, `kind ∈ {constant, problem,
  field}`, `rank ∈ {scalar, tensor}`, `treat_as_coeff`, `space` (for `field`: e.g. `DG0`, `CG1`,
  `R`), `bounds`, `scaling_factor`, `log_parametrization` (store `θ` with `coef = exp(θ)` so the
  coefficient stays positive during optimization).
- [ ] Config:
  ```yaml
  elliptic:
    coefficients:
      kappa: {kind: constant, value: 1.0, treat_as_coeff: true, bounds: [0.1, 10.0], scaling: 1.0}
      mu:    {kind: field, space: DG0, init: problem, treat_as_coeff: true, log_parametrization: true}
      c:     {kind: constant, value: 0.0}
  ```
  `kind: problem` means the problem-data class supplies a UFL expression (spatial variation,
  anisotropic tensors); it is never `eval`'d from the cfg. Missing coefficients default to `1`, `1`, `0`.
- [ ] Model-coefficient interface on `EllipticProblem`, matching `Dynamics`/`Hyperviscosity`
  (`treat_as_coeffs` pattern): `has_coeff`, `get_coeff_var`, `get_coeff_list`,
  `get_coeff_spaces`, `get_coeff_test_trial_vars`, `set_coeffs(parameters, coeff_sub)`,
  `get_coeff_size`, `get_coeff_bounds`, `get_coeff_scaling_factors`. Coefficients with
  `treat_as_coeff: false` are baked into the forms as `Constant`/expressions. Coefficients with
  `treat_as_coeff: true` enter the forms through `coeff_split`, so `ufl.derivative` with respect to
  them works.
- [ ] `differential_forms.weighted_hodge_laplacian_expr(u, μ, κ, c, ...)` for MMS:
  `σ = μ⁻¹ δu`, `f = d σ + δ(κ du) + c u`; natural data become weighted fluxes
  (`g_du = κ du`). Add a `VariableCoefficientSineProblem` with smooth, non-constant, anisotropic
  coefficients.
- [ ] Riesz-map preconditioners (§4.12) and eigenproblems (§4.11) take the same weights.
- [ ] **Sensitivities / steady-state optimization:**
  - `EllipticProblem.solve_adjoint(dJdx)` — the system is symmetric, so this reuses the forward
    operator: solve `Aᵀ λ = dJ/dx` with the homogenized BCs, and the same harmonic treatment.
  - `EllipticProblem.coefficient_gradient(λ)` = `−λᵀ ∂R/∂m` assembled with `ufl.derivative`.
  - A `SteadyStateConstrainedOptimization` in `optimize.py` that reuses `ReducedProblem`,
    `scipy_options`, `taylor_test`, the L2/observation objectives and bounds/scaling handling,
    with no time stepping (one "block", one observation step).
  - Optional: route `workflows/optimize_params.py` to it when the model is time-invariant (one
    truth run, recover `κ`/`μ`).
- [ ] Validation: tensor coefficients need the matching shape (`dim×dim` on vector-proxy spaces,
  scalar on 0-/n-form spaces); a coefficient on an absent block (`μ` at k=0, `κ` at k=n)
  → error.

### 4.11 Hodge Laplacian eigenproblems (separate task, `dimswe/elliptic_eigen.py`)
Find `λ`, `(σ, u) ≠ 0`:
```
(μ σ, τ) − (u, dτ)          = 0
(dσ, v) + (κ du, dv)        = λ (u, v)
```
This is a generalized problem `A x = λ M x` where `M` (the `u`-block mass) is singular on the `σ`
block. For k=0 it is the primal `(κ du, dv) = λ (u, v)`.

- [ ] Refactor `elliptic.py` so that space construction, BC construction and bilinear-form
  builders are free functions shared by `EllipticProblem` and the eigenproblem class
  (no duplication).
- [ ] `HodgeLaplacianEigenproblem` class (`model.type: elliptic-eigen`, dispatched from
  `get_model`) built on Firedrake `LinearEigenproblem`/`LinearEigensolver` (SLEPc):
  shift-and-invert with a configurable target, MUMPS for the inner solve, `nev`/`ncv`/tolerance
  from the cfg. Shift-and-invert sends the infinite eigenvalues (from the singular `M`) to 0.
- [ ] Homogeneous BCs only (essential via `DirichletBC` with `bc_shift` set so that BC rows
  don't produce spurious eigenvalues in the target window; natural need no terms); periodic as usual.
- [ ] Optional deflation of the known harmonic forms (§4.4), or keep them and report their count.
  Zero-eigenvalue multiplicity is reported as a statistic and compared to the §1.4 rule.
- [ ] Output: eigenvalues (and residual norms) as statistics; eigenpairs `(σ_i, u_i)` saved as
  checkpoint indices `idx = i` (one output "step" per mode), so the existing loaders and plotting
  can loop over modes. VTK writes one `.pvd` entry per mode.
- [ ] `exact_box_eigenvalues(k, dim, lengths, periodic_dirs, face_bcs, n)` (pure Python) for
  tests. On a box with one BC type per face, eigenforms separate per `dx^I` component. Per
  direction, the 1D factor is:
  - in `I`: essential → `cos`, `m ≥ 0`; natural → `sin`, `m ≥ 1`;
  - not in `I`: essential → `sin`, `m ≥ 1`; natural → `cos`, `m ≥ 0`;
  - periodic: `cos`/`sin` with `(2π m / L)²`, `m ≥ 0` (one mode at `m = 0`, two otherwise).
  
  `λ = Σ (m_i π / L_i)²`; the spectrum is the multiset union over `|I| = k`. Multiplicities matter.
  Mixed per-direction BCs (essential on one face, natural on the other) give quarter-wave modes
  `((m + ½) π / L)²`.
- [ ] Plotting: eigenvalue-index plot (computed vs exact) and mode plots in 1D/2D.
- [ ] Later reuse: the numerical harmonic forms in §8 are the zero-eigenvalue eigenvectors from
  this solver.

### 4.12 Scalable iterative solvers (separate task)
Goal: iteration counts that stay bounded under mesh refinement (and, where possible, under order
and coefficient changes) for every (dim, complex, k, family), in parallel.

- [ ] **Riesz-map block preconditioner** (Arnold–Falk–Winther): MINRES (the system is symmetric
  indefinite) with block-diagonal `P = diag(R_{k-1}, R_k)`, where `R_j` is the weighted
  `HΛ^j` inner product: `R_{k-1}: (μ σ, τ) + (dσ, dτ)`, `R_k: (u, v) + (κ du, dv) + (c u, v)`.
  Pass it as `aP` to `LinearVariationalProblem` and split the fields with PETSc `fieldsplit`
  (additive).
- [ ] Block solvers per space type (with selectable variants):
  - H1 (CG/S): BoomerAMG or GAMG; `P1PC`/`PMGPC` p-multigrid for high order.
  - H(curl): Hypre AMS (`firedrake.HypreAMS`); 2D H(div) = rotated H(curl), also AMS.
  - H(div) in 3D: Hypre ADS (`firedrake.HypreADS`).
  - L2 (DG/DPC): DG mass is block-diagonal → `pc_type: bjacobi` + `sub_pc_type: lu` (exact).
  - Fallback for elements/cells AMS/ADS do not support (high order, quads/hex, serendipity —
    **verify support in the installed Firedrake**): geometric multigrid on a `MeshHierarchy`
    with vertex-star patch smoothers (`ASMStarPC`/`PCPATCH`, the AFW smoother); for high-order
    `Q`, `FDMPC` + p-multigrid.
- [ ] Mesh support for multigrid: `mesh.refinement_levels` builds a `MeshHierarchy` (or
  `ExtrudedMeshHierarchy`), and the model works on the finest level. Boundary registry and
  periodicity must survive refinement.
- [ ] Harmonic forms with iterative solvers: the `nullspace` approach is the default (the `R`
  space adds a dense row/column that does not scale). For `lagrange`, put a Schur/`R` block in the
  fieldsplit, document it and test it, but recommend against it at scale.
- [ ] Alternative for k=n (mixed Poisson/Darcy): Schur-complement fieldsplit with a
  diagonal-mass approximation, for comparison.
- [ ] Solver parameters can't be static dicts (field indices depend on which blocks exist, the
  space types and the harmonic treatment), so add a builder
  `get_elliptic_solver_parameters(name, k, dim, complex, family, harmonic_constraint)` in
  `parameters.py` (or `dimswe/elliptic_solvers.py`). Names: `elliptic-direct`,
  `elliptic-direct-singular`, `riesz-amg`, `riesz-gmg`, `riesz-pmg`, `schur-mass`.
- [ ] Reuse in eigenproblems (§4.11): iterative inner solve for shift-and-invert (preconditioned
  with the same Riesz maps) as an option.
- [ ] Logging: KSP iteration counts and timings (PETSc log events, like the existing
  `"time step"` event) stored as statistics.
- [ ] `workflows/elliptic_solver_scaling.py` (+ `.sh`): sweep `nx` / order / MPI ranks /
  coefficient contrast, record iterations and times, write a summary table and plot.

---

## 5. Implementation order (each step py_compile-clean and independently reviewable)
1. `differential_forms.py` + unit tests for the §1.2 table.
2. `harmonic_forms.py` index-set rule + unit tests (pure Python).
3. Mesh prerequisites (§4.1) + boundary registry (§4.2).
4. `elliptic_problems.py` (`SineProblem` first).
5. `elliptic.py`: k = 0 (primal) on all dims/BCs → then k = n → then middle k, with
   `lagrange` first, then `nullspace` + rhs projection helper, then inhomogeneous data.
6. Driver/output integration (§4.7) + example cfgs.
7. Integration/convergence tests (§6) for `Q`.
8. New FEEC families (§4.9): refactor `DeRhamComplex` (Q unchanged) → `P-` → `P+` → `S-` → `S`;
   extend the convergence matrix to each family as it lands.
9. Variable coefficients (§4.10): constant/problem/field coefficients → model-coefficient
   interface → adjoint + coefficient gradient + Taylor test → steady-state optimization.
10. Eigenproblems (§4.11): shared-builder refactor → `HodgeLaplacianEigenproblem` → exact box
    spectra tests.
11. Scalable iterative solvers (§4.12): solver-parameter builder → `riesz-amg` → nullspace +
    iterative → `riesz-gmg`/`riesz-pmg` → scaling workflow.
12. Plotting, convergence workflow, README/workflows README updates.

Steps 8–11 are independent of each other once step 7 is done and can go in any order. §4.12's
coefficient robustness tests need §4.10.

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
  face, `lump_mass: true` rejected, bad `harmonic_constraint`, family/cell mismatch).
- [ ] `test_harmonic_forms.py::test_project_rhs_orthogonal_to_harmonic` — after projecting a UFL
  source and an assembled rhs `Cofunction`, `(f, h_i)` / `b(h_i)` ≈ 0; projecting twice changes
  nothing; data that are already orthogonal stay the same.
- [ ] `test_harmonic_forms.py::test_remove_harmonic_component` — result orthogonal to basis.
- [ ] `test_derham_families.py` — for each family × cell × dim: spaces build, `d V^k ⊂ V^{k+1}`
  (interpolate `d` of a random discrete function into `V^{k+1}` exactly), and the `Q` family
  produces identical spaces to the current implementation.
- [ ] `test_elliptic_coefficients.py` — coefficient config parsing/validation (shapes, absent
  blocks, bounds, log-parametrization round-trip); `get_coeff_*` sizes/bounds/scaling match the
  config; `treat_as_coeff: false` gives an empty coefficient space.
- [ ] `test_elliptic_coefficients.py::test_coefficient_gradient_taylor` — Taylor test
  (2nd-order remainder) for `dJ/dm` for scalar `R` coefficients and DG0 field coefficients, at
  k=0, k=n and one middle k, using `optimize.taylor_test`.
- [ ] `test_elliptic_eigen.py::test_exact_box_eigenvalues` — pure Python: known 1D/2D spectra
  and multiplicities (e.g. unit-square k=0 Neumann: 0, π², π², 2π², …; k=1 multiplicities on
  the torus), and the zero-eigenvalue count equals the §1.4 harmonic dimension.
- [ ] `test_elliptic_solvers.py` — solver-parameter builder produces the right fieldsplit
  indices for every block layout (with/without `σ`, with/without `R`) — pure Python.

Integration tests (`tests/integration/test_elliptic.py`):
- [ ] Parametrized MMS convergence over the full matrix (12 (dim, complex, k) combos ×
  {periodic, essential, natural, mixed}) at two resolutions; assert observed L2 rate for `u`,
  `σ`, `du` ≥ expected − 0.3 (expected: `order + 1` for k=0 `u`, `order` otherwise — confirm
  empirically and record). Keep 3D cases at `order: 1`, small `nx` and mark `@pytest.mark.slow`.
- [ ] Harmonic constraint: fully periodic case with `f` having a constant-form component ⇒
  `p_h` matches the projection of `f`, and `(u_h, h_i) ≈ 0` — for **both** `lagrange` and
  `nullspace`.
- [ ] `lagrange` vs `nullspace` agreement: same `(σ_h, u_h, p_h)` to solver tolerance on every
  case with `𝔥^k ≠ {0}` (and `nullspace` with `𝔥 = 0` gives the same result as no constraint).
- [ ] Inhomogeneous essential, natural and mixed data (`boundary_data: exact`) reproduce MMS
  rates in every (dim, complex, k); `BoundaryDrivenProblem` runs and gives a nonzero solution.
- [ ] Inhomogeneous data combined with harmonic forms (e.g. cylinder k=1) for both constraint
  approaches.
- [ ] Family matrix: repeat the MMS convergence tests for `P-`, `P+`, `S-`, `S` on their supported
  cells (subset of BCs, small meshes, `slow` marker), against a per-family expected-rate table.
- [ ] Variable coefficients: `VariableCoefficientSineProblem` reproduces MMS rates for scalar
  and anisotropic tensor `μ`, `κ`, and `c > 0` (harmonic constraint disabled automatically).
- [ ] Steady-state optimization: recover a scalar `κ` (and a smooth DG0 `μ` field with a small
  regularization) from a truth run with explicit seeds; loss decreases and the recovered value is
  within tolerance.
- [ ] Eigenproblems: first `N` computed eigenvalues match `exact_box_eigenvalues` with
  multiplicities, with no spurious modes in the window, and the eigenvalue error converges at
  `≈ 2·order`. Cover all 12 (dim, complex, k) combos (3D small/`slow`) and at least one case per
  BC type, including mixed and partially periodic.
- [ ] Iterative solvers: for `riesz-amg` (and `riesz-gmg` where supported), the solution matches
  `elliptic-direct` to tolerance; iteration counts over 3 refinements stay below a bound and grow
  by at most a small factor; coefficient-contrast sweep (`κ` jump 1→10³) for robustness; nullspace
  case converges. Parallel smoke test under `mpiexec -n 2` (`slow`/`mpi` marker).
- [ ] Driver test: `run_model(parameters)` with `tests/elliptic.cfg` writes `sim.h5` containing
  `u`, `sigma` and the error statistics (CLI/pipeline path requirement); same for
  `tests/elliptic-eigen.cfg` (eigenvalues + modes).
- [ ] Shared fixtures (`elliptic_parameters(dim, complex, k, bcs)` builder) in `tests/conftest.py`.
- Deterministic: problems are analytic; no randomness except seeded test vectors in the
  integration-by-parts test.

---

## 7. Decisions (resolved)
1. Inhomogeneous BC data: **supported** for essential and natural BCs (§1.6, §4.5), from the
   exact solution or from problem-class overrides.
2. Harmonic constraint: **both** Lagrange multipliers (`R` space) and PETSc nullspace (§1.5,
   §4.4), plus a helper that projects a rhs (UFL source or assembled `Cofunction`) to be
   orthogonal to the harmonic space.
3. `lump_mass: true` is **rejected** for elliptic problems.
4. New FEEC families are a **separate task** (§4.9): simplicial `P-` and `P+`, and serendipity
   `S-` and `S`.
5. `EllipticProblem` lives in **`dimswe/elliptic.py`** (dispatched from `models.get_model`), not
   in `models.py`.

---

## 8. Deferred / future work
- Numerical harmonic forms for general meshes (`mesh.type: file`): take the zero-eigenvalue
  eigenvectors from the §4.11 solver, then orthonormalize.
- Weighted `u` inner product `(w u, v)` (and hence weighted harmonic forms), which needs the
  numerical harmonic-form path.
- Mixed BCs on sub-portions of a face (needs mesh markers from `file` meshes; harmonic forms then
  require the numerical path).
