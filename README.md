# DIMSWE

# Description
- DIMSWE (Data-Integrated Moist thermal Shallow Water) is a Firedrake based code for simulation and PDE-constrained optimization of neutral and charged viscous fluids with learnable neural network components.
- It can mimic the spatial and temporal numerics used in the HOMME dycore when the H1 formulation and appropriate forcings and LieSplit time integration is used

# Design
- It is designed around geometric mechanics formulations, specifically Hamiltonian formulations for inviscid fluids and metriplectic formulations for viscous fluids.
- The key objects that define a model are Poisson brackets, metric brackets, Hamiltonians and entropies.

- There are currently three formulations implemented:
  - CF: Curl form Hamiltonian formulation for arbitrary advected quantities, discretized with H(div) for the velocity and the appropriate finite element spaces for the advected quantities (currently only advected densities)
  - LP: Lie-Poisson Hamiltonian formulation for arbitrary advected quantities, discretized with DG for the momentum and the appropriate finite element spaces for the advected quantities (currently only advected densities)
  - H1: Curl form Hamiltonian formulation for advected densities, discretized with H^1 (CG) for the velocity and a choice of either H^1 or DG for the densities
- There are currently two Hamiltonians implemented:
  - Thermal shallow water
  - Moist thermal shallow water

- Forcing terms that fall outside of the geometric mechanics framework can be added, currently there are three forcings:
  - Three way physics for moist thermal shallow water (CF, LP and H1 formulations)
  - DG1 transport for the DG densities in the H1 formulation
  - Hyperviscosity for the H1 formulation

- Time stepping is done using either a general RK scheme, or a composed LieSplit scheme
  - Currently only explicit RK method are implemented
  - In the LieSplit integrator, the dynamics and the forcings can be composed using an arbitrary choice of general RK schemes and substepping
  - The time stepping schemes can take both forward and adjoint steps, where the latter leverages the built-in derivative capabilities of UFL
  - The explicit RK and LieSplit schemes can also take tangent-linear and second-order adjoint steps, whose forms are UFL directional derivatives of the forward and adjoint step forms

- PDE-constrained optimization is done using scipy.optimize routines combined with the adjoint time stepping capabilities.
- Gradients (Jacobians) are computed with the discrete adjoint, so any first-order scipy method (for example L-BFGS-B) can be used
- Hessian-vector products are computed with a forward-over-adjoint scheme, following equation 2.11 of `petsc-opt.pdf`: a tangent-linear forward sweep, then a combined first- and second-order adjoint backward sweep. This enables the second-order scipy methods Newton-CG, trust-ncg, trust-krylov and trust-constr (only trust-constr also handles bounds)
- Hessian-vector products are available for coefficients, initial states, both together, and analytic initial-condition parameters, with the explicit RK and LieSplit time steppers. The full Hessian matrix is never formed, so methods that need it (dogleg, trust-exact) are not supported
- See `docs/hessian_vector_products.md` for the derivation and design, and `workflows/README.md` for how to choose a method in the optimization workflow and compare methods


- Elliptic Hodge Laplacian problems `(d delta + delta d) u = f` for k-forms in 1D/2D/3D can be solved with `model.type: elliptic` (see `elliptic.cfg`, `TODO-elliptic.md` and `docs/hodge_laplacian.tex`)
  - Mixed FEEC formulation with essential, natural, periodic or mixed per-face boundary conditions and harmonic forms (Lagrange multipliers or a PETSc nullspace)
  - FEEC families `Q` (tensor product), `P-`/`P+` (simplicial) and `S` (serendipity, 2D) via `spatial-discretization.family` (`S-` is defined but rejected: the installed FIAT cannot assemble trimmed serendipity elements)
  - Run with `python run_model.py elliptic.cfg`; convergence sweeps with `workflows/elliptic_convergence.py`
  - Variable (scalar or SPD tensor) coefficients `mu`, `kappa`, `c` via `elliptic.coefficients`, optionally as model coefficients with adjoint gradients and steady-state optimization (`optimize.SteadyStateConstrainedOptimization`)
  - Hodge Laplacian eigenproblems with `model.type: elliptic-eigen` (SLEPc shift-and-invert; see `elliptic-eigen.cfg`), compared against exact box spectra

# Running the tests
- `pytest` runs everything; `pytest -m "not slow"` skips the 3D convergence/eigen cases.
- The suite is large (parametrized convergence, eigenvalue and FEEC-family matrices) and parallelizes well with
  [pytest-xdist](https://pypi.org/project/pytest-xdist/): `pip install pytest-xdist`, then `pytest -n auto`
  (one serial Firedrake process per worker; every test writes only to its own temporary directory).
- The first run after a Firedrake update or cache clear is much slower: Firedrake compiles and caches a kernel for
  every distinct form. Later runs reuse the cache.
