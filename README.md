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

- PDE-constrained optimization is done using scipy.optimize routines combined with the adjoint time stepping capabilities.
  - Currently only a Jacobian can be calculated, limiting the model to 1st order optimization algorithms

- Elliptic Hodge Laplacian problems `(d delta + delta d) u = f` for k-forms in 1D/2D/3D can be solved with `model.type: elliptic` (see `elliptic.cfg`, `TODO-elliptic.md` and `docs/hodge_laplacian.tex`)
  - Mixed FEEC formulation with essential, natural, periodic or mixed per-face boundary conditions and harmonic forms (Lagrange multipliers or a PETSc nullspace)
  - FEEC families `Q` (tensor product), `P-`/`P+` (simplicial) and `S` (serendipity, 2D) via `spatial-discretization.family` (`S-` is defined but rejected: the installed FIAT cannot assemble trimmed serendipity elements)
  - Run with `python run_model.py elliptic.cfg`; convergence sweeps with `workflows/elliptic_convergence.py`
  - Variable (scalar or SPD tensor) coefficients `mu`, `kappa`, `c` via `elliptic.coefficients`, optionally as model coefficients with adjoint gradients and steady-state optimization (`optimize.SteadyStateConstrainedOptimization`)
  - Hodge Laplacian eigenproblems with `model.type: elliptic-eigen` (SLEPc shift-and-invert; see `elliptic-eigen.cfg`), compared against exact box spectra
