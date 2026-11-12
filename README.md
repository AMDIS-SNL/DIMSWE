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
