# DIMSWE

# Description
- DIMSWE (Data-Integrated Moist thermal Shallow Water) is a Firedrake based code for simulation and PDE-constrained optimization of neutral and charged viscous fluids with learnable neural network components.

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
