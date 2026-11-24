# DIMSWE

DIMSWE (Data-Integrated Moist thermal Shallow Water) is a [Firedrake](https://www.firedrakeproject.org/)-based
code for simulation and PDE-constrained optimization of neutral and charged viscous fluids with learnable
neural network components.

DIMSWE is built around geometric mechanics formulations: Hamiltonian formulations for inviscid fluids and
metriplectic formulations for viscous fluids. A model is defined by its Poisson brackets, metric brackets,
Hamiltonians and entropies. On top of this, DIMSWE provides:

- explicit Runge–Kutta and composed LieSplit time integrators with forward, adjoint, tangent-linear and
  second-order adjoint steps;
- PDE-constrained optimization with discrete-adjoint gradients and forward-over-adjoint
  Hessian-vector products, driven by `scipy.optimize`;
- mixed finite element exterior calculus (FEEC) solvers for Hodge Laplacian problems and eigenproblems
  on box meshes, curved domains and immersed surfaces.

Start with {doc}`getting_started` to install DIMSWE and run a first model, or read the {doc}`overview` for
the formulations and methods.

```{note}
This documentation is under construction. The configuration reference, the remaining user-guide and theory
pages, and docstrings for every module are still being added.
```

```{toctree}
:maxdepth: 2
:caption: Introduction

getting_started
overview
```

```{toctree}
:maxdepth: 1
:caption: User guide

user_guide/workflows
```

```{toctree}
:maxdepth: 1
:caption: Theory

theory/hessian_vector_products
```

```{toctree}
:maxdepth: 1
:caption: Reference

api/index
```
