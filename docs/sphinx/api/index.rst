API reference
=============

One page per module, generated from the docstrings. Each page starts with summary tables of the
module's classes and functions, followed by the full entries; class entries list their base
classes, so hierarchies such as ``BracketBase`` → ``PoissonBracket`` → the concrete brackets can be
followed by clicking through.

The ``dimswe`` package re-exports most of its modules with ``from .x import *``, so in user code
``dimswe.get_model`` and ``dimswe.models.get_model`` are the same object. This reference lists every
object only under the module that defines it.

.. Every table below writes its stub pages to api/generated/ (not under version control); the
   page layout comes from _templates/autosummary/module.rst.

Core model
----------

The geometric-mechanics building blocks: brackets, Hamiltonians and entropies, the variables they
act on, and the models and dynamics assembled from them.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :nosignatures:

   dimswe.models
   dimswe.variables
   dimswe.operators
   dimswe.poisson_brackets
   dimswe.metric_brackets
   dimswe.hamiltonians
   dimswe.entropies
   dimswe.dynamics
   dimswe.dissipation

Discretization and physics
--------------------------

Meshes, discrete operators and UFL helpers, initial conditions, and the forcings that fall
outside the geometric-mechanics framework.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :nosignatures:

   dimswe.meshes
   dimswe.transport_operators
   dimswe.ufl_helpers
   dimswe.bernstein
   dimswe.assembly
   dimswe.initial_conditions
   dimswe.physics
   dimswe.positivity

Time stepping
-------------

Explicit Runge–Kutta and LieSplit integrators with forward, adjoint, tangent-linear and
second-order adjoint steps.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :nosignatures:

   dimswe.timestepping

Optimization
------------

PDE-constrained optimization with ``scipy.optimize``: reduced problems, discrete-adjoint gradients
and forward-over-adjoint Hessian-vector products.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :nosignatures:

   dimswe.optimize

Elliptic problems
-----------------

Mixed FEEC solvers for Hodge Laplacian problems and eigenproblems: differential forms, boundary
conditions, harmonic forms, problem data, variable coefficients and solver configurations.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :nosignatures:

   dimswe.elliptic
   dimswe.elliptic_eigen
   dimswe.elliptic_problems
   dimswe.elliptic_coefficients
   dimswe.elliptic_solvers
   dimswe.differential_forms
   dimswe.boundary_conditions
   dimswe.harmonic_forms
   dimswe.numerical_harmonic
   dimswe.topology

Input, output and diagnostics
-----------------------------

Configuration files, checkpoint output, diagnostics and statistics, logging and profiling.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :nosignatures:

   dimswe.parameters
   dimswe.output
   dimswe.diagnostics
   dimswe.statistics
   dimswe.logger
   dimswe.profiling
   dimswe.numpy_helpers

Plotting
--------

Plotting routines for saved runs (``plot_model.py``).

.. autosummary::
   :toctree: generated
   :template: module.rst
   :nosignatures:

   dimswe.plotting.plotting
   dimswe.plotting.plot_adv_dens
   dimswe.plotting.plot_maxwell
   dimswe.plotting.plot_elliptic

Workflows
---------

Scripts and helpers for parameter sweeps, batch runs, comparisons, optimization and validation.
Each script module is also a command-line entry point; see the workflows guide for how they fit
together.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :nosignatures:

   workflows.common
   workflows.opt_helpers
   workflows.losses
   workflows.generate_cfgs
   workflows.run_batch
   workflows.compare_runs
   workflows.optimize_params
   workflows.steady_state
   workflows.validate
   workflows.plot_validation
   workflows.compare_optimizers
   workflows.profile_hvp
   workflows.elliptic_convergence
   workflows.elliptic_solver_scaling
