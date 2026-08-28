# Changes from upstream DIMSWE

## Baseline decision

`BASE_CANDIDATE = d0eb61598a2cb1049628c3cc054ab9a1f3143bf6`

`CONFIDENCE = HIGH`

This is the most defensible state supplied by Chris before the learned-physics
development sequence because:

- it is the direct ancestor immediately before the 2026-08-04--07 Arjun Sharma
  development sequence;
- its message is `starting to add coefficient scaling`, consistent with the
  pre-existing coefficient/adjoint exploration in DIMSWE;
- all learned-physics, JAX-moist, production-HVP, and Test 1A/1B commits are
  descendants of it; and
- the commit is present in the independent collaborator history and is an
  ancestor of `d2f5d66ecb5500aad24eca37280f8a52e22a250f`.

The GitHub `main` ref points farther back at `149fe55`; it is not an equally
plausible handoff point because the coefficient-scaling work through `d0eb615`
was already in the supplied branch. No competing local branch or tag identifies
a later Chris baseline.

## What remains original

The original finite-element model is still the core of every deployed run:

- mixed-state definitions in `dimswe/variables.py`, especially
  `MoistThermalShallowWaterVariables_CF_H1` at lines 244--247;
- problem/model assembly in `dimswe/models.py`, including
  `get_forcing_terms` and `AdvDensH1Model`;
- analytical moist rates and exact source structure in
  `dimswe/physics.py:7-104` (`qsat`, `ThreeWayPhysics.rhs`);
- the Firedrake Runge--Kutta and Lie-split machinery in
  `dimswe/timestepping.py`; and
- pre-existing adjoint work in `ode_adjoint/` and coefficient handling in the
  model/timestep layers.

These routines were not replaced by a JAX solver. Learned physics is an opt-in
replacement for the final moist child of the existing split.

## Tracked development after the baseline

The baseline-to-frozen-tree tracked diff contains 94 files, approximately
42,906 insertions and 44 deletions. The committed development sequence is:

1. environment and baseline characterization (`27545b2` through `5bb9a5b`);
2. NumPy ODE and Firedrake HVP prototypes (`c0c3562` through `8581a81`);
3. production hyperviscosity, dry-Lie, and complete MTSWE split derivative
   infrastructure (`62be672` through `59fae2f`);
4. PyROL state/control/Hessian-vector adapters (`7c7caa9` through `47bfde1`);
5. analytical JAX moist replica and derivative actions (`ea185b7` through
   `51ae869`);
6. opt-in full-split JAX moist integration (`51e7eeb` through `3185088`); and
7. learned-physics framework and Test 1A/1B preparation/certification
   (`59f429d` through `d2f5d66`).

The frozen accepted state also had ten tracked files modified beyond `HEAD`:

`dimswe/configs/resolved_hidden_c0_pilot.cfg`, `dimswe/initial_conditions.py`,
`dimswe/jax_moist_adapter.py`, `dimswe/jax_moist_hvp.py`,
`dimswe/moist_backend.py`, `dimswe/mtswe_split_hvp.py`,
`dimswe/resolved_hidden_c0.py`, `dimswe/resolved_hidden_c0_driver.py`,
`dimswe/timestepping.py`, and `docs/LEARNED_PHYSICS_EXPERIMENTS.md`.

Those modifications are part of the scientific snapshot; `HEAD` alone was not
accepted as complete.

## Added scientific layers

- `dimswe/jax_moist.py`: float64 JAX replica of the analytical local moist
  algebra.
- `dimswe/jax_moist_adapter.py` and `dimswe/moist_backend.py`: Firedrake/JAX
  primal bridge and opt-in timestep integration.
- `dimswe/jax_moist_hvp.py`, `dimswe/mtswe_split_hvp.py`,
  `dimswe/dry_lie_hvp.py`, and `dimswe/hyperviscosity_hvp.py`: exact discrete
  tangent, reverse, and second-order actions.
- `dimswe/test2a_operator.py`, `dimswe/test2b_rain_learning.py`: neural models,
  feature maps, normalization, and representation-specific output maps.
- `dimswe/test2a_discrete_offline.py`, `dimswe/test2a_discrete_training.py`,
  `dimswe/test2a_trajectory.py`: fixed deployed-discrete and recursive
  objectives.
- `dimswe/test2a_pyrol.py`: JAX-pytree/PyROL bridge.
- Test 2A/Test 2B campaign drivers, configs, postprocessors, tests, and synthesis
  reports listed in `CANONICAL_EXPERIMENTS.md`.

The `firedrake_hvp_prototype/`, `ode_adjoint/hvp.py`, Test 1A/1B material, and
early Test 2 preparation remain useful derivative/provenance evidence but are
not the primary final Representation A/B/C campaign.
