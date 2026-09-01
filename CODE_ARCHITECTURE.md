# Code architecture and call graph

## Module responsibility map

### ORIGINAL_UPSTREAM

| Module | Responsibility |
|---|---|
| `dimswe/variables.py` | mixed state and finite-element space layout |
| `dimswe/models.py` | DIMSWE model and forcing-term construction |
| `dimswe/physics.py` | analytical `ThreeWayPhysics`, including exact `A`, `R`, and source form |
| `dimswe/operators.py` | weak-form transport/forcing operators |
| `dimswe/output.py`, `dimswe/diagnostics.py` | run output and diagnostics |

### MODIFIED_UPSTREAM

| Module | Responsibility |
|---|---|
| `dimswe/timestepping.py` | original RK/Lie stepping plus backend dispatch and cached tangent/adjoint/HVP entry points |
| `dimswe/initial_conditions.py` | original initial-condition family with accepted later changes |
| `ode_adjoint/adjoint_timesteppers.py`, `ode_adjoint/dynamics.py` | upstream adjoint prototypes extended for derivative certification |

### NEW_LEARNED_PHYSICS

| Module | Responsibility |
|---|---|
| `dimswe/jax_moist.py` | pure float64 JAX analytical moist kernel |
| `dimswe/jax_moist_adapter.py` | Firedrake mixed state ↔ broken-GLL arrays ↔ JAX ↔ weak source/mass solve |
| `dimswe/moist_backend.py` | opt-in JAX replacement for only the final moist Euler child |
| `dimswe/jax_moist_hvp.py` | state/parameter JVP, VJP, and differentiated-VJP for that child |
| `dimswe/mtswe_split_hvp.py` | cached six-child split, fixed prefix, complete tangent/reverse/HVP composition |
| `dimswe/test2a_operator.py` | Representation A MLP, normalization, features, and parameter I/O |
| `dimswe/test2b_rain_learning.py` | Representation A/B/C and B+ provider, scaling, output maps, local derivatives |
| `dimswe/test2a_discrete_offline.py` | literal fixed deployed-discrete M2-X operations |
| `dimswe/test2a_discrete_training.py` | cache/certification and fast M2-X objective |
| `dimswe/test2a_trajectory.py` | H1/H2/H5 windows, forward tapes, exact reverse, PyROL adapter |
| `dimswe/test2a_pyrol.py` | JAX pytree flattening and generic PyROL objective adapters |
| `dimswe/test2b_m1y_campaign.py` | post-prefix Y-state cache, offline M1-Y objective, certification, and independent fits |
| `dimswe/test2b_m1y_evaluation.py` | held-out Y-state cache and matched M1-X/M1-Y evaluation |
| `dimswe/learned_physics/` | earlier generic learned-physics interfaces and historical objective vocabulary |

### EXPERIMENT_ONLY

`dimswe/test2a_*`, `dimswe/test2b_*`, their configs, `scripts/test2*`, and the
representation postprocessors construct campaigns and syntheses around the core
above. Test 1A/1B and `ode_adjoint/hvp.py` are certification/provenance
infrastructure, not a separate production timestep. The superseded isolated
Firedrake prototype is under
`archive/development-history/firedrake_hvp_prototype/`.

### UNKNOWN / PRESERVED

M1-only output trees named `representation-BTP` and `representation-BTPL` exist,
but they do not establish the completed B+ campaign described by the preparation
report. They are retained locally and classified `UNKNOWN`.

## Canonical Representation A/B/C execution path

```text
dimswe/test2b_rain_learning_campaign.py:main/train
  -> load_configuration / load_preparation
  -> initial_parameters or load_parameters
  -> RainActiveNeuralMoistPhysics(...)
       -> RainMLPConfiguration + DenseMLP
       -> normalized features (h,S,Qv,Qc,B)
       -> representation-specific output map
       -> local source dictionary (S,Qv,Qc,Qr)
  -> build_neural_case
       -> original DIMSWE model/problem construction
       -> timestepping.get_timestepper / get_time_integrator
       -> moist_backend.build_moist_integrator
       -> JAXMoistEulerIntegrator
       -> JAXMoistEulerPrimal + JAXMoistEulerHVP
       -> ProductionMTSWESplitHVP
  -> objective selected by representation-independent objective axis
       M1-X: OperatorObjective on x_features/x_A/x_R
       M2-X: FixedObjective / fixed source-to-state matrices
       H1/H2/H5: NeuralTrajectoryObjective
  -> JAXPytreeObjective or TrajectoryPyROLObjective
  -> ROL.Problem -> ROL.Solver (line-search L-BFGS)
  -> fit_result.json + final_parameters.{npz,json}
  -> representation_[a|b|c]_postprocess.py
  -> representation_[a|b|c]_final_comparison.json
  -> docs/TEST2B_REPRESENTATION_*_FINAL_SYNTHESIS.md
```

The production campaign constructs objective choices at
`dimswe/test2b_rain_learning_campaign.py:478-528` and invokes PyROL in
`train` at lines 816--896. The same source/provider is used across objectives;
the representation is not inferred from the objective name.

## Canonical M1-Y execution path

```text
dimswe.test2b_m1y_campaign prepare
  -> load historical truth and frozen normalization
  -> _postprefix
       -> ProductionMTSWESplitHVP.take_forward_step_cached(X*, t, dt)
       -> select boundary_states[-2] = Y*=P(X*)
  -> analytical provider at Y*
  -> cache normalized (h,S,Qv,Qc,B), A*, R*, carrier weights

dimswe.test2b_m1y_campaign train
  -> load_m1y_preparation
  -> initial_parameters(representation)          # independent seed zero
  -> OperatorObjective(Y features, Y targets)
  -> JAXPytreeObjective
  -> PyROL Problem -> Solver (line-search L-BFGS)
  -> checkpoints + final_parameters + fit_result

dimswe.test2b_m1y_evaluation
  -> prepare held-out Y* states 81--160
  -> fixed-network A/B/C inference
  -> matched M1-X/M1-Y direct and deployed diagnostics
```

The prefix runs only while preparing the immutable Y arrays. The M1-Y
optimization hot loop has no Firedrake step, no recursive state, and no
differentiate-through-prefix path. `dimswe/configs/test2b_m1y_20260828.json`
freezes the contract and `tests/test_test2b_m1y_campaign.py` protects state
location, features, target order, normalization, and initialization.

## One learned moist child in detail

```text
Firedrake mixed Function X
  -> JAXMoistEulerPrimal.interpolate_and_pack
       interpolate h,S,Qv,Qc and B to broken CG3 GLL carrier
       copy cell-local 16-point arrays
  -> jax.device_put(float64 arrays)
  -> RainActiveNeuralMoistPhysics.combined_parameterized_kernel
       normalize [h,S,Qv,Qc,B]
       DenseMLP(theta,z)
       scale/map output as A, B, C, or prepared B+
       construct source densities
  -> jax.device_get(source arrays)
  -> unpack_carrier
  -> assemble weak mixed Cofunction
  -> solve complete mixed mass matrix
  -> PETSc Vec AXPY: X_out = X_in + dt * tendency
```

The concrete bridge is `JAXMoistEulerPrimal.evaluate` in
`dimswe/jax_moist_adapter.py:571-663`. Packing is at lines 307--340, host/device
transfer at 378--400, weak assembly at 525--551, and the Firedrake/PETSc update
at 553--634. The adapter is intentionally serial, quadrilateral, order-3, and
CPU-JAX only (`jax_moist_adapter.py:192-245`).

## Where JAX first enters

The default `moist_backend="ufl"` does not import JAX. When a caller explicitly
selects the JAX backend, `build_moist_integrator` in
`dimswe/moist_backend.py:125-136` constructs `JAXMoistEulerIntegrator`, whose
constructor imports `JAXMoistEulerPrimal` locally at lines 74--90. This is the
runtime JAX hook into DIMSWE.

For learned physics, `local_physics` must also be supplied. The accepted
`physics_mode` values are checked at `jax_moist_adapter.py:165-190`.
`ProductionMTSWESplitHVP._forward_child` passes explicit `neural_parameters`
only to child six (`mtswe_split_hvp.py:885-914`).

## Theta ownership and flow

Frozen inference parameters are owned by
`RainActiveNeuralMoistPhysics._parameters` and returned only via a copy
(`test2b_rain_learning.py:345-356,462-464`). During optimization, PyROL owns a
flat ROL vector. `PytreeVectorCodec` maps it to the same structured JAX pytree.
That pytree flows through:

```text
ROL vector
  -> PytreeVectorCodec.pytree_from_vector
  -> objective(parameters)
  -> ProductionMTSWESplitHVP.take_forward_step_cached(...,
                                                       neural_parameters=parameters)
  -> JAXMoistEulerPrimal.evaluate(..., neural_parameters=parameters)
  -> RainActiveNeuralMoistPhysics.combined_parameterized_kernel(..., parameters)
  -> DenseMLP(parameters, normalized_features)
```

Parameter files store the pytree arrays in NPZ plus a JSON metadata/provenance
record (`test2b_rain_learning.py:283-330`).

## Objective and cache paths

### M1-X

`OperatorObjective` operates entirely on fixed arrays and a pure JAX function.
The historical driver supplies X-state features/targets. JAX `value_and_grad`
provides the parameter gradient.

### M1-Y

`prepare_m1y` computes truth-derived `Y*=P(X*)` once, then
`m1y_objective` supplies Y-state features and Y-state analytical targets to
the same fixed-array `OperatorObjective`. The accepted M1-Y A/B/C fits start
from seed zero and do not continue the H1/H2/H5 ladder.

### M2-X

Literal preparation applies exact Firedrake interpolation/source-assembly/mass
operators once. `FixedDiscreteCache` (`test2a_discrete_training.py:153-335`)
stores feature/target arrays and sparse/tensor operator data.
`FastFixedDiscreteObjective` executes the algebra in JAX with zero hot-loop
Firedrake/PETSc solves.

### H1 / M2-Y

`take_fixed_prefix_cached` computes `Y_k=P(X_k*)` once.
`prepare_h1_cache` serializes the fixed post-prefix arrays/operators. H1 uses the
same one-neural-child discrete map on every optimization evaluation and resets
to truth for every `k`.

### H2 / H5

`NeuralTrajectoryObjective` receives fixed first-prefix caches from its
constructor (`test2a_trajectory.py:289-371`). `_forward_window` completes the
first step from that cache, then executes complete split steps from predicted
states. `_tape` caches the full forward graph under a SHA-256 of the parameter
pytree (`lines 434-468`) so a same-parameter value/gradient pair reuses it.
`_gradient_window` reverses target duals through child six to one and accumulates
the parameter VJP.

## Differentiation path

```text
loss state residual
  -> GlobalMixedMassMetric.value_and_dual
  -> reverse through complete split (children 6,5,...,1)
       moist child:
         Firedrake mass/source assembly transposes
         JAX local state VJP
         JAX parameter VJP -> pytree gradient contribution
       other children:
         certified Firedrake variational adjoints
  -> sum parameter pytrees over steps/windows
  -> flatten into ROL gradient vector
```

Forward tangents use the same graph in child order. HVPs differentiate the
reverse: local JAX differentiated VJPs are composed with incremental Firedrake
adjoints in `JAXMoistEulerHVP` and `ProductionMTSWESplitHVP`. The final campaign
uses exact gradients with L-BFGS and does not request production HVPs.

## Manuscript-facing evaluation path

```text
frozen checkpoints and cache sidecars
  -> postprocessing/ml_results_20260829/scripts/
  -> compact quantitative CSV/JSON
  -> tables/{main,supplement}
  -> figures/{main,supplement}

frozen A/B/C models + truth restart states
  -> postprocessing/deployed_hybrid_dynamics_20260830/scripts/replay_spatial_maps.py
  -> external replay-map NPZ + versioned hash sidecar
  -> render_spatial_package.py
  -> accepted common-scale galleries / external movies

accepted analytical truth
  -> postprocessing/ground_truth_figures_20260829/scripts/
  -> external truth-map caches + compact diagnostics
  -> accepted deterministic manuscript figures
```

The path helpers in the two later packages use `DIMSWE_REPOSITORY` and
documented optional external-root variables. Historical absolute paths remain
only in immutable provenance records.
