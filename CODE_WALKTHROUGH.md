# Routine and subroutine walkthrough

The tables use `UPSTREAM`, `MODIFIED_UPSTREAM`, `NEW`, and `EXPERIMENT` as
short forms of the requested provenance classes. “All” representations means
A/B/C and the prepared constrained-rain variants where accepted by the provider.

## Original model and timestep

| File / routine | Arguments → returns | Mathematical object | Provenance / technology | Theta, representations, objectives, experiments |
|---|---|---|---|---|
| `dimswe/variables.py:244` `MoistThermalShallowWaterVariables_CF_H1` | spaces, tracer names → mixed-variable descriptor | `X=(v,h,S,Qv,Qc,Qr)` | UPSTREAM / Firedrake | no theta; all deployed experiments |
| `dimswe/models.py:16` `get_forcing_terms` | parameters, variables, spaces, IC → forcing list | assembles model RHS terms | UPSTREAM / Firedrake-UFL | no theta; all representations retain this model construction |
| `dimswe/models.py:184` `AdvDensH1Model` | parameters, logger → model | dynamics, spaces, forcings, coefficients | UPSTREAM / Firedrake | no direct theta; Test 1/2 cases |
| `dimswe/physics.py:7` `qsat` | `h,s,B,q0,H0,g` → saturation field | analytical `q_sat` | UPSTREAM / UFL | no theta; analytical oracle and Rep A rain |
| `dimswe/physics.py:68` `ThreeWayPhysics.rhs` | state, time, coefficients, tests → weak residual | analytical `A,R,G(A,R)` | UPSTREAM / Firedrake-UFL | no theta; oracle for all; analytical R in A |
| `dimswe/timestepping.py:12` `get_time_integrator` | scheme/model/options/backend → child integrator | selects RK/Euler and moist backend | MODIFIED_UPSTREAM / Python+Firedrake | learned provider enters via backend args; all deployed objectives |
| `dimswe/timestepping.py:46` `get_timestepper` | split configuration → stepper | production Lie composition | MODIFIED_UPSTREAM / Firedrake | theta later enters final child; all deployed runs |
| `dimswe/timestepping.py:758` `LieSplittingIntegrator` | child integrators, subcycles → composed stepper | `Phi=F6 o ... o F1` | MODIFIED_UPSTREAM / Firedrake-PETSc | all representations; primal execution |
| `dimswe/timestepping.py:991` `take_forward_step` | destination/source/time/dt → assigned destination | one complete split step | MODIFIED_UPSTREAM / Firedrake-PETSc | neural final child if selected; autonomous evaluations |

## JAX/local physics boundary

| File / routine | Arguments → returns | Mathematical object | Provenance / technology | Theta, representations, objectives, experiments |
|---|---|---|---|---|
| `dimswe/jax_moist.py:74` `_moist_algebra` | packed state/fields/physical params → rate dictionary | analytical `C,E,A,R` | NEW / JAX | no theta; analytical parity/oracle, Rep A R |
| `dimswe/jax_moist.py:172` `moist_rates_and_source_density_jax` | same → rates and four sources | exact `G(A,R)` | NEW / JAX | no theta; oracle/certification |
| `dimswe/moist_backend.py:125` `build_moist_integrator` | UFL child, backend, local provider → child | opt-in backend dispatch | NEW / Python+mixed | provider carries theta; all learned reps/objectives |
| `dimswe/moist_backend.py:51` `JAXMoistEulerIntegrator` | UFL oracle + provider → integrator wrapper | child-six primal interface | NEW / Firedrake+JAX | fixed inference theta via provider; deployed split |
| `dimswe/jax_moist_adapter.py:151` `JAXMoistEulerPrimal` | model/options/provider → adapter | `X+ = X + dt M^-1 b(G_theta)` | NEW / mixed Firedrake-PETSc-JAX | accepts A/B/C/B+ provider; every deployed objective |
| `dimswe/jax_moist_adapter.py:307` `interpolate_and_pack` / `pack_carrier` | scalar expression/carrier → owned `(cells,16)` float64 array | evaluation operator `I` at broken-CG3 GLL nodes | NEW / Firedrake+NumPy | upstream of theta; all learned reps |
| `dimswe/jax_moist_adapter.py:378` `_to_device_tree` | NumPy mapping → JAX device mapping | host-to-device boundary | NEW / NumPy+JAX | state/features/physical params; all learned reps |
| `dimswe/jax_moist_adapter.py:525` `_assemble_source_dual` | source arrays → mixed Cofunction | weak operator `b(G)` | NEW / Firedrake-UFL | after theta; all learned reps/objectives |
| `dimswe/jax_moist_adapter.py:553` `solve_mass` | source Cofunction → mixed Function | `M^-1 b` | NEW / Firedrake-PETSc | after theta; M2-X/H1/H2/H5 and rollout |
| `dimswe/jax_moist_adapter.py:571` `evaluate` | state, dt, optional physical and neural params → primal cache | complete moist Euler map/tape | NEW / mixed | explicit theta at lines 601--625; all reps and deployed objectives |

## Neural model, features, and output maps

| File / routine | Arguments → returns | Mathematical object | Provenance / technology | Theta, representations, objectives, experiments |
|---|---|---|---|---|
| `dimswe/test2a_operator.py:75` `MLPConfiguration` | dimensions/layers/activation/dtype/seed → validated config | network architecture | EXPERIMENT / Python | default Rep A 5-32-32-1, tanh, float64, seed 0; all Test2A objectives |
| `dimswe/test2a_operator.py:149` `initialize_mlp` | config → parameter pytree | Glorot-uniform weights, zero biases | NEW / JAX+NumPy | creates theta; Test2A and reused by Test2B |
| `dimswe/test2a_operator.py:189` `DenseMLP.__call__` | theta, normalized features → raw output | dense affine/activation composition | NEW / JAX | theta enters every layer; A/B/C/B+; all objectives |
| `dimswe/test2a_operator.py:207` `NormalizationMetadata` | offsets/scales → normalized coordinates | `(z-mu)/sigma`, `A/sigma_A` | EXPERIMENT / JAX+NumPy | Rep A Test2A; M1-X/M1-Y/M2-X/H1/H2/H5 |
| `dimswe/test2a_operator.py:365` `LocalAFeatureMap` | packed state/context → five-vector | `(h,S,Qv,Qc,B)` | EXPERIMENT / JAX | Rep A; all Test2A objectives |
| `dimswe/test2a_operator.py:383` `HybridAMoistOutputMap` | state/context/oracle/raw output → source | `G(A_theta,R_analytic)` | EXPERIMENT / JAX | theta through raw A; Rep A only |
| `dimswe/test2b_rain_learning.py:67` `RainMLPConfiguration` | representation → fixed config | 5-32-32-d tanh network | EXPERIMENT / Python | A:1281, B/B+:1314, C:1380 parameters; all Test2B objectives |
| `dimswe/test2b_rain_learning.py:113` `RainLearningNormalization` | frozen offsets/scales/provenance → scaling object | mass-weighted input/output scaling | EXPERIMENT / JAX+NumPy | all reps; fit only on truth 0--80 |
| `dimswe/test2b_rain_learning.py:205` `bplus_physical_rates` | raw output, `h,Qc,qprecip`, scales → `A,R` | thresholded linear-exceedance + softplus rain | EXPERIMENT / JAX | B+ preparation only |
| `dimswe/test2b_rain_learning.py:342` `RainActiveNeuralMoistPhysics` | rep, theta, normalization → local provider | feature → MLP → output map → `G_theta` | NEW/EXPERIMENT / JAX | owns frozen theta; A/B/C/B+; every Test2B objective and rollout |
| `dimswe/test2b_rain_learning.py:466` `combined_with_parameters` | state/fields/physical params/theta → rates+source | explicit-parameter local source | NEW / JAX | optimization theta; M1-X/M1-Y/M2-X/H1/H2/H5 |
| `dimswe/test2b_rain_learning.py:469-493` parameter JVP/VJP/joint differentiated VJP | local primal/covectors/directions → pytree actions | `G_theta dtheta`, `G_theta^T lambda`, derivative of pullback | NEW / JAX | all reps; gradients and HVP infrastructure |

The frozen Test 2B architecture contract is exactly 5-32-32-`d`, with
`d=1,2,4` for A/B/C, tanh on both hidden layers, a linear output, float64,
and seed-zero Problem-A Glorot-uniform weights with zero biases. The resulting
parameter counts are 1281, 1314, and 1380. M1-X, M1-Y, M2-X, H1, H2, and H5
do not change this architecture or the X-fitted normalization.

## Split derivatives and trajectory objectives

| File / routine | Arguments → returns | Mathematical object | Provenance / technology | Theta, representations, objectives, experiments |
|---|---|---|---|---|
| `dimswe/jax_moist_hvp.py:199` `JAXMoistEulerHVP` | primal adapter → derivative helper | derivative graph for moist Euler child | NEW / mixed | all learned reps; M2-X/H1/H2/H5 |
| `dimswe/jax_moist_hvp.py:455` `take_forward_step_cached` | state/time/dt/theta → primal cache | `M_theta(X)` and tape | NEW / mixed | explicit theta; deployed-discrete/trajectory |
| `dimswe/jax_moist_hvp.py:491` `take_tangent_step` | primal cache, state direction → tangent cache | `F_x dx` | NEW / Firedrake+JAX JVP | all reps; certification/HVP |
| `dimswe/jax_moist_hvp.py:556` `take_parameter_tangent_step` | primal cache, theta direction → tangent cache | `F_theta dtheta` | NEW / Firedrake+JAX JVP | all reps; derivative checks/HVP |
| `dimswe/jax_moist_hvp.py:605` `take_adjoint_step_cached` | primal cache, output dual → reverse result | `F_x^T lambda` | NEW / Firedrake+JAX VJP | all reps; H2/H5 reverse |
| `dimswe/jax_moist_hvp.py:664` `take_parameter_adjoint_step` | primal cache, output dual → state reverse + pytree | `F_theta^T lambda` | NEW / Firedrake+JAX VJP | theta gradient; M2-X/H1/H2/H5 |
| `dimswe/jax_moist_hvp.py:680,753` incremental-adjoint methods | primal/tangent/covector directions → HVP terms | differentiated VJP | NEW / Firedrake+JAX JVP-of-VJP | all reps; second-order certification, not final L-BFGS HVP |
| `dimswe/mtswe_split_hvp.py:745` `ProductionMTSWESplitHVP` | production Lie stepper → derivative composer | exact six-child discrete map | NEW / mixed | all reps; M2-X/H1/H2/H5 |
| `dimswe/mtswe_split_hvp.py:917` `take_forward_step_cached` | state/time/dt/theta → six-child tape | `Phi_theta(X)` | NEW / mixed | explicit theta to child six; recursive objectives/autonomous evaluation |
| `dimswe/mtswe_split_hvp.py:965` `take_fixed_prefix_cached` | truth state/time/dt → five-child cache | `P(X*)` | NEW / Firedrake-PETSc | theta-independent; H1 cache and first step of H2/H5 |
| `dimswe/mtswe_split_hvp.py:1011` `take_forward_step_from_prefix` | prefix, theta → complete tape | `M_theta(P(X*))` | NEW / mixed | theta at child six; H1/H2/H5 first step |
| `dimswe/mtswe_split_hvp.py:1057` `take_neural_parameter_tangent_step` | primal/state+theta directions → tangent | `Phi_x dx + Phi_theta dtheta` | NEW / mixed | all reps; derivative checks/HVP |
| `dimswe/mtswe_split_hvp.py:1107` `take_neural_parameter_adjoint_step` | primal/output dual → state dual+pytree | complete-step VJP | NEW / mixed | all reps; H1/H2/H5 gradients |
| `dimswe/test2a_trajectory.py:123` `reset_windows` | starts, horizon, loss mode, weights → specs | truth-reset/non-overlap sampling | EXPERIMENT / Python | H1/H2/H5; reused by A/B/C campaigns |
| `dimswe/test2a_trajectory.py:238` `GlobalMixedMassMetric` | helper, denominator → metric | global normalized mixed mass norm | EXPERIMENT / Firedrake-PETSc | all reps; H1/H2/H5 |
| `dimswe/test2a_trajectory.py:289` `NeuralTrajectoryObjective` | case, truth, windows, metric → objective | accumulated fixed/recursive rollout loss | NEW/EXPERIMENT / mixed | explicit theta; H1/H2/H5, all reps |
| `dimswe/test2a_trajectory.py:373` `_forward_window` | theta, window spec → window tape | first cached prefix, then recursion | NEW / mixed | H1 stops after one; H2/H5 feed predicted states |
| `dimswe/test2a_trajectory.py:486` `_gradient_window` | theta, forward tape → pytree gradient | reverse-through-time discrete adjoint | NEW / mixed | H1/H2/H5 exact parameter gradients |

## Fixed objectives, PyROL, and campaign entry points

| File / routine | Arguments → returns | Mathematical object | Provenance / technology | Theta, representations, objectives, experiments |
|---|---|---|---|---|
| `dimswe/test2a_discrete_offline.py:282` `ProductionDiscreteOfflineOperations` | split helper/provider → literal operations | exact source/interpolation/mass deployed map | EXPERIMENT / mixed | A (and shared concept for A/B/C); M2-X |
| `dimswe/test2a_discrete_training.py:152` `FixedDiscreteCache` | arrays/operators/metadata → validated cache | cached M2-X/H1 discrete operators | EXPERIMENT / NumPy/SciPy | theta-independent cache; M2-X/H1 |
| `dimswe/test2a_discrete_training.py:349` `FastFixedDiscreteObjective` | cache/model config → objective | fixed JAX deployed-discrete loss | EXPERIMENT / JAX | theta explicit; M2-X/H1 |
| `dimswe/test2a_horizon_curriculum.py:414` `prepare_h1_cache` | config, path → written certified cache | `Y_k=P(X_k*)` fixed data | EXPERIMENT / mixed | Rep A; H1/M2-Y |
| `dimswe/test2a_h1_m2_equivalence.py:407` `run_equivalence_audit` | configs/caches/output → audit record | H1/M2-Y value/gradient equivalence | EXPERIMENT / mixed | Rep A; dedicated regression support |
| `dimswe/test2b_m1y_campaign.py:158` `load_m1y_configuration` | config path → resolved source + frozen record | M1-Y state/support/architecture contract | EXPERIMENT / Python | no theta; A/B/C M1-Y |
| `dimswe/test2b_m1y_campaign.py:367` `_postprefix` | analytical case, `X_k*`, step → `Y_k*` | `Y_k*=P(X_k*)` | EXPERIMENT / Firedrake-PETSc | no learned theta; M1-Y cache preparation only |
| `dimswe/test2b_m1y_campaign.py:376` `prepare_m1y` | config, immutable manifest, output → certified NPZ/sidecar | fixed Y-state features and analytical A/R targets | EXPERIMENT / mixed | no optimization theta; A/B/C M1-Y |
| `dimswe/test2b_m1y_campaign.py:340` `representation_target` | rep, normalized Y features, A*, R*, scales → target array | A: `A*`; B: `(A*,R*)`; C: exact four-source vector | EXPERIMENT / NumPy+JAX scaling | target axis for A/B/C M1-Y |
| `dimswe/test2b_m1y_campaign.py:751` `m1y_objective` | config, cache, representation → `OperatorObjective` | offline `J_M1-Y` on Y features/targets | EXPERIMENT / JAX | theta explicit in fixed-array MLP; M1-Y only |
| `dimswe/test2b_m1y_campaign.py:879` `train_m1y` | config/cache/validation/rep/output → fit record | independent seed-zero full-batch M1-Y optimization | EXPERIMENT / JAX+PyROL-ROL | creates theta; A/B/C M1-Y; no prefix/recursion in loop |
| `dimswe/test2b_m1y_evaluation.py:108` `prepare_y_heldout` | config, training cache, output → held-out Y cache | truth-derived Y states 81--160 | EXPERIMENT / mixed | no training; matched M1-X/M1-Y evaluation |
| `dimswe/test2b_m1y_evaluation.py:396` `evaluate_representation` | config/caches/rep/output → matched JSON | fixed-model cross-state direct/deployed metrics | EXPERIMENT / JAX+NumPy+accepted evaluation | frozen M1-X/M1-Y theta; A/B/C |
| `dimswe/test2b_m1y_report.py:222` `build_results` | output directory → results CSV/JSON/manifest | compact M1-X/M1-Y synthesis | EXPERIMENT / Python | frozen theta hashes only; accepted A/B/C comparison |
| `dimswe/test2a_pyrol.py:39` `PytreeVectorCodec` | initial pytree → flat codec | theta pytree ↔ ROL vector isomorphism | NEW / JAX+NumPy | all reps/objectives |
| `dimswe/test2a_pyrol.py:90` `JAXPytreeObjective` | JAX objective + initial theta → ROL Objective | value/gradient/HVP adapter | NEW / JAX+PyROL | M1-X, M1-Y, and fixed JAX objectives |
| `dimswe/test2a_pyrol.py:213` `CallbackPytreeObjective` | value/gradient callbacks + theta → ROL Objective | external exact-gradient adapter | NEW / NumPy+PyROL | deployed discrete/trajectory callbacks |
| `dimswe/test2a_trajectory.py:554` `TrajectoryPyROLObjective` | trajectory objective + theta → ROL Objective | cached value/gradient bridge | NEW / PyROL+mixed | H1/H2/H5 |
| `dimswe/test2a_pyrol.py:294` `build_test2a_lbfgs_parameters` | optimizer config → ROL parameter list | line-search L-BFGS controls | EXPERIMENT / PyROL-ROL | all final campaigns; no production HVP |
| `dimswe/test2b_rain_learning_campaign.py:478` `objectives` | preparation+representation → M1-X/M2-X objects | objective-axis factory | EXPERIMENT / JAX | A/B/C; historical M1-X and M2-X |
| `dimswe/test2b_rain_learning_campaign.py:521` `trajectory_objective` | config/data/rep/theta/horizon → trajectory object | H1/H2/H5 factory | EXPERIMENT / mixed | A/B/C; H1/H2/H5 |
| `dimswe/test2b_rain_learning_campaign.py:816` `train` | config/artifacts/rep/objective/limits → fit files | selected optimization campaign | EXPERIMENT / JAX+Firedrake+PyROL | A/B/C final fits; prepared variants have only partial evidence |
| `dimswe/test2b_representation_[a,b,c]_postprocess.py` | configs + frozen fits → comparison JSON | cross-objective/autonomous evidence synthesis | EXPERIMENT / Python+mixed | one representation each; final Test2B reports |

## Technology boundary summary

- **Firedrake/UFL** owns spaces, interpolation, weak residual assembly, and
  discrete finite-element operators.
- **PETSc** owns mixed vectors/matrices, mass solves, AXPY updates, and dual
  action underneath Firedrake.
- **JAX** owns local analytical/neural algebra, dense networks, parameter
  pytrees, and local JVP/VJP/differentiated-VJP operations.
- **PyROL/ROL** owns line search, L-BFGS state, iteration acceptance, and calls
  the supplied `value`/`gradient` interfaces.
- **NumPy/Python** owns file formats, frozen caches, manifests, orchestration,
  hashes, and postprocessing.
