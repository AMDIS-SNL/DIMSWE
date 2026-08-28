# Learned-physics equations and algorithms

## 1. Original DIMSWE state and split

The deployed mixed state is

\[
X=(v,h,S,Q_v,Q_c,Q_r),
\]

where `v` is velocity, `h` is depth, `S` is entropy density, and the three
`Q` fields are water densities. The concrete variable ordering is defined by
`MoistThermalShallowWaterVariables_CF_H1` in `dimswe/variables.py:244-247`;
the carrier spaces are assembled in `AdvDensVariables_CF_H1` at lines 160--173.

One production step is a composition

\[
\Phi_\theta = M_\theta\circ P,
\]

where `P` is the fixed five-child prefix and `M_theta` is the final explicit
moist Euler child. `ProductionMTSWESplitHVP._child_specs` in
`dimswe/mtswe_split_hvp.py:860-887` records the actual order:

1. dry RK4 half-step;
2. dry RK4 half-step;
3. hyperviscosity Euler step;
4. DG SSPRK43 half-step;
5. DG SSPRK43 half-step;
6. moist Euler step.

The ordinary primal executes the same child ordering through
`LieSplittingIntegrator.take_forward_step` in `dimswe/timestepping.py:991-1000`.
`take_fixed_prefix_cached` (`mtswe_split_hvp.py:969-1013`) exposes `P`, and
`take_forward_step_from_prefix` (lines 1015--1052) applies only `M_theta`.

## 2. Analytical moist physics

With

\[
q_v=Q_v/h,\quad q_c=Q_c/h,\quad s=S/h,\quad \beta_2=gL,
\]

the saturation value and thermodynamic factor are

\[
q_{sat}=q_0\frac{H_0}{h+B}\exp\left[20(1-s/g)\right],\qquad
\gamma_v=(1+20q_{sat}\beta_2/g)^{-1}.
\]

The deployed condensation, evaporation, net vapour, and rain rates are

\[
C=\max(0,\gamma_v(q_v-q_{sat})/\tau_v),
\]

\[
E=\min(q_c/\Delta t,\max(0,\gamma_v(q_{sat}-q_v)/\tau_v)),
\]

\[
A=E-C,\qquad
R=\max(0,\gamma_r(q_c-q_{precip})/\tau_r),
\]

with `tau_v=tau_r=configured_dt`. The exact moist source density is

\[
G(A,R;X)=
\begin{bmatrix}
h\beta_2A\\ hA\\-h(A+R)\\hR
\end{bmatrix}_{(S,Q_v,Q_c,Q_r)}.
\]

This is the original UFL implementation in `ThreeWayPhysics.rhs`,
`dimswe/physics.py:68-104`. The equivalent differentiable JAX algebra is
`_moist_algebra` and `moist_source_density_jax` in
`dimswe/jax_moist.py:74-204`. Velocity and depth sources are structural zeros.
The map conserves total water and satisfies the thermodynamic coupling
`S_t-beta2*Qv_t=0` for structured representations.

## 3. Physics representations

Let `f_theta` be the normalized MLP evaluated on

\[
z=\operatorname{normalize}(h,S,Q_v,Q_c,B).
\]

### Representation A

The network returns one coordinate, mapped to `A_theta`; rain remains the exact
analytical `R(X)`:

\[
G_A(X;\theta)=G(A_\theta(X),R(X);X).
\]

This is `RainActiveNeuralMoistPhysics.combined` at
`dimswe/test2b_rain_learning.py:390-394`. The earlier Test 2A equivalent is
`HybridAMoistOutputMap` in `dimswe/test2a_operator.py:383-402`.

### Representation B

The network returns `(A_theta,R_theta)`, and both rates pass through the exact
source map:

\[
G_B(X;\theta)=G(A_\theta(X),R_\theta(X);X).
\]

See `test2b_rain_learning.py:395-398`. Unconstrained `R_theta` is not projected
positive.

### Representation C

The network directly returns the four source components:

\[
G_C(X;\theta)=(S_t,Q_{v,t},Q_{c,t},Q_{r,t})_\theta.
\]

No two-rate manifold or source invariant is imposed; see
`test2b_rain_learning.py:399-401`. The older Test 2A “Problem B” is this
representation on the earlier no-rain support.

### B+ preparation

The prepared linear-exceedance map (historical identifiers `BPLUS`/`BTPL`) is

\[
A_\theta=\sigma_A y_0,\qquad
R_\theta=\sigma_R\max\left(0,\frac{q_c-q_{precip}}{\delta q}\right)
\frac{\operatorname{softplus}(y_1)}{\log 2}.
\]

It is implemented by `bplus_physical_rates` in
`dimswe/test2b_rain_learning.py:205-230`. A related `BTP` variant uses a hard
threshold and positive gate (`btp_physical_rates`, lines 236--253). These maps
are prepared and derivative-certified. The evidence does not establish a
completed B+ M1/M2-X/H1/H2/H5 campaign; BTP/BTPL M1-only files are retained as
uncertain partial evidence.

## 4. Training objectives

Physics representation chooses `G_theta`; objective chooses where and how it is
evaluated.

### M1: direct local regression

At fixed truth samples, M1 compares a learned rate or tendency to its analytical
target. For A/B the loss is in normalized rate coordinates; for C it is in four
normalized source coordinates. `OperatorObjective` in
`dimswe/test2b_rain_learning_campaign.py:72-117` is the rain-active production
implementation. No Firedrake timestep is inside the optimization loop.

### M2-X: deployed-discrete at boundary truth

For each boundary truth state `X_k*`,

\[
J_{M2-X}(\theta)=\sum_k
\frac{\|\Phi_\theta^{moist}(X_k^*)-\Phi_*^{moist}(X_k^*)\|_M^2}{D}.
\]

The fixed feature arrays and exact interpolation/weak-source/mass-update
operators are cached. `ProductionDiscreteOfflineOperations.predict` in
`dimswe/test2a_discrete_offline.py:346-383` is the literal bridge;
`FastFixedDiscreteObjective` in `dimswe/test2a_discrete_training.py:350-436`
executes its cacheable JAX form. It is offline deployed-discrete training, not
recursive “online” training.

### H1 / M2-Y: truth-reset post-prefix objective

Define `Y_k*=P(X_k*)`. Then

\[
J_{H1}(\theta)=\sum_k
\frac{\|M_\theta(Y_k^*)-X_{k+1}^*\|_M^2}{D}.
\]

Every sample resets to truth before applying the fixed prefix; no learned state
feeds a later learned evaluation. `prepare_h1_cache` in
`dimswe/test2a_horizon_curriculum.py:416-552` constructs the exact fixed cache.
`dimswe/test2a_h1_m2_equivalence.py:88-409` states and audits its exact
deployed-discrete equivalence. H1 is therefore also offline/cacheable here.

### H2 and H5: recursive trajectories

For non-overlapping windows of length `H`,

\[
\widehat X_{k,0}=X_k^*,\quad
\widehat X_{k,j+1}=\Phi_\theta(\widehat X_{k,j}),\quad
J_H=\sum_{k,j=1}^{H}w_j
\frac{\|\widehat X_{k,j}-X_{k+j}^*\|_M^2}{D}.
\]

At `j=0`, the prefix is cached at truth. At every later step the complete model
receives the preceding predicted state. This is implemented in
`NeuralTrajectoryObjective._forward_window`,
`dimswe/test2a_trajectory.py:373-432`. H2 is the first objective with genuine
model-generated-state feedback; H5 extends the same recursion.

`GlobalMixedMassMetric` (`test2a_trajectory.py:238-287`) uses a common global
denominator and returns the dual derivative `2 w M(error)/D` (there is no hidden
factor of one half).

## 5. Discrete derivatives

For one child map `x+=F(x,theta)`, the tangent is

\[
\delta x_+=F_x\delta x+F_\theta\delta\theta,
\]

and reverse accumulation is

\[
\lambda=F_x^T\lambda_+,\qquad
g_\theta\mathrel{+}=F_\theta^T\lambda_+.
\]

Firedrake variational/mass operators provide the spatial interpolation,
assembly, solves, and their transposes. JAX supplies local neural/source
`jvp`, `vjp`, and differentiated-VJP actions. The concrete moist implementations
are `JAXMoistEulerHVP.take_tangent_step` (lines 491--547),
`take_parameter_tangent_step` (556--603), `take_adjoint_step_cached`
(605--662), `take_parameter_adjoint_step` (664--678), and the two incremental
adjoint methods (680--829) in `dimswe/jax_moist_hvp.py`.

The complete split composes child tangents forward and child adjoints backward
in `ProductionMTSWESplitHVP` (`dimswe/mtswe_split_hvp.py:921-1332`). Recursive
trajectory gradients reverse over each window in
`NeuralTrajectoryObjective._gradient_window`,
`dimswe/test2a_trajectory.py:486-518`.

## 6. ROL optimization

Neural parameters are JAX pytrees. `PytreeVectorCodec` in
`dimswe/test2a_pyrol.py:39-88` uses JAX flatten/unflatten operations to expose a
single float64 vector to PyROL. `JAXPytreeObjective` and
`CallbackPytreeObjective` implement ROL's `value`, `gradient`, and `hessVec`
contract at lines 90--277. The trajectory-specific adapter is
`TrajectoryPyROLObjective` at `test2a_trajectory.py:554-607`.

Production uses line-search L-BFGS (`build_test2a_lbfgs_parameters`,
`test2a_pyrol.py:294-304`). ROL requests scalar values and flat gradients; the
adapters obtain exact JAX or discrete-adjoint gradients and copy them into the
ROL vector. HVP actions exist and are certified infrastructure, but the final
Test 2A/Test 2B campaign configs set `production_HVP=false`.
