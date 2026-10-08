# DIMSWE development notes

This is a living note for the first writable DIMSWE milestone.  Statements are
labelled as mathematics, demonstrated code behavior, physical interpretation,
or unresolved intent so that current implementation details are not mistaken
for scientific decisions.

## 1. Active state and finite-element spaces

**Demonstrated code behavior.** `AdvDensVariables_CF_Base` puts velocity first
and then all density-like variables in the mixed state
(`dimswe/variables.py:133-146`).  `AdvDensVariables_CF_H1` assigns velocity to
`CGV` and active/CG-inactive densities to `CG`; DG-inactive densities are
explicit DG1 Functions (`dimswe/variables.py:160-173`).  The quadrilateral
de Rham complex constructs spectral `CG(order)`/`CGV(order)` spaces at
`dimswe/meshes.py:84-118`.

- Dry TSWE state: `[v, h, S]`, with vector CG3 velocity and scalar CG3 `h,S`
  in the milestone configuration.
- Moist TSWE state: `[v, h, S, Qv, Qc, Qr]`; the moist constructor makes all
  three water densities DG1 (`dimswe/variables.py:244-247`).
- Hyperviscosity adds diagnostic Laplacian variables for `[v,h,S]` in matching
  spaces (`dimswe/dissipation.py:61-74`).
- Trainable hyperviscosity coefficients are scalar `R0` Functions ordered
  `[s,c0]` (`dimswe/dissipation.py:29-34,50-56`).

## 2. Continuous subsystem equations

**Mathematical reading of the implemented weak form.** Let

\[
\zeta=\partial_xv_y-\partial_yv_x,\qquad
F=hv,
\]

and let `rho` range over the active densities `h,S`.  On a periodic smooth
domain, the split volume form in `dimswe/dynamics.py:468-497` is consistent
with

\[
\begin{aligned}
v_t +(f+\zeta)v^\perp+\nabla B_h+\frac{S}{h}\nabla B_S &=0,\\
h_t+\nabla\!\cdot(hv)&=0,\\
S_t+\nabla\!\cdot(Sv)&=0,
\end{aligned}
\]

before adding split children.  The code directly demonstrates the weak form;
the displayed strong form additionally assumes the periodic integration by
parts and smooth product rules.

The moist child adds local phase conversion.  With condensation `C=Dqv`,
evaporation `E=Dqc`, rain conversion `R=Dqr`, and
`beta2=g*L`, `dimswe/physics.py:68-104` gives

\[
Q_{v,t}=h(E-C),\quad Q_{c,t}=h(C-E-R),\quad
Q_{r,t}=hR,\quad S_t=h\beta_2(E-C).
\]

**Physical interpretation.** `Qv,Qc,Qr` are depth-weighted vapour, cloud, and
rain quantities; `S` is the thermal density.  The max/min rate definitions at
`dimswe/physics.py:83-90` make the map nonsmooth on switching surfaces.  Tests
therefore use states strictly on a selected branch.

**Configured/applied timestep consequence.** On the switch-safe branch used
for characterization, `ThreeWayPhysics` divides the active rates by its
configured timestep while Euler multiplies the assembled tendency by the
independently applied timestep.  For a branch-fixed state this gives

\[
\Delta Q_v =
\frac{dt_{\mathrm{applied}}}{dt_{\mathrm{configured}}}G(Q_v,Q_c,Q_r,S,h).
\]

The non-gating outside-sandbox run measured integrated vapour increments
`-1.339285714285711e13`, `-6.696428571428555e12`, and
`-2.678571428571422e13` for `(configured, applied)` values `(100,100)`,
`(100,50)`, and `(50,100)`, respectively.  Their measured ratios to the first
case were `1.0`, `0.5`, and `2.0`, confirming the implemented scaling without
changing or endorsing its physics.

## 3. Hamiltonian and derivatives

**Mathematics implemented in code.** The continuous-form Hamiltonian density
is

\[
\mathcal H(v,h,S)=\tfrac12h|v|^2+\tfrac12hS+hB.
\]

`ThermalShallowWater_Hamiltonian_CF.compute_total_energy` implements this at
`dimswe/hamiltonians.py:136-143`.  Its derivative expressions are

\[
F=\frac{\delta H}{\delta v}=hv,\qquad
B_h=\frac{\delta H}{\delta h}=\tfrac12|v|^2+\tfrac12S+B,\qquad
B_S=\frac{\delta H}{\delta S}=\tfrac12h,
\]

at `dimswe/hamiltonians.py:145-157`.

**Unresolved intent.** The Hamiltonian owns a topography Function with an
initializer (`dimswe/hamiltonians.py:86-94`), but the active model initializer
does not call it.  The TC5 characterization therefore found this `B` equal to
zero while the dynamics-owned topography was nonzero.  See
`docs/DIMSWE_CHARACTERIZATION.md`.

## 4. PDE to weak form to semidiscrete system

**Transformation.** Write the strong split PDE abstractly as

\[
x_t+\mathcal F(x;\theta)=0.
\]

Testing each component and integrating the configured spatial terms produces
the UFL residual `R(x;theta)`.  `AdvDensCF_H1_Dynamics.rhs` selects the model
and forcing residuals at `dimswe/dynamics.py:499-507`.  `GeneralRK` then forms
each stage with an explicit leading minus sign and the mass pairing at
`dimswe/timestepping.py:149-172`.  Hence the assembled semidiscrete equation
is

\[
M\dot x+R(x;\theta)=0,
\]

and an explicit stage solves `M F_i = -R(x_i;theta)`.  The RK update
`x_{n+1}=x_n+dt*sum(b_i F_i)` is at
`dimswe/timestepping.py:346-369`.

This sign chain explains why the negative weak source terms in
`ThreeWayPhysics.rhs` become the positive tendencies written in Section 2.

## 5. Split timestep

**Demonstrated code behavior.** `LieSplittingIntegrator` constructs one RK
child per configured term list and advances them sequentially, including
subcycles, at `dimswe/timestepping.py:494-537`.

- Tiny dry ROL map: model/RK4, then hyperviscosity/Euler
  (`tests/tswe_rol_small.cfg:43-53`).
- Tiny moist map: model/RK4, hyperviscosity/Euler, DG transport/SSPRK43, then
  phase conversion/Euler (`tests/mtswe_small.cfg:48-59`).

**Mathematical consequence.** This is an ordered Lie composition; swapping a
child or changing limiter placement changes the method.  No such change was
made in this milestone.

## 6. Moist conversion invariants

**Derivation.** Adding the three water tendencies gives

\[
\partial_t(Q_v+Q_c+Q_r)=h[(E-C)+(C-E-R)+R]=0.
\]

Likewise,

\[
\partial_t(S-\beta_2Q_v)=h\beta_2(E-C)
-\beta_2h(E-C)=0.
\]

Because `S` is CG3 and the water variables are DG1, the permanent tests
assemble these cross-space quantities as integrals instead of comparing
coefficients (`tests/test_mtswe_baseline.py:56-64`).

**Demonstrated result.** On the switch-safe nonzero conversion state, total
water remained `7.874999999999995e13`, the thermal-vapour integral remained
`1.7834953499999994e17`, and both measured differences were exactly `0.0`.
The `h` and `v` L2 changes were `0.0`, while the `Qv` L2 change was nonzero
(`2.678571428571422e6`), excluding a vacuous no-update result.  The complete
one-step map was finite and repeated bit-for-bit; the declared 64-epsilon
absolute envelope reached `1.0629237529771114e-10`.

## 7. Existing discrete-adjoint call flow

**Demonstrated code behavior.** The terminal objective is
`0.5*integral(|x_N-data|^2)` in `L2Objective.evaluate`
(`dimswe/optimize.py:52-79`).  A coefficient value call:

1. maps normalized coefficients to physical values
   (`dimswe/optimize.py:148-153`);
2. resets and recomputes stored forward states
   (`dimswe/optimize.py:154-165`);
3. evaluates the terminal mismatch.

The gradient call resets adjoint storage, repeats the forward trajectory,
projects the terminal mismatch, and sweeps backward
(`dimswe/optimize.py:169-220`).  Each explicit RK child reconstructs forward
stages, solves stage adjoints in reverse, assembles coefficient derivatives,
and advances the state adjoint (`dimswe/timestepping.py:375-423`).  The Lie
split adjoint reverses both child and subcycle order at
`dimswe/timestepping.py:539-574`.

**Boundary honored.** The existing adjoint was reused unchanged.  The existing
empty `hessp` at `dimswe/optimize.py:236-237` was neither exposed nor called.

## 8. Normalized coefficient coordinates

**Implemented transformation.** Hyperviscosity declares physical coefficient
order `[s,c0]`, scaling `d=[3.2,0.07]`, and physical bounds
`[2,0.01] <= [s,c0] <= [4,2]`
(`dimswe/dissipation.py:39-56`).  The existing reduced objective consumes
normalized coefficients through `theta=d*theta_hat`, and its adjoint returns
the chain-rule-scaled gradient (`dimswe/optimize.py:148-153,217-219`).

For the one-scalar adapter,

\[
c_0=d_{c0}z,\qquad
\frac{dJ}{dz}=d_{c0}\frac{dJ}{dc_0},\qquad
z_l=\frac{c_{0,l}}{d_{c0}},\quad
z_u=\frac{c_{0,u}}{d_{c0}}.
\]

Thus `d_c0=0.07`, `z_l=0.14285714285714285`, and
`z_u=28.57142857142857`.  `s=3.2` is held at normalized value `1.0`.

## 9. ROL value and gradient call flow

**Demonstrated code behavior.** `ScalarC0Objective` validates coefficient order
and scaling at `dimswe/rol_adapter.py:58-85`.  For every value or gradient
call it packs `[fixed_s_normalized,z]` at
`dimswe/rol_adapter.py:87-93`.  `value` delegates directly to the existing
reduced objective (`dimswe/rol_adapter.py:95-100`); `gradient` delegates to the
existing discrete adjoint, overwrites the one-element output, and selects only
entry 1 (`dimswe/rol_adapter.py:102-114`).  Counters and histories are updated
on both paths.  Bounds and first-order L-BFGS parameters are built at
`dimswe/rol_adapter.py:8-55`.

No distributed vector, Hessian callback, forward duplicate, or adjoint
duplicate exists in this path.  The older SciPy path is untouched.

## 10. Milestone predictions and actual results

| Prediction derived before testing | Actual result |
| --- | --- |
| Moist total-water and thermal-vapour integrals cancel algebraically | Both differences exactly `0.0` |
| Moist-only update leaves `h,v` unchanged | Both L2 changes exactly `0.0` |
| Branch-fixed moist increment scales as applied/configured `dt` | Measured ratios `1.0`, `0.5`, and `2.0` for `(100,100)`, `(100,50)`, and `(50,100)` |
| Tiny full MTSWE map is finite and repeatable near solver roundoff | All fields finite; repeat difference `0.0` |
| Adapter equals direct objective/adjoint | Value `2.9293330343144117e13` and gradient `-4.556740275600571e13` matched exactly |
| Centered difference agrees with scalar adjoint | FD `-4.556740275508789e13`; relative error `2.0142037004271277e-11` |
| Sampled profile identifies a useful one-dimensional truth | Unique sampled minimum `J=0` at `c0=0.14` |
| Bounded first-order ROL decreases the objective and moves toward truth | `c0: 0.02 -> 0.13999999999999957`; `J: 5.207703172114747e13 -> 9.028607444196265e-18` |

ROL used six iterations, 17 value calls, and seven gradient calls during the
recorded solve/final check.  It exited on step tolerance.  Exact recovery is
not a general contract; the test asserts only decrease, feasibility, and
movement toward truth, supported here by the sampled local profile.

**Final verification.** The authoritative 2026-08-04 outside-sandbox full
suite completed in `469.66 s` with `31 passed`, one separately guarded
optional Matplotlib plotting skip, one pre-existing ODE optimization xfail,
and `12200 warnings`.  The optional JAX and PyROL tests ran without skipping.
The permanent MTSWE invariant, ROL optimization, adjoint, and
finite-difference tests all passed without weakening their assertions.

## 11. Unresolved characterization findings

**Current observations, not specifications:**

- active split step did not call the DG limiter hook;
- Hamiltonian topography remained zero for a nonzero TC5 mountain;
- on one switch-safe active branch, the moist conversion update has the
  explicit scaling `applied_dt/configured_dt`; measured increment ratios were
  `1.0`, `0.5`, and `2.0` for `(100,100)`, `(100,50)`, and `(50,100)`;
- isolated periodic DG transport changed a cosine `Qr` field;
- the tested hyperviscosity mode amplification was `0.9999894512733702`.

The code evidence, possible intended behavior, and required author decisions
are detailed in `docs/DIMSWE_CHARACTERIZATION.md`.  No discrepancy was fixed.

## 12. Planned stages

The next planned work, in dependency order, is:

1. NumPy ODE HVP prototype;
2. exact dry DIMSWE HVP;
3. ROL HVP callback;
4. physics-interface refactor;
5. analytic JAX physics;
6. invariant-preserving neural physics.

None of these stages, including HVP or JAX-physics work, was started in this
milestone.
