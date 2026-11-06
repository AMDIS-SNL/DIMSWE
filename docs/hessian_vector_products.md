# Hessian-vector products in DIMSWE: design notes

This note covers steps 1 and 2 of the plan in `TODO.md`.

1. It maps the existing first-order optimization path in `dimswe/optimize.py` and
   `dimswe/timestepping.py`.
2. It translates the forward-over-adjoint Hessian-vector product of `petsc-opt.pdf`
   (Zhang, Constantinescu and Smith, *PETSc TSAdjoint*, eqs. 2.9-2.11, 2.16) into DIMSWE's
   discrete objects and names.

Later steps (timestepper API, storage, implementation, tests) should refer back to the
notation defined here. Names marked *(new)* do not exist yet. They are proposals for step 3
onward.

---

## 1. The existing first-order path

### 1.1 Call chain

```
scipy.optimize.minimize(problem.fun, z0, jac=problem.jac, ...)       _ODEConstrainedOptimization.optimize
  ReducedProblem.fun(z) / .jac(z)                                    memoized on last z, counts nfev / njev
    x_full = to_full(z)          x_full = x0_full; x_full[free] = z * scale[free]
    full_fun(x_full) / full_jac(x_full)                              lambdas built in reduced_problem(opt_type)
      'coeffs'     -> obj/jac(coeffs, None)
      'ics'        -> obj/jac(None, ics.reshape(nblocks, local_x_size), coeffs0)
      'coeffs+ics' -> obj/jac(x[:ncoeff], x[ncoeff:].reshape(...))
      'ic-params'  -> obj(None, ic_map.forward(p)[None,:]);  jac = ic_map.backward(jac(None, ic_map.forward(p)[None,:]))
        Lagrangian_ODEConstrainedOptimization.jac(coeffs_arr, ics_arr, coeffs0)
    g_z = g_full[free] * scale[free]                                 chain rule through the scaling
```

### 1.2 `obj` (forward only)

For each block `i`, the code does the following:

1. Set the coefficients with `_set_coeffs`, which calls `timestepper.set_numpy_coeff` or `set_coeff(coeffs0)`.
2. Call `reset_internal_vars()`.
3. Call `_compute_block_states(i, ics_arr)`, which runs `compute_states`. That stores **every**
   step `x_0..x_N` in `self.states[n]` (the `[x, aux]` list) and `self.tns[n]`. The start state is
   either `data_blocks[i][0]` or `ics_arr[i]` (copied through `self.ic`).
4. For each `step in obs_steps`, call `_assign_mismatch` (`mismatch = states[step][0] - data`) and
   then `assemble(loss_form)`.

Storage is full forward storage. `self.states` is reused for every block. Only the x-part
`states[n][0]` enters the loss. The aux part `states[n][1]` is just an initial guess, because for
explicit RK the aux variables are recomputed from `x` in every stage.

### 1.3 `jac` (forward + reverse sweep)

For each block `i`:

1. Zero `lambda_np1` and `ic`, call `reset_internal_vars()`, and run the forward pass (same as `obj`).
2. **Observation sources.** For each `k, step`, assign the mismatch and solve `lambda_np1_projector`.
   That gives `lambda_source = M^{-1} s_step`, which is stored in `obs_sources[k]`, where
   `s_step = assemble(source_form)` is a dual vector. `s_step` is also added straight into
   `grad_ics[i]`. All of this happens before the reverse sweep because the sweep is not guaranteed
   to leave `self.states` untouched.
3. **Reverse sweep** for `n = N..1`:
   - If `n` is an observation step: `lambda_np1 += obs_sources[obs_index[n]]`.
   - `reset_internal_vars()`.
   - `take_adjoint_step(delta_grad_coeff, delta_lambda, lambda_np1, states[n-1], tns[n], dt)`. This
     recomputes the forward stages from `x_{n-1}` (recompute-from-stored-state checkpointing), then
     solves the stage adjoints and returns `(delta_lambda_rhs, delta_grad_rhs)` as flattened dual
     arrays. It also sets `delta_lambda = M^{-1} delta_lambda_rhs`.
   - `lambda_np1 += delta_lambda`, `grad_coeffs += delta_grad_rhs`, `grad_ics[i] += delta_lambda_rhs`.
4. Return `grad_coeffs`, `ravel(grad_ics)`, or their `hstack`, depending on which of
   `coeffs_arr` / `ics_arr` is `None`.

After the sweep, `grad_ics[i]` is the **dual** vector `M lambda_0`, built by telescoping:
`M lambda_0 = sum_obs s_n + sum_n delta_lambda_rhs_n`. `grad_coeffs` is summed over blocks.

### 1.4 Inside `GeneralRK` (explicit RK only; DIRK/IRK raise `NotImplementedError`)

Forward step `x_{n+1} = N(x_n; theta)`. The stage values are `F_i = Fi[i][0][0]` and
`W_i = Fi[i][0][1]` (aux).

```
xi_i     = x_n + dt * sum_{j<i} A_ij F_j                 (xi_splits[i]; aux slot replaced by W_i, t -> t + c_i dt)
L_W W_i  = g_i(xi_i, theta)                              auxsolvers[i]   (residuals_aux[i], constant Jacobian)
M F_i    = f_i(xi_i, W_i, theta)                         Fsolvers[i]     (residuals_F[i],  f_i = -model.rhs at stage i)
x_{n+1}  = x_n + dt * sum_i b_i F_i
```

`f_i` and `g_i` are UFL 1-forms in the test function `xhat`. `M` is the x mass matrix.

First-order adjoint step (`take_adjoint_step`). The stage adjoints are `mu_i = mui[i][0][0]`
and `nu_i = mui[i][0][1]` (aux). Stage `i` is processed from last to first:

```
M mu_i     = dt b_i M lambda_{n+1} + sum_{j>i} (df_j/dF_i)^T mu_j + sum_{j>i} (dg_j/dF_i)^T nu_j    musolvers[i]
M_W nu_i   = (df_i/dW_i)^T mu_i                                                                     muauxsolvers[i]
r^lam_n    = sum_i (df_i/dx)^T mu_i + (dg_i/dx)^T nu_i          L_delta_lambda  -> delta_lambda_rhs (dual)
r^theta_n  = sum_i (df_i/dtheta)^T mu_i + (dg_i/dtheta)^T nu_i  L_delta_grad    -> delta_grad_rhs   (dual)
M delta_lambda = r^lam_n                                        deltalambdasolver
```

The `muaux` residual uses `inner(nu_i, xhat[1])`, a mass matrix. So the code assumes the aux
LHS operator `L_W` is the aux mass matrix `M_W` (see the comment "THIS IS ASSUMING THAT LHS W IS
LINEAR IN W"). The HVP inherits this assumption.

`LieSplittingIntegrator` composes `GeneralRK` sub-integrators. Its forward pass runs the substeps
and stores the substep states in `self.xks`. Its adjoint pass recomputes those substeps and then
calls each sub-integrator's `take_adjoint_step` in reverse. It carries the substep adjoint in
`lambda_k` and sums the dual increments.

### 1.5 Where HVPs plug in (no second optimization stack)

| Layer | First-order today | HVP counterpart |
|---|---|---|
| scipy | `jac=problem.jac` | `hessp=problem.hessp` (step 7) |
| `ReducedProblem` | `jac(z)`: `g_full[free]*scale[free]`, `njev` | `hessp(z, v_z)` *(new)*: `v_full = 0; v_full[free] = scale[free]*v_z`; return `(H v_full)[free]*scale[free]`; `nhvp` (step 6) |
| `reduced_problem(opt_type)` | builds `full_jac` | also builds `full_hessp(x_full, v_full)` *(new)*, with the same reshapes per `opt_type` |
| `Lagrangian_ODEConstrainedOptimization` | `jac(coeffs, ics)` | `hessp(coeffs, ics, coeff_dir, ics_dir)` *(new)*, replacing the placeholder |
| timestepper | `take_forward_step`, `take_adjoint_step` | `take_tangent_step`, `take_second_order_adjoint_step` *(new)* (step 3) |

Notes for later steps, found while mapping the code:

- **Scaling.** The reduced Hessian is `H_z = S_f P^T H_full P S_f`, where `P` puts the free
  entries into full length and `S_f = diag(scale[free])`. Fixed entries get a zero direction.
- **Memoization.** Newton-CG and the trust-region methods call `hessp(z, ·)` many times at one
  `z`. The forward states `x_n` depend only on `z`, but the reverse quantities in an HVP also depend
  on the direction. The first implementation should recompute everything per call. Caching
  `x_n` and `lambda_n` per `z` is a step 20 optimization.
- **`reset_internal_vars()` repeatability.** `jac` calls it before every block and every reverse
  step. The HVP sweeps must do the same, and must also reset any new tangent and second-order
  stage variables.
- **`take_adjoint_step` recomputes the stages** from `x_n`. It does not reuse whatever stage
  values the forward pass left behind. The TLM and SOA steps should follow the same rule.
- **`grad_ics` is always computed**, even for `opt_type='coeffs'`, because it is cheap. The
  same goes for `H_x0 v`.
- **Coefficient parallel convention.** The coefficient-block HVP must be summed, and MPI-reduced
  if at all, exactly like `delta_grad_rhs` in `jac`. Then the gradient and the HVP match in parallel.
- **No observations at step 0.** `obs_steps` lie in `1..N`, so there is no direct loss term on `x_0`.

---

## 2. PDF eq. 2.11 to DIMSWE

### 2.1 Correspondence table

The paper uses Euclidean column vectors with transposes. DIMSWE stores adjoints as **primal
Riesz representers**: `lambda` is a `Function`, and the dual (cofunction) vector is `M lambda`.

| `petsc-opt.pdf` | Meaning | DIMSWE equivalent |
|---|---|---|
| `u_n` | state at step n | `self.states[n][0]` (x-part only) |
| `N(u_n)` (2.2) | one time step | `timestepper.take_forward_step` (explicit RK stages `F_i`, `W_i` inside) |
| `eta` | initial condition | `ics_arr[b]`, per block `b`, copied through `self.ic` |
| `p` (2.12, augmented) | parameters | coefficients `theta`: `timestepper.coeff`, set by `set_numpy_coeff` |
| `eta` for `ic-params` | — | `x0 = ic_map.forward(p)`: an L2 projection of UFL expressions in `p` |
| `psi` | objective | `J = sum_b sum_{n in obs_steps} l(x_n^b - d_n^b)`, with `l = loss_form` |
| `(dpsi/du)^T` | terminal / obs adjoint source | `s_n = assemble(source_form)` (dual). `M^{-1} s_n = obs_sources[k]` |
| integral `r(t,u)` (2.3) | trajectory loss | the discrete sum over `obs_steps`, added as a source in the reverse sweep instead of the paper's augmented `q` |
| `lambda_n` (2.6) | first-order adjoint | `M lambda_n` (dual), where DIMSWE `lambda_n` = `lambda_np1` during the sweep |
| `(dN/du)^T lambda_{n+1}` | adjoint step | `lambda_{n+1} + M^{-1} r^lam_n`, from `take_adjoint_step` plus `lambda_np1 += delta_lambda` |
| `mu_n` (2.16, parameter adjoint) | partial gradient wrt `p` | running sum of `delta_grad_rhs`, i.e. `self.grad_coeffs` |
| stage adjoints (not named in the paper) | RK internals | `mu_i = mui[i][0][0]`, `nu_i = mui[i][0][1]`. **Do not confuse these with the paper's `mu_n`.** |
| `v` | direction | full-space `v = (v_theta, v_x0^1..v_x0^B)`, from reduced `v_z` via `v[free] = scale[free] * v_z` |
| `(du_n/deta) v` (boxed) | TLM state | `w_n` *(new)*: `self.tangent_states[n]` |
| `p`-part of the TLM | — | `theta_dot = v_theta`, constant in time (augmented `p' = 0`) |
| `(dlambda_n/deta) v` | second-order adjoint | `lambda_dot_n` *(new)*, primal. Dual is `M lambda_dot_n` |
| `lambda_{n+1}^T (d^2N/du^2)(u_n) w_n` | vector-Hessian-vector term | second derivatives of the stage forms `f_i`, `g_i`, contracted with `mu_i`, `nu_i` and the stage tangents (sec. 2.3) |
| `d/du (dpsi/du)^T w_N` | loss second variation | `sdot_n = source_form` with `mismatch -> w_n` (sec. 2.4) |
| `d lambda_0/deta v = nabla^2 psi v` | HVP result, IC part | `H_x0 v = M lambda_dot_0` (dual, rank-local, like `grad_ics`) |
| — (augmented `p` rows) | HVP result, coefficient part | `H_theta v = sum_b sum_n rdot^theta_n` (like `grad_coeffs`) |

### 2.2 Tangent-linear (TLM) step, the boxed terms

This step differentiates the forward step in the direction `(w_n, theta_dot)`. The proposed
variables are `Fdot_i`, `Wdot_i` *(new)* in `GeneralRK`.

```
w_0^b       = v_x0^b                    ('ics', 'coeffs+ics')
            = T(p) v_p                  ('ic-params', T = ic_map.tangents() stacked as columns)
            = 0                         ('coeffs')
xidot_i     = w_n + dt * sum_{j<i} A_ij Fdot_j
M_W Wdot_i  = dg_i/dxi [xidot_i] + dg_i/dtheta [theta_dot]
M Fdot_i    = df_i/dxi [xidot_i] + df_i/dW [Wdot_i] + df_i/dtheta [theta_dot]
w_{n+1}     = w_n + dt * sum_i b_i Fdot_i
```

These are linear solves with the same operators `M` and `M_W` as `Fsolvers` and `auxsolvers`. The
RHS is `ufl.derivative(f_i, [xk, F_0..F_{s-1}, W_i, coeff], [w, Fdot_0.., Wdot_i, theta_dot])`.
The stage forms depend on `x_n` and `F_j` only through `xi_i`, so UFL applies the chain rule if we
differentiate with respect to the `Function`s `xk` and `Fi[j]` themselves. The forward stages at
`x_n` must be current, which needs the same recompute as in `take_adjoint_step`.

### 2.3 Second-order adjoint (SOA) step, the forward-over-adjoint part of eq. 2.11

Write the first-order adjoint of one step through a scalar stage functional

```
Phi_n(x, F, W, theta; mu, nu) = sum_i  f_i(xi_i, W_i, theta)[mu_i] + g_i(xi_i, theta)[nu_i]
```

This is the 1-forms `f_i` and `g_i` acted on the stage adjoints, i.e. `action(f_i, mu_i)`. Every
first-order adjoint form in `GeneralRK.__init__` is a partial derivative of `Phi_n`:

```
M mu_i    = dt b_i M lambda_{n+1} + dPhi/dF_i        M_W nu_i = dPhi/dW_i
r^lam_n   = dPhi/dx                                  r^theta_n = dPhi/dtheta
```

Differentiating these in the direction of the full tangent `tau = (w_n, Fdot, Wdot, theta_dot)` and
the adjoint tangents `(mu_dot, nu_dot)` gives the SOA step. The proposed variables are `mu_dot_i`,
`nu_dot_i`, `lambda_dot`, `delta_lambda_dot` *(new)*:

```
M mu_dot_i   = dt b_i M lambda_dot_{n+1} + dPhi/dF_i [mu_dot, nu_dot] + D(dPhi/dF_i)[tau]
M_W nu_dot_i = dPhi/dW_i [mu_dot]        + D(dPhi/dW_i)[tau]
rdot^lam_n   = dPhi/dx [mu_dot, nu_dot]  + D(dPhi/dx)[tau]          M delta_lambda_dot = rdot^lam_n
rdot^theta_n = dPhi/dtheta [mu_dot, nu_dot] + D(dPhi/dtheta)[tau]
lambda_dot_n = lambda_dot_{n+1} + delta_lambda_dot   (+ M^{-1} sdot_n at observation steps)
```

- The `[mu_dot, nu_dot]` terms are the **existing** first-order adjoint forms with
  `mu -> mu_dot` and `nu -> nu_dot` (they are linear in the adjoints). So `residuals_mu`,
  `residuals_muaux`, `residual_xk` and `residual_grad` can be reused through `ufl.replace`, and the
  same `M` / `M_W` solvers apply. This matches eq. 2.11's `(dN/du)^T (dlambda_{n+1}/deta) v`.
- The `D(...)[tau]` terms are the paper's `lambda_{n+1}^T d^2N/du^2 (du_n/deta) v`, with the RK
  stages expanded. In UFL they are
  `derivative(derivative(Phi_n, Y, test), [xk, F_*, W_*, coeff], tau)` for `Y` in
  `{F_i, W_i, xk, coeff}`. They are matrix-free vector-Hessian-vector contractions, which is what
  step 3 asks for.
- `M` and `M_W` are state independent, so they contribute no second-derivative terms.
- With coefficients, `tau` includes `theta_dot`. This gives the mixed `theta`-`x` Hessian blocks
  automatically, which `coeffs+ics` needs. That is the augmented-system argument of sec. 2.3 of
  the paper applied to `theta`.

### 2.4 Loss second derivatives

Both objectives are quadratic in the mismatch, so `d^2 l` is constant and `source_form` is linear
in `mismatch`:

```
sdot_n = replace(source_form, {mismatch: w_n})
       = loss_scale * sum_var weight_var * inner(xtest_var, w_var) dx     (ObservationL2Objective)
       = inner(xtest, w_n) dx                                              (L2Objective)
```

So `M^{-1} sdot_n` is just `w_n` reweighted per variable. In the unweighted, unscaled case it is
exactly `w_n`. Like `obs_sources`, it is added to `lambda_dot` at step `n`, and `sdot_n` is added to
`H_x0 v`. A future non-quadratic objective needs a new `_Objective.build_forms` output, a
`hessian_form(mismatch, direction)`, rather than this replacement.

### 2.5 Assembling the full-space HVP

For each block `b`, run the TLM forward, then a combined reverse sweep that advances `lambda` and
`lambda_dot` together. At step `n` the SOA needs `x_{n-1}`, `w_{n-1}`, the first-order stage
adjoints `mu_i` and `nu_i` built from `lambda_n`, and `lambda_dot_n`. The results are

```
H_x0^b v   = M lambda_dot_0^b = sum_obs sdot_n + sum_n rdot^lam_n          (dual, rank-local)
H_theta v  = sum_b sum_n rdot^theta_n
```

Return them in the same layout as `jac`: `coeffs`, `ravel(ics)`, or the `hstack` of both.

**`ic-params`** needs one extra term that the first-order path does not have, because `x0(p)` is
nonlinear in `p` (it is linear only in the projected expression):

```
H_p v = T^T (H_x0 (T v))  +  [ (d^2 x0/dp_j dp_k) v_k ]_j^T (M lambda_0)
```

The second term is the projection of the second Gateaux derivative of the IC expressions,
paired with `grad_ics` and `allreduce`d like `ParametricInitialCondition.backward`. This needs
a new `ParametricInitialCondition` method that builds second-derivative expressions with nested
`ufl.derivative`. Dropping the term gives a Gauss-Newton-like approximation, not the true HVP.

**`LieSplittingIntegrator`.** `N = N_S ∘ ... ∘ N_1`, so TLM and SOA compose the same way the
existing adjoint does. The TLM runs forward over the substeps and stores the substep tangents next
to `xks`. The SOA runs backward and carries `lambda_k` and a new `lambda_dot_k` through the
substeps, calling each sub-integrator's SOA step.

### 2.6 What steps 3-5 need from this

- Per `GeneralRK`: new tangent stage variables (`Fdot_i`, `Wdot_i`), adjoint-tangent stage
  variables (`mu_dot_i`, `nu_dot_i`), a coefficient direction Function `coeff_dir`, an `xk`
  tangent `w_var`, and a `lambda_dot_var`. Forms are built once in `__init__` from `Phi_n` via
  `ufl.derivative` and `replace`. Solvers reuse the constant-Jacobian `M` / `M_W` configurations.
- Per step, the SOA needs **simultaneously**: `x_n` (stored), `w_n` (store the TLM trajectory
  like `self.states`, which is the simplest choice for step 4), and the first-order adjoint
  `lambda_{n+1}` and stage adjoints `mu_i`, `nu_i` (recomputed in the same reverse sweep, so
  there is no extra storage).
- Supported scope for the first implementation: explicit `GeneralRK` methods (`Euler`, `RK4`,
  `SSPRK3`, `SSPRK43`) and `LieSplit` built from them. That is the same scope as `jac`.

---

## 3. Timestepper API (step 3, implemented)

The second-order API is additive. The existing `take_forward_step` / `take_adjoint_step` are unchanged except that `GeneralRK.take_adjoint_step` now uses the shared `_solve_forward_stages` / `_solve_first_order_adjoint_stages` helpers. The new methods exist on both `GeneralRK` (explicit only) and `LieSplittingIntegrator`.

| Method | Meaning |
|---|---|
| `set_coeff_direction(f)` / `set_numpy_coeff_direction(arr)` | set `theta_dot` (`None` zeroes it). It is an input like `coeff`, so `reset_internal_vars` leaves it alone |
| `take_tangent_step(wnp1, wn, xn, tn, dt, xnp1=None)` | `w_{n+1} = dN/dx w_n + dN/dtheta theta_dot`. If `xnp1` is given it also writes `x_{n+1}` |
| `take_second_order_adjoint_step(delta_lambda, delta_lambda_dot, lambda_np1, lambda_dot_np1, xn, wn, tnp1, dt)` | combined first- and second-order adjoint step. It returns `(delta_lambda_rhs, delta_grad_rhs, delta_lambda_dot_rhs, delta_hess_coeff_rhs)`. The first two equal `take_adjoint_step`'s outputs, the last two are this step's contributions to `H_x0 v` and `H_theta v` |

**Construction.** The tangent-linear and second-order forms are not hand-derived. They are the `directional_derivative` of the existing residual forms:

- `residuals_F` and `residuals_aux` are differentiated in all forward directions `(xk, F_j, W_j, coeff) -> (wk, Fdot_j, Wdot_j, coeff_dir)`.
- `residuals_mu`, `residuals_muaux`, `residual_xk` and `residual_grad` are differentiated in the forward directions plus the adjoint directions `(lambda, mu_j, nu_j, delta_lambda, delta_grad) -> (lambda_dot, mu_dot_j, nu_dot_j, delta_lambda_dot, delta_grad_dot)`.

The resulting HVP is therefore the exact derivative of the discrete gradient that `jac` computes, including its aux mass-matrix assumption. Each derivative is linear in its own unknown, with the same constant mass operator as the matching first-order solve, so `create_linear_solver_from_residual` and the existing solver-parameter keys are reused. Forms and solvers are built lazily on the first second-order call (`_setup_second_order`), so plain forward runs pay nothing.

**LieSplit time fix.** `LieSplittingIntegrator.take_adjoint_step` used to pass the substep *start* time `tnp1 - dt + k*sub_dt` as the sub-integrator's end time, so the adjoint was evaluated one substep early. This only matters for time-dependent right-hand sides. It now passes `tnp1 - dt + (k+1)*sub_dt`, consistent with `take_forward_step` and with the new second-order step.

**Tests.** `tests/unit/test_timestepping_second_order.py` runs a Taylor test of each step against the first-order step it differentiates, for the `tests/tswe_small.cfg` LieSplit and for plain RK4, with state-only, coefficient-only and combined directions.

---

## 4. Storage and recomputation (step 4, decided and implemented)

### 4.1 What each reverse step needs

`take_second_order_adjoint_step` for step `n -> n-1` (code index `n = N..1`) needs:

| Quantity | Source |
|---|---|
| `x_{n-1}` | stored: `self.states[n-1]` (existing full forward storage) |
| `w_{n-1}` | **stored**: `self.tangent_states[n-1]` *(new)*, x-space only |
| `lambda_n` | running buffer `self.lambda_np1`, advanced in the same sweep |
| `lambda_dot_n` | running buffer `self.lambda_dot_np1` *(new)* |
| `theta`, `theta_dot` | `timestepper.coeff`, `timestepper.coeff_dir` (constant over the sweep) |
| `M^{-1} s_n`, `M^{-1} sdot_n` at observation steps | precomputed: `self.obs_sources`, `self.obs_tangent_sources` *(new)* |
| forward stages `F_i, W_i`, tangent stages, `mu_i, nu_i`, `mu_dot_i, nu_dot_i` | **recomputed** inside the step from the rows above, never stored |

### 4.2 Decisions

1. **Store the full tangent trajectory.** This matches the existing full-state storage. Recomputing `w_{n-1}` from `w_0` at every reverse step would cost `O(N^2)` tangent steps, and checkpointing is out of scope (step 20). Memory per block becomes `2(N+1)` x-space Functions instead of `N+1`. Aux tangents are never stored, since aux variables are recomputed in every stage. So `create_tangent_states` uses one-element lists `[w_n]` rather than `[x, aux]` pairs.
2. **The HVP forward pass computes states and tangents together.** `compute_states_and_tangents` calls `take_tangent_step(..., xnp1=states[n+1])`, so the state trajectory costs no extra forward solves. It calls `reset_internal_vars()` before every step, exactly like `compute_states`.
3. **One combined reverse sweep.** `lambda` and `lambda_dot` are advanced together, because the second-order step needs `lambda_n` and the first-order stage adjoints at the same time. So there is no separate first-order sweep, and no first-order adjoint storage. As a by-product, each HVP also yields the gradient and `M lambda_0`. The `ic-params` curvature term needs `M lambda_0`, and a later optimization could use the gradient to fill `ReducedProblem`'s jac cache.
4. **Precompute the observation sources before the reverse sweep**, as `jac` does, because the sweep may overwrite `self.states`. `_compute_observation_sources` fills both source lists and returns the dual sums `(sum s_n, sum sdot_n)`. `sdot_n` is built as `derivative(source_form, mismatch, tangent_mismatch)`. That equals `source_form` with the mismatch replaced by `w_n` for the current quadratic losses, but stays exact for any loss whose source form depends on the state only through the mismatch.
5. **Lazy allocation.** `_setup_hvp()` creates all HVP storage and the tangent-source projector on first use. Runs that only use first-order methods allocate nothing extra.
6. **No cross-call caching yet.** Each HVP call redoes the forward pass, even though `x_n` depends only on the optimization point. Caching `x_n` per point, or reusing `jac`'s trajectory, belongs to step 20.

**Cost per HVP**, counted in solves per step: forward pass 2 (forward and tangent stages), reverse sweep 4 (forward, tangent, first-order adjoint and second-order adjoint stages). One gradient costs 1 + 2 by the same count, so an HVP costs about 2x a gradient.

**Consistency caveat.** With iterative stage solvers (the production `basic_linear_system` is CG/Jacobi), the stages recomputed in the reverse sweep match the forward pass only to solver tolerance. That is already true of `jac`. `take_tangent_step` also skips the aux initial guess `xn[1]` that `take_forward_step` uses, so with iterative solvers the HVP trajectory differs from `obj`'s at tolerance level. Taylor tests use direct solvers (`tests/gradient_helpers.SOLVER_PARAMETERS`), where all of these agree to round-off.

### 4.3 Implemented helpers (`dimswe/optimize.py`)

- `create_tangent_states(model, nsteps)` and `compute_states_and_tangents(xns, tns, wns, timestepper, nsteps, dt, x0, w0, t0)`: module level, next to `create_states` / `compute_states`.
- `Lagrangian_ODEConstrainedOptimization._setup_hvp()`, `_set_coeff_direction(arr)`, `_compute_block_states_and_tangents(block_id, ics_arr, ics_dir_arr)` and `_compute_observation_sources(block_id)`. Step 5's `hessp` composes these with the combined reverse sweep.

**Tests.** `tests/unit/test_optimize_hvp_storage.py` checks three things: the HVP forward pass reproduces `compute_states`; the stored tangent trajectory passes Taylor tests against the state trajectory for initial-state, coefficient and combined directions; and the observation tangent sources (unweighted and weighted) pass Taylor tests against the first-order sources.

---

## 5. Full-space HVP (step 5, implemented)

In `Lagrangian_ODEConstrainedOptimization` (`dimswe/optimize.py`):

- `_hvp_sweep(coeffs_arr, ics_arr, coeffs_dir_arr, ics_dir_arr, coeffs0)` is the algorithm of sec. 2.5 with the storage of sec. 4. For each block it calls `_compute_block_states_and_tangents`, then `_compute_observation_sources`, then a combined reverse sweep over `take_second_order_adjoint_step`. It returns `(grad_coeffs, grad_ics, hess_coeffs, hess_ics)`; the gradient part matches `jac`. The coefficient direction is cleared in a `finally` block, so a failed call leaves no stale `theta_dot` behind.
- `hessp(coeffs_arr, ics_arr, coeffs_dir_arr=None, ics_dir_arr=None, coeffs0=None)` replaces the old placeholder. It takes the point the same way `jac` does and returns `H v` in `jac`'s layout. Direction parts for variables that are not being optimized are ignored.
- `ic_params_hessp(ic_map, params, direction, coeffs0)` computes `T^T H_x0 (T v) + (d^2 x0/dp^2 [v])^T (M lambda_0)` from one sweep in the direction `T v`, since that sweep also returns `M lambda_0`.
- `full_hessp(opt_type, coeffs0, ic_map)` returns a callable `(x_full, v_full) -> H v_full`, laid out like `reduced_problem`'s `full_fun` / `full_jac` for the same `opt_type`. Step 6 plugs this into `ReducedProblem`.

New methods on `ParametricInitialCondition`:

- `tangent_action(v)`: `T v`, rank-local.
- `second_tangents()`: projections of `d^2 expr / dp_j dp_k`. The symbolic forms are built with nested `ufl.derivative` on first use; only the upper triangle is projected, and entries that are identically zero are skipped.
- `second_order_backward(grad_x0, v)`: the curvature term, `allreduce`d like `backward`.

**Tests.** `tests/unit/test_optimize_hvp.py` covers each opt type with a gradient-difference Taylor test of `H v` (helper `hessian_taylor_test` in `tests/gradient_helpers.py`) and a symmetry test `u.Hv = v.Hu`. It also checks that the sweep's gradient equals `jac`, and that `H v` is linear in `v`.

---

## 6. Reduced-space HVP (step 6, implemented)

`ReducedProblem` now takes an optional `full_hessp(x_full, v_full)`.

- `hessp(z, p)` returns `(H_full(to_full(z)) direction_to_full(p))[free] * scale[free]`. The map `direction_to_full(p)` sets `v[free] = p * scale[free]` and leaves zeros on fixed entries; it is the linear part of `to_full`. So `H_z = S P^T H_full P S`, the second-order counterpart of `jac`'s chain rule.
- `nhvp` counts HVP calls separately from `nfev` and `njev`. HVPs are not memoized, because scipy passes a new direction on every call.
- `has_hessp()` reports whether HVPs are available. `hessp` raises a clear `ValueError` if they are not, or if `full_hessp` returns the wrong shape.
- `_ODEConstrainedOptimization.reduced_problem` passes `self.full_hessp(opt_type, coeffs0, ic_map)` when the class provides it (`Lagrangian_ODEConstrainedOptimization` does). Building the callable compiles nothing; that happens on the first `hessp` call.

**Tests.** `tests/unit/test_reduced_problem_hvp.py` is pure Python, on a quadratic with known Hessian. It checks the exact `S P^T A P S`, agreement with reduced-`jac` differences, the zero direction on fixed entries, evaluation at `to_full(z)`, separate counting, and the error paths. `test_reduced_hessp_taylor_with_scaling_and_free_mask` in `tests/unit/test_optimize_hvp.py` runs a gradient-difference Taylor test in reduced coordinates on the model problem, with scaling and one fixed coefficient.

---

## 7. Second-order methods in `optimize()` (step 7, implemented)

`_ODEConstrainedOptimization.optimize` gains `use_hessp=None`, `use_bounds=True` and `xtol=None`. Its defaults keep every existing call unchanged.

| Method | `hessp` passed | Bounds | Notes |
|---|---|---|---|
| `Newton-CG` | yes (auto) | **rejected** | scipy would ignore bounds. Without `hessp` it falls back to finite differences of `jac`. `xtol` is its only tolerance |
| `trust-ncg`, `trust-krylov` | yes (required) | **rejected** | `gtol` |
| `trust-constr` | yes (auto) | supported | `gtol`, `xtol`. The callback is called as `(xk, state)`, which `scipy_callback(zk, *args)` accepts |
| first-order methods (`L-BFGS-B`, `TNC`, `CG`, ...) | never | unchanged | `CG`/`BFGS` still get scipy's warn-and-ignore behaviour for bounds |

- `resolve_hessp_use(method, use_hessp, use_jacobian, has_hessp, has_bounds)` holds all the validation and raises `ValueError` with a clear message. The error cases are: HVPs requested for a method that doesn't use them or a problem that doesn't provide them; HVPs or `Newton-CG`/trust-region methods without a gradient; `trust-ncg`/`trust-krylov` without HVPs; and bounds with `Newton-CG`/`trust-ncg`/`trust-krylov`. `HESSP_METHODS`, `method_uses_hessp` and `method_supports_bounds` are exported for step 8's workflow validation.
- **Bounds restriction.** `'coeffs'` always applies the model's coefficient bounds when none are given. So unbounded second-order methods need `use_bounds=False`, which optimizes without bounds; otherwise use `trust-constr`.
- `scipy.optimize.minimize` now gets one call, with `jac`/`hessp` added as keyword arguments only when used.
- The returned `OptimizeResult` carries `problem_nfev`, `problem_njev` and `problem_nhvp`. These count actual model evaluations after memoization, because scipy's own counters differ between methods. Verbose output now prints these counts.

**Tests.** `tests/unit/test_optimizer_methods.py` is pure Python. It covers the method capabilities, every `resolve_hessp_use` branch and the second-order `scipy_options`. It also runs `optimize()` end to end on a scaled quadratic for all four HVP methods (convergence to the known minimizer, with `nhvp > 0`), `trust-constr` with bounds and a callback, `L-BFGS-B` never calling `hessp`, and the Newton-CG bounds rejection and its `use_bounds=False` escape.

---

## 8. Workflow optimizer specs (step 8, implemented)

`workflows/optimize_params.py` adds three optimizer keys. Their defaults keep existing configs unchanged.

```yaml
optimizer:
  method: trust-constr   # or Newton-CG / trust-ncg / trust-krylov; L-BFGS-B stays the default
  use_hessp: null        # null: pass HVPs to methods that use them; true: require; false: never
  use_bounds: true       # false: optimize without bounds (including the model's coefficient bounds)
  xtol: null             # Newton-CG / trust-constr step tolerance; null keeps scipy's default
```

`validate_optimizer_spec(optimizer_spec, target, parameters)` runs in `load_spec`, so bad combinations fail before any model is built. It rejects:

- unknown method names, and `dogleg`/`trust-exact`, which need a full Hessian;
- non-boolean `use_hessp`/`use_bounds`;
- `use_hessp: true` with a method that does not use HVPs, and `use_hessp: false` with `trust-ncg`/`trust-krylov`;
- `Newton-CG`/`trust-ncg`/`trust-krylov` on the `coeffs` target while bounds are on. That target is always bounded by the model's coefficient bounds, so the message suggests `trust-constr` or `use_bounds: false`;
- `use_bounds: false` when parameters set explicit bounds, which would otherwise be ignored silently;
- an invalid `xtol`, or `xtol` with a method that does not use it.

First-order methods are only checked for a known name. `ic-params` (bounds are rejected) and `ic-field` (no bounds) accept every HVP method. With `use_bounds: false`, `build_problem` skips the start-inside-bounds checks for coefficients.

**Validation location.** The workflow validates without Firedrake, so it keeps pure-Python copies of the method tables (`HESSP_METHODS`, `BOUNDED_METHODS`, ...). `tests/unit/test_workflow_optimizer_tables.py` keeps them in sync with `dimswe.optimize`, and checks that anything the workflow accepts also passes the runtime `resolve_hessp_use`. `run_job` forwards `use_hessp`, `use_bounds` and `xtol` to `optimize()`, so the runtime check still applies too.

**Tests.** The pure-Python tests in `tests/integration/test_optimization_workflows.py` check that the defaults are unchanged, that valid second-order specs are accepted, and that every rejection gives a clear message. They also run `load_spec` on copies of the example specs switched to `trust-constr` and `Newton-CG`.

---

## 9. HVP metrics in workflow outputs (step 9, implemented)

- **`result.yaml` → `optimizer`** is now built by `optimizer_summary(optimizer_spec, result, problem, gtol)` in `workflows/optimize_params.py`. It holds:
  - the resolved spec: `method`, `use_hessp`, `use_bounds`, `xtol` and the existing keys;
  - the run outcome: `success`, `status`, `message`, `gtol_used` and `nit`;
  - the problem's evaluation counts `nfev`, `njev` and the new `nhvp`. These count actual model evaluations;
  - the new `hessp_used`, which records whether HVPs reached scipy, because `use_hessp: null` resolves per method.

  The top-level `final_loss` is unchanged. Results written before this change lack `nhvp` and `hessp_used`; readers should default them to `0` and `false`.
- **`history.csv`** gets cumulative `nfev`, `njev` and `nhvp` columns, appended after the existing iteration, loss, gradient-norm and parameter columns, so existing consumers are unaffected. `validate.py` and `plot_validation.py` read columns and keys by name, and none of the names changed.
- `optimize()` sets `res.hessp_used`, and the job log prints a per-job evaluation line.
- **`summary.csv`** (written by `validate.py`): `summary_row` appends `method`, `njev`, `nhvp` and `hessp_used` after `nfev` (step 13), so first- and second-order runs can be compared from the sweep table. Older results default to `""`, `0`, `0` and `False`.

**Tests.** Pure-Python tests in `tests/integration/test_optimization_workflows.py` check `optimizer_summary` for a second-order result and for a first-order result without HVP information, and check the `history.csv` column order with and without counters.

---

## 10-12. Test coverage

**Step 10 (pure-Python plumbing)** was complete before this pass. `tests/unit/test_reduced_problem_hvp.py` covers scaling, `free_mask`, counting and error paths. `tests/unit/test_optimizer_methods.py` covers method selection for supported and unsupported second-order methods, and the workflow-level selection tests are in `tests/integration`.

**Step 11 (Firedrake HVP Taylor tests)** is now complete. `tests/gradient_helpers.reduced_hessian_taylor_test` is the HVP counterpart of `reduced_taylor_test`. `tests/unit/test_optimize_hvp.py` mirrors the existing gradient tests in reduced, scaled coordinates:

| Gradient test | HVP test |
|---|---|
| `test_coeff_gradient_taylor[1,5]` (final-state L2) | `test_final_state_coeff_hessp_taylor[1,5]` (with scaling) |
| `test_ic_gradient_taylor[1,5]` | `test_final_state_ic_hessp_taylor[1,5]` |
| `test_trajectory_loss_ic_gradient_with_weights` | `test_weighted_trajectory_ic_hessp_taylor` |
| `test_ic_params_gradient_uses_ufl_tangents` | `test_ic_params_reduced_hessp_taylor` (all free, and `oy` fixed) |
| `test_*_gradient_vanishes_at_truth` | `test_hessp_at_truth_has_positive_curvature` (zero gradient, `v.Hv > 0`) |

The full-space tests (`test_hessp_taylor`, `test_hessp_symmetric`, ...) and `test_reduced_hessp_taylor_with_scaling_and_free_mask` from step 5 remain. The per-step building blocks are tested in `test_timestepping_second_order.py` and `test_optimize_hvp_storage.py`.

**Step 12 (repeatability).**

- Every HVP Taylor test in `test_optimize_hvp.py` runs twice through `assert_repeatable` and requires identical remainders and rates. `assert_repeatable` now passes through the extra `change` entry.
- `tests/unit/test_optimize_hvp_repeatability.py` adds targeted checks, each to round-off (`rtol 1e-12`):
  - an HVP is unchanged by interleaved `obj`/`jac`/`hessp` calls at another point;
  - `obj` and `jac` are unchanged by an HVP, and `theta_dot` is zero afterwards;
  - a coefficient direction from one HVP does not leak into a later `ics` HVP;
  - `theta_dot` is cleared after an HVP that fails mid-sweep (simulated with `monkeypatch`), and the next HVP is unaffected;
  - repeated `ReducedProblem.hessp` calls agree and are each counted.

## 13. Workflow-level tests

Pure-Python tests in `tests/integration/test_optimization_workflows.py`:

- **Spec acceptance and rejection** (from step 8): second-order methods are accepted per target; unknown methods, full-Hessian methods, bad `use_hessp`/`use_bounds`/`xtol` values, and Newton-CG or trust-ncg on the bounded `coeffs` target are rejected. `load_spec` is exercised on copies of the examples. The bounds error names the method, the target and the remedies (`trust-constr`, `L-BFGS-B`, `optimizer.use_bounds: false`).
- **Reporting:**
  - `optimizer_summary` survives a `result.yaml` round trip with numpy scalars;
  - `run_job` forwards `use_hessp`, `use_bounds` and `xtol` to `optimize()` (checked by inspection, since `run_job` needs Firedrake);
  - `validate.summary_row` records the method and HVP counts, with defaults for older results;
  - a `summary.csv` mixing an L-BFGS-B row and a trust-constr row reads back with the right per-method columns;
  - `history.csv` counter columns (from step 9).

## 14. Example configs

Each L-BFGS-B example in `workflows/examples/` has a second-order counterpart that solves the same problem. Only `optimization_name` and the `optimizer` section differ:

- `optimize_coeff_trust_constr.cfg` (`coeffs`) uses `trust-constr`, the only HVP method that handles the coefficient bounds. It stops on `gtol` (projected gradient) or `xtol` (trust radius).
- `optimize_ic_params_newton_cg.cfg` (`ic-params`) uses `Newton-CG`. The target has no bounds. Newton-CG stops on `xtol`, and `gtol` is not passed to it.

Both set `use_hessp: true` explicitly. Tests check that each pair matches outside the optimizer settings and name, that the second-order examples use HVPs, and that `workflows/README.md` lists them.

## 15. Optimizer comparison tooling

`workflows/compare_optimizers.py` is pure Python and never runs the model. It reads the `result.yaml` of every job under the given job directories or optimization roots, using `workflows.common.find_job_dirs`.

- **Problem identity.** Jobs solve the same problem when they share target and job name (`<truth-run>__<loss-label>`). The paired examples from step 14 satisfy this.
- **Rows.** `comparison_row` builds one row per job: method, `use_hessp`, `hessp_used`, success, status, message, `nit`, `nfev`, `njev`, `nhvp`, initial and final loss, `loss_reduction`, `optimize_seconds`, and per-parameter relative errors. Missing entries in older results get defaults.
- **Per-problem comparison.** `add_problem_comparisons` fills in `final_loss_vs_best` (relative to the lowest final loss of the problem) and `nmethods`.
- **Outputs.** `write_comparison` saves `optimizer-comparison.csv`, one row per job with fixed leading columns. It also saves `optimizer-comparison.yaml`, which groups jobs by problem and records the best jobs by `lowest_final_loss`, `fewest_gradients` and `shortest_time`.
- **Printing.** `format_comparison_table` renders a plain-text table.
- **Wall time.** `run_job` now records `optimize_seconds`, the wall time of the optimizer loop alone, in `result.yaml`.

Tests: `tests/integration/test_compare_optimizers.py`, on fake roots written to `tmp_path`. The command-line entry point and shell wrapper are step 16.

## 16. Comparison script

`python -m workflows.compare_optimizers <job-dir-or-root>...` and the wrapper `workflows/compare_optimizers.sh` print the comparison table and save `optimizer-comparison.csv` and `.yaml`. The wrapper only activates the Firedrake venv and uses no MPI. The name avoids `compare_runs.sh`, which compares simulation outputs.

- **Output directory.** `--output-dir` defaults to the common parent of the optimization roots: the root itself for a single root, `runs/optimizations` for the example pairs.
- **Options.** `--no-save`, `--only <substring>...` (matched against `<optimization>/<job>`) and `--columns <column>...` (any CSV column).
- **Errors.** Unknown columns, and an `--only` that matches nothing, raise `ValueError`.
- **Docs and tests.** Usage is documented in `workflows/README.md` under "Comparing optimizer methods". Tests are in `tests/integration/test_compare_optimizers.py`.

## 17. User documentation

- **`README.md`.** The "only a Jacobian" statement is replaced. The README now describes:
  - adjoint gradients;
  - forward-over-adjoint HVPs (eq. 2.11);
  - the supported second-order scipy methods, targets and time steppers;
  - pointers to this note and to `workflows/README.md`.
- **`workflows/README.md`:**
  - The spec example shows `use_hessp`, `use_bounds` and `xtol`.
  - A new "Optimizer methods" section has a method table (order, HVP use, bounds, stopping tolerances), a short forward-over-adjoint explanation, and the restrictions (bounds, no full Hessian, `use_hessp`, `xtol`, time stepping).
  - The outputs section has a table of the `result.yaml` optimizer fields and describes the `history.csv` counters. `summary.csv` lists its new columns. The comparison script is documented, and the package-changes and limitations lists are updated.
- **Tests.** `tests/integration/test_optimization_workflows.py` checks the key statements, the method names and every optimizer spec key, so the docs fail loudly if they drift from the code.

## 18. Staged verification

`workflows/verify_hessp.sh` runs the verification sequence in order and stops at the first failure. `--from <stage>` restarts at a stage.

1. **plumbing:** `test_reduced_problem_hvp.py`, `test_optimizer_methods.py`, `test_workflow_optimizer_tables.py` (pure Python).
2. **taylor:** `test_timestepping_second_order.py`, `test_optimize_hvp_storage.py`, `test_optimize_hvp.py`, `test_optimize_hvp_repeatability.py`. It also reruns the first-order gradient tests, to show they are unchanged.
3. **reporting:** `test_optimization_workflows.py`, `test_compare_optimizers.py`.
4. **end-to-end:** `tests/integration/test_optimizer_end_to_end.py`. It recovers `ox`, `oy` (`ic-params`) on `tests/tswe_small.cfg` with L-BFGS-B and with Newton-CG, and checks that:
   - both runs reduce the loss by at least `1e-4` and cut the parameter error by at least 20x;
   - only Newton-CG performs HVPs;
   - `optimize()` reports the problem's own counters;
   - the `History` counters are cumulative.

   It then writes both runs as optimization roots in `run_job`'s `result.yaml` layout, runs `compare_optimizers.main`, and checks that the CSV holds exactly the recorded `nit`, `nfev`, `njev`, `nhvp`, final loss and `hessp_used`.
5. **workflow** (`--workflow <ranks>`): the double-vortex truth sweep, `optimize_ic_params.cfg` and `optimize_ic_params_newton_cg.cfg` through `optimize_params`, and then `compare_optimizers` on both roots. Jobs are reused with `--skip-existing`. Results written before HVP support have no `nhvp`; delete them to rerun.

A test in `test_compare_optimizers.py` checks that the script references only existing test files and examples, and that it runs every `test_*hvp*.py` unit test.

**Models without coefficients (bug found in stage 5).** The workflow's `ic-params` and `ic-field` targets run with every `treat_as_coeffs` false, as in `mtswe.cfg`, so `model.get_coeff_var` returns `None`. `LieSplittingIntegrator.reset_internal_vars` and `take_adjoint_step` assigned to that `None` without a guard. That code predates the HVP work, and it broke every first-order evaluation too. Both now check `model.has_coeff()`, as `GeneralRK` does, and `take_adjoint_step` returns `delta_grad_rhs = None` in that case.

The unit tests had missed this because `tests/tswe_small.cfg` sets `hyperviscosity.treat_as_coeffs: true`. `tests/unit/test_optimize_without_coefficients.py` now covers the case: gradient and HVP Taylor tests for `ics` and `ic-params` without coefficients, through LieSplit. Separately, `optimize_params.py` now prints the full traceback of a failed job.

## 19. Representative comparison and regression baseline

`workflows/optimizer_baseline.sh <ranks> (--save | --check) [--all-jobs]` runs the double-vortex `ic-params` problem (`ox`, `oy`) with L-BFGS-B and with Newton-CG, using the step 14 example pair. By default it runs the final-state loss only, one job per truth run. It then runs `compare_optimizers.py` with `--save-baseline` or `--check-baseline workflows/baselines/doublevortex-params.yaml`.

- **What a baseline holds.** Per job `<optimization>/<job>`: the `BASELINE_METRICS` (method, target, success, status, `nit`, `nfev`, `njev`, `nhvp`, initial and final loss) and `optimize_seconds`.
- **What counts as a regression** (`check_against_baseline`):
  - the job is missing, or its method or target changed;
  - the job converged before and doesn't now;
  - a count exceeds `(1 + count_rtol) * baseline + 1`;
  - the final loss exceeds `loss_factor * baseline`. A factor is used because converged losses are round-off dominated.
- **Wall time.** It is checked only with `--time-factor`; otherwise it is reported as a ratio.
- **Notes.** Improvements and new jobs are reported as notes and don't fail the check.
- **Tests.** `tests/integration/test_compare_optimizers.py` covers save/check round trips, each regression type, the count tolerance and the exit status.

The baseline file must be produced on the target machine, with `--save`, after the stage 5 workflow run passes. It is not committed with this change.

## 20. Performance measurement tooling

The performance plan is in `docs/hvp_performance_plan.md`. Its section 2 (measure first) is implemented as follows:

- **`dimswe/profiling.py`.** Nested timing regions (`region`, `@timed`), keyed by the path of enclosing regions. Each one also opens a PETSc log event, so it shows up in `-log_view` flamegraphs. Timing is off unless a driver calls `enable()`, so normal runs only pay for the PETSc event calls. The module never imports petsc4py itself, so it cannot initialize PETSc before Firedrake has read the command-line options.
- **Instrumentation.**
  - `obj`, `jac`, `hessp` / `ic_params_hessp`, and their passes and sweeps.
  - The LieSplit substep recompute and the substep adjoints, with one region per sub-integrator (`LieSplittingIntegrator.profile_labels`).
  - Inside GeneralRK: the four kinds of stage pass, the increment assemblies and solves, the updates, and `_setup_second_order`. Its form construction moved to `_build_second_order_solvers`, so that it can be timed on its own.
  - The forward stage solves of `take_forward_step` now go through `_solve_forward_stages`: the same solves in the same order.

  None of this changes the numerics.
- **`workflows/profile_hvp.py` / `profile_hvp.sh`.**
  - Builds a workflow job as `optimize_params.py` does.
  - Times a first call and steady-state calls of `obj`, `jac` and `hessp` (two directions) at the starting point.
  - Writes `profile.yaml` / `profile-tree.txt`: setup vs steady state, measured ratios and stage passes against the cost model, a per-region breakdown, the per-sub-integrator cost of a stage pass, and an optimizer run-time estimate from `result.yaml`.
- **Tests.** `tests/unit/test_profiling.py` and `tests/integration/test_profile_hvp.py`. The latter includes a Firedrake test asserting that the instrumented gradient and HVP of `tests/tswe_small.cfg` perform exactly the stage passes of the plan's cost model: 3 forward passes per substep for `jac`, and 3 forward plus 3 tangent passes for `hessp`. Plan items A2/A3 must update that table (`COST_MODEL_STAGE_PASSES`).

The step 19 baseline already gives about 3.8 gradients per HVP, against 2 in the solve-count model, and HVPs taking 82-91% of the Newton-CG run time (plan section 2.2). The profiler runs fill in plan section 2.3.

## 21. Persistent increment assembly (plan item B0)

The profile in plan section 2.3 showed the bare `assemble()` calls on the increment forms taking 52% of an HVP and 31% of a gradient. The forms are `L_delta_lambda` and `L_delta_grad` in `take_adjoint_step`, and additionally `L_delta_lambda_dot` and `L_delta_grad_dot` in `take_second_order_adjoint_step`. Each call repeated the form's symbolic processing and parloop construction, taking 0.1-0.5 s, while the kernels themselves take milliseconds.

`GeneralRK._assemble_increment(name)` now keeps one `dimswe.assembly.PersistentAssembler` per form. It assembles into a reused Cofunction and returns a new flattened array, as before.
- **Strategies.** The persistent API depends on the Firedrake version, so the assembler tries `get_assembler(form).assemble(tensor=)`, then `create_assembly_callable(form, tensor=)`, then `assemble(form, tensor=)`.
- **Validation.** Each strategy is checked against a plain `assemble()` on its first two uses (relative tolerance 1e-10), and the decision is made collectively across ranks.
- **Fallback.** A strategy that disagrees, for example by missing coefficient updates or accumulating into the old tensor, or that raises, is dropped. If none works, the assembler falls back to plain assembly.

The numerics are unchanged: the same forms are assembled with the same kernels.
