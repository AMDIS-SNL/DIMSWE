# NumPy ODE exact Hessian-vector-product prototype

## Purpose and scope

This prototype certifies an exact discrete Hessian-vector product (HVP) for
the isolated NumPy ODE package.  It does not add second-order functionality to
Firedrake, DIMSWE, PyROL, JAX, or neural-network code.  The implementation is
matrix-free with respect to timestep Jacobians and tensor-free with respect to
second derivatives.  Existing dense first-derivative matrices remain part of
the small NumPy dynamics API.

## Notation and derivative interface

Let the state be \(x\in\mathbb R^{n_x}\), the constant parameter vector be
\(p\in\mathbb R^{n_p}\), and the dynamics be
\(f(x,t,p)\in\mathbb R^{n_x}\).  A combined direction is \((w,q)\), with
\(w\in\mathbb R^{n_x}\) and \(q\in\mathbb R^{n_p}\).  The dynamics API exposes
the following actions:

| Code method | Mathematical action | Output shape |
| --- | --- | --- |
| `jac_x_action` | \(f_x w\) | `(nx,)` |
| `jac_params_action` | \(f_p q\) | `(nx,)` |
| `jacT_x_action` | \(f_x^T\ell\) | `(nx,)` |
| `jacT_params_action` | \(f_p^T\ell\) | `(nparams,)` |
| `directional_jacT_x_action` | \([f_{xx}[w]+f_{xp}[q]]^T\ell\) | `(nx,)` |
| `directional_jacT_params_action` | \([f_{px}[w]+f_{pp}[q]]^T\ell\) | `(nparams,)` |

The last two methods directly form contracted vectors.  No third-order array
is constructed.  `LotkaVolterra` and `LogisticEquation` implement this new
contract, while their existing RHS and Jacobian methods remain intact.  The
test suite also supplies a two-state/two-parameter nonlinear dynamics with
nonzero \(f_{xx}\), \(f_{xp}/f_{px}\), and \(f_{pp}\), plus the scalar reference
\(f(x,p)=p^2x\).

## Explicit Euler derivation

For

\[
x_+ = x + \Delta t f(x,p),\qquad
A=I+\Delta t f_x,\qquad B=\Delta t f_p,
\]

the tangent is

\[
w_+ = w + \Delta t(f_xw+f_pq).
\]

For an output adjoint \(\lambda_+\), ordinary reverse gives

\[
\lambda=A^T\lambda_+,\qquad g_{\rm step}=B^T\lambda_+.
\]

Differentiating this reverse graph gives

\[
\mu=A^T\mu_+ + [dA(w,q)]^T\lambda_+,
\]

and the parameter HVP contribution is

\[
h_{\rm step}=B^T\mu_+ + [dB(w,q)]^T\lambda_+.
\]

For one Euler step of \(f=p^2x\), with \(x=2\), \(p=3\),
\(\Delta t=0.1\), terminal datum \(d=1\), and \(q=0.5\), direct
differentiation of \(x_+=x(1+\Delta t p^2)\) predicts

| Quantity | Value |
| --- | ---: |
| \(x_+\) | 3.8 |
| \(J_p\) | 3.36 |
| \(Hq\) | 1.28 |
| \(H\) | 2.56 |

The implementation reproduces these values without using them internally.

## Generic explicit Runge--Kutta algorithm

For the existing explicit tableau \((a_{ij},b_i,c_i)\),

\[
Y_i=x_n+\Delta t\sum_{j<i}a_{ij}K_j,\qquad
K_i=f(Y_i,t_n+c_i\Delta t,p),
\]

\[
x_{n+1}=x_n+\Delta t\sum_i b_iK_i.
\]

The forward tangent stages, evaluated in increasing stage order, are

\[
W_i=w_n+\Delta t\sum_{j<i}a_{ij}V_j,
\qquad V_i=f_xW_i+f_pq,
\]

\[
w_{n+1}=w_n+\Delta t\sum_i b_iV_i.
\]

The exact ordinary reverse graph is evaluated in decreasing stage order:

\[
\bar K_i=\Delta t b_i\lambda_{n+1}
 +\Delta t\sum_{j>i}a_{ji}\bar Y_j,
\qquad \bar Y_i=f_{x,i}^T\bar K_i,
\]

\[
\lambda_n=\lambda_{n+1}+\sum_i\bar Y_i,
\qquad g_n=\sum_i f_{p,i}^T\bar K_i.
\]

The incremental reverse graph uses the cached \(W_i\):

\[
\delta\bar K_i=\Delta t b_i\mu_{n+1}
 +\Delta t\sum_{j>i}a_{ji}\delta\bar Y_j,
\]

\[
\delta\bar Y_i=f_{x,i}^T\delta\bar K_i
 +[f_{xx,i}[W_i]+f_{xp,i}[q]]^T\bar K_i,
\]

\[
\mu_n=\mu_{n+1}+\sum_i\delta\bar Y_i,
\]

\[
h_n=\sum_i\left(f_{p,i}^T\delta\bar K_i
 +[f_{px,i}[W_i]+f_{pp,i}[q]]^T\bar K_i\right).
\]

The implementation is in `_GeneralRK`; Euler, classical RK4, SSPRK3, and
SSPRK43 therefore use exactly the same tangent and reverse algorithms.  No
tableau-specific HVP is present.

## Multiple timesteps and terminal objective

For \(x_{n+1}=\Phi_n(x_n,p)\), the prototype stores timestep states, tangents,
and the primal/tangent stage data for every step.  For

\[
J=\tfrac12\lVert x_N-d\rVert^2,
\]

the terminal data are

\[
\lambda_N=x_N-d,\qquad \mu_N=w_N.
\]

Each cached RK step applies the stage algorithms above in reverse order.  The
total parameter derivatives are

\[
g=\sum_{n=0}^{N-1}g_n,\qquad Hq=\sum_{n=0}^{N-1}h_n.
\]

No dense timestep matrices \(A_n\) or \(B_n\) are formed.  The returned
`hvp_initial_state` is \(\mu_0\), so the same call certifies combined initial
state and parameter directions.  With two scalar Euler steps and the data
above, direct differentiation of \(x_2=x_0(1+\Delta t p^2)^2\) predicts and
the tests obtain:

| Quantity | Value |
| --- | ---: |
| \(x_1\) | 3.8 |
| \(x_2\) | 7.22 |
| \(J_p\) | 28.3632 |
| \(Hq\) | 19.6024 |
| \(H\) | 39.2048 |

## Operator composition

`OperatorComposition` implements a small child protocol with the same six
first- and second-directional actions.  In child order it evaluates

\[
x^{(i)}=\Phi_i(x^{(i-1)},p),\qquad
w^{(i)}=A_iw^{(i-1)}+B_iq.
\]

In reverse child order it evaluates

\[
\lambda^{(i-1)}=A_i^T\lambda^{(i)},
\]

\[
\mu^{(i-1)}=A_i^T\mu^{(i)}
 +[dA_i(w^{(i-1)},q)]^T\lambda^{(i)},
\]

and accumulates

\[
Hq\mathrel{+}=B_i^T\mu^{(i)}
 +[dB_i(w^{(i-1)},q)]^T\lambda^{(i)}.
\]

The two-child test uses a parameter-dependent linear child followed by a
parameter-independent quadratic state child.  Although the second child has
\(B_2=0\), it changes the gradient from `[0.769, -0.477]` to
`[1.33016001, -0.55024612]` and the HVP from `[0.6565, -0.1385]` to
`[1.721557, -0.14764624]` by transporting tangent and adjoint information.

`OperatorComposition.terminal_least_squares_gradient` is an independent
first-order oracle.  It performs a primal child pass and an ordinary-adjoint
reverse pass only; it neither calls the HVP method nor evaluates tangent or
incremental-adjoint actions.  Composition centered differences evaluate this
method at \(p\mathbin{\pm}\epsilon q\).

## Exact versus Gauss--Newton

For residual map \(r(p)=x_N(p)-d\),

\[
H_{\rm exact}q=r'^Tr'q+\sum_k r_k\,r_k''q,
\qquad H_{\rm GN}q=r'^Tr'q.
\]

`terminal_least_squares_gauss_newton_hvp` applies the ordinary adjoint to the
terminal tangent.  It is explicitly separate from the exact incremental
adjoint.  In the measured RK4 case the exact and Gauss--Newton vectors are
identical at zero residual, `[0.02799139, -0.00923229]`.  Away from zero
residual, the exact vector is `[0.09425734, -0.04499217]`, the Gauss--Newton
vector remains `[0.02799139, -0.00923229]`, and their 2-norm difference is
`7.5299e-2`.

## Code and call-graph map

| File/API | Role |
| --- | --- |
| `ode_adjoint/dynamics.py` | First-derivative actions and contracted second-directional transpose actions |
| `ode_adjoint/adjoint_timesteppers.py::forward_step_data` | Copied primal RK stages |
| `ode_adjoint/adjoint_timesteppers.py::linearize_step` | Tangent RK stages |
| `ode_adjoint/adjoint_timesteppers.py::reverse_step` | Exact ordinary stage reverse |
| `ode_adjoint/adjoint_timesteppers.py::reverse_hvp_step` | Coupled ordinary/incremental stage reverse |
| `ode_adjoint/hvp.py::terminal_least_squares_gradient` | Independent certified discrete gradient |
| `ode_adjoint/hvp.py::terminal_least_squares_hvp` | Multistep exact HVP and initial-state block |
| `ode_adjoint/hvp.py::terminal_least_squares_gauss_newton_hvp` | Terminal least-squares Gauss--Newton HVP |
| `ode_adjoint/hvp.py::OperatorComposition` | Independent ordinary gradient plus generic exact-HVP composition prototype |
| `ode_adjoint/ode_optimize.py` | Existing block objective and legacy gradient; its `hessp` stub is not used by this isolated API |

The exact call graph is

`terminal_least_squares_hvp -> linearize_step -> dynamics Jacobian actions ->
reverse_hvp_step -> reverse_step + contracted second-directional actions`.

The legacy `take_adjoint_step` wrapper computes \(t_n=t_{n+1}-\Delta t\)
and now rejects a stale or inconsistent `_last_step` before calling
`reverse_step`.  It checks scalar time/step inputs, strict parameter shape,
and cached step size, starting time, and parameters.  Comparisons use an
eight-machine-epsilon relative and absolute tolerance: parameters and step
size were copied directly into the cache, while starting time is reconstructed
by one subtraction.  A mismatch raises a field-specific `ValueError`.  State
is intentionally not checked because the legacy signature does not receive
\(x_n\).

Arrays in returned caches are copies.  Tests verify input non-mutation,
one-dimensional shape enforcement, floating output dtype, and bitwise
repeatability of repeated HVP calls.

## Verification measurements

Centered differences use

\[
\frac{g(p+\epsilon q)-g(p-\epsilon q)}{2\epsilon}.
\]

ODE differences use `terminal_least_squares_gradient`; composition differences
use `OperatorComposition.terminal_least_squares_gradient`.  Both invoke only
ordinary adjoint actions and are independent of the incremental-adjoint
implementation.

| `eps` | Euler, one step error | RK4, seven step error | Composition error |
| ---: | ---: | ---: | ---: |
| `1e-1` | `5.9999e-6` | `1.6869e-4` | `5.4921e-4` |
| `1e-2` | `5.9999e-8` | `1.6867e-6` | `5.4921e-6` |
| `1e-3` | `5.9999e-10` | `1.6867e-8` | `5.4921e-8` |
| `1e-4` | `5.9178e-12` | `1.6918e-10` | `5.4867e-10` |
| `1e-5` | `2.8499e-12` | `3.4056e-12` | `9.8266e-12` |
| `1e-6` | `2.0537e-11` | `3.2741e-11` | `1.9967e-10` |
| `1e-7` | `2.3487e-10` | `3.0284e-9` | `1.8520e-9` |
| `1e-8` | `2.5643e-9` | `6.7712e-9` | `2.6522e-8` |

The useful truncation range is approximately `1e-1` through `1e-4`, with
second-order error reduction under decade refinement.  The best errors occur
near `1e-5`; roundoff dominates from approximately `1e-6` downward.  At three
steps, the minimum HVP errors for Euler, RK4, SSPRK3, and SSPRK43 were between
`3.5e-12` and `9.2e-12`, all at `eps=1e-5`.

For three deterministic RK4 parameter directions, the symmetry checks gave:

| Pair | \(u^THv\) | \(v^THu\) | Absolute error |
| --- | ---: | ---: | ---: |
| 0, 1 | `2.1748284252543479e-2` | `2.1748284252543482e-2` | `3.469e-18` |
| 0, 2 | `1.2642152237605925e-2` | `1.2642152237605928e-2` | `3.469e-18` |
| 1, 2 | `3.8905168950366741e-2` | `3.8905168950366741e-2` | `0.000e+0` |

The analytic references, centered-difference sequences, symmetry checks, and
repeat calls provide independent and repeatable oracles.

The authoritative completed outside full-suite command is recorded in
`/private/tmp/dimswe-ode-hvp-full-suite.log`: `46 passed, 1 skipped, 1
xfailed, 12200 warnings in 450.57s`.  The focused NumPy ODE command is rerun
after each prototype hardening change; after cached-step and composition-oracle
hardening it reports `28 passed, 1 xfailed, 6 warnings`.

## Limitations before any Firedrake transfer

- The prototype supports explicit RK, constant parameters, fixed step size,
  and a terminal least-squares objective.  Running costs, parameter-dependent
  initial data, variable steps, event handling, implicit stages, and mass
  matrices need separate derivations.
- First Jacobians are still dense in the existing NumPy dynamics classes;
  only timestep Jacobians and second derivatives are action based.
- `OperatorComposition` currently assumes equal input/output state sizes.
- The standalone HVP is intentionally not wired into SciPy optimization or
  the legacy `Lagrangian_ODEConstrainedOptimization.hessp` stub.
- Stage data are stored per step for clarity.  A production implementation
  will need an explicit checkpoint/recompute policy and mutation/lifetime
  rules appropriate to Firedrake objects.
- No conclusion here validates a Firedrake tape, PyROL `hessVec`, DIMSWE
  physics, JAX physics, or neural-network Hessian action.
