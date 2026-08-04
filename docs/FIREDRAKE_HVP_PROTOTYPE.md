# Firedrake exact Hessian-vector-product prototype

## Purpose and isolation

`firedrake_hvp_prototype` transfers the certified NumPy discrete-HVP graph to
a deliberately small variational finite-element problem.  It supports fixed
step explicit Euler and classical RK4 through one generic stage algorithm.
The implementation is independent of the DIMSWE timesteppers and optimizer,
does not expose a PyROL `hessVec`, and does not use pyadjoint to obtain any
derivative.

The implemented model is

\[
  u_t=\kappa\Delta u+p^2u.
\]

Setting \(\kappa=0\) gives the reaction certification ladder.  Positive
\(\kappa\), a nonconstant field, and homogeneous Dirichlet boundary
conditions give the nontrivial spatial case.

## Primal and dual spaces

For a scalar finite-element space \(V_h\), the primal objects

\[
u,\ K_i,\ Y_i,\ w,\ V_i,\ W_i,\psi_i,\delta\psi_i\in V_h
\]

are Firedrake `Function` objects.  The adjoint objects

\[
\lambda^\star,\ \mu^\star,\ \bar K_i^\star,
\ \delta\bar K_i^\star\in V_h^*
\]

are `Cofunction` objects on `V.dual()`.  Scalar parameter gradients and HVPs
are Python `float` values.

The L2 mass/Riesz map is

\[
  \langle Mz,\eta\rangle=\int_\Omega \eta z\,dx.
\]

`WeakStageModel.mass_map` assembles the one-form and returns a `Cofunction`.
`WeakStageModel.l2_riesz_representative` explicitly solves \(Mz=\ell\) and
returns a `Function`.  It never copies dual coefficients into primal storage.
The natural pairing is evaluated by `assemble(action(dual, primal))`.

The tests demonstrate that these three objects are different:

- the coefficient array of \(Mz\), which depends on the finite-element mass
  matrix;
- the coefficient array of the primal Riesz representative \(z\);
- the scalar pairing \(\langle Mz,v\rangle=\int_\Omega zv\,dx\).

They also apply the mass map after the Riesz solve and recover the original
dual object to approximately \(2\times10^{-15}\).

## Weak primal and tangent stages

The spatial sign convention follows integration by parts for
\(+\kappa\Delta u\).  With homogeneous Dirichlet data, the stage tendency is
defined by

\[
 \int_\Omega \eta K_i\,dx
 =p^2\int_\Omega\eta Y_i\,dx
 -\kappa\int_\Omega\nabla\eta\mathbin\cdot\nabla Y_i\,dx.
\]

Equivalently, the residual written on the left is

\[
 \int_\Omega\eta K_i\,dx
 +\kappa\int_\Omega\nabla\eta\mathbin\cdot\nabla Y_i\,dx
 -p^2\int_\Omega\eta Y_i\,dx=0.
\]

Every \(K_i\) is obtained from a Firedrake mass solve, including when
\(\kappa=0\) and the result happens algebraically to be \(p^2Y_i\).

For a combined direction \((W_i,q)\), UFL directionally differentiates the
primal right-hand-side one-form before the tangent mass solve:

\[
 \int_\Omega\eta V_i\,dx
 =p^2\int_\Omega\eta W_i\,dx
 +2pq\int_\Omega\eta Y_i\,dx
 -\kappa\int_\Omega\nabla\eta\mathbin\cdot\nabla W_i\,dx.
\]

Thus the code supports both parameter-only directions, with incoming
\(w=0\), and combined incoming-state/parameter directions.

## Ordinary reverse mass solve

Let the stage right-hand side be the one-form \(B(Y_i,p)\), so
\(MK_i=B(Y_i,p)\).  Given \(\bar K_i^\star\in V_h^*\), ordinary reverse first
solves for a primal auxiliary field:

\[
 M\psi_i=\bar K_i^\star.
\]

It then differentiates the already-contracted scalar form

\[
 \langle B(Y_i,p),\psi_i\rangle
 =p^2\int_\Omega Y_i\psi_i\,dx
 -\kappa\int_\Omega\nabla\psi_i\mathbin\cdot\nabla Y_i\,dx.
\]

UFL assembly gives the dual state pullback

\[
 \bar Y_i^\star
 =p^2M\psi_i-\kappa K^*\psi_i
\]

and the ordinary scalar parameter contribution

\[
 g_i=2p\int_\Omega Y_i\psi_i\,dx.
\]

For Euler, \(\bar K^\star=\Delta t\lambda_+^\star\) and
\(\lambda_u^\star=\lambda_+^\star+\bar Y^\star\).  The implementation does
not use the reaction cancellation to bypass \(M\psi=\bar K^\star\).

## Incremental reverse and exact HVP

The tangent of the reverse graph supplies
\(\delta\bar K_i^\star\).  The second primal auxiliary solve is

\[
 M\delta\psi_i=\delta\bar K_i^\star.
\]

The incremental state pullback is assembled from contracted UFL derivatives:

\[
 \delta\bar Y_i^\star
 =B_Y^*\delta\psi_i
 +d(B_Y^*\psi_i)[W_i,q].
\]

For this model,

\[
 \delta\bar Y_i^\star
 =p^2M\delta\psi_i-\kappa K^*\delta\psi_i
 +2pqM\psi_i.
\]

The exact scalar parameter HVP contribution is

\[
 h_i=B_p^*\delta\psi_i+d(B_p^*\psi_i)[W_i,q]
 =2q\int Y_i\psi_i\,dx
 +2p\int W_i\psi_i\,dx
 +2p\int Y_i\delta\psi_i\,dx.
\]

All second terms are directional derivatives of contracted scalar or
one-form expressions.  The code constructs no third-order tensor and never
finite-differences a gradient internally.

## Generic explicit Runge--Kutta graph

For an explicit tableau \((a_{ij},b_i,c_i)\), increasing stage order computes

\[
 Y_i=u_n+\Delta t\sum_{j<i}a_{ij}K_j,
 \qquad MK_i=B(Y_i,p),
\]

\[
 W_i=w_n+\Delta t\sum_{j<i}a_{ij}V_j,
 \qquad MV_i=dB(Y_i,p)[W_i,q].
\]

The step updates are

\[
 u_{n+1}=u_n+\Delta t\sum_i b_iK_i,
 \qquad
 w_{n+1}=w_n+\Delta t\sum_i b_iV_i.
\]

Decreasing stage order computes

\[
 \bar K_i^\star=\Delta t b_i\lambda_{n+1}^\star
 +\Delta t\sum_{j>i}a_{ji}\bar Y_j^\star,
\]

\[
 \delta\bar K_i^\star=\Delta t b_i\mu_{n+1}^\star
 +\Delta t\sum_{j>i}a_{ji}\delta\bar Y_j^\star.
\]

`ExplicitRungeKutta` contains the only forward, tangent, ordinary-reverse,
and incremental-reverse algorithms.  Euler and classical RK4 differ only in
their immutable `ButcherTableau` data.

## Terminal objective and multistep reversal

For

\[
 J=\tfrac12\int_\Omega(u_N-d)^2\,dx,
\]

the terminal dual data are assembled as

\[
 \lambda_N^\star=M(u_N-d),\qquad \mu_N^\star=Mw_N.
\]

`terminal_least_squares_gradient` performs an independent primal and ordinary
reverse pass.  It does not call the HVP or tangent routines, so centered
differences of this gradient are independent checks of incremental reverse.
`terminal_least_squares_hvp` reverses timestep caches from \(N-1\) to zero and
stages from last to first, accumulating the shared scalar gradient and HVP.

## Reaction analytic certification

The reaction domain is a unit interval and constant fields therefore reduce
to the unit-volume scalar reference without changing the finite-element
algorithm.  For \(u_0=2\), \(d=1\), \(p=3\), \(\Delta t=0.1\), \(q=0.5\),
and \(w_0=0\), one Euler step gives:

| Quantity | Certified value |
| --- | ---: |
| \(K_0\) | 18 |
| \(u_1\) | 3.8 |
| \(J\) | 3.92 |
| \(w_1\) | 0.6 |
| \(g_p\) | 3.36 |
| \(Hq\) | 1.28 |
| \(H\) | 2.56 |

The actual reverse auxiliaries are \(\psi=0.28\) and
\(\delta\psi=0.06\).  The three HVP terms are `0.56`, `0`, and `0.72`.

Two Euler steps give:

| Quantity | Certified value |
| --- | ---: |
| \(u_1\) | 3.8 |
| \(u_2\) | 7.22 |
| \(J\) | 19.3442 |
| \(g_p\) | 28.3632 |
| \(Hq\) | 19.6024 |
| \(H\) | 39.2048 |

One, two, and five steps are compared with direct scalar first- and
second-derivative formulas for both Euler and RK4.

## Reaction-diffusion certification

The spatial problem uses a six-cell `UnitIntervalMesh`, continuous linears,
\(\kappa=0.08\), and homogeneous Dirichlet values at both endpoints.  The
deterministic fields are

\[
u_0=\sin(\pi x),\qquad d=0.2x(1-x),\qquad
w_0=x(1-x)(1+0.5x),
\]

with \(p=0.7\), \(q=0.3\), and \(\Delta t=0.02\).  This exercises the genuine
spatial action \(M^{-1}(p^2M-\kappa K)\).  Euler at one and three steps and
RK4 at one and three steps match a test-only dense assembled interior
\(M/K\) oracle within \(5\times10^{-13}\), including the combined-direction
parameter HVP.  The dense matrices and `numpy.linalg.solve` occur only in the
test oracle; the prototype never assembles a full timestep Jacobian.

## Centered-difference convergence

The centered differences use

\[
 \frac{g(p+\epsilon q)-g(p-\epsilon q)}{2\epsilon},
\]

and for a combined direction also use
\(u_0\mathbin\pm\epsilon w_0\).  The measured errors are:

| \(\epsilon\) | reaction objective to gradient | reaction RK4 HVP | diffusion Euler combined HVP | diffusion RK4 combined HVP |
| ---: | ---: | ---: | ---: | ---: |
| 0.04 | 2.9546895e-4 | 6.3308652e-5 | 4.4373654e-6 | 4.7594494e-6 |
| 0.02 | 7.3853825e-5 | 1.5826570e-5 | 1.1093302e-6 | 1.1898469e-6 |
| 0.01 | 1.8462618e-5 | 3.9566055e-6 | 2.7733186e-7 | 2.9746076e-7 |
| 0.005 | 4.6156020e-6 | 9.8914909e-7 | 6.9332922e-8 | 7.4365127e-8 |

Every error decreases by essentially four when epsilon halves, demonstrating
the expected second-order centered-difference regime before roundoff.

## Solver, mutation, and cache rules

The L2 mass matrix is assembled as PETSc AIJ and solved in serial with
`ksp_type=preonly` and `pc_type=lu`.  The inverse mass matrix is never formed.
Stage right-hand sides and pullbacks are assembled one-forms.  The full RK
timestep Jacobian is neither assembled nor stored.

Firedrake 2026.4.1 has two relevant local API details:

- scaled `Cofunction` arithmetic can promote a concrete dual object to a UFL
  `FormSum`; dual accumulation therefore uses PETSc AXPY on owned output
  storage;
- `DirichletBC.apply` rejects a `Cofunction`.  Passing the boundary conditions
  to one-form assembly zeros constrained dual entries naturally, so the
  prototype uses that operation rather than pretending the dual is primal.

All inputs are treated as read-only.  `Function.copy(deepcopy=True)` and
`Cofunction.copy(deepcopy=True)` are used at API and cache boundaries.  Each
step and stage cache owns its primal/tangent data; returned trajectory fields
also do not alias cache fields.  Reverse mass solves own a copy of their dual
right-hand side.  Tests mutate original inputs after cache creation, verify
cached values remain unchanged, verify dual inputs survive reverse calls, and
obtain bitwise-identical repeated results.

## Limitations before a DIMSWE transfer

This prototype has one scalar parameter, a fixed step size, explicit Euler or
classical RK4, a linear reaction-diffusion stage operator, homogeneous fixed
boundary data, and a terminal least-squares objective.  It does not address
mixed/vector spaces, nonsymmetric spatial operators, parameter-dependent mass
matrices or boundary conditions, nonlinear solves, running objectives,
variable timesteps, checkpoint scheduling, distributed cache ownership, or
production solver preconditioning.

Those issues require separate derivation and certification before any DIMSWE
transfer.  No DIMSWE production HVP, PyROL `hessVec`, JAX physics, or
neural-network work is part of this milestone.
