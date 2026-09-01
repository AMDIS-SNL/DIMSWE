# Exact DoubleVortex case specification

## Scope and provenance

This document reconstructs the executable `DoubleVortex` initial condition used by the frozen DIMSWE truth runs. It was derived from the authoritative read-only repository at

`/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615`.

The immutable Test 2A truth is `external-results/test1b-production/truth_c0_0.14` and records Git checkpoint `2543681af71a02dd8892694f0fef9e5f2eb806fc`. The immutable Test 2B truth is `external-results/test2b-rain-active-truth/production-n64-zeta-m0p06-dt100-t16000` and records checkpoint `d2f5d66ecb5500aad24eca37280f8a52e22a250f`. The latter path was verified to exist and to contain complete metadata, 161 restart arrays, 161 Firedrake checkpoints, 161 diagnostic records, and 161 spectra (steps 0 through 160).

The authoritative working tree is not clean: its present `zeta` plumbing is an uncommitted modification relative to `d2f5d66...`. Therefore the reconstruction is pinned both to the immutable run metadata and to the following current source hashes:

- `dimswe/initial_conditions.py`: `4f341133e1dcf23241ca10f626763fb010ef5e6f0b61e045044efac57b671909`
- `dimswe/physics.py`: `78b16349225021bee1ebbdf9e513669004e8e3189352e4e294b2765e99885ee8`
- `dimswe/ufl_helpers.py`: `67daf96a8d52d0ccffa10b949f02ef8176982a54770e34b2793abb6498f4ebd6`
- `dimswe/configs/resolved_hidden_c0_pilot.cfg`: `e3dd7fe40aeb4f5d2391e365673e4a012397f48c33a7d50d4513abb3ffef5a44`
- `scripts/run_test2b_rain_active_truth_production.sh`: `1a29a57dc01459f7495a5c09172a8fa2161f68bc9ce2248ef086d5d204a764ce`

## Domain, mesh, and topography

The physical domain is the flat doubly periodic square

\[
\Omega=[0,L_x)\times[0,L_y),\qquad L_x=L_y=5{,}000{,}000\ \mathrm{m}.
\]

The executable mesh is a Firedrake `PeriodicRectangleMesh` with quadrilateral cells. Test 2A uses (16\times16) cells; Test 2B uses (64\times64) cells. The topography/geometric height field is identically zero,

\[
B(x,y)=0.
\]

Provenance: `dimswe/initial_conditions.py:439-453,493`; `dimswe/meshes.py:51-54`; `dimswe/configs/resolved_hidden_c0_pilot.cfg:30-36`; the two truth `metadata.json` files under `domain`, `mesh`, and `finite_element_spaces`.

## Constants and vortex geometry

The initializer sets

\[
\begin{aligned}
g&=9.80616\ \mathrm{m\,s^{-2}}, & f&=6.147\times10^{-5}\ \mathrm{s^{-1}},\\
H_0&=750\ \mathrm{m}, & \Delta h&=75\ \mathrm{m},\\
\sigma_x&=\sigma_y=\frac{3}{40}L_x=375{,}000\ \mathrm{m},\\
o_x&=o_y=0.1, & (x_c,y_c)&=(L_x/2,L_y/2),\\
c&=0.05, & a&=1/3,\\
D&=L_x/2=2{,}500{,}000\ \mathrm{m}, & q_0&=0.002,\\
L&=10, & \beta_2&=gL=98.0616\ \mathrm{m\,s^{-2}},\\
q_{\mathrm{precip}}&=10^{-4}, & \gamma_r&=10^{-3}.
\end{aligned}
\]

The vortex centers are

\[
(x_1,y_1)=(0.4L_x,0.4L_y)=(2000,2000)\ \mathrm{km},
\]

\[
(x_2,y_2)=(0.6L_x,0.6L_y)=(3000,3000)\ \mathrm{km}.
\]

`U0=0` is assigned by the constructor but is not used in the returned initial condition.

Provenance: `dimswe/initial_conditions.py:439-464,483`; `dimswe/configs/resolved_hidden_c0_pilot.cfg:15-27`.

## Exact sine-periodicized vortex construction

The implementation does **not** use the usual piecewise minimum-image distance
(\min_{n\in\mathbb Z}|x-x_i+nL_x|). Instead it uses a smooth sine mapping. For each vortex (i\in\{1,2\}), define

\[
X_i(x)=\frac{L_x}{\pi\sigma_x}
\sin\!\left(\frac{\pi(x-x_i)}{L_x}\right),\qquad
Y_i(y)=\frac{L_y}{\pi\sigma_y}
\sin\!\left(\frac{\pi(y-y_i)}{L_y}\right),
\]

and

\[
G_i(x,y)=\exp\!\left[-\frac12\left(X_i^2+Y_i^2\right)\right].
\]

Near a center, (X_i\sim(x-x_i)/\sigma_x) and (Y_i\sim(y-y_i)/\sigma_y), so the local convention is the standard Gaussian
(\exp[-r^2/(2\sigma^2)]): `sigmax` and `sigmay` are standard-deviation-like widths, not e-folding radii.

The exact height is

\[
h(x,y,0)=H_0-\Delta h\left[
G_1(x,y)+G_2(x,y)-
\frac{4\pi\sigma_x\sigma_y}{L_xL_y}
\right].
\]

Thus the vortices are negative depth anomalies. The constant correction is exactly the expression above; it should not be reinterpreted as an exactly evaluated mean of the sine-periodicized Gaussian.

Provenance: `dimswe/initial_conditions.py:475-485`.

## Exact derivatives and geostrophic velocity

Define the auxiliary quantities used literally by the code,

\[
\widetilde X_i(x)=\frac{L_x}{2\pi\sigma_x}
\sin\!\left(\frac{2\pi(x-x_i)}{L_x}\right),\qquad
\widetilde Y_i(y)=\frac{L_y}{2\pi\sigma_y}
\sin\!\left(\frac{2\pi(y-y_i)}{L_y}\right).
\]

Since

\[
\frac{\partial G_i}{\partial x}
=-\frac{\widetilde X_i}{\sigma_x}G_i,
\qquad
\frac{\partial G_i}{\partial y}
=-\frac{\widetilde Y_i}{\sigma_y}G_i,
\]

the analytical height derivatives are

\[
\boxed{
\frac{\partial h}{\partial x}
=\frac{\Delta h}{\sigma_x}\sum_{i=1}^{2}\widetilde X_iG_i,
\qquad
\frac{\partial h}{\partial y}
=\frac{\Delta h}{\sigma_y}\sum_{i=1}^{2}\widetilde Y_iG_i.}
\]

The code initializes the velocity components and momentum density as

\[
\boxed{
u(x,y,0)=-\frac{g}{f}\frac{\partial h}{\partial y}
=-\frac{g\Delta h}{f\sigma_y}\sum_{i=1}^{2}\widetilde Y_iG_i,}
\]

\[
\boxed{
v(x,y,0)=\frac{g}{f}\frac{\partial h}{\partial x}
=\frac{g\Delta h}{f\sigma_x}\sum_{i=1}^{2}\widetilde X_iG_i,}
\qquad
\boldsymbol m=h\boldsymbol u.
\]

Here ($\\boldsymbol u=(u,v)$); the repository stores that vector under the state key `v`.

The repository rotation is

\[
\mathcal R(a,b)=(-b,a).
\]

Consequently

\[
\mathcal R(\boldsymbol u)
=\left(-\frac{g}{f}h_x,-\frac{g}{f}h_y\right)
=-\frac{g}{f}\nabla h,
\]

and the intended constant-(g), constant-(f) balance is exact analytically:

\[
\boxed{f\,\mathcal R(\boldsymbol u)+g\nabla h=\boldsymbol0.}
\]

This statement is specifically about the Coriolis/constant-(g) height-gradient pair. The complete moist thermal model also contains the nonuniform (S/h) field, splitting, limiting, and hyperviscosity, so it does not assert that every term of the fully discrete initial tendency vanishes.

Provenance: `dimswe/initial_conditions.py:479-490`; `dimswe/ufl_helpers.py:5-12`.

## Exact (b/S), saturation, and condensate initialization

The initializer uses the symbol (s=S/h) for the specific thermal/buoyancy-like scalar. Its exact initial value is

\[
s(x,y,0)=g\left[1+c\exp\!\left(
-\frac{(x-L_x/2)^2+(y-L_y/2)^2}{a^2D^2}
\right)\right],
\]

\[
\boxed{S(x,y,0)=h(x,y,0)s(x,y,0).}
\]

If the report denotes this specific scalar by (b), then (b=s=S/h). Unlike the vortex height anomalies, this central scalar bump is coded directly in Cartesian distance and is not sine-periodicized.

With (B=0), the saturation law is

\[
q_{\mathrm{sat}}(h,s,B)=q_0\frac{H_0}{h+B}
\exp\!\left[20\left(1-\frac{s}{g}\right)\right].
\]

The water variables are areal/density variables (Q_k=hq_k). Initially,

\[
\boxed{Q_v=h(1-\zeta)q_{\mathrm{sat}},\qquad Q_c=0,\qquad Q_r=0.}
\]

Therefore

\[
\frac{q_v}{q_{\mathrm{sat}}}=1-\zeta.
\]

- Test 2A omits `zeta`; the constructor default is (zeta=0), so it is analytically exactly saturated: (q_v/q_{\mathrm{sat}}=1).
- Test 2B passes (zeta=-0.06), so it is analytically 6% supersaturated: (q_v/q_{\mathrm{sat}}=1.06).

Provenance: `dimswe/initial_conditions.py:455-464,489-496`; `dimswe/physics.py:7-8`; `scripts/run_test2b_rain_active_truth_production.sh:25-38`. The Test 2A default is also confirmed by the absence of `zeta` in its immutable `metadata.json` initial-condition record; Test 2B records `zeta: -0.06`.

## Moist source laws needed to interpret the truth

For ($q_v=Q_v/h$), ($q_c=Q_c/h$), ($q_r=Q_r/h$), ($s=S/h$), and
($\tau_v=\tau_r=\Delta t$), define

\[
\gamma_v=\left(1+20q_{\mathrm{sat}}\frac{\beta_2}{g}\right)^{-1},
\]

\[
C=\max\!\left(0,\frac{\gamma_v(q_v-q_{\mathrm{sat}})}{\tau_v}\right),
\]

\[
E=\min\!\left(\frac{q_c}{\Delta t},
\max\!\left(0,\frac{\gamma_v(q_{\mathrm{sat}}-q_v)}{\tau_v}\right)\right),
\]

\[
\boxed{A=E-C,\qquad
R=\max\!\left(0,\frac{\gamma_r(q_c-q_{\mathrm{precip}})}{\tau_r}\right).}
\]

Thus (A<0) denotes net condensation and (A>0) denotes net evaporation. The coupled density sources are

\[
(\dot S,\dot Q_v,\dot Q_c,\dot Q_r)_{\rm moist}
=h\left(\beta_2A,\ A,\ -(A+R),\ R\right).
\]

These imply the pointwise source invariants

\[
\dot S-\beta_2\dot Q_v=0,
\qquad
\dot Q_v+\dot Q_c+\dot Q_r=0.
\]

Provenance: `dimswe/physics.py:16-31,68-104`; independently mirrored in `dimswe/jax_moist.py:74-162,172-185`.

## Discretization and run controls

Both frozen truths use:

- quadrilateral periodic mesh;
- vector CG(3) spectral velocity, scalar CG(3) spectral (h,S), and scalar DG(1) spectral (Q_v,Q_c,Q_r);
- GLL-lumped mass measure;
- Lie splitting with `[RK4, Euler, SSPRK43, Euler]`, subcycles `[2,1,2,1]`, expanded into dry RK4 twice, hyperviscosity Euler, DG limiter/advection SSPRK43 twice, and moist Euler;
- (c_0=0.14), hyperviscosity exponent (s=3.2), UFL moist backend, and seed 0;
- (Delta t=100\ \mathrm{s}), 160 steps, final time (16{,}000\ \mathrm{s}), and output every step (161 stored states including (t=0)).

The difference in truth resolution is deliberate: Test 2A is (16\times16), whereas Test 2B is (64\times64). Test 2B additionally stores its velocity spectrum on a (128\times128) diagnostic sampling grid. These values come from each immutable `metadata.json`; Test 2B is independently fixed by `dimswe/configs/test2b_rain_active_case.json:4-23` and `scripts/run_test2b_rain_active_truth_production.sh:25-38`.

## Compact LaTeX block

The following is the self-contained report-ready construction:

```latex
\begin{gathered}
\Omega=[0,L_x)\times[0,L_y),\quad L_x=L_y=5\times10^6\ {\rm m},\quad B=0,\\
(x_1,y_1)=(0.4L_x,0.4L_y),\qquad (x_2,y_2)=(0.6L_x,0.6L_y),\\
\sigma_x=\sigma_y=\frac{3L_x}{40},\quad H_0=750\ {\rm m},\quad
\Delta h=75\ {\rm m},\quad g=9.80616\ {\rm m\,s^{-2}},\quad
f=6.147\times10^{-5}\ {\rm s^{-1}},\\
X_i=\frac{L_x}{\pi\sigma_x}\sin\!\left(\frac{\pi(x-x_i)}{L_x}\right),\qquad
Y_i=\frac{L_y}{\pi\sigma_y}\sin\!\left(\frac{\pi(y-y_i)}{L_y}\right),\\
G_i=\exp\!\left[-\frac12(X_i^2+Y_i^2)\right],\\
h=H_0-\Delta h\left(G_1+G_2-\frac{4\pi\sigma_x\sigma_y}{L_xL_y}\right),\\
\widetilde X_i=\frac{L_x}{2\pi\sigma_x}\sin\!\left(\frac{2\pi(x-x_i)}{L_x}\right),\qquad
\widetilde Y_i=\frac{L_y}{2\pi\sigma_y}\sin\!\left(\frac{2\pi(y-y_i)}{L_y}\right),\\
h_x=\frac{\Delta h}{\sigma_x}\sum_{i=1}^2\widetilde X_iG_i,\qquad
h_y=\frac{\Delta h}{\sigma_y}\sum_{i=1}^2\widetilde Y_iG_i,\\
u=-\frac{g}{f}h_y,\qquad v=\frac{g}{f}h_x,\qquad
f\mathcal R(u,v)+g\nabla h=0,\quad \mathcal R(a,b)=(-b,a),\\
s=g\left[1+0.05\exp\!\left(-\frac{(x-L_x/2)^2+(y-L_y/2)^2}
{(1/3)^2(L_x/2)^2}\right)\right],\qquad S=hs,\\
q_{\rm sat}=0.002\,\frac{750}{h+B}
\exp\!\left[20\left(1-\frac{s}{g}\right)\right],\\
Q_v=h(1-\zeta)q_{\rm sat},\qquad Q_c=Q_r=0,\\
\zeta=0\quad({\rm Test\ 2A}),\qquad
\zeta=-0.06\quad({\rm Test\ 2B}).
\end{gathered}
```
