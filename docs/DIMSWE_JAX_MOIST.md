# DIMSWE exact JAX moist primal and derivative boundary

Status: **J1/J2/J3 COMPLETE — EXTERNALLY CERTIFIED, SERIAL CPU ONLY.**

This document describes the opt-in JAX replica of the deployed moist Euler
child.  The existing UFL `ThreeWayPhysics` term remains unchanged, independent,
and the default production implementation.  J1 did not install a runtime
backend switch; J3 installs a narrowly scoped complete-split selector while
retaining that default.

## Scope

J1 implements

```text
mixed Firedrake state
  -> exact cell-local broken-CG3/GLL values
  -> pure float64 JAX rates and coupled source densities
  -> production-GLL weak source assembly
  -> complete mixed mass solve
  -> explicit Euler state update
```

J2 adds an opt-in, independent derivative helper for this one moist Euler
child.  J3 reuses those J1/J2 helpers behind child 6 of the complete deployed
split and threads the same backend through the existing reduced and PyROL
paths.  It does not add a neural network, neural parameters, state
normalization, checkpointing, MPI certification, or accelerator certification.

The new modules are optional.  `dimswe.__init__` does not import them, so an
ordinary `import dimswe` does not acquire a JAX dependency.

## Exact deployed algebra

The local inputs are `h`, `S`, `Qv`, and `Qc`, plus topography `B`.  Velocity
and `Qr` are omitted because the deployed local rates do not depend on them.

```text
qv = Qv / h
qc = Qc / h
s  = S / h

beta2 = g * L

qsat = q0 * H0 / (h + B) * exp(20 * (1 - s / g))
gamma_v = 1 / (1 + 20 * qsat * beta2 / g)

c_argument = gamma_v * (qv - qsat) / configured_dt
C = where(c_argument < 0, 0, c_argument)

e_argument = gamma_v * (qsat - qv) / configured_dt
E_positive = where(e_argument < 0, 0, e_argument)
evaporation_cap = qc / configured_dt
E = where(evaporation_cap < E_positive,
          evaporation_cap,
          E_positive)

r_argument = gamma_r * (qc - qprecip) / configured_dt
R = where(r_argument < 0, 0, r_argument)

A = E - C
```

Both deployed relaxation times are exactly `configured_dt`.  The arithmetic is
not algebraically simplified: in particular `20 * qsat * beta2 / g` remains in
that order.

Strict `where` comparisons reproduce the installed UFL equality convention:
at a max/min tie, the second operand is selected.  `jnp.maximum` and
`jnp.minimum` are not used.

No singular denominator is clipped, no epsilon is added, and no NaN, Inf,
negative cloud water, or inactive branch is sanitized.  The finite and
nonfinite behavior is part of the replica contract.

## Invariant-null-space sources

The JAX kernel constructs all four source densities from the same `A` and `R`
values:

```text
S  =  h * beta2 * A
Qv =  h * A
Qc = -h * (A + R)
Qr =  h * R
```

Consequently

```text
Qv_source + Qc_source + Qr_source = 0
S_source - beta2 * Qv_source = 0
```

up to local float64 operation roundoff.  The sources are not predicted or
computed independently.

## Float64 contract

`dimswe.jax_moist` raises `JAXMoistConfigurationError` when
`jax_enable_x64` is false.  Every input leaf must already have dtype `float64`;
float32 arrays are rejected instead of silently promoted.  Constants in the
kernel are explicitly float64.

The pure functions accept scalar shape `[]`, arbitrary leading batch shapes,
and the deployed cell-major shape `[owned_cells, 16]`.  Unjitted functions and
jitted wrappers are both public.  Boolean masks and reduced margins are
returned only by the separate diagnostic API, not in a differentiated output
pytree.

## Why the boundary is at GLL points

The production order-three quadrilateral method evaluates source forms at a
tensor four-by-four Gauss-Lobatto-Legendre rule.  Its fields do not share a raw
DOF layout: `h` and `S` are CG3 while `Qv`, `Qc`, and `Qr` are DG1.  The
moist source is assembled weakly and followed by a mixed mass solve.

The adapter therefore uses `spaces.CG.broken_space()` as a cell-local carrier.
It has 16 values per cell at the exact production GLL points.  Normal spectral
DG3 is rejected because its nodes are Gauss-Legendre interior points, not the
production GLL points.  Raw mixed state DOFs, DG1 moisture nodes,
`VertexOnlyMesh`, and the stock Firedrake `JaxOperator` are not used as the
primary boundary.

Packing uses a copied cell-node map and copied C-contiguous NumPy arrays.  The
cell map's tensor ordering has the first physical coordinate varying fastest;
the adapter records that exact order explicitly.  Unpacking uses the same map
into independent broken-space Functions.

Forms containing the broken carrier are compiled explicitly with

```python
form_compiler_parameters={"mode": "vanilla"}
```

because the installed default spectral compiler failed on this coefficient
combination during J0 probing.

## Weak assembly and mass solve

The four broken-GLL source carriers are tested against the corresponding
components of the complete mixed test function and assembled using the
production `spaces.dx`.  The resulting object is a complete mixed
`Cofunction`; its velocity and depth blocks are structural/numerical zeros.

The adapter assembles the same complete mixed mass form used by the production
Euler stage and constructs a `LinearSolver` with the production
`erkstage-f` solver parameters.  The source dual is copied before the solve.
The returned tendency and Euler output are independently owned Functions.

## Configured versus applied timestep

The rate kernel receives only the configured physics timestep captured by
`ThreeWayPhysics`.  The adapter separately applies

```text
state_out = state_in + applied_dt * tendency
```

Tests cover applied timesteps equal to, different from, and zero relative to
the configured physics timestep.

## Diagnostics and ownership

Each returned cache owns:

- the input and output states;
- configured and applied timesteps;
- copied parameter values;
- read-only packed state and topography arrays;
- read-only `A`, `R`, and source arrays;
- full GLL diagnostic arrays, masks, signatures, and margins;
- legacy DG1-node masks, signatures, and margins;
- the assembled source `Cofunction`;
- the mass-solved tendency Function.

The legacy diagnostic contract is retained for parity.  It samples DG1 nodes,
whereas the new diagnostic additionally records every actual source-evaluation
GLL point.  Tests demonstrate that these grids have different cardinalities
and margins.

## J2 exact derivative chain

J2 differentiates the unchanged J1 map

```text
Y(x) = x + applied_dt * M^-1 A f(Px),
```

where `P` is the exact J1 broken-CG3/GLL interpolation and packing, `f` is the
unchanged four-channel local source, `A` is the exact J1 weak assembly, and `M`
is the complete mixed mass operator.  No nominal replacement for any of these
operators is introduced.

The certified forward chain is explicitly

```text
x
  --P--> exact broken-GLL q
  --JAX f--> source density
  --A--> mixed source Cofunction
  --M^-1--> tendency
  --> applied Euler update.
```

`dimswe.jax_moist_derivatives` is Firedrake-free and exposes:

- `moist_source_jvp` and `moist_source_jvp_jit`, implemented with `jax.jvp`;
- `moist_source_vjp` and `moist_source_vjp_jit`, implemented with `jax.vjp`;
- `moist_source_differentiated_vjp` and its jitted wrapper, implemented by
  applying `jax.jvp` to a VJP map whose active arguments are both local state
  and output covector.

The differentiated VJP returns exactly

```text
J(q)^T dbar_source + D[J(q)^T bar_source][dq].
```

Production code forms no dense Jacobian or Hessian and defines no custom JVP
or VJP.  Pullback closures are consumed inside a call and are not retained in
public caches.

## Exact transposes around JAX

`P*` uses the adjoint-interpolation API in the installed Firedrake version:

```python
assemble(interpolate(TestFunction(source_space), carrier_cofunction))
```

The packed Euclidean covector is first placed in the algebraic dual of the
same broken carrier with the copied J1 `cell_node_map`.  Firedrake then applies
the transpose of its nodal interpolation into each original CG3 or DG1 state
space.  The four results are installed in a genuine mixed `Cofunction`; the
velocity and `Qr` input blocks are structural zeros.  Point assignment into a
mixed primal space is not used as an approximation to `P*`.

`A*` is assembled as four carrier-dual weak forms,

```text
integral(carrier_test * psi_channel) * production_dx,
```

using the same vanilla form-compiler mode required by J1.  Packing those
carrier `Cofunction` coefficients with the same cell/GLL map includes the GLL
weights, cell geometry, field test functions, carrier ordering, and all four
source channels.  Independent multi-cell tests directly check both transpose
pairings to near roundoff before any reverse-oracle comparison.

The reverse remains dual-native.  An incoming mixed `Cofunction` is scaled by
`applied_dt`, the same complete mixed mass system solves for the primal
auxiliary `psi`, and `A*`, the JAX VJP, and `P*` are applied in that order.  No
Riesz conversion occurs between the child boundary and source reverse.  The
incremental reverse repeats the mass solve for the incoming incremental
adjoint and uses the differentiated VJP for the ordinary-transpose plus local
state-Hessian contributions.

```text
incoming mixed Cofunction
  --M^-*--> primal reverse auxiliary
  --A*--> packed source covector
  --JAX VJP--> packed state covector
  --P*--> mixed Cofunction.
```

For incremental reverse, the JAX differentiated VJP supplies

```text
delta_bar_q =
    J(q)^T delta_bar_source
    + D[J(q)^T bar_source][dq],

mu_x = mu_plus + P* delta_bar_q.
```

## J2 helper ownership and scope

`dimswe.jax_moist_hvp.JAXMoistEulerHVP` owns cache/result dataclasses for the
primal, tangent, reverse, and incremental reverse.  Firedrake Functions and
Cofunctions are deep-copied; packed JAX-facing arrays are copied, C-contiguous,
and read-only.  Results do not alias helper scratch or inputs.  The J1 primal
adapter and the UFL `ProductionMoistEulerHVP` remain available and independent.

The full helper intentionally accepts only the existing fixed moist
parameters.  At the pure local level, state, fields, and parameters remain
separate pytrees; tiny dense tests demonstrate state-parameter,
parameter-state, and parameter-parameter second-order blocks as **local
capability only**.  These blocks are not in the production MTSWE HVP or PyROL
contract.

The strict J1 `jnp.where` algebra is unchanged.  Classical derivative and HVP
claims apply only away from switches.  Every finite-difference certification
requires unchanged legacy DG1 and actual GLL signatures plus scale-separated
GLL margins.  Equality cases record JAX's selected AD result but do not claim a
classical derivative, HVP, or Hessian symmetry.  A dedicated construction
shows a GLL switch pattern that legacy DG1 sampling alone misses.

J2 certification is serial CPU only.  J3 complete-split integration, neural
closure, neural parameters in PyROL, MPI, accelerators, and checkpointing are
explicitly excluded.

## J3 complete-split backend contract

J3 adds the keyword-only construction API

```python
get_timestepper(..., moist_backend="ufl")  # default
get_timestepper(..., moist_backend="jax")  # opt in
```

Only the exact deployed MTSWE graph accepts `"jax"`.  Invalid values and a JAX
request on another split fail during construction.  The four production
integrator slots remain `[RK4, Euler, SSPRK43, moist]`, the subcycles remain
`[2,1,2,1]`, and the expanded children remain

```text
dry_rk4_0
dry_rk4_1
hyperviscosity_euler
dg_ssprk43_0
dg_ssprk43_1
moist_euler
```

For `"ufl"`, the final slot is the unchanged production `Euler` object and
`ProductionMoistEulerHVP` differentiates its retained UFL form.  For `"jax"`,
`JAXMoistEulerIntegrator` uses `JAXMoistEulerPrimal` for the final primal child
and retains the independently constructed UFL Euler object only as an oracle
and coefficient store.  `ProductionMTSWESplitHVP` then selects
`JAXMoistEulerHVP` from that same child choice.  The legacy generic reverse is
rejected for the JAX wrapper so it cannot silently combine a JAX primal with a
UFL derivative.

The first five expanded children, their start times, applied steps, solver
objects, and derivative helpers are not dispatched.  Complete reverse order
remains moist, DG, DG, hyperviscosity, dry, dry, with genuine mixed
`Cofunction`s exchanged directly between children.  The JAX fixed-moist
control wrappers expose exact structural zeros for the moist child physical
`c0` gradient and HVP.  The only physical scalar control therefore remains
hyperviscosity `c0`.

The existing PyROL objective constructors and the normalized map `c0=0.07*z`
are unchanged.  They receive an already configured complete split.  For a JAX
split, active-set qualification uses the actual broken-CG3/GLL diagnostic;
the cache also retains the legacy DG1 diagnostic for historical parity.

`tests/test_jax_moist_full_split.py` is the focused J3 source and external
runtime-certification target.  It contains 11 tests grouped into
backend/graph, one- and three-step primal, tangent, reverse, incremental
reverse/HVP and symmetry, reduced objective, PyROL parity, and
ownership/restoration layers.  Parity failures report timestep, child,
field/block, absolute and relative error, reference norm, both active-set
representations, configured physics step, and applied child step.  Tolerances
are scale-aware float64 multiples of machine epsilon; the accepted production
natural-pairing symmetry standard remains separate.

## Authoritative J3 external certification

The externally executed J3 sequence reported:

```text
J3 complete focused suite:
  11 passed, 11251 warnings in 655.43s

J2 regression:
  34 passed, 236 warnings in 90.95s

J1 regressions:
  36 passed, 223 warnings in 52.62s

Production MTSWE HVP regression:
  22 passed, 29690 warnings in 923.39s

Complete repository:
  260 passed, 1 skipped, 1 xfailed,
  72031 warnings in 2191.37s
```

No `FAILED` or `ERROR` section occurred.  These results certify UFL/JAX
one- and three-step primal, tangent, reverse, incremental-reverse/HVP,
reduced-objective, reduced-gradient, reduced-HVP, and existing PyROL scalar,
initial-condition, and combined-interface parity for the tested serial CPU
configuration.  The J1/J2 helper regressions and the independent production
MTSWE UFL oracle remain green.

## Authoritative J2 external certification

The externally executed J2 sequence reported:

```text
P/P* and A/A* operator transpose tests:
  2 passed, 38 warnings in 45.01s

Tangent:
  3 passed, 147 warnings in 57.07s

Reverse:
  2 passed, 56 warnings in 58.10s

Incremental reverse / HVP:
  3 passed, 83 warnings in 66.65s

Complete J2 derivative file:
  34 passed, 236 warnings in 88.08s

J1 primal regression:
  36 passed, 223 warnings in 50.60s

Accepted production MTSWE HVP regression:
  22 passed, 29690 warnings in 929.70s

Complete repository:
  249 passed, 1 skipped, 1 xfailed,
  60885 warnings in 1808.49s
```

No `FAILED` or `ERROR` section occurred.  The independently reported operator
pairings were:

```text
operator  left                    right                   abs error       rel error
P/P*      1.3227015709416408      1.32270157094164        6.661338e-16   5.036162e-16
A/A*      1.3846557112240454e12   1.3846557112240452e12   2.441406e-4    1.763186e-16
```

The nonzero absolute `A/A*` difference is at a `1.38e12` pairing scale; its
relative error is approximately machine precision.  Representative tangent
absolute/relative discrepancies were

```text
(1.936112702526155e-11,  7.024880901761053e-16)
(7.4113962232369e-10,    1.7814008610261646e-16)
(1.4439336762522103e-11, 7.484409421059483e-16)
(5.686947199638788e-10,  1.9527326243649433e-16)
```

Representative reverse relative discrepancies were approximately `4.98e-16`;
incremental-reverse/HVP relative discrepancies ranged from approximately
`4.76e-16` to `6.01e-16`.

## Authoritative J1 external certification

The externally executed J1 sequence and complete repository suite reported:

```text
Pure JAX:
  20 passed in 2.88s

Named carrier/local Firedrake parity:
  7 passed, 59 warnings in 23.10s

Complete JAX/Firedrake J1 file:
  16 passed, 223 warnings in 41.41s

Accepted moist baseline:
  2 passed, 110 warnings in 109.49s

Accepted production MTSWE HVP regression:
  22 passed, 29690 warnings in 1013.72s

Complete repository:
  215 passed, 1 skipped, 1 xfailed,
  60634 warnings in 1757.04s
```

No `FAILED` or `ERROR` section occurred.  The characterized warnings are the
PyOP2/NumPy shape deprecation, FINAT quadrilateral DG-to-DQ warning, UFL
quadrature-metadata warning, SciPy L-BFGS-B `hessp`/`disp` warnings, and PETSc
seeing pytest's `-q` option.

Coverage includes:

- independent NumPy scalar/batch formulas;
- feasible active branches and ties;
- singular `h`, `h+B`, and gamma denominator cases;
- exponential overflow classification;
- float32 rejection and x64 enforcement;
- JIT/non-JIT parity and immutability;
- multiple-cell GLL coordinates, ordering, and round trips;
- UFL/JAX `C`, `E`, `A`, `R`, `qsat`, and `gamma_v` parity;
- all six mixed source-dual blocks;
- complete mixed mass-solved tendency and Euler-output parity;
- fixed and Real-coefficient parameter modes;
- local, weak, and post-solve invariant checks;
- repeated evaluation, deep ownership, and exception safety.

For a representative finite evaporation/cap/rain case with configured timestep
100 and applied timestep 37, the independently reported comparisons were:

```text
quantity       absolute error          relative error
A              8.470329472543e-22      4.235164736272e-16
R              4.135903062765e-25      4.135903062765e-16
source dual    1.957400529864e-10      2.661180189901e-16
tendency       1.225217394601e-10      1.665741992550e-16
Euler output   4.906538933387e-11      1.326418410191e-21
```

The source-dual and tendency values use the mixed natural L2 norm.  Parity
tests use blockwise, scale-aware tolerances; velocity and depth source/tendency
blocks are checked as exact numerical zeros.

## Certification boundary

J1 certification is serial CPU only.  It establishes the complete primal
chain while preserving its separate numerical layers: pure local JAX algebra,
exact broken-CG3/GLL carrier, weak source assembly, complete mixed mass solve,
and final Euler update.

J2 separately certifies the isolated child's JAX tangent, reverse, and
incremental reverse on serial CPU.  J3 now externally certifies its opt-in
integration through the complete split, reduced maps, and existing PyROL
interfaces for that same serial CPU scope.  Neither J2 nor J3 implies MPI,
GPU/TPU, neural closure, neural parameters in PyROL, or checkpointing support.
