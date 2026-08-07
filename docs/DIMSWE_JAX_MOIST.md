# DIMSWE exact JAX moist primal boundary

Status: **J1 COMPLETE — EXTERNALLY CERTIFIED, SERIAL CPU ONLY.**

This document describes the opt-in JAX replica of the deployed moist Euler
child.  The existing UFL `ThreeWayPhysics` term remains unchanged, independent,
and the only default production implementation.  J1 does not install a runtime
backend switch.

## Scope

J1 implements only

```text
mixed Firedrake state
  -> exact cell-local broken-CG3/GLL values
  -> pure float64 JAX rates and coupled source densities
  -> production-GLL weak source assembly
  -> complete mixed mass solve
  -> explicit Euler state update
```

It does not implement JAX JVPs, VJPs, differentiated VJPs, HVP integration,
PyROL integration, a neural network, checkpointing, MPI certification, or
accelerator certification.

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

## Authoritative external certification

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

It does not imply MPI, GPU/TPU, JAX derivative, neural-closure, split runtime
backend-integration, or checkpointing support.  No derivative or neural
certification follows from the primal tests.
