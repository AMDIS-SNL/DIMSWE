# 2026-08-06 DIMSWE exact JAX moist J0/J1/J2/J3 audit

Status: **J1/J2/J3 COMPLETE — EXTERNALLY CERTIFIED, SERIAL CPU ONLY.**

## Repository boundary

J1 was implemented on branch `dev/dimswe-jax-moist-closure` from required base
HEAD `47bfde1721ed37b91c8f8ea7a186ad958f701511`.

The existing UFL physics, `dimswe/timestepping.py`, and
`dimswe/mtswe_split_hvp.py` were not modified.  No runtime switch was added.
The ordinary package import remains independent of JAX.

No remote modification, push, complete-split derivative integration, PyROL
integration, neural network, checkpointing, MPI certification, or accelerator
certification is part of this audit.

J2 work begins on branch `dev/dimswe-jax-moist-derivatives` at certified J1
checkpoint `f3451cd85c6406afe46e01a10faee22cfb94cc8d`.  The existing UFL
oracle, production mathematics, timestepping, split composition, and J1 primal
files remain independent and unchanged.

## J0 findings carried into J1

The deployed moist operator is a quadrature-evaluated weak source followed by
a complete mixed mass solve.  It is not a pointwise map over raw mixed state
DOFs.  CG3 thermal/depth fields and DG1 water fields are evaluated at the
production four-by-four GLL points.

The exact local rates are condensation `C`, capped evaporation `E`, rain `R`,
and net vapour rate `A = E - C`.  The source is the two-coordinate invariant
null-space representation

```text
(S, Qv, Qc, Qr)
  = h*A*(beta2, 1, -1, 0) + h*R*(0, 0, -1, 1).
```

The configured physics timestep appears in every local relaxation rate and in
the evaporation cap.  The applied Euler timestep is a separate multiplier
after the mass solve.

Installed UFL selects the second max/min operand at equality.  J1 implements
the equivalent strict comparisons with `jnp.where`.

## Architecture decision

The accepted J1 boundary is:

```text
Firedrake interpolation to broken CG3/GLL
  -> copied cell-major float64 arrays
  -> CPU JAX local kernel
  -> copied source arrays
  -> independent broken-GLL source Functions
  -> vanilla-compiled production-GLL weak form
  -> complete mixed mass solve
  -> Euler update
```

Rejected primary boundaries were raw DOF arrays, normal spectral DG3,
DG1 diagnostic nodes, `VertexOnlyMesh`, the stock first-order-only JAX external
operator, generated non-JAX point kernels, and opaque pure callbacks.

The multiple-cell layout test found that Firedrake's broken-space
`cell_node_map` stores the first physical coordinate fastest, opposite the
axis-table ordering returned by the FInAT tensor dual basis.  The adapter now
records the actual packed ordering and permanently tests it on four cells.

## Files introduced

- `dimswe/jax_moist.py`
- `dimswe/jax_moist_adapter.py`
- `tests/test_jax_moist_local.py`
- `tests/test_jax_moist_firedrake.py`
- `docs/DIMSWE_JAX_MOIST.md`
- `docs/audits/2026-08-06-dimswe-jax-moist-design.md`

## Authoritative external certification

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

The tests establish local formula parity, branch/tie behavior, nonfinite
classification, x64-only execution, exact multiple-cell GLL layout, UFL rate
parity, six-block source-dual parity, production-equivalent mass-solved
tendency, Euler output at multiple applied timesteps, fixed/Real parameter
modes, invariant consequences, diagnostic-grid distinction, ownership,
repeatability, and induced-exception safety.

One independent representative diagnostic reported relative errors
`4.24e-16` for `A`, `4.14e-16` for `R`, `2.66e-16` for the mixed source dual,
`1.67e-16` for the mixed tendency, and `1.33e-21` for the Euler output.  The
corresponding absolute errors were `8.47e-22`, `4.14e-25`, `1.96e-10`,
`1.23e-10`, and `4.91e-11`, respectively; their physical scales differ by many
orders of magnitude, so the tests retain blockwise scale-aware criteria.

## Certification boundary and deferred implementation

J1 certification is serial CPU only.  It covers the distinct primal layers:
pure local JAX algebra, exact broken-CG3/GLL carrier, weak source assembly,
complete mixed mass solve, and final Euler update.  It does not imply MPI,
GPU/TPU, JAX derivative, neural-closure, split runtime backend-integration, or
checkpointing support.

J2 certifies JVP, VJP, differentiated VJP, and incremental reverse for the
isolated moist child.  J3 remains responsible for the complete split runtime
switch and PyROL parity.  Neural work remains deferred to J4.

## J2 implementation audit

The certified derivative graph is the exact J1 composition

```text
x
  --P--> exact broken-GLL q
  --JAX f--> source density
  --A--> mixed source Cofunction
  --M^-1--> tendency
  --> applied Euler update.
```

The reverse is

```text
incoming mixed Cofunction
  --M^-*--> primal reverse auxiliary
  --A*--> packed source covector
  --JAX VJP--> packed state covector
  --P*--> mixed Cofunction.
```

Incremental reverse uses

```text
delta_bar_q =
    J(q)^T delta_bar_source
    + D[J(q)^T bar_source][dq]

mu_x = mu_plus + P* delta_bar_q.
```

The pure local API in `dimswe/jax_moist_derivatives.py` uses `jax.jvp`,
`jax.vjp`, and `jax.jvp` of a state-and-covector VJP map.  Parameters and
topography are explicit separate pytrees but fixed in the full helper.  Dense
Jacobians and Hessians occur only in tiny tests.  No custom derivative rule is
used and no Boolean diagnostic enters a differentiated output pytree.

`dimswe/jax_moist_hvp.py` introduces the independent
`JAXMoistEulerHVP`.  Its `P*` is the installed Firedrake adjoint interpolation
from a broken-carrier `Cofunction` into each original state-space dual.  Its
`A*` is a vanilla-compiled weak carrier-dual assembly against the primal mixed
auxiliary.  These choices preserve the copied J1 cell/GLL ordering and include
quadrature weights, geometry, basis values, and field placement.  Reverse and
incremental reverse remain dual-native and use the same complete mixed mass
solve; there is no intermediate Riesz conversion.

Owned dataclasses cover primal, tangent, reverse, and incremental-reverse
results.  Public caches contain deep-copied Functions/Cofunctions and
read-only copied arrays, not JAX pullback closures.  Tests exercise
repeatability, non-aliasing, immutability, structural zeros, and recovery after
an induced exception.

Both active-set grids are retained.  Finite differences require equality of
base/plus/minus legacy DG1 and actual GLL signatures and scale-separated GLL
margins.  A piecewise CG3 construction is included for which every legacy DG1
condensation sample is active while the production GLL samples contain both
branches, demonstrating why the legacy signature is insufficient.

The authoritative external J2 certification is:

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

No `FAILED` or `ERROR` section occurred.  The certified transpose pairings
were:

```text
P/P*:
  left  = 1.3227015709416408
  right = 1.32270157094164
  abs   = 6.661338147750939e-16
  rel   = 5.036161061643471e-16

A/A*:
  left  = 1.3846557112240454e12
  right = 1.3846557112240452e12
  abs   = 2.44140625e-4
  rel   = 1.763186494815942e-16
```

Representative tangent absolute/relative discrepancies were
`(1.936112702526155e-11, 7.024880901761053e-16)`,
`(7.4113962232369e-10, 1.7814008610261646e-16)`,
`(1.4439336762522103e-11, 7.484409421059483e-16)`, and
`(5.686947199638788e-10, 1.9527326243649433e-16)`.  Reverse relative
discrepancies were approximately `4.98e-16`; incremental-reverse/HVP relative
discrepancies ranged from approximately `4.76e-16` to `6.01e-16`.

The local parameter-block test is explicitly local capability only.  Trainable
moist parameters are not exposed through the full helper or PyROL.  J2 is
certified on serial CPU only.  No J3 complete-split switch, neural closure,
neural parameters in PyROL, MPI or accelerator claim, checkpointing, remote
change, or push is part of J2.

## J3 final source and certification audit

J3 started from the externally certified J2 checkpoint
`51ae8692e5f28389a304a0387f1542577fb54198` on branch
`dev/dimswe-jax-moist-full-split`.  The only pre-existing untracked paths were
the protected `.DS_Store` and `docs/.DS_Store`; the index was empty.

The opt-in API is `get_timestepper(..., moist_backend="jax")`; omission and
explicit `"ufl"` select the unchanged UFL child.  `dimswe.moist_backend`
validates the two-value selector and wraps only the final deployed
`terms=['threewayphysics']` Euler object.  The wrapper calls the certified
`JAXMoistEulerPrimal`, keeps the separately constructed UFL Euler as an oracle,
and rejects the legacy generic reverse to prevent a mixed-backend derivative.

`ProductionMTSWESplitHVP` reads the same selector from the parent split.  It
uses `ProductionMoistEulerHVP` for UFL and the certified `JAXMoistEulerHVP` for
JAX.  The dry RK4, hyperviscosity Euler, and DG SSPRK43 helpers are unchanged.
The JAX moist result is augmented only with structural physical-`c0` zeros;
all derivative Functions, Cofunctions, and packed arrays remain the owned J2
objects.  No Riesz map is inserted between children.

The existing PyROL public vector and objective types are unchanged.  Its
active-set selector now uses actual GLL margins when that diagnostic exists
and the legacy UFL diagnostic otherwise.  JAX caches retain both GLL and DG1
representations; no threshold is loosened.

The focused `tests/test_jax_moist_full_split.py` collects 11 tests.  It covers
the default and graph contract, every one-step boundary, three-step primal
trajectories, all three control-direction blocks for tangent and incremental
reverse, complete reverse boundaries, reduced value/gradient/HVP, natural
bilinear symmetry, existing scalar/IC/combined PyROL interfaces, ownership,
repeatability, and induced-exception recovery.  The accepted production suite
remains unchanged as the deep UFL oracle.

The final source audit confirms that omission and explicit `"ufl"` still use
the unchanged UFL child and that `"jax"` remains opt-in.  Backend dispatch
changes only child 6.  The first five children, six-child order, child start
times, child timestep allocation, and complete reverse order are unchanged.
Physical `c0` remains hyperviscosity-only; the JAX moist child contributes
structural zeros to its `c0` gradient and HVP.  Child reverse maps exchange
genuine mixed Cofunctions without an inter-child Riesz conversion.  Existing
PyROL scalar, initial-condition, and combined vector/objective interfaces and
their normalization are unchanged.  J1/J2 helpers are reused rather than
reimplemented.

The authoritative external J3 certification is:

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

No `FAILED` or `ERROR` section occurred.  The focused result certifies
JAX/UFL one- and three-step primal, tangent, reverse, incremental-reverse/HVP,
reduced objective/gradient/HVP, and existing PyROL scalar, IC, and combined
parity in the tested serial CPU configuration.  The J2, J1, and independent
production-MTSWE-oracle regressions all remain green.

No neural network, neural parameters, state normalization, checkpointing,
MPI/accelerator certification, remote change, push, stage, or commit is part
of J3.
