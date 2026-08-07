# 2026-08-06 DIMSWE exact JAX moist J0/J1 audit

Status: **J1 COMPLETE — EXTERNALLY CERTIFIED, SERIAL CPU ONLY**

## Repository boundary

J1 was implemented on branch `dev/dimswe-jax-moist-closure` from required base
HEAD `47bfde1721ed37b91c8f8ea7a186ad958f701511`.

The existing UFL physics, `dimswe/timestepping.py`, and
`dimswe/mtswe_split_hvp.py` were not modified.  No runtime switch was added.
The ordinary package import remains independent of JAX.

No remote modification, push, derivative integration, PyROL integration,
neural network, checkpointing, MPI certification, or accelerator certification
is part of this audit.

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

J2 remains responsible for JVP, VJP, differentiated VJP, incremental reverse,
and HVP integration.  J3 remains responsible for the complete split runtime
switch and PyROL parity.  Neural work remains deferred to J4.
