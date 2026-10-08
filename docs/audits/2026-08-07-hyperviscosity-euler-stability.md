# Exact deployed hyperviscosity Euler-child stability audit

## Trigger and scope

An external 32-by-32 `doublevortex` pilot at `dt=400` remained well behaved
for `c0=0.07`, while its `c0=0.14` pair developed suspicious late-time growth
after initially dissipative behavior.  The observed hyperviscosity tendency
norm was about `3.57e6`; kinetic energy and projected enstrophy grew; sampled
high-k fraction reached about `1.28e-5`.  This is not accepted as physical
small-scale population.

This audit adds an opt-in diagnostic only.  It changes no production
hyperviscosity, timestepper, J1/J2/J3, or J4A mathematics and advances no state
when producing a resolution table.

## Exact operator and sign

For each active production field `i in {v,h,S}`, direct inspection of
`dimswe/dissipation.py` and `dimswe/timestepping.py` gives

```text
M_i Q_i = -K_i x_i,
M_i F_i = c0 r^s K_i Q_i,
x_i^+ = [I-dt*c0*r^s*(M_i^-1 K_i)^2]x_i,
r=max(mesh.dx/order,mesh.dy/order).
```

The second sign follows because `Hyperviscosity.rhs` contains the negative
weak form and `GeneralRK` forms its stage right-hand side as `-model.rhs`.
Qv, Qc, and Qr are inactive.  If `K_i phi=mu_i M_i phi`, then
`lambda_i=r^s mu_i^2`, and explicit Euler is non-growing exactly when
`dt*c0*lambda_i,max <= 2`.

## Sparse method and reporting

`dimswe/hyperviscosity_stability.py` assembles the exact configured M and K as
PETSc AIJ matrices.  Serial CSR is retained.  After verifying diagonal GLL M
and symmetric K, the code uses the symmetric similar matrix
`B=M^-1/2 K M^-1/2` with deterministic ARPACK Lanczos.  It reports the Ritz
residual and a conservative maximum-absolute-row-sum upper bound.  Production
matrices are never dense.

For each v/h/S, nx/ny, c0, dt, and s row the diagnostic records lambda_max,
its conservative upper bound, sigma, largest-mode amplification, conservative
Euler amplification bound, estimated and conservative dt_max, and
`0.8*conservative_dt_max` as the documented recommended dt.  Classification
distinguishes a Ritz-detected violation, a conservative stability certificate,
and an inconclusive gap.

## Tiny oracle and focused validation

Dense conversion is guarded by an explicit dof ceiling and used only for the
2-by-2 oracle.  The oracle also applies one actual deployed hyperviscosity
child to an owned deterministic random state.

- v: 72 dofs; h and S: 36 dofs each.
- sparse/dense eigenvalue relative discrepancy: at most `1.36e-16`.
- matrix/deployed-child relative discrepancy: at most `1.64e-16`.
- inactive Qv/Qc/Qr change: exactly zero.
- focused result: `19 passed, 44 warnings in 4.29s` for
  `tests/test_hyperviscosity_stability.py` plus
  `tests/test_resolved_hidden_c0_prep.py`.

Two earlier oracle attempts were blocked before a result because Firedrake and
pytools selected read-only virtual-environment caches.  Redirecting
`PYOP2_CACHE_DIR`, `FIREDRAKE_TSFC_KERNEL_CACHE_DIR`, `XDG_CACHE_HOME`, and
`MPLCONFIGDIR` to `/tmp` resolved the environment limitation; the unchanged
oracle then passed.

## Analysis semantic correction

Pilot analysis now reports finite-state checks independently of numerical
stability.  It never infers stability from finite coefficients or positive h.
Separate serialized heuristics can warn about late median growth in kinetic
energy, projected enstrophy, hyperviscosity tendency, and high-k fraction.
They are explicitly neither necessary nor sufficient proofs of instability.
A high-k threshold crossing is no longer called dynamically populated modes.

## Gate

No further resolved Test-1B run is authorized by this audit.  The external
16/32/64 sparse stability table must be obtained and reviewed first.  An
inconclusive upper-bound gap is not permission to run.  No flow, resolution,
timestep, or duration is selected here.
