# Test 2A stored-UFL/JAX offset audit

Date: 2026-08-09

This is a read-only follow-on to the completed H=1/Method-2 equivalence
audit.  It performs no optimization and loads only truth states 0 through
80.  The complete machine-readable record is
`external-results/test2a/backend-offset-audit/backend_offset_audit.json`.

## Finding

The reported `1.2452933152003626e-6` relative mixed-state discrepancy was
not a UFL-versus-analytical-JAX backend offset.  The earlier audit used the
trajectory case's `moist_helper`.  That case is deliberately constructed
with the frozen neural-A local physics.  A parameterless call to such a
helper evaluates its stored frozen neural parameters; it does not switch the
helper back to analytical A.

The relevant routing is explicit in the source:

- `test2a_trajectory_certification._build_case` loads compatible neural
  physics and supplies it to the JAX case;
- `ProductionMTSWESplitHVP` passes that local physics to
  `JAXMoistEulerHVP`;
- `JAXMoistEulerPrimal.__init__` selects `local_physics.combined_kernel`
  when local physics is present; and
- `JAXMoistEulerPrimal.evaluate(..., neural_parameters=None)` calls the
  already selected combined kernel.  Here, “no explicit parameter
  argument” means “use the helper's frozen neural parameters,” not “use
  analytical A.”

The completed equivalence audit called this parameterless path while
constructing its purported analytical post-prefix target.  This follow-on
audit instead constructs a separate helper with `local_physics=None` for a
genuinely analytical JAX child.

## Three-way decomposition

For each transition `k=0,...,79`, the audit reconstructed

`Y_k = (C5 o C4 o C3 o C2 o C1)(X*_k)`

with the production child ordering and times, and then formed

- `A_k = Z^UFL_k - X*_{k+1}` (stored/provenance component),
- `B_k = Z^JAX,analytical_k - Z^UFL_k` (genuine backend component), and
- `C_k = Z^JAX,analytical_k - X*_{k+1}` (total).

The maximum over all 80 transitions was:

| component | max absolute mixed-mass norm | max relative mixed-mass norm |
|---|---:|---:|
| stored/provenance `A_k` | 0 | 0 |
| analytical JAX minus UFL `B_k` | 2.1049165046639062e-6 | 5.672172660597087e-17 |
| analytical JAX minus stored `C_k` | 2.1049165046639062e-6 | 5.672172660597087e-17 |

`C_k = A_k + B_k` held with zero maximum coefficient discrepancy.  Fresh
UFL thus reproduced every stored next state bitwise.  Genuine analytical JAX
agreed with fresh UFL to float64 operation-order accuracy.  For comparison,
the frozen-neural helper reproduced the earlier maximum relative mixed-state
offset exactly: `1.2452933152003626e-6` (absolute mixed-mass norm
`46205.931692944214`).

The historical post-prefix `Y_k` states were not stored, so direct bitwise
comparison with an original intermediate is impossible.  Nevertheless, the
truth metadata records the UFL backend, `dt=100`, and every-step boundary
output; the current reconstruction uses the production order and times; and
fresh prefix plus fresh UFL child reproduces every stored boundary bitwise.
There is no observed provenance/split/time/copy discrepancy.

## Field decomposition

Maximum field errors for the genuine analytical-JAX-minus-UFL component are:

| field | absolute L2 norm | relative L2 norm | max coefficient difference |
|---|---:|---:|---:|
| v | 0 | 0 | 0 |
| h | 0 | 0 | 0 |
| S | 1.834818306835024e-6 | 4.969767094839437e-17 | 9.094947017729282e-13 |
| Qv | 8.946700735358055e-9 | 1.2693910767165289e-15 | 5.773159728050814e-15 |
| Qc | 9.00468473984425e-9 | 2.1845364306121444e-12 | 6.00380679958816e-15 |
| Qr | 0 | 0 | 0 |

The stored/provenance component is exactly zero in every field.  The total
component is therefore identical to the table above.  The earlier
frozen-neural comparison also left `v`, `h`, and `Qr` exactly unchanged, but
its maximum coefficient differences were `0.07794655734141998` in `S` and
about `2.97400305711e-4` in each of `Qv` and `Qc`.

At the local source level, the water identity residual was exactly zero and
the maximum `S-beta2*Qv` source residual was
`1.734723475976807e-18`.  After projection into the distinct `S` and moisture
spaces, the weak/integrated identities are subject to cancellation of
roundoff-sized residuals; coefficientwise `S-beta2*Qv` is not a valid test
because those fields have different coefficient layouts.

## Rate and downstream audit

Across 80 post-prefix states (327,680 cell-local 4x4 GLL samples):

| rate | max absolute difference | RMS difference | relative RMS | sign disagreements |
|---|---:|---:|---:|---:|
| analytical A, JAX vs UFL | 1.0257163672749426e-19 | 2.718914151537897e-20 | 2.6406116398783088e-12 | 0 |
| analytical R, JAX vs UFL | 0 | 0 | zero reference | 0 |

Both condensation and evaporation branches were represented.  Their maximum
A discrepancies were approximately `1.026e-19`; exact-zero A samples stayed
exactly zero.  In contrast, the frozen neural A used by the earlier audit had
relative RMS error `0.06426965192269384`, maximum absolute error
`1.4604953961801801e-8`, and 40,374 sign disagreements out of 327,680 samples.
Its original R remained identical and exactly zero on this support.

The first observed float64 differences between genuine analytical UFL and
JAX are in derived local arithmetic (`qv`, `qc`, `qsat`, and `gamma_v`) at
roughly `1e-20` to `1e-15`; the directly packed `h,S,Qv,Qc,B` inputs agree
exactly.  These propagate to:

- maximum local source differences of `7.751219676147784e-17` in `Qv/Qc`
  and `7.600961960535502e-15` in `S`;
- maximum source-dual natural relative discrepancy
  `5.99844963797776e-12`;
- an exactly identical JAX and UFL mass solve when supplied the same RHS;
  and
- maximum tendency natural relative discrepancy
  `5.9984586379663056e-12`.

Thus there is no non-roundoff stage in the genuine analytical UFL/JAX chain.
In the earlier comparison, the first material discrepancy is the local A
provider selection: frozen neural A versus analytical UFL A.  Cell ordering,
the 4x4 GLL points, source injection, weak assembly, and the mass solve are
not the cause.

## Scale relative to the moist update

The genuine analytical-JAX/UFL state difference is at most
`1.081489146382033e-11` of the true moist increment in mixed mass norm.
Maximum fieldwise ratios are `9.978174786585026e-12` for `S`,
`3.269675725407536e-11` for `Qv`, and `3.3517328323844925e-11` for `Qc`.
Relative to the physical A target, its RMS discrepancy is
`2.6406116398783088e-12`.

The previously reported frozen-neural offset is instead as much as
`0.11959805796756147` of the moist increment in mixed mass norm; maximum
fieldwise ratios reach `0.11564586720295648` for `S` and approximately
`1.07544` for each of `Qv` and `Qc`.  This is neural approximation error at
post-prefix states, not backend error.

## Reconciliation with earlier certification

The J1 moist tests compared analytical UFL expressions and analytical JAX
rates at identical fresh GLL inputs, followed by source-dual, mass-solved
tendency, and Euler-output parity.  The J3 full-split test compared fresh UFL
and analytical-JAX complete steps and required bitwise equality through the
first five child boundaries before checking child 6 at scale-aware float64
tolerances.  Stored Test-1B truth states were not targets in those tests.

Those certifications are logically consistent with this result and need no
revision.  The completed H=1/Method-2 audit's *interpretation* of its
“analytical JAX versus stored UFL” control must be revised: that control was
actually frozen-neural JAX versus stored analytical-UFL.  Its stored-target
H=1 equivalence, fixed-prefix result, and H=2 cross-time conclusion are not
invalidated.

## Target recommendation

Define a future M2-Y/H=1 physical target with the genuinely analytical JAX
child, `C6_star(Y_k)`.  This directly asks the neural model to approximate the
intended deployed analytical A law in the deployment backend.  Keep the
stored UFL boundary target as an independent provenance/parity check.  On
this trajectory the two analytical targets differ only at float64
operation-order scale, but the explicit analytical-JAX definition prevents a
future backend/provenance discrepancy from being folded into the learned A
law.

