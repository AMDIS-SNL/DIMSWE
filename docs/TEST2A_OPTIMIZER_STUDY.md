# Test 2A-1 controlled optimizer study

## Frozen comparison

This study changes only the optimizer.  All methods use the same
`doublevortex_A_operator.npz` dataset, 331,776 samples, feature order
`(h,S,Qv,Qc,B)`, physical A target, 5-32-32-1 tanh float64 model, seed-0
Glorot parameter pytree, training normalization, and deterministic full-batch
normalized MSE.  The common starting-pytree SHA-256 is
`6d4ac7bafe775b90a70e2a199ef3305308c3d02333f7daeac1c870b6f18e0975`.

The study reads the prepared local operator dataset only.  It does not open
truth snapshots, advance DIMSWE, inspect states after 80, or warm-start one
method from another.

## Installed ROL capability audit

The installed ROL exposes unconstrained line-search quasi-Newton and
trust-region methods.  The tested methods are:

- line-search `Limited-Memory BFGS`, memory 10, 500 outer iterations;
- line-search `Limited-Memory BFGS`, memory 20, 500 outer iterations; and
- trust region with `Truncated CG`, exact objective HVPs, at most 100 outer
  iterations and 50 inner CG iterations.

The trust-region configuration explicitly sets `Use as Hessian=false` and
`Use as Preconditioner=false`, so truncated CG calls the certified exact JAX
HVP rather than a secant action.  A tiny analytic test confirms real HVP
callbacks.  L-BFGS makes no HVP callbacks.

Dense full BFGS is not exposed by this ROL installation.  Its
`SecantFactory` supports limited-memory BFGS/DFP/SR1, Barzilai-Borwein, and a
user-defined type.  A dense 1,281-by-1,281 float64 matrix would require about
13.1 MB, so storage would be reasonable; availability is the blocker.  The
study does not mislabel large-memory L-BFGS as full BFGS.  No SciPy, Adam, or
AdamW method is used.

Each callback family is JIT-warmed before timing.  Warm-up time and measured
ROL solver time are reported separately.  Callback counters are reset after
warm-up, so objective/gradient/HVP counts describe the ROL solve itself.
Accepted-iterate histories are separate from all line-search trial objective
calls.

## Convergence and cost result

All three methods exhausted their configured outer-iteration limit.  None is
reported as converged.

| method | accepted iterations | termination | final J | final gradient norm | objective / gradient / HVP calls | solver wall (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| L-BFGS m=10 | 500 | `MAXITER` | 0.2624279837 | 1.3748734610 | 1045 / 501 / 0 | 28.0243 |
| L-BFGS m=20 | 500 | `MAXITER` | 0.2539424566 | 0.9221656692 | 1041 / 501 / 0 | 28.1173 |
| trust-region truncated CG, exact HVP | 100 | `MAXITER` | 0.4693190769 | 0.1449791698 | 101 / 82 / 720 | 38.1170 |

Every method starts at `J=0.9135568694` and gradient norm
`0.2890821398`.  The final L-BFGS gradient norms are larger than the initial
norm despite much lower objectives, so their last iterates are not credible
stationary points.  Trust-region truncated CG has a lower final gradient norm
than either L-BFGS result but is still far from the study's transparent
stationarity screen, `||g|| <= max(1e-6, 1e-4 ||g_initial||)`.

The HVP count conclusion applies only to this local neural objective.  It says
nothing about future solver-in-loop HVP cost.

## Physical accuracy

| method | relative RMS | active `1e-3` relative RMS | active `1e-6` relative RMS | correlation | sign accuracy | physical MAE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100-iteration L-BFGS reference | 0.818622 | 0.816569 | 0.817220 | 0.572916 | 0.540720 | 3.38988e-9 | 1.01282e-7 |
| L-BFGS m=10 | 0.512277 | 0.501536 | 0.502573 | 0.858598 | 0.696129 | 2.30235e-9 | 7.07055e-8 |
| L-BFGS m=20 | 0.503927 | 0.497904 | 0.498925 | 0.864432 | 0.710626 | 2.17654e-9 | 8.41959e-8 |
| trust-region truncated CG | 0.685069 | 0.677304 | 0.678056 | 0.732962 | 0.658710 | 2.90084e-9 | 8.57434e-8 |

The affine baseline has relative RMS `0.919464`, correlation `0.390297`, and
sign accuracy `0.621556`.  All extended runs improve RMS and correlation.
L-BFGS m=20 is the strongest accuracy/cost compromise for this fixed
objective: it slightly improves every central accuracy measure over memory 10
at essentially the same wall time.  It does not dominate every statistic
(memory 10 has the smaller maximum absolute error), and neither run is
stationary.

## Scientific decision

The accepted classification is
`OPTIMIZATION_AND_MODEL_BOTH_UNRESOLVED`:

- extending the identical memory-10 method from 100 to 500 iterations lowers
  relative RMS from `0.819` to `0.512`, proving the old 100-iteration result
  was budget-limited;
- nevertheless, no tested method reaches convincing stationarity;
- the best current relative RMS remains about `0.504`, active-region relative
  RMS about `0.498`, and sign accuracy about `0.711`.

Thus it is not yet possible to attribute the remaining error primarily to
representation or optimization.  The fixed MLP is not ready for embedding.
Beating the affine baseline does not override the O(1) relative error and weak
sign accuracy.

The next controlled step is one explicitly labeled continuation of the
memory-20 L-BFGS result with a larger budget and stationarity monitoring.  It
must not replace this common-start comparison.  Test 2A-2 should begin only if
that continuation either reaches credible stationarity with strong accuracy,
or establishes that a stationary solution of this fixed model remains
inadequate.  Architecture/features/normalization should not be changed until
that distinction is resolved.

That continuation has now been completed.  Over 1,500 additional accepted
iterations it reduced relative RMS from `0.504` to `0.205`, but remained at
`MAXITER` with material recent progress and sign accuracy `0.813`.  The updated
classification is `CONTINUING_OPTIMIZER_LIMITED`, and the model is still not
ready for embedding.  See `docs/TEST2A_CONTINUATION.md`.

## Outputs

Generated outputs remain untracked under
`external-results/test2a/optimizer-study/`:

```text
optimizer_comparison.json
convergence_cost_summary.json
physical_accuracy_summary.json
objective_vs_iteration.png
relative_rms_comparison.png
<method>.json
<method>_parameters.npz
<method>_parameters.json
```
