# Test 2A-1 memory-20 L-BFGS continuation

## Contract

This is a warm-started continuation of the accepted memory-20 L-BFGS
500-iteration optimizer-study result.  It does not restart from seed 0 and
does not change the dataset, features, normalization, architecture,
activation, objective, optimizer, memory, or full-batch policy.

The source parameter artifact was verified before ROL execution:

```text
NPZ SHA-256
  81e37295e262c92673729d2c5e7785256e1f82131b066a2478c73b64060b923e
pytree-value SHA-256
  baa8d12f649c3ecccb538b5ba5c6f9544defc8bd6dcf0151098cb07f9bb3b988
parameter count
  1,281
```

The machine-readable contract is
`dimswe/configs/test2a_m20_continuation.json`.  ROL uses line-search
limited-memory BFGS with memory 20, exact JAX gradients, gradient tolerance
`1e-8`, step tolerance `1e-12`, and a limit of 1,500 additional accepted
iterations.  No HVP is used.

## Checkpoint/restart behavior

Parameter checkpoints and physical metrics are written at additional
iterations `+100,+250,+500,+1000,+1500`.  A progress JSON stores artifact
fingerprints, accepted history available at the checkpoint, cumulative
callback counts, and the latest checkpoint path.  `--resume` restarts from
those exact parameter values rather than the original iteration-500 fit.

ROL's secant pairs live only inside its running C++ process and are not exposed
for serialization.  A normal uninterrupted run retains memory-20 history for
all 1,500 steps, as this accepted run did.  A process-level resume necessarily
starts with empty secant history and records `secant_history_restored=false`;
this is parameter restartability, not exact optimizer-state checkpointing.

## Periodic physical metrics

| additional accepted iterations | J | relative RMS | active `1e-3` relative RMS | correlation | sign accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.253942457 | 0.503927 | 0.497904 | 0.864432 | 0.710626 |
| 100 | 0.220645279 | 0.469729 | 0.460910 | 0.882950 | 0.711859 |
| 250 | 0.175480154 | 0.418904 | 0.407943 | 0.908425 | 0.751712 |
| 500 | 0.120869788 | 0.347663 | 0.339159 | 0.937472 | 0.757274 |
| 1000 | 0.069830616 | 0.264255 | 0.258995 | 0.964371 | 0.801513 |
| 1500 | 0.041968771 | 0.204863 | 0.199548 | 0.978761 | 0.813273 |

At the endpoint, active-`1e-6` relative RMS is `0.200699`, physical RMSE is
`1.91647e-9`, MAE is `9.03576e-10`, and maximum absolute error is
`3.70024e-8`.

Relative to the continuation start:

```text
objective decrease                 0.2119736851
relative RMS-error reduction       59.3467 percent
correlation increase               0.1143289670
sign-accuracy increase             0.1026473503
```

The affine baseline remains much worse: relative RMS `0.919464`, correlation
`0.390297`, and sign accuracy `0.621556`.

## Convergence and stationarity diagnostics

ROL terminated at `EXITSTATUS_MAXITER`; this is not convergence.  The solve
used 3,148 objective evaluations, 1,501 gradient evaluations, zero HVPs, and
84.18 seconds of measured solver time.

```text
starting gradient norm             0.9221656692
final gradient norm                0.1401590419
final/start gradient ratio         0.1519890044
final step / final parameter norm  2.366159e-4
last-50 objective decrease         0.0018012090 (4.115 percent)
last-100 objective decrease        0.0036729015 (8.047 percent)
```

The objective is still decreasing materially, the relative step is not tiny,
and the gradient has not met either ROL's tolerance or the continuation's
multi-signal plateau screen.  No stationarity claim is made from objective
change alone.

## Decision

The classification is `CONTINUING_OPTIMIZER_LIMITED`.  Continued optimization
substantially improves the same fixed representation and no practical plateau
is evident.  This is much stronger evidence of optimizer limitation than the
500-iteration study alone.

The operator is still not ready for Test 2A-2.  Although relative RMS and
active-region errors have reached about 0.20, sign accuracy is only 0.813 and
the solution has not stabilized.  The next step is another explicitly bounded
memory-20 continuation from the `+1500`/final checkpoint, retaining all frozen
model and data choices.  Architecture changes and solver embedding remain
premature.

Generated results are under
`external-results/test2a/optimizer-study/continuation-m20/` and remain
untracked.
