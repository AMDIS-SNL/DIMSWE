# Test 2A-1 convergence-to-plateau continuation

## Frozen problem and provenance

This study continues the selected Test-2A memory-20 ROL line-search L-BFGS
fit from the completed `+1500` continuation endpoint. It changes none of the
dataset, five features, normalization, 5-32-32-1 tanh architecture, seed,
full-batch normalized-MSE objective, optimizer, or L-BFGS memory.

The source artifact was verified before execution:

```text
source objective                 0.04196877146795583
source gradient norm             0.14015904191539993
NPZ SHA-256                      346691b4792a62b8aebab15a72d7988637489d85b739410b231703adba115c09
pytree-value SHA-256             8ab4c1556a9d5125552e4aff0bd4eb2c20d3fabbc72b43c565caa1e2c579a676
parameter count                  1,281
```

The machine-readable study contract is
`dimswe/configs/test2a_m20_plateau_continuation.json`. The run allowed 5,000
additional accepted iterations with exact JAX gradients, no production HVP,
and checkpoints at `+250,+500,+1000,+2000,+3000,+4000,+4500,+4750,+4900,+5000`.
The uninterrupted process retained its process-local memory-20 secant history.
The prior 500-iteration optimizer-study and 1,500-iteration continuation
outputs were not overwritten.

## Result

ROL reached the additional-iteration limit. `MAXITER` is recorded as the
actual termination and is not interpreted as convergence.

| quantity | +1500 source | +5000 endpoint |
| --- | ---: | ---: |
| normalized MSE `J` | 0.0419687715 | 0.00946959743 |
| relative RMS(A) | 0.204862811 | 0.097311857 |
| active-`1e-3` relative RMS(A) | 0.199547773 | 0.095015401 |
| active-`1e-6` relative RMS(A) | 0.200698541 | 0.095816521 |
| correlation | 0.978760598 | 0.995242602 |
| complete existing sign accuracy | 0.813272930 | 0.829248266 |
| sign accuracy, `|A| > 1e-3 max|A|` | 0.900443637 | 0.924937292 |
| sign accuracy, `|A| > 1e-2 max|A|` | 0.961715543 | 0.989371498 |
| sign accuracy, `|A| > 1e-1 max|A|` | 1.0 | 1.0 |

The endpoint physical RMSE is `9.10341e-10`, MAE is `4.69221e-10`, and
maximum absolute error is `1.77850e-8`. Relative RMS fell 52.50 percent from
this continuation's source and is about 89.4 percent below the affine
baseline value `0.919464385`. The complete sign metric remains deliberately
reported, but it includes targets down to `1e-6 max|A|`; the activity-stratified
values distinguish those near-zero samples from physically stronger phase
change.

The additional segment used 10,430 objective evaluations, 5,001 gradient
evaluations, zero HVPs, and 287.34 seconds of measured solver time.

## Mathematical versus practical convergence

```text
actual ROL gradient norm                    0.0563421686
gradient / continuation-start gradient     0.401987398
final parameter-step / parameter norm      4.48533e-5
last-50 objective reduction                 0.5146 percent
last-100 objective reduction                1.0641 percent
last-100 relative-RMS reduction             0.5335 percent
```

The ROL gradient tolerance (`1e-8`) and step tolerance (`1e-12`) were not met.
More importantly for the requested practical screen, the final 100 accepted
steps still reduced the objective by about one percent, the gradient has only
fallen to 0.402 of this continuation's starting value, the relative parameter
step is not small, and relative RMS changed by 0.534 percent between the
`+4900` and `+5000` physical checks. Correlation and sign metrics were more
stable, but all required plateau signals do not agree.

## Decision

The classification is `STILL_OPTIMIZER_LIMITED`. Accuracy is now substantially
stronger, especially in the more active sign strata, but the requested
convergence-to-plateau study did not establish a plateau within the additional
5,000-iteration cap. Therefore this result alone does not yet authorize
Test-2A-2 embedding under the stated decision rule. This conclusion separates
mathematical nonstationarity from physical approximation quality; it does not
claim that ROL's absolute `1e-8` gradient tolerance is necessary for eventual
embedding.

Generated parameters, JSON records, checkpoints, and plots are under
`external-results/test2a/optimizer-study/continuation-m20-plateau/` and remain
untracked.
