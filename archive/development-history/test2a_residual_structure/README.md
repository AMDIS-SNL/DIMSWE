# Test 2A-1 frozen-operator residual structure

This read-only diagnostic uses the final `+5000` memory-20 continuation
parameters and all 331,776 deployed GLL samples from states 0 through 80. It
does not optimize, update parameters, open truth snapshots, or access states
81 through 160.

## Sign and sign-regime metrics

| support | sign accuracy |
| --- | ---: |
| existing non-negligible, `|A| > 1e-6 max|A|` | 0.829248 |
| `|A| > 1e-3 max|A|` | 0.924937 |
| `|A| > 1e-2 max|A|` | 0.989371 |
| `|A| > 1e-1 max|A|` | 1.0 |

For truth `A<0`, relative RMS is `0.094285`, sign accuracy is `0.942689`, and
the regime contributes 44.72 percent of total residual squared energy. For
truth `A>0`, relative RMS is `0.099273`, sign accuracy over every positive
sample is `0.633383`, and the regime contributes 54.49 percent. The low
all-positive sign score is dominated by values arbitrarily close to zero; the
activity-stratified global scores above are the physically more useful sign
controls. Exact-zero targets are 0.518 percent of samples and contribute only
0.786 percent of residual squared energy.

## Magnitude, time, and switching structure

Only 4.66 percent of residual squared energy lies at
`|A| <= 1e-3 max|A|`. Samples above `1e-2 max|A|` contribute 83.89 percent,
and samples above `1e-1 max|A|` contribute 51.02 percent. The latter contain
92.31 percent of target squared energy and have only 7.23 percent relative RMS,
so their large absolute-error contribution mainly reflects the RMS objective's
legitimate weighting of strong events. The intermediate `1e-2` to `1e-1`
band is less accurate, with relative RMS `0.2023`.

The exact deployed switch is

```text
qv     = Qv/h
s      = S/h
q_sat  = q0 H0/(h+B) exp(20(1-s/g))
delta  = qv-q_sat
gamma_v = 1/(1+20 q_sat (g L)/g)
C      = max(0, gamma_v delta/dt)
E      = min(Qc/(h dt), max(0, -gamma_v delta/dt))
A      = E-C
```

Thus `delta=0` is the condensation/evaporation saturation switch. The
cloud-water cap is a second kink; because the deployed law does not clip its
cap below at zero, negative `Qc/h` can also change the realized evaporation
rate's sign. Re-evaluating this certified JAX algebra reproduced every dataset
target exactly (`max error 0`).

The neighborhood `|delta/q_sat| <= 1e-6` contains 15.06 percent of samples but
only 0.244 percent of residual squared energy. The remaining error is therefore
not a phase-boundary-localized failure. Supersaturated condensation candidates
have relative RMS `0.09119` and 36.28 percent of residual energy. Combined
sub-saturated branches carry 63.72 percent; uncapped evaporation is the least
accurate major branch at relative RMS `0.12653`, but no one branch is singular.

Time dependence is present but not localized: steps 0 through 9 contribute
23.93 percent of residual squared energy versus 12.35 percent under uniform
per-state weighting, while the single largest state (step 0) contributes 6.80
percent. Per-state relative RMS ranges from `0.0736` to `0.1254`, and residuals
remain present across all states 0 through 80.

## Recommendation

`CONTINUE_OPTIMIZATION`.

The residual is active-event weighted and moderately biased toward
sub-saturated/evaporation and early-time samples, but it is not concentrated
at the nonsmooth saturation boundary, in one sign alone, or at a small set of
times. Combined with the source run's continuing objective decrease and lack
of stationarity, this supports continuing the same frozen optimization before
changing representation. The operator should not yet be frozen for embedding.

Machine-readable summaries and plots are under
`external-results/test2a/residual-structure-m20-plus5000/`.
