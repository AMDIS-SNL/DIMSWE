# Production M1 state-location audit

**Conclusion:** production M1 constructs its network input and analytical
target at the timestep-boundary truth state `X_n^*`.  It does **not** apply
the deterministic split prefix `P` and does **not** construct M1 samples at
`Y_n^*=P(X_n^*)`.

This conclusion follows from the production config-to-objective code path and
direct reconstruction of immutable stored samples.  Synthesis prose was not
used as evidence.

## State definitions used here

- `X_n^*`: restart snapshot `restart/step_NNNNNNNN.npy`, loaded directly as
  the truth state at the timestep boundary.
- `Y_n^*=P(X_n^*)`: state after split children 1--5 and immediately before
  child 6, `moist_euler`.  In the cached full-step API this is
  `complete.boundary_states[-2]`.

The observed child order in both numerical reconstructions was
`dry_rk4_0`, `dry_rk4_1`, `hyperviscosity_euler`, `dg_ssprk43_0`,
`dg_ssprk43_1`, `moist_euler`.

## Frozen Test-2A Representation-A M1

### Production trace

1. The production long-fit config
   `dimswe/configs/test2a_fair_operator_200k.json:4-7` selects
   `dimswe/configs/test2a_selected_operator.json` and the immutable
   `external-results/test2a/dataset/doublevortex_A_operator.npz`.  The
   selected config records feature order `(h,S,Qv,Qc,B)`, target
   `cache.rates['A']`, and truth states 0--80
   (`dimswe/configs/test2a_selected_operator.json:3-19`).
2. The production CLI dispatches `train-operator` to `train_operator_long`
   (`dimswe/test2a_fair_longfit.py:640-672`).  That routine loads the already
   prepared dataset, applies its frozen normalization, and sends those arrays
   directly to the M1 operator objective
   (`dimswe/test2a_fair_longfit.py:120-138`).  It contains no model timestep or
   prefix call in sample construction.
3. The dataset constructor loads each truth snapshot directly from
   `restart/step_NNNNNNNN.npy` as `trajectory.states[step]`
   (`dimswe/resolved_hidden_c0_inference.py:82-117`).  For each step 0--80 it
   executes exactly one
   `adapter.evaluate(trajectory.states[step], case.dt)` and takes both packed
   features and `cache.rates['A']` from that same cache
   (`dimswe/test2a_operator.py:750-779`).
4. The adapter interpolates the passed state itself into packed
   `h,S,Qv,Qc`, passes those arrays and `B` to the analytical kernel, and
   returns its rates in the same cache
   (`dimswe/jax_moist_adapter.py:571-647`).  The deployed feature map stacks
   exactly those components (`dimswe/test2a_operator.py:364-379`).

Therefore the exact state passed to the feature construction is `X_n^*`, and
the exact state passed to `A^*` is the same `X_n^*`.  No `P` is evaluated by
production Test-2A M1 preparation or training.

### Numerical spot check

Frozen case:

- truth run: `external-results/test1b-production/truth_c0_0.14`;
- dataset: `external-results/test2a/dataset/doublevortex_A_operator.npz`;
- `n=40`, `t=4000`, flattened deployed point 3479 (cell 217, local GLL 7);
- restart SHA256:
  `5acb6d6a8960987437e52672cd49ac378a0c32bc525b881791b1e5486118c4f7`;
- dataset SHA256:
  `5e800c22a8945cbb1ed2449f97026111e7a1c46acb74feed6ce7ac770399cb0a`.

| quantity | frozen M1 row | independent `X_40^*` | independent `Y_40^*` |
|---|---:|---:|---:|
| normalized `h` | -0.007478952905067314 | -0.007478952905067314 | 0.007903307737827388 |
| normalized `S` | -0.01925668864067062 | -0.01925668864067062 | 0.001565401989462463 |
| normalized `Qv` | -0.04029698893997063 | -0.04029698893997063 | -0.038542645870950736 |
| normalized `Qc` | 2.622443139020776 | 2.622443139020776 | 2.6296746633108996 |
| normalized `B` | 0 | 0 | 0 |
| `A^*` | -2.1807073638294305e-8 | -2.1807073638294305e-8 | -3.378377050022745e-8 |

The frozen feature row is bitwise equal to reconstructed `X`; its maximum
absolute difference from `Y` is `2.0822090630133086e-2`.  The frozen `A^*`
matches both the independently transcribed formula and production adapter at
`X` exactly, while it differs from the `Y` value by
`1.1976696861933145e-8`.

## Frozen rain-active Test-2B M1

### Production trace

1. `dimswe/configs/test2b_rain_active_learning.json:4-10` selects the frozen
   truth run and states 0--80; lines 17--27 specify the common
   `(h,S,Qv,Qc,B)` input and A/B/C output contracts.
2. The CLI `train` command reaches `train(...)`, which loads `objectives(...)`
   and selects `fixed['M1']` for stage M1
   (`dimswe/test2b_rain_learning_campaign.py:816-830,898-930`).
3. `objectives(...)` supplies `x_features`, `x_A`, and `x_R` to M1; it
   supplies the separate `y_*` arrays only to H1
   (`dimswe/test2b_rain_learning_campaign.py:478-503`).  Representation A's
   M1 consumes `x_A`; Representation B consumes `x_A,x_R`; Representation C
   forms the four conservative source targets from those same X-state rates
   (`dimswe/test2b_rain_learning_campaign.py:72-110`).
4. Preparation loads every `truth[step]` directly from its restart snapshot
   (`dimswe/test2b_rain_learning_campaign.py:206-215`).  `_analytical_arrays`
   makes one `adapter.evaluate(state,dt)` call and takes both the five input
   components and `A/R` from that same result
   (`dimswe/test2b_rain_learning_campaign.py:196-203`).  `prepare_data` passes
   `[truth[i] for i in range(81)]` to this routine to make `x_*`
   (`dimswe/test2b_rain_learning_campaign.py:218-231`).
5. Only afterward, in a separate branch, preparation advances a full step,
   retains `boundary_states[-2]`, and creates `y_*`; those data feed H1, not
   M1 (`dimswe/test2b_rain_learning_campaign.py:232-250`).

Thus all Test-2B A/B/C M1 inputs and analytical targets are evaluated at the
same `X_n^*`.  The prefix is evaluated during shared preparation only to
construct the separately named H1 arrays; it is not applied in the M1 path.

### Numerical spot checks

Frozen case:

- truth run:
  `external-results/test2b-rain-active-truth/production-n64-zeta-m0p06-dt100-t16000`;
- dataset:
  `external-results/test2b-rain-active-learning/preparation/fixed_learning_data.npz`;
- `n=60`, `t=6000` (rain-onset training regime);
- restart SHA256:
  `e821cf6aff775d61c0433e1eca02d53cadb3497fcecd16a48c7d56dab1d7b62f`;
- dataset SHA256:
  `6e159015234fd94881b0b97888b7481eb049a02dcd96c571078920c0bedc901c`.

At flattened point 62064 (cell 3879, local GLL 0), the stored Representation-A
M1 feature differs from reconstructed `X_60^*` by at most
`4.440892098500626e-16` and from reconstructed `Y_60^*` by
`3.566547078699234e-2`.  Its target is
`A^*=1.9066869679218322e-8`, exactly the independent and adapter value at `X`;
the independent `Y` value is `3.0889498399827324e-8`, a difference of
`1.1822628720609002e-8`.

A second stored point was selected specifically to test the rain target used
by Representation-B M1: flattened point 48307 (cell 3019, local GLL 3).

| quantity | frozen M1 row | independent `X_60^*` | independent `Y_60^*` |
|---|---:|---:|---:|
| normalized `h` | -0.14171676696955296 | -0.14171676696955296 | -0.1434870667733267 |
| normalized `S` | -0.21969550319932288 | -0.21969550319932288 | -0.221722295762373 |
| normalized `Qv` | 0.01630126351252047 | 0.01630126351252047 | 0.016099264978141268 |
| normalized `Qc` | 1.2162909678665177 | 1.2162909678665175 | 1.2142187694868478 |
| normalized `B` | 0 | 0 | 0 |
| `A^*` | -2.266647765399186e-9 | -2.2666477653991863e-9 | -2.7410366348952805e-9 |
| `R^*` | 7.112575638169817e-12 | 7.112575638169817e-12 | 6.808710835478906e-12 |

Here the maximum stored-feature difference is `2.220446049250313e-16` versus
`X`, but `2.072198379669876e-3` versus `Y`.  Stored `R^*` equals the independent
and adapter X-state value exactly and differs from the Y-state value by
`3.0386480269091125e-13`.  Stored `A^*` agrees with the independent X formula
to `4.14e-25` and differs from Y by `4.743888694960946e-10`.

The independent calculation is a NumPy transcription of the analytical
formula, not a call back into the production JAX law.  The diagnostic invokes
one full forward step only to construct the counterfactual `Y` comparison; it
does not modify truth or participate in sample generation.

## Report equation

The M1 report equation must use **`X_n^*`**, unambiguously:

```text
z_n = Normalize([h,S,Qv,Qc,B](X_n^*))
A_target,n = A^*(X_n^*)
R_target,n = R^*(X_n^*)              # when the representation learns R
```

Using `Y_n^*` in an M1 training equation would be factually wrong.  `Y_n^*`
is the correct input location for H1/M2-Y and for the network when deployed at
the moist split child, so those equations should remain explicitly distinct
from M1.

Machine-readable results are in `M1_STATE_LOCATION_TEST2A.json`,
`M1_STATE_LOCATION_TEST2B.json`, and
`M1_STATE_LOCATION_TEST2B_RAIN_TARGET.json`.  Their reconstruction code is
`scripts/feature_sufficiency_20260828_m1_state_spotcheck.py`.
