# Supplementary table. Local moist-physics error on deployed pre-moist states

Stored local errors evaluated on model-generated pre-moist states Yhat=P(Xhat). Coverage exists for all four main models and all representations, but each representation learns a different quantity.

| physical_case | representation | model_label | state | regime | quantity | physical_RMS_error | normalized_RMS_error | relative_RMS_error | target_RMS | sample_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Test 2B | A | M1-Y | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | A | 6.988e-09 | 0.07719 | 0.1078 | 6.48e-08 | 10485760 |
| Test 2B | A | H1 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | A | 4.148e-09 | 0.04582 | 0.06421 | 6.46e-08 | 10485760 |
| Test 2B | A | H2 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | A | 4.146e-09 | 0.0458 | 0.06418 | 6.46e-08 | 10485760 |
| Test 2B | A | H5 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | A | 4.124e-09 | 0.04555 | 0.06383 | 6.46e-08 | 10485760 |
| Test 2B | B | M1-Y | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | A | 3.389e-09 | 0.03744 | 0.05249 | 6.456e-08 | 10485760 |
| Test 2B | B | M1-Y | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | R_all | 1.351e-13 | 0.00679 | 0.02917 | 4.633e-12 | 10485760 |
| Test 2B | B | M1-Y | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | R_truth_active | 5.59e-13 | 0.02809 | 0.0249 | 2.245e-11 | 453004 |
| Test 2B | B | H1 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | A | 3.245e-09 | 0.03585 | 0.05027 | 6.455e-08 | 10485760 |
| Test 2B | B | H1 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | R_all | 2.697e-12 | 0.1355 | 0.5566 | 4.846e-12 | 10485760 |
| Test 2B | B | H1 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | R_truth_active | 1.337e-11 | 0.6718 | 0.5545 | 2.412e-11 | 429032 |
| Test 2B | B | H2 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | A | 3.233e-09 | 0.03572 | 0.0501 | 6.454e-08 | 10485760 |
| Test 2B | B | H2 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | R_all | 2.701e-12 | 0.1357 | 0.5566 | 4.853e-12 | 10485760 |
| Test 2B | B | H2 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | R_truth_active | 1.338e-11 | 0.6725 | 0.5544 | 2.414e-11 | 429374 |
| Test 2B | B | H5 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | A | 3.186e-09 | 0.0352 | 0.04937 | 6.454e-08 | 10485760 |
| Test 2B | B | H5 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | R_all | 2.679e-12 | 0.1346 | 0.5568 | 4.811e-12 | 10485760 |
| Test 2B | B | H5 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | R_truth_active | 1.329e-11 | 0.668 | 0.5546 | 2.397e-11 | 428020 |
| Test 2B | C | M1-Y | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | S_source | 0.000269 | 0.04033 | 0.05654 | 0.004758 | 10485760 |
| Test 2B | C | M1-Y | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | Qv_source | 2.712e-06 | 0.03987 | 0.05589 | 4.853e-05 | 10485760 |
| Test 2B | C | M1-Y | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | Qc_source | 2.677e-06 | 0.03935 | 0.05517 | 4.853e-05 | 10485760 |
| Test 2B | C | M1-Y | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | Qr_source | 3.567e-10 | 0.02366 | 0.1227 | 2.907e-09 | 10485760 |
| Test 2B | C | H1 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | S_source | 0.0009774 | 0.1465 | 0.2017 | 0.004845 | 10485760 |
| Test 2B | C | H1 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | Qv_source | 9.495e-06 | 0.1396 | 0.1922 | 4.941e-05 | 10485760 |
| Test 2B | C | H1 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | Qc_source | 9.668e-06 | 0.1421 | 0.1957 | 4.941e-05 | 10485760 |
| Test 2B | C | H1 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | Qr_source | 2.064e-09 | 0.1369 | 9.274e+298 | 0 | 10485760 |
| Test 2B | C | H2 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | S_source | 0.001191 | 0.1785 | 0.2438 | 0.004884 | 10485760 |
| Test 2B | C | H2 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | Qv_source | 1.13e-05 | 0.1662 | 0.2269 | 4.981e-05 | 10485760 |
| Test 2B | C | H2 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | Qc_source | 1.152e-05 | 0.1693 | 0.2313 | 4.981e-05 | 10485760 |
| Test 2B | C | H2 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | Qr_source | 2.015e-09 | 0.1337 | 9.058e+298 | 0 | 10485760 |
| Test 2B | C | H5 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | S_source | 0.002374 | 0.3558 | 0.4605 | 0.005154 | 10485760 |
| Test 2B | C | H5 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | Qv_source | 1.889e-05 | 0.2776 | 0.3593 | 5.256e-05 | 10485760 |
| Test 2B | C | H5 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | Qc_source | 2.035e-05 | 0.2991 | 0.3872 | 5.256e-05 | 10485760 |
| Test 2B | C | H5 | model-generated pre-moist state Yhat=P(Xhat) | ALL deployed calls | Qr_source | 1.552e-09 | 0.1029 | 6.974e+298 | 0 | 10485760 |
