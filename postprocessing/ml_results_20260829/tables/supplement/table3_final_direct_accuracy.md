# Table 3. Final direct prediction accuracy

## Representation A

| Model | Support | A RMS | A rel. RMS | A bias | A corr. |
| --- | --- | --- | --- | --- | --- |
| M1-X | train 0--80 | 4.950e-10 | 5.468e-03 | -8.630e-13 | 1 |
| M1-X | held-out 81--160 | 2.907e-09 | 2.23 | 2.447e-11 | 0.276 |
| M1-Y | train 0--80 | 9.005e-10 | 9.948e-03 | 4.530e-11 | 1 |
| M1-Y | held-out 81--160 | 6.400e-09 | 4.92 | 6.345e-10 | 0.143 |
| M2-X-independent | train 0--80 | 4.987e-10 | 5.509e-03 | 7.296e-13 | 1 |
| M2-X-independent | held-out 81--160 | 4.503e-09 | 3.46 | 5.351e-10 | 0.203 |
| warm M2-X | train 0--80 | 4.589e-10 | 5.070e-03 | -5.239e-13 | 1 |
| warm M2-X | held-out 81--160 | 2.589e-09 | 1.99 | 6.694e-11 | 0.327 |
| H1 | train 0--80 | 1.127e-09 | 0.0125 | 9.256e-11 | 1 |
| H1 | held-out 81--160 | 3.676e-09 | 2.82 | -6.258e-10 | 0.221 |
| H2 | train 0--80 | 1.129e-09 | 0.0125 | 9.188e-11 | 1 |
| H2 | held-out 81--160 | 3.679e-09 | 2.83 | -6.332e-10 | 0.221 |
| H5 | train 0--80 | 1.135e-09 | 0.0125 | 9.128e-11 | 1 |
| H5 | held-out 81--160 | 3.695e-09 | 2.84 | -6.343e-10 | 0.22 |

## Representation B

| Model | Support | A rel. RMS | R rel. RMS | active-R rel. RMS | R FP rate | R FN rate |
| --- | --- | --- | --- | --- | --- | --- |
| M1-X | train 0--80 | 6.291e-03 | 0.014 | 9.594e-03 | 0.29 | 0 |
| M1-X | held-out 81--160 | 2.9 | 0.0207 | 0.0189 | 0.571 | 0 |
| M1-Y | train 0--80 | 0.0106 | 0.0241 | 0.0163 | 0.294 | 0 |
| M1-Y | held-out 81--160 | 6.34 | 0.0328 | 0.0261 | 0.2 | 0 |
| M2-X-independent | train 0--80 | 5.476e-03 | 4.46 | 1.33 | 0.111 | 1 |
| M2-X-independent | held-out 81--160 | 3.44 | 1.65 | 1.22 | 0.108 | 1 |
| warm M2-X | train 0--80 | 5.044e-03 | 0.434 | 0.429 | 0.821 | 1.765e-03 |
| warm M2-X | held-out 81--160 | 0.949 | 0.461 | 0.461 | 0.767 | 1.420e-03 |
| H1 | train 0--80 | 0.0102 | 0.563 | 0.542 | 0.826 | 0 |
| H1 | held-out 81--160 | 1.61 | 0.558 | 0.557 | 0.795 | 0 |
| H2 | train 0--80 | 0.0102 | 0.563 | 0.542 | 0.826 | 0 |
| H2 | held-out 81--160 | 1.6 | 0.558 | 0.557 | 0.795 | 0 |
| H5 | train 0--80 | 0.0102 | 0.563 | 0.542 | 0.826 | 0 |
| H5 | held-out 81--160 | 1.58 | 0.558 | 0.557 | 0.795 | 0 |

## Representation C

| Model | Support | S RMS | Qv RMS | Qc RMS | Qr RMS | effective A rel. | effective R rel. | off-manifold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1-X | train 0--80 | 3.935e-05 | 3.992e-07 | 4.005e-07 | 5.331e-11 | 6.397e-03 | 0.0414 | 1.297e-03 |
| M1-X | held-out 81--160 | 2.066e-04 | 2.145e-06 | 2.437e-06 | 2.647e-10 | 2.39 | 0.0588 | 6.393e-03 |
| M1-Y | train 0--80 | 6.550e-05 | 6.625e-07 | 6.527e-07 | 8.914e-11 | 0.0109 | 0.0687 | 1.030e-03 |
| M1-Y | held-out 81--160 | 3.043e-04 | 3.060e-06 | 3.027e-06 | 3.802e-10 | 3.32 | 0.0824 | 2.553e-03 |
| M2-X-independent | train 0--80 | 2.991e-05 | 2.936e-06 | 4.232e-06 | 1.980e-08 | 0.0337 | 15.5 | 0.0521 |
| M2-X-independent | held-out 81--160 | 3.750e-04 | 7.797e-06 | 1.140e-05 | 2.090e-08 | 5.55 | 4.69 | 0.164 |
| warm M2-X | train 0--80 | 2.914e-05 | 1.511e-06 | 1.285e-06 | 1.590e-09 | 0.0145 | 1.23 | 0.0171 |
| warm M2-X | held-out 81--160 | 2.985e-04 | 4.423e-06 | 4.974e-06 | 4.275e-09 | 4.34 | 0.926 | 0.031 |
| H1 | train 0--80 | 6.015e-05 | 1.953e-06 | 1.917e-06 | 2.167e-09 | 0.0195 | 1.68 | 0.0259 |
| H1 | held-out 81--160 | 4.089e-04 | 5.161e-06 | 5.370e-06 | 3.735e-09 | 5.1 | 0.813 | 0.0348 |
| H2 | train 0--80 | 6.442e-05 | 2.167e-06 | 2.025e-06 | 2.166e-09 | 0.0221 | 1.68 | 0.026 |
| H2 | held-out 81--160 | 3.951e-04 | 5.455e-06 | 5.374e-06 | 3.727e-09 | 5.15 | 0.811 | 0.0374 |
| H5 | train 0--80 | 1.417e-04 | 2.835e-06 | 3.537e-06 | 2.173e-09 | 0.0361 | 1.69 | 0.0365 |
| H5 | held-out 81--160 | 4.154e-04 | 6.050e-06 | 5.959e-06 | 3.715e-09 | 5.62 | 0.809 | 0.0441 |

## Test 2A scope

Test 2A accuracy is available only on training support. Representation A has complete direct-A diagnostics; Representation C has stored component RMS and structural diagnostics but not a historically matched held-out suite. These values remain in `data/final_direct_metrics.csv` and are not mixed into the Test 2B panels.
