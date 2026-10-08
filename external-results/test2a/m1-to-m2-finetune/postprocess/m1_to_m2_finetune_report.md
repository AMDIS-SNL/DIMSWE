# Test 2A M1 to M2 fine-tuning diagnostic

This is a sequential workflow diagnostic, not a replacement for the matched seed-zero comparison.

| iteration | J_op | J_disc | relative RMS(A) | correlation |
|---:|---:|---:|---:|---:|
| 0 | 0.000373006108793 | 0.000834686430905 | 0.0193133660658 | 0.999812982276 |
| 1000 | 0.000456160694186 | 0.000705452459443 | 0.02135791877 | 0.999771937629 |
| 5000 | 0.000508883179247 | 0.000659878976152 | 0.0225584392024 | 0.999745486755 |
| 10000 | 0.000540777008142 | 0.000624144201594 | 0.0232546126208 | 0.999730271317 |
| 25000 | 0.000568126151023 | 0.000573913338939 | 0.0238353970184 | 0.999716455081 |
| 50000 | 0.000617766931678 | 0.000516735962957 | 0.0248549176558 | 0.999691377878 |

All autonomous evaluations use only truth states 0..80 and do not influence optimization stopping.
Method-1 secant history is not transferred into the Method-2 optimizer.
