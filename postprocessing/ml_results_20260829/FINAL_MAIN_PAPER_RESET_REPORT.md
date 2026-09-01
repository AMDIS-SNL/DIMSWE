# Final main-paper Machine-Learning Results reset

Status: **PASSED**

The main Results package now uses only M1-Y, H1/M2-Y, H2, and H5. The earlier
M1-X, independent M2-X, warm M2-X, X/Y-control, and full objective-matrix
figures remain in `figures/supplement/`; no accepted artifact was deleted.

## Main figure sequence

1. `ML1_main_optimization` — M1-Y and H1 saved-iteration histories; H2/H5
   initial/final endpoints only.
2. `ML2_main_training_evaluation` — the fitted M1-Y and H1 objectives on
   training and later evaluation data.
3. `ML3_main_callsite_physical_accuracy` — final local-law accuracy at
   truth-derived pre-moist states Y*=P(X*).
4. `ML4_main_deployed_physical_diagnostics` — deployed diagnostics for the
   four main methods.
5. `ML5A/B/C_main_global_trajectories` — truth and the four main methods for
   Representations A, B, and C.

Every main figure was opened and inspected at its rendered size. The sole
layout correction was removal of the crowded 5k tick from the 10k M1-Y axes in
Figures 1 and 2. Their plotted CSV SHA-256 values remained unchanged.

## Fitted-objective histories

M1-Y and H1 training/evaluation histories are available for A/B/C. M1-Y uses
Y*_0..80 and Y*_81..160. H1 uses Y*_0..79 -> X*_1..80 and
Y*_81..159 -> X*_82..160. All six final training objectives reproduce the
accepted values; maximum absolute difference is 1.016e-19.
The H1 evaluation denominator is the same normalized target-energy formula
applied to the later one-step windows. H2/H5 recursive histories were not
computed.

## Final local-law relative RMS errors at Y*

### Representation A: A

| model | training | evaluation |
| --- | --- | --- |
| M1-Y | 0.00493914 | 2.23616 |
| H1 | 0.00676067 | 1.32521 |
| H2 | 0.00675979 | 1.32651 |
| H5 | 0.00676641 | 1.33214 |

### Representation B: A and R

| model | A train | A eval | R all train | R all eval | R active train | R active eval |
| --- | --- | --- | --- | --- | --- | --- |
| M1-Y | 0.00534115 | 2.92705 | 0.0241042 | 0.0327439 | 0.0162778 | 0.0261076 |
| H1 | 0.00591962 | 0.785577 | 0.562805 | 0.55764 | 0.542372 | 0.556559 |
| H2 | 0.00591716 | 0.780347 | 0.562811 | 0.557645 | 0.542373 | 0.556564 |
| H5 | 0.00592671 | 0.771215 | 0.562784 | 0.557617 | 0.542334 | 0.556537 |

### Representation C: source components and normalized source vector

Training states:

| model | S | Qv | Qc | Qr | source vector |
| --- | --- | --- | --- | --- | --- |
| M1-Y | 0.00514134 | 0.00514893 | 0.00514778 | 0.0685739 | 0.00616917 |
| H1 | 0.00630188 | 0.0321911 | 0.0300094 | 1.66459 | 0.0867043 |
| H2 | 0.00913225 | 0.0357141 | 0.0320222 | 1.66362 | 0.0874371 |
| H5 | 0.0234998 | 0.0461292 | 0.0542196 | 1.66918 | 0.093632 |

Evaluation states:

| model | S | Qv | Qc | Qr | source vector |
| --- | --- | --- | --- | --- | --- |
| M1-Y | 1.53695 | 1.51476 | 1.49988 | 0.0821967 | 0.268231 |
| H1 | 2.03068 | 2.49884 | 2.61423 | 0.80783 | 0.892698 |
| H2 | 1.96036 | 2.63524 | 2.61348 | 0.805993 | 0.893289 |
| H5 | 2.05885 | 2.93409 | 2.91101 | 0.80345 | 0.910524 |

The complete metric rows, including Representation-B activation diagnostics,
are in `tables/main/table3_main_callsite_accuracy.csv` and
`data/final_callsite_y_metrics.csv`.

## Deployed call-site coverage

Stored autonomous diagnostics provide model-generated pre-moist-state
Yhat=P(Xhat) local errors for all 12 representation/model combinations. They
are tabulated in `tables/supplement/tableS_deployed_callsite_yhat_accuracy.*`.
Coverage is uniform, but no common A/B/C scalar is reported because the three
representations learn different quantities.

## Scientific completeness of the main subset

Removing X-based methods creates no gap in the deployment-consistent main
narrative: M1-Y, H1, H2, and H5 all have final Y* accuracy and deployed
diagnostics. The M1-X comparison remains scientifically useful as a controlled
sampling-location result and is retained in the supplement. H2/H5 do not have
training/evaluation recursive histories, by the explicit cost gate; Figure 1
therefore shows endpoints and Figure 3 supplies their fixed-array final Y*
accuracy.

## Safety and provenance

No training, optimization, truth generation, prefix/timestep integration,
recursive-history campaign, autonomous rollout, or spatial rerun occurred.
No accepted checkpoint or accepted scientific data was modified. Twelve final
checkpoint hashes were verified. The authoritative repository and M1-Y
workspace retain branch `dev/dimswe-learned-physics-framework`, HEAD `d2f5d66ecb5500aad24eca37280f8a52e22a250f`, and
their frozen status/diff fingerprints.
