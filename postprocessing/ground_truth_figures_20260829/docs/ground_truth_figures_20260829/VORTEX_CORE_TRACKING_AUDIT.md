# Vortex-core tracking discontinuity audit

## Conclusion

The apparent jumps near 1.9 h and 4.1 h are gridpoint-winner switches, not physical discontinuities of the vorticity cores. A continuously tracked, fixed-physical-radius centroid has no corresponding jump. This audit uses only the previously extracted Test 2B relative-vorticity cache; Firedrake and the truth extractor were not rerun.

## Raw-maximum frame inspection

At 6700--6800 s, the raw maximum for core 1 switches from
\((2009.657,2009.657)\) km to \((2009.657,1974.718)\) km. Core 2 switches symmetrically from
\((2990.343,2990.343)\) km to \((2990.343,3025.282)\) km. These are adjacent GLL locations separated by 34.9386 km in \(y\).

| frame | old-location \(\zeta\) | new-location \(\zeta\) | winning point | relative gap |
|---|---:|---:|---|---:|
| step 67, 6700 s | 15.0141287 | 15.0135736 | old | \(3.697\times10^{-5}\) |
| step 68, 6800 s | 15.0177708 | 15.0181904 | new | \(2.794\times10^{-5}\) |

At 14,800--14,900 s, core 1 switches from
\((2009.657,1974.718)\) km to \((1974.718,1974.718)\) km, and core 2 switches symmetrically from
\((2990.343,3025.282)\) km to \((3025.282,3025.282)\) km. These are adjacent GLL locations separated by 34.9386 km in \(x\).

| frame | old-location \(\zeta\) | new-location \(\zeta\) | winning point | relative gap |
|---|---:|---:|---|---:|
| step 148, 14,800 s | 15.4546795 | 15.4541435 | old | \(3.468\times10^{-5}\) |
| step 149, 14,900 s | 15.4525900 | 15.4533567 | new | \(4.962\times10^{-5}\) |

Here \(\zeta\) is in cached units of \(10^{-5}\ {\rm s^{-1}}\). An eight-neighbor test finds one discrete local maximum in each frame: the old point is the local maximum before the switch and the new adjacent point after it. There are not two spatially distinct vortex-scale extrema. The broad peak drifts smoothly through a tie between adjacent samples, while an argmax necessarily changes discontinuously.

## Continuous fixed-radius definition

For each core and each saved frame, the revised center is

\[
\boldsymbol c_n =
\boldsymbol c_{n-1}
+\frac{\sum_{\boldsymbol x_j\in D(\boldsymbol c_{n-1},600\,{\rm km})}
\zeta_j^+\,\boldsymbol\delta_j}
{\sum_{\boldsymbol x_j\in D(\boldsymbol c_{n-1},600\,{\rm km})}\zeta_j^+},
\]

where \(\zeta_j^+=\max(\zeta_j,0)\), \(\boldsymbol\delta_j\) is the periodic minimum-image displacement from the previous center, and the result is wrapped back to the 5000 km periodic domain. The radius is fixed in physical units, both disks remain disjoint, and the center is never recentered on the raw argmax.

Across the two raw switches, the continuous centroid motions are:

| transition | raw argmax motion | continuous core-1 motion | continuous core-2 motion |
|---|---:|---:|---:|
| 6700--6800 s | 34.9386 km | 0.11683 km | 0.11683 km |
| 14,800--14,900 s | 34.9386 km | 0.13625 km | 0.13625 km |

The maximum continuous one-frame motion anywhere in the 161-state record is 0.14356 km.

## Continuous motion results

Both vortex centroids have the same maximum displacement from their respective step-0 centers:

\[
d_{\max}=18.0029\ {\rm km}
=0.00360058\,L
=0.0480077\,\sigma.
\]

Thus the maximum is 0.3601% of the 5000 km domain and 4.8008% of the 375 km vortex width. Pair separation increases smoothly from 1438.4651 to 1465.4986 km, a maximum change of 27.0334 km (1.879% of the initial centroid separation). Pair orientation changes from \(45.0000^\circ\) to \(45.9385^\circ\), and the pair midpoint remains fixed to approximately \(9\times10^{-9}\) km.

## Threshold and radius sensitivity

An independent centroid retaining only points above 50% of the local maximum, within the same previous-center 600 km disk, gives:

- maximum core displacement: 18.2126 km;
- maximum one-frame motion: 3.1879 km;
- maximum separation change: 28.9014 km;
- maximum orientation change: \(0.8805^\circ\).

The few-kilometer threshold-track steps are caused by GLL points entering or leaving the hard threshold, but are still an order of magnitude smaller than the raw 34.9386 km argmax switches.

Across all-positive radii of 375, 500, and 600 km and threshold fractions 0.3, 0.5, and 0.7, maximum core displacement is 15.69--20.90 km, maximum separation change is 25.97--34.75 km, and maximum orientation change is below \(0.94^\circ\). The stationary-skeleton conclusion is insensitive to these reasonable definitions.

## Artifacts and outputs

The revised CSV and NPZ retain raw maxima, continuous fixed-radius centroids, and the 50%-threshold centroids as distinct columns. The JSON metadata contains every competing vorticity value, grid-index switch, sensitivity result, method constant, source-cache hash, and output hash:

- **outputs/ground_truth_figures_20260829/data/test2b_vortex_core_tracks.csv**
- **outputs/ground_truth_figures_20260829/data/test2b_vortex_core_tracks.npz**
- **outputs/ground_truth_figures_20260829/data/test2b_vortex_core_tracks.json**
- **outputs/ground_truth_figures_20260829/figures/figure5_test2b_vortex_core_motion.png**
- **outputs/ground_truth_figures_20260829/figures/figure5_test2b_vortex_core_motion.pdf**

Figure 5 now uses the continuous fixed-radius centroid. Figures 1--4 are unchanged.
