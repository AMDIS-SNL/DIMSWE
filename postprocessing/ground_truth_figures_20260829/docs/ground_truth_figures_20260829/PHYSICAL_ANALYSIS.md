# Verified truth physics and event chronology

## Immutable truth records and diagnostic timing

The analysis reads, without modifying, these accepted truth records:

- Test 2A: **external-results/test1b-production/truth_c0_0.14**
- Test 2B: **external-results/test2b-rain-active-truth/production-n64-zeta-m0p06-dt100-t16000**

The requested complete Test 2B path was verified rather than inferred. Both records contain 161 states, steps 0--160, at a natural cadence of 100 s through 16,000 s.

Saved-state fields are diagnosed at their labeled time. The rates \(A\) and \(R\) are recomputed by replaying split children 1--5 and applying the exact analytical moist law to the post-prefix state entering child 6. They are therefore the tendencies acting over the following 100 s moist Euler child. Spatial extrema and active fractions use the complete GLL diagnostic support, not the reduced image grid.

## Test 2A and Test 2B distinction

| Property | Test 2A | Test 2B |
|---|---:|---:|
| mesh | \(16\times16\) | \(64\times64\) |
| vapor parameter \(\zeta\) | \(0\) | \(-0.06\) |
| analytical \(q_v/q_{\rm sat}\) at \(t=0\) | \(1\) | \(1.06\) |
| analytical initial regime | saturated | 6% supersaturated |
| \(Q_c,Q_r\) at \(t=0\) | both zero | both zero |
| first \(R>0\) | never | step 51, 5100 s |
| maximum local \(q_c\) | \(3.11771\times10^{-5}\) at 8500 s | \(1.05179\times10^{-4}\) at 8900 s |
| rain threshold \(q_{\rm precip}\) | \(10^{-4}\) | \(10^{-4}\) |
| final integrated \(Q_c\) | \(1.05434\times10^{11}\ {\rm m^3}\) | \(1.55151\times10^{12}\ {\rm m^3}\) |
| final integrated \(Q_r\) | \(0\) | \(2.30733\times10^8\ {\rm m^3}\) |

The comparison is a regime comparison, not a grid-convergence experiment: initial vapor loading and resolution both change.

### Initial saturation and the sign of \(A\)

With \(A=E-C\), negative \(A\) means condensation and positive \(A\) means evaporation.

- Test 2A is exactly saturated in the analytical initializer. Projection between the CG and DG spaces makes the deployed post-prefix step-0 saturation ratio range from 0.952934 to 1.009264. Consequently step 0 has condensation on 58.3984% of the full GLL support, \(A=0\) elsewhere, and no evaporation. Every saved state from step 1 onward has both signs of \(A\). This is a discrete representation effect, not an analytical supersaturation imposed by the case.
- Test 2B is analytically 6% supersaturated. Its deployed step-0 post-prefix ratio is 1.056853--1.060699 and \(A<0\) everywhere. Both condensation and evaporation occur at every saved state from step 1 onward as the flow reorganizes saturation and condensate.

The global full-record \(A\) ranges are

\[
\begin{aligned}
{\rm Test\ 2A}:&\quad -1.09005\times10^{-7}
\le A\le 1.04482\times10^{-7}\ {\rm s^{-1}},\\
{\rm Test\ 2B}:&\quad -8.52867\times10^{-7}
\le A\le 4.77115\times10^{-8}\ {\rm s^{-1}}.
\end{aligned}
\]

### Cloud and rain evolution

Test 2A creates cloud water through projection-triggered and subsequently flow-induced condensation/evaporation. Its local cloud maximum occurs at 8500 s but reaches only 31.18% of the rain threshold. Integrated cloud water fluctuates and reaches its stored maximum at the final state. \(R\) and \(Q_r\) are exactly zero at every stored state.

Test 2B undergoes strong initial condensation and rapid cloud accumulation. The local cloud threshold is first crossed at step 51. Local \(q_c\) and local \(R\) both peak at 8900 s; the integrated production rate peaks later at 12,000 s because the rain-active support has expanded. Integrated \(Q_c\) fluctuates but reaches its stored maximum at the final state. Since \(R\ge0\) and there is no rain sink in this source law, integrated \(Q_r\) grows monotonically after onset.

## Independently verified Test 2B chronology

| Event | Step | Time | Diagnostic definition and value |
|---|---:|---:|---|
| initial | 0 | 0 s | \(Q_c=Q_r=0\); deployed ratio 1.056853--1.060699; \(A<0\) everywhere |
| last clearly pre-rain | 50 | 5000 s | \(R=0\) on all 65,536 GLL samples and integrated \(hR=0\); \(\max q_c=9.98910\times10^{-5}\) |
| first certifiable rain | 51 | 5100 s | 4/65,536 GLL samples have \(R>0\); \(\max q_c=1.001832\times10^{-4}\); \(\int hR\,dA=0.129326\ {\rm m^3\,s^{-1}}\) |
| sustained-rain start | 51 | 5100 s | beginning of uninterrupted meaningful \(R\) support through the record |
| sustained-rain certification | 61 | 6100 s | 11 consecutive states/1000 s; mean active fraction \(0.00641147\); accumulated rain \(5.38707\times10^5\ {\rm m^3}\), versus a \(1.05123\ {\rm m^3}\) float64 floor |
| peak local cloud and \(R\) | 89 | 8900 s | \(\max q_c=1.05179096\times10^{-4}\); \(\max R=5.17910\times10^{-11}\ {\rm s^{-1}}\) |
| peak integrated \(R\); mature rain | 120 | 12,000 s | \(\int hR\,dA=3.05591\times10^4\ {\rm m^3\,s^{-1}}\); rain-active fraction 0.0766602 |
| peak integrated cloud | 160 | 16,000 s | \(\int Q_c\,dA=1.55151132\times10^{12}\ {\rm m^3}\) |
| final | 160 | 16,000 s | \(\int Q_r\,dA=2.30733007\times10^8\ {\rm m^3}\); active fraction 0.100464 |

“Certifiable” means that the exact deployed formula produces positive \(R\) on the full GLL support, not merely that an interpolated image contains a nonzero-looking pixel. “Sustained” additionally requires uninterrupted positive support and rain accumulation far above a conservative float64 roundoff floor. The accepted record remains rain-active at every state from step 51 through step 160.

## Conservation checks

| Diagnostic | Test 2A | Test 2B |
|---|---:|---:|
| initial total water \(\int(Q_v+Q_c+Q_r)\,dA\) | \(3.48932785\times10^{13}\) | \(3.69868752\times10^{13}\) |
| maximum absolute drift | \(0.109375\) | \(0.65625\) |
| maximum relative drift | \(3.13456\times10^{-15}\) | \(1.77428\times10^{-14}\) |
| max moist-source water residual | \(0\) | \(1.10842\times10^{-21}\) |
| max \(S-\beta_2Q_v\) source residual | \(1.73472\times10^{-18}\) | \(1.38778\times10^{-17}\) |

For Test 2B, the time-integrated rain source is \(2.30733007\times10^8\ {\rm m^3}\), equal to the final integrated \(Q_r\) to roundoff. These checks support the event chronology and show that the rain signal is not conservation noise.

## Physical interpretation

The imposed Test 2B supersaturation first drives condensation throughout the domain, building a broad cloud reservoir. The double-vortex circulation then breaks the initially uniform saturation ratio into paired annular and spiral structures. Cloud-water depressions and threshold-crossing bands wrap around the compact vorticity cores; \(R\) localizes on thin crescent-shaped arcs rather than filling the cores. Those arcs lengthen and overlap as rain support expands, so the domain-integrated production peaks after the local production maximum.

The vortices therefore materially reorganize moisture: the rain field is not a spatially uniform response to the 6% initial supersaturation. It is a thresholded response concentrated along rotating cloud-water gradients around the two vortices. Test 2A supplies a useful non-raining contrast: it develops similar vortex-centered condensation/evaporation structure, but its cloud reservoir never approaches the autoconversion threshold closely enough to activate \(R\).

Small negative \(Q_c\) or \(Q_r\) values visible in raw interpolated map caches are finite-element numerical undershoots. Publication panels clip those undershoots only for display; the raw arrays and all exact audit quantities are retained.

## Cached-map vortex-core motion

The two positive-vorticity cores were tracked through all 161 Test 2B cached maps without rerunning Firedrake. Each raw core is the maximum in a 750 km periodic association disk centered on the preceding core centroid. The accepted smoother location is instead a positive-vorticity-weighted periodic centroid within a fixed 600 km disk centered on the previous centroid; it never recenters on the winning gridpoint and the two disks remain disjoint.

The raw gridpoint maxima move by at most 49.411 km (0.988% of the 5000 km domain and 13.18% of the 375 km width). Their visible steps at 6800 and 14,900 s are switches between immediately adjacent GLL points whose vorticity differs by only \(2.8\)--\(5.0\times10^{-5}\) in relative terms. Across those same frames the continuous centroid moves only 0.117 and 0.136 km. The smoother centroids move by at most 18.003 km for either core:

\[
\frac{18.003}{5000}=0.003601,\qquad
\frac{18.003}{375}=0.04801.
\]

The centroid pair separation increases smoothly from 1438.465 to 1465.499 km, a maximum change of 27.033 km. Its orientation changes only from \(45.000^\circ\) to \(45.939^\circ\), and the pair midpoint displacement is below \(9\times10^{-9}\) km (stored numerical precision). The equal and opposite core motion is therefore a slight symmetric deformation of the pair, not bulk translation. A 50%-of-local-maximum threshold centroid and radius sensitivity tests give the same conclusion; full switch values and definitions are in the [vortex tracker audit](VORTEX_CORE_TRACKING_AUDIT.md).

This directly supports the stationary-skeleton interpretation. The vorticity cores remain fixed to within about 4.8% of one vortex width while cloud and rain features develop long rotating annuli and crescents around them. The striking motion in the moisture movie is primarily transport and wrapping relative to an approximately stationary vortex pair, rather than advection of the entire pair across the domain. The added result is reported as Figure 5; Figures 1--4 were not altered.

## Remaining worthwhile diagnostics

- A Lagrangian or phase-relative measure of the angular propagation speed of the rain arcs would quantify vortex transport more directly.
- A radial/azimuthal composite about each moving vorticity maximum could separate core, ring, and inter-vortex moisture responses.
- If a future report needs a controlled attribution to supersaturation alone, matched-resolution \(16^2\) and \(64^2\) runs at both \(\zeta=0\) and \(-0.06\) would be required; the present accepted truths do not form that factorial experiment.
- Domain-integrated energy or moist available-potential-energy diagnostics are not stored in the immutable truth artifacts and were not reconstructed here.
