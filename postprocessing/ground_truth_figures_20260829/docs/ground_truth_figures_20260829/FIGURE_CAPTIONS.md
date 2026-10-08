# Draft report captions

## Figure 1 — DoubleVortex initial physical state

Exact executable reconstruction of the DoubleVortex initial condition: (a) depth anomaly and analytically geostrophic flow, (b) analytical relative vorticity, and (c) thermal/buoyancy anomaly \(100(b/g-1)\), where the report variable \(b=S/h\). The two negative depth anomalies are centered at \((2000,2000)\) and \((3000,3000)\) km and use the source code's sine-periodicized Gaussian coordinates with \(\sigma_x=\sigma_y=375\) km. The velocity satisfies \(u=-(g/f)h_y\), \(v=(g/f)h_x\), and hence \(f\mathcal R(\boldsymbol u)+g\nabla h=0\). White crosses mark the prescribed vortex centers. The separate thermal/buoyancy anomaly is a central Cartesian Gaussian.

## Figure 2 — Combined truth chronology and regime comparison

Chronology and regime comparison from all 161 accepted states at 100 s cadence. Panels show (a) integrated cloud water, (b) maximum cloud-water excess above \(q_{\rm precip}=10^{-4}\), so positive values certify threshold crossing somewhere in the domain, (c) integrated rain water, (d) the explicitly labeled domain min--max range of \(A=E-C\) for Test 2B, (e) domain-integrated Test 2B rain production, and (f) domain minimum and maximum saturation departure. Test 2B's imposed approximately 6% supersaturation is consumed during the first moist adjustment; rain is first certifiable at 5100 s, sustained-rain behavior is certified at 6100 s, and integrated production peaks at 12,000 s. Test 2A remains below the rain threshold and has \(M_r=0\) throughout. Rate diagnostics are evaluated at the exact post-prefix state entering the following moist split child. Test 2A and Test 2B differ in both initial vapor loading and spatial resolution (\(16^2\) versus \(64^2\)); the comparison identifies different physical regimes and is not a grid-convergence study.

## Figure 3 — Test 2B event-state gallery

Physically selected Test 2B truth states: initial supersaturation (0 s), last pre-rain state (5000 s), sustained-rain certification (6100 s), mature rain at peak integrated production (12,000 s), and the final/peak-integrated-cloud state (16,000 s). The vortex circulation reorganizes initially broad condensation into paired annular and spiral cloud structures. Rain production occupies thin crescent-shaped threshold-crossing arcs around the compact vortex cores and expands through the mature regime. Fixed limits are used across columns; small negative condensate undershoots are clipped only for display.

## Test 2B truth-evolution movie

All 161 Test 2B truth states from 0 to 16,000 s. Fixed panels show relative vorticity, supersaturation, cloud water, rain water, net phase-change rate \(A=E-C\), and rain-production rate \(R\). Event labels identify the last pre-rain state, first certifiable rain, sustained-rain certification, peak local production, and peak integrated production. The movie exposes the transition from broad initial condensation to vortex-wrapped cloud gradients and localized raining crescents.

## Figure 5 — Test 2B vortex-core motion

Continuous positive-vorticity-centroid motion through the full Test 2B trajectory. The step-0 centroid positions are \((1991.426,1991.426)\) km and \((3008.574,3008.574)\) km. Panel (a) shows each local trajectory \((\Delta x_i,\Delta y_i)\), with small open start and filled end markers; panel (b) shows periodic displacement from the corresponding initial position; and panel (c) shows pair separation and unwrapped orientation. Either core moves by at most 18.003 km, 0.360% of the domain or 4.80% of the 375 km vortex width. The pair midpoint is stationary to numerical precision, separation changes by 27.033 km, and orientation by only \(0.939^\circ\). Thus the vortex skeleton is approximately stationary while the moisture, cloud, and rain structures deform over much larger distances.

## Test 2A truth-evolution movie

All 161 Test 2A truth states from 0 to 16,000 s. Relative vorticity, saturation departure, cloud water, and \(A\) show the coarse vortex-centered condensation/evaporation pattern. Rain panels are omitted because \(R=Q_r=0\) throughout.
