# Ground-truth figure and movie plan

## Design principles

- Use the immutable Test 2A and Test 2B restart arrays as the state source.
- Evaluate (A) and (R) at the exact post-children-1--5 input to the moist Euler child, matching the accepted truth audit.
- Plot saved state variables at the labeled time; label rate panels as the moist tendency acting over the following (100\ \mathrm{s}) split child.
- Convert specific cloud water to (mathrm{g\,kg^{-1}}), specific rain water to (mathrm{\mu g\,kg^{-1}}), (A) to (mathrm{g\,kg^{-1}\,h^{-1}}), and (R) to (mathrm{\mu g\,kg^{-1}\,h^{-1}}).
- Express coordinates in km, vorticity in (10^{-5}\ \mathrm{s^{-1}}), and integrated (Q_c,Q_r) as (int Q_k\,dA) in model water-volume units ((mathrm{m^3}) under unit reference density).
- Use fixed limits across event columns, comparisons, and movie frames. Signed fields use a perceptually balanced diverging map centered at zero; nonnegative water/rain fields use sequential maps.
- Retain all 161 saved states for each rendered movie. The movie cadence is one frame per stored (100\ \mathrm{s}) state, with playback accelerated to a compact report-viewing rate.

## Figure 1 — Exact DoubleVortex initial state

Analytical, high-resolution \(1\times3\) reconstruction directly from the executable formulas:

1. \(h-H_0\) with velocity streamlines and vortex centers;
2. analytical relative vorticity \(\partial_xv-\partial_yu\);
3. thermal/buoyancy anomaly \(100(b/g-1)\), using report notation \(b=S/h\).

This figure establishes the physical geometry without a separate moisture-contract panel.

## Figure 2 — Combined chronology and regime comparison

A clean \(2\times3\) time history using both truths and the exact GLL/moist-child audit:

1. integrated cloud water for Tests 2A and 2B;
2. maximum \(q_c-q_{\rm precip}\), with zero denoting threshold crossing;
3. integrated rain water for both tests;
4. explicitly labeled domain min--max \(A\) for Test 2B;
5. domain-integrated Test 2B rain-production rate on its own axis;
6. domain minimum and maximum saturation departure for both tests.

Common markers identify 5100 s (first certifiable \(R>0\)), 6100 s (sustained-rain certification), and 12,000 s (peak integrated rain-production rate). A single figure-level key identifies tests and events. The rain-active GLL fraction is omitted.

## Figure 3 — Test 2B event-state gallery

Five physically selected columns:

- (0\ \mathrm{s}): initial supersaturation;
- (5000\ \mathrm{s}): last clearly pre-rain saved state;
- (6100\ \mathrm{s}): sustained-rain certification;
- (12000\ \mathrm{s}): peak integrated rain-production rate, used as the mature-rain state;
- (16000\ \mathrm{s}): final state and peak stored integrated (Q_c,Q_r).

Rows show supersaturation, (q_c), (q_r), and (R), with relative-vorticity contours overlaid where legible. The first (R>0) state at (5100\ \mathrm{s}) is intentionally marked in Figure 2 rather than given a nearly indistinguishable gallery column.

## Movies

### Test 2B truth evolution (priority)

A (2\times3) layout showing saved-state relative vorticity, supersaturation, (q_c), (q_r), and exact post-prefix (A,R). All panels use fixed limits. Event labels change from `initial` to `pre-rain`, `rain onset`, `sustained rain`, `mature rain`, and `final` according to the verified chronology.

### Test 2A truth evolution

A scientifically useful (2\times2) comparison movie showing relative vorticity, saturation departure, (q_c), and (A). Identically zero (q_r,R) panels are omitted. It documents reversible condensation/evaporation without threshold crossing.

The existing environment has Matplotlib and Pillow but no `ffmpeg`; therefore the native rendered deliverable is an animated GIF. The scripts will also emit all PNG frames and record an exact optional MP4 command for a later environment that already provides `ffmpeg`.

## Post-plan addendum — Figure 5

At the user's later request, a compact vortex-motion diagnostic was added. The publication figure uses only the finalized continuous positive-vorticity centroids. Its three panels show local displacement-plane trajectories, displacement magnitude, and pair separation/orientation; raw gridpoint-max tracks remain available only in the machine-readable audit data.
