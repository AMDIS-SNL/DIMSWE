# Test2B controlled rain-active double-vortex case design

## 1. Scope and decision

This document records a bounded physical case-design sweep after the unchanged
64-by-64 refinement proved dry.  It does not contain ML training or a long
production truth run.  The authoritative machine-readable sweep is
`external-results/test2b-rain-active-case-design/case_design_summary.json`;
the proposed production contract is
`dimswe/configs/test2b_rain_active_case.json`.

The selected control is the signed initial saturation deficit `zeta` already
present in the analytic `DoubleVortex` initial condition:

`Qv(t=0) = h (1-zeta) qsat`, with `Qc(t=0)=Qr(t=0)=0`.

The historical value is `zeta=0`.  The selected value is `zeta=-0.06`, hence
the boundary state is uniformly 6% supersaturated.  Cloud and rain water are
not inserted.  The first moist child dynamically produces cloud; rain remains
zero until the evolved specific cloud water crosses the unchanged threshold.

The selected configuration keeps `qprecip=1e-4`, `gamma_r=0.001`, and the
analytical `A` and `R` laws exactly unchanged.  It is ready for a manual
64-by-64 production truth run to `t=16000`; that run was not launched here.

## 2. Why the unchanged baseline does not rain

The completed dry-refinement evidence is:

| truth | mesh | peak `qc/qprecip` | peak time | behavior after peak |
|---|---:|---:|---:|---|
| accepted Test2A | 16 x 16 | about `0.3118` | `8500` | declines |
| dry refinement | 64 x 64 | `0.21809694213470482` | `7900` | declines |

The 64-by-64 source is
`external-results/test2b-rain-active-truth/segment1-n64-dt100-t16000/`
(truth and audit both `complete`).  Its peak specific `Qc` is
`2.1809694213470483e-5`, leaving a `7.819030578652952e-5` threshold margin;
`R` and total `Qr` mass are exactly zero.  Relative total-water drift is
`5.5974247328800295e-15`.

The baseline begins exactly saturated rather than supersaturated.  Dry and DG
dynamics create local supersaturation and analytical `A` condenses vapor, but
the same source decreases `S` through `S_t=h beta2 A`.  Because
`qsat ~ exp(20(1-(S/h)/g))`, that thermodynamic response raises the local
saturation value during condensation and is stabilizing.  Finite vapor,
advection, limiter/upwind mixing, and hyperviscous spreading then limit the
local cloud maximum.  The observed peak-and-decline at both resolutions is
direct evidence that unchanged time extension is not a route to precipitation.

## 3. Physical-control archaeology

The exact source is `dimswe.initial_conditions.DoubleVortex` plus
`dimswe.physics.ThreeWayPhysics`.

| candidate | class | role | case-design decision |
|---|---|---|---|
| `zeta` in `Qv=h(1-zeta)qsat` | physical initial condition | uniform vapor loading / saturation ratio | selected one-parameter family |
| direct initial `Qc` | physical initial condition | inserts condensate by hand | rejected |
| initial `Qr` | physical initial condition | inserts rain by hand | rejected |
| entropy-bump amplitude `c=0.05` | physical initial condition | changes `S`, `qsat`, and thermodynamics spatially | not varied; less isolated |
| vortex height amplitude `dh=75` | physical initial condition | changes height and geostrophic velocity together | not varied; confounds dynamics |
| vortex offsets `ox=oy=0.1` | physical initial condition | changes vortex geometry | not varied |
| `q0=0.002`, `H0=750`, `L=10` | moist-law/physical constants | saturation and thermo coupling | frozen |
| `qprecip`, `gamma_r` | rain microphysics | onset and rain conversion | frozen by contract |
| external moisture forcing | physical forcing | none exists in this case | no control available |
| mesh, `dt`, `c0`, `s`, limiter | numerical parameters | resolution, stepping, dissipation | fixed, not used to force rain |

Negative `zeta` is the cleanest control because it changes only initial vapor
loading, has an exact dimensionless interpretation, retains zero initial cloud
and rain, and does not alter the flow, entropy field, source map, or
microphysics.  The historical omission of `zeta` remains exactly equivalent to
`zeta=0`; the new configuration path is backward compatible.

## 4. Bounded 64-by-64 sweep

All runs use the same `5e6`-m periodic domain, quadrilateral Q3/DG1 spaces,
`dt=100`, `c0=0.14`, `s=3.2`, analytical UFL moist backend, six-child split,
and output every step.  Each begins with exact boundary saturation ratio
`1-zeta`; post-prefix ranges differ slightly because children 1--5 act before
the audited child 6.

| `zeta` | boundary ratio | end time | peak `qc` | `qc/qprecip` | first meaningful `R` | classification |
|---:|---:|---:|---:|---:|---:|---|
| `0` | `1.00` | `16000` | `2.1809694e-5` | `0.21810` | none | dry baseline |
| `-0.03` | `1.03` | `8000` | `6.3338188e-5` | `0.63338` | none | dry |
| `-0.04` | `1.04` | `8000` | `7.7257422e-5` | `0.77257` | none | dry |
| `-0.05` | `1.05` | `8000` | `9.1165930e-5` | `0.91166` | none | near-onset, dry |
| `-0.06` | `1.06` | `12000` | `1.05179096e-4` | `1.05179` | step 51, `t=5100` | sustained rain-active under the proposed criterion |

The selected run's first step reaches only `0.85284 qprecip`, so it does not
begin rain-active.  It supplies 51 saved PRE_RAIN boundary states before
onset.  Rain remains physically meaningful at every saved state from step 51
through the bounded endpoint at step 120.

Selected `zeta=-0.06` diagnostics:

| quantity | value |
|---|---:|
| first exact / meaningful `R` | step 51, `t=5100` |
| peak `qc` | `1.0517909572531444e-4` at `t=8900` |
| maximum `R` | `5.179095725314434e-11` |
| maximum spatial RMS `R` | `6.950747206789994e-12` |
| space-time RMS `R` | `4.049209992138056e-12` |
| maximum meaningful active GLL fraction | `0.07666015625` |
| final / maximum total `Qr` mass at `t=12000` | `120774905.21082714` |
| integrated deployed rain-source mass | `120774905.2108267` |
| relative maximum total-water drift | `1.7742780285355334e-14` |
| maximum local water-source residual | `1.108422020821057e-21` |
| maximum local `S-beta2 Qv` source residual | `1.3877787807814457e-17` |
| KE relative change to `t=12000` | `-0.10090462950027512` |
| projected-enstrophy relative change | `-0.0811335596054712` |

The equality of final rain mass and time-integrated deployed rain-source mass
to floating-point accuracy independently confirms that rain is generated by
the analytical source and conservatively transported.

All saved states are finite; minimum post-prefix GLL `h` and `Qv` densities
are `626.0193005278404` and `0.550806939353457`.  As in the accepted dry
DG1 run, `Qc` has a bounded transport/interpolation undershoot (minimum
conservative density `-0.0027681977347767846`).  Rain transport introduces a
much smaller `Qr` conservative-density undershoot, `-7.395397161050754e-6`,
corresponding to order `1e-8` in specific moisture.  Stored boundary
coefficient minima are smaller still: the exact all-restart minima are
`Qc=-1.5199660151721811e-4` and `Qr=-1.732894942399503e-6`.  These do not
activate rain and do not grow or impair
global positivity of water mass; they are recorded as a limitation of the
existing DG transport rather than hidden.  No negative `h` or `Qv` occurs.

## 5. Proposed sustained-rain criterion

No repository-wide canonical definition exists.  For this case design,
`SUSTAINED_RAIN_ACTIVE` is **proposed**, not silently generalized, as the first
continuous saved-state interval satisfying all of:

1. duration at least 1,000 physical time units (11 saved states at cadence
   100, not merely ten states spanning 900);
2. physically meaningful `R` and positive domain-integrated deployed rain
   production at every saved state;
3. mean physically meaningful active GLL fraction at least `1e-4`; and
4. positive increase in total `Qr` mass above `128 eps` times total water.

At 64 by 64, onset is `t=5100` and the criterion is certified at `t=6100`:
the 11-state mean active fraction is `0.006411465731534091`, and rain mass
grows by `538706.5819210776` against a `1.0512302194880667` numerical floor.
Activity then persists through `t=12000`.  A single threshold crossing could
not satisfy this test.

## 6. 32-by-32 sensitivity

The selected physical parameter was repeated at 32 by 32 through `t=12000`;
the production recommendation was not selected by which resolution rains more.

| quantity | 32 x 32 | 64 x 64 |
|---|---:|---:|
| onset time | `4400` | `5100` |
| sustain certification time | `5400` | `6100` |
| peak `qc` | `1.07366624e-4` | `1.05179096e-4` |
| peak time | `7600` | `8900` |
| maximum `R` | `7.36662387e-11` | `5.17909573e-11` |
| final `Qr` mass | `121029038.4128055` | `120774905.21082714` |
| relative water drift | `2.53468290e-15` | `1.77427803e-14` |

Onset is resolution-sensitive by 700 time units and the pointwise peak rate is
more sensitive, as expected for a thresholded local law.  Peak `qc` differs by
about 2.08%, while final integrated rain mass differs by only about 0.21%.
Both cases are stable, sustained, and conservative.  This is concrete
sensitivity to report, not a reason to downgrade the intended 64-by-64 truth.

## 7. Production truth contract

The selected production case is:

- the same `5e6 x 5e6` physical domain and double-vortex expressions;
- 64 by 64 quadrilateral cells, Q3/DG1 spaces, 128-by-128 diagnostics;
- `zeta=-0.06`, initial `Qc=Qr=0`;
- `dt=100`, 160 complete steps, final time 16,000;
- every boundary state saved (cadence 100);
- `c0=0.14`, `s=3.2`, seed 0, analytical UFL child;
- unchanged `qprecip=1e-4`, `gamma_r=0.001`, and unchanged `A/R` laws.

The bounded evidence certifies PRE_RAIN through `t=5000`, ONSET at `t=5100`,
and sustained activity through `t=12000`.  Continued activity to `t=16000`
is expected but not claimed as already observed.  Even if rain decays after
the pilot, the run already has a 6,900-unit certified active interval; the
additional 40 steps provide post-peak evolution and future chronological
partition flexibility.

Every output has an atomic flat restart, Firedrake checkpoint, diagnostics,
and spectra.  An interrupted run with the identical configuration fingerprint
resumes from its latest valid saved step; a mismatched configuration is
refused.  The 161 states are projected at 1.095 GB.  The completed dry 160-step
run took 189.22 seconds; the warm selected 120-step run took 109.93 seconds and
its audit 54.45 seconds.  A cold production run plus audit is therefore
estimated at roughly 4--5 minutes on this Mac, with setup/cache variation; it
is not a promise.

Saving every complete step retains boundary `X_k`.  The certified split can
reconstruct post-prefix `Y_k`, analytical `A`, analytical `R`, and the full
four-tendency truth for structured `A+R`, structured learned `(A,R)`, and
black-box four-output comparisons.  The future chronological train/held-out
partition remains unfrozen until the final truth's regime coverage is audited.

## 8. Manual production launch (not executed)

```bash
cd /Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615
mkdir -p external-results/test2b-rain-active-truth
nohup caffeinate -i bash scripts/run_test2b_rain_active_truth_production.sh \
  > external-results/test2b-rain-active-truth/production-n64-zeta-m0p06-master.log 2>&1 &
```

The runner validates the frozen case file, runs only the new selected truth
root, performs the read-only rain audit, and verifies completion.  It does not
train an ML model.

## 9. Status

Only bounded case-design and resolution-sensitivity simulations were run.
No long production truth and no ML/training/optimization job was launched.

**STATUS: TEST2B_RAIN_ACTIVE_CASE_READY.**
