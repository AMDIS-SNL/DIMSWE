# Test2B rain-active double-vortex truth preparation

## 1. Scope and present decision

This record prepares, but does not launch, the higher-resolution truth case
needed after the frozen no-rain Test2A Problems A and B.  It preserves the
accepted double-vortex initial condition, physical constants, six-child split,
and analytical `A/R` laws.  No ML, parameter inference, or production truth
campaign was run during this preparation.

### Completed dry-refinement outcome (2026-08-11)

The prepared 64-by-64 `dt=100` segment subsequently completed through
`t=16000` with 161 states under
`external-results/test2b-rain-active-truth/segment1-n64-dt100-t16000/`.
Its peak specific cloud water was `2.1809694213470483e-5` at `t=7900`, only
`21.809694213470482%` of `qprecip`; it declined thereafter.  Analytical `R`
and rain-water mass remained exactly zero, while relative maximum total-water
drift was `5.5974247328800295e-15`.  For comparison, the accepted 16-by-16
case reached about `31.18%`.  Thus neither refinement nor unchanged time
extension provides a route to rain.  This is a completed dry scientific
result, not a failed run.  The subsequent controlled physical-case design is
recorded separately in `docs/TEST2B_RAIN_ACTIVE_CASE_DESIGN.md`; this document
retains the historical refinement preparation and its original readiness
decision.

The requested `AMDIS_Learned_Physics_Research_Plan.md` filename is not present
in this checkout.  The repository-equivalent contracts inspected were
`docs/LEARNED_PHYSICS_EXPERIMENTS.md`, `docs/RESOLVED_HIDDEN_C0.md`,
`dimswe/configs/test1b_selected_plan.json`, and the frozen Problem-A/Problem-B
syntheses.  Source and accepted truth metadata settle the implementation
details below.

**Readiness decision:** the 64-by-64 numerical refinement is stable and
operationally practical, but the bounded evidence does not establish that the
unchanged physical problem will ever enter the rain-active regime.  The
accepted 16-by-16 trajectory moves away from the threshold after its peak, and
the matched-timestep 64-by-64 pilot remains below it at `t=8000`.  A first
`t=16000` refinement segment is prepared as a diagnostic continuation, but it
must not yet be represented as a certified rain-active production truth.

## 2. Accepted Test2A truth archaeology

The authoritative source is
`external-results/test1b-production/truth_c0_0.14/metadata.json` (status
`complete`).  The accepted truth is not a 32-by-32 Firedrake mesh.

| quantity | accepted value | meaning |
|---|---:|---|
| physical domain | `[0,5,000,000) x [0,5,000,000)` m | doubly periodic rectangle |
| mesh | `16 x 16` | 256 periodic quadrilateral cells |
| velocity | vector CG(3), spectral | 4,608 global scalar coefficients including components |
| `h`, `S` | scalar CG(3), spectral | 2,304 coefficients each |
| `Qv`, `Qc`, `Qr` | scalar DG(1), spectral | 1,024 coefficients each |
| full state | 12,288 coefficients | field order `(v,h,S,Qv,Qc,Qr)` |
| deployed moist samples | `16 x 16 x 4 x 4 = 4,096` per state | cell-major broken-CG3 GLL representation; shared CG points repeat by cell |
| spectral diagnostic samples | `32 x 32` | cell-centred uniform physical samples for FFT diagnostics only |
| `dt` | 100 | complete model timestep |
| steps / duration | 160 / 16,000 | stored boundary states 0 through 160 |
| output cadence | every step | physical cadence 100; 161 restart/checkpoint states |
| backend | analytical UFL | certified JAX analytical law is a float64-equivalent diagnostic oracle |
| `c0`, `s` | `0.14`, `3.2` | deployed grid-scaled hyperviscosity convention |
| seed | 0 | recorded provenance; the double-vortex IC itself is deterministic |

The current truth consumed 30.137276 seconds and 95,996,061 bytes in its
completed serial run.  It writes an atomic flat restart array and Firedrake
HDF5 checkpoint at every output, plus JSON diagnostics and spectral NPZs.
Incomplete runs resume from the latest finite output-stride restart whose
configuration fingerprint matches.

### 2.1 Initial condition

`dimswe.initial_conditions.DoubleVortex` fixes

- `Lx=Ly=5e6`, `H0=750`, `dh=75`, `g=9.80616`, and Coriolis
  `f=6.147e-5`;
- vortex centres `(0.4Lx,0.4Ly)` and `(0.6Lx,0.6Ly)` with
  `sigma_x=sigma_y=3Lx/40`;
- the periodic Gaussian height and geostrophic two-vortex velocity expressions
  in `DoubleVortex.get_value`;
- flat topography `B=0`;
- entropy `s=g[1+0.05 exp(-r^2/((1/3)^2(Lx/2)^2))]` and `S=hs`;
- `q0=0.002`, `zeta=0`, `Qv=h qsat`, and `Qc=Qr=0` initially.

There is no random perturbation and no resolution-dependent resampling of an
array: the same UFL initial-condition expressions are interpolated on each
mesh.

### 2.2 Complete split and diffusion

The production Lie split is exactly

1. dry RK4 half-step at `t_n`;
2. dry RK4 half-step at `t_n+dt/2`;
3. hyperviscosity Euler full-step;
4. DG SSPRK43 half-step at `t_n`;
5. DG SSPRK43 half-step at `t_n+dt/2`;
6. analytical moist Euler full-step.

The configured lists are `[RK4,Euler,SSPRK43,Euler]`, terms
`[[model],[hyperviscosity],[dg1limiter],[threewayphysics]]`, and subcycles
`[2,1,2,1]`.  The model is `advdens-cf-h1` with `mtswe` Hamiltonian, no
Poisson/metric bracket, `alpha_s=1`, velocity and total-density upwinding,
split forms for `h` and `S`, GLL mass lumping, and no additional tracer fields.
Masses use the configured GLL-lumped volume measure.  The
hyperviscosity operator is

`H = r^s (M^-1 K)^2`, with `r=max(mesh.dx/order,mesh.dy/order)`,

so retaining the dimensionless `c0=0.14` and `s=3.2` is the repository's
grid-refinement convention; copying a dimensional coefficient would be wrong.
The DG1 limiter and all upwind/split-form choices remain unchanged.

## 3. Analytical phase-change and rain laws

At the post-prefix state passed to child 6, let

`qv=Qv/h`, `qc=Qc/h`, `s=S/h`,

`qsat = q0 H0/(h+B) exp(20(1-s/g))`, and `beta2=gL`.

With `L=10`, `gamma_r=0.001`, `qprecip=1e-4`, and the repository convention
`tau_v=tau_r=dt`, the analytical rates are

`C=max(0, gamma_v(qv-qsat)/tau_v)`,

`E=min(qc/dt, max(0,gamma_v(qsat-qv)/tau_v))`,

`A=E-C`, and

`R=max(0, gamma_r(qc-qprecip)/tau_r)`.

Thus exact rain activation requires the **specific** cloud water `Qc/h` to
exceed `1e-4`; a large conservative field coefficient `Qc` alone is not the
criterion.  The source map is

`Qv_t=hA`, `Qc_t=-h(A+R)`, `Qr_t=hR`, `S_t=h beta2 A`.

It algebraically conserves `Qv+Qc+Qr` and preserves the source relation
`S-beta2 Qv`.

## 4. Why the accepted trajectory has no rain

The read-only audit
`external-results/test2b-preparation/test2a_truth_rain_threshold_audit.json`
loads all 161 accepted boundary states, replays the accepted complete step only
to recover the exact post-children-1..5 state, and evaluates the certified
analytical JAX/UFL-equivalent local law there.

| diagnostic | value |
|---|---:|
| maximum `qc=Qc/h` | `3.11771075970985e-5` |
| time / step of maximum | `8500 / 85` |
| fraction of rain threshold reached | `0.311771075970985` |
| remaining threshold margin | `6.882289240290151e-5` |
| `qc` RMS at that time | `6.643225073044002e-6` |
| conservative `Qc` maximum / RMS at that time | `2.3685502006091007e-2 / 5.0339714470978925e-3` |
| maximum analytical `R` | exactly `0` |
| total `Qr` mass | exactly `0` |
| maximum analytical `|A|` at the peak state | `7.444054119642776e-8` |

The maximum rises through the early condensation phase, peaks at `t=8500`,
then falls to `2.4323845225947928e-5` at `t=16000`.  The late trajectory is
not converging toward `qprecip`: extending the same 16-by-16 state is case B,
not case A.  Total cloud-water mass continues to spread while its local
specific maximum falls, so domain-integrated cloud mass is not a proxy for
rain onset.

The exact discrete total-water integral begins at
`3.4893278484426496e13`; its maximum drift is `0.109375`, or
`3.1345578504128253e-15` relative.  Across the local source audit, the water
identity is bit-exact and the largest `S-beta2 Qv` residual is
`1.734723475976807e-18`.  These establish the conservation/numerical floor.

## 5. Proposed 64-by-64 physical refinement

The frozen preparation configuration is
`dimswe/configs/test2b_rain_active_truth.json`; its canonical-JSON
configuration fingerprint is
`5371b96df5d2daf4539cdfc7e9be77345d16be19083b776427764f4b614daf4e`.
The accepted Test2A source metadata file SHA256 is
`5c5b47736362ccace69944df7c226f83238a3eb453b63717035040dc27b60545`.

| quantity | Test2A | 64-by-64 candidate |
|---|---:|---:|
| physical domain | `5e6 x 5e6` | unchanged |
| cells | `16 x 16` | `64 x 64` |
| polynomial spaces | CG3/DG1 | unchanged |
| state coefficients | 12,288 | 196,608 |
| diagnostic FFT samples | `32 x 32` | `128 x 128` |
| moist GLL samples/state | 4,096 | 65,536 |
| `dt` | 100 | 100 |
| saved-state cadence | 100 | 100 (every model step) |
| `c0`, `s` | `0.14`, `3.2` | unchanged |
| analytical `A/R` and all moist constants | accepted | unchanged |
| IC, forcing, boundaries, split | accepted | unchanged |

The 64-by-64 exact hyperviscosity audit gives, at `dt=100,c0=0.14`,
`sigma=lambda_max dt c0 = 1.1201445455631` and a certified row-sum upper
bound `1.8860793301517`, both inside the Euler interval `[0,2]`.  The Ritz
`dt_max` is 178.5483853778 and the certified conservative `dt_max` is
106.0400783799.  The old 0.8 convenience recommendation is 84.8320627039,
so `dt=100` has less than that optional safety margin but is still formally
certified stable for the exact hyperviscosity child.

Retaining `dt=100` is scientifically important: `ThreeWayPhysics` defines
`tau_v=tau_r=dt`.  A preliminary `dt=50` pilot was stable, but it doubled the
number of relaxation actions per physical interval and therefore was not a
clean mesh-only refinement.  It is retained only as bounded engineering
evidence under `external-results/test2b-preparation/`; it is not the proposed
truth.

Keeping `dt=100` also retains the physical durations of future H1/H2/H5
windows (100/200/500).  Every model-step boundary is saved, so future work can
reconstruct post-prefix `Y_k`, analytical `A/R`, full tendencies, and any
short-window schedule without missing internal timesteps.

## 6. Rain-onset contract

The accepted Test2A diagnostic contract is retained:

- **exact onset:** first deployed child-6 GLL sample with `R != 0`;
- **float64-resolved onset:** `|R| > 64 eps` times the run's analytical rate
  comparison scale;
- **physically meaningful onset:** additionally
  `|dt h R| > 1e-12 RMS(Qr)` at the same deployed state;
- first strictly positive exact discrete `integral Qr dx` is recorded
  separately.

At every saved state the audit records maximum/RMS/activity of `A` and `R`,
`Qc` mass/max/RMS and threshold margin, `Qr` mass, total water, exact source
invariants, KE, projected enstrophy, field minima, and finite-state status.
This is deliberately more informative than `R != 0` alone.

No canonical repository definition of **sustained rain** exists.  The config
therefore marks the following as a candidate requiring explicit scientific
approval, not as a silently frozen pass criterion:

1. physically meaningful `R` at ten consecutive saved states (1,000 physical
   time units at the candidate cadence);
2. at least `1e-4` of the block's space-time GLL samples above `1e-6` of the
   block maximum `|R|`; and
3. positive exact `Qr` mass above a `128 eps` total-water-scaled numerical
   floor.

`RAIN_ONSET` and this candidate `SUSTAINED_RAIN_ACTIVE` would delimit
PRE_RAIN, ONSET, and RAIN_ACTIVE intervals.  The sustained definition must be
approved before it controls scientific stopping or data selection.

## 7. Bounded 64-by-64 certification

The accepted candidate pilot is
`external-results/test2b-preparation/bounded-n64-dt100-t8000`; its read-only
rain audit is
`external-results/test2b-preparation/bounded_n64_dt100_t8000_rain_activity_audit.json`.
It is a bounded 80-step, `t=8000` evaluation, not a production truth.

| bounded certificate | measured result |
|---|---:|
| status / finite saved states | complete / 81 of 81 |
| maximum `qc=Qc/h` | `2.1809694213470483e-5` |
| time / step of maximum | `7900 / 79` |
| rain threshold reached | `0.21809694213470482` |
| remaining threshold margin | `7.819030578652952e-5` |
| analytical `R` / `Qr` mass | exactly zero / exactly zero |
| maximum relative total-water drift | `5.5974247328800295e-15` |
| maximum local water-source residual | exactly zero |
| maximum local `S-beta2 Qv` source residual | `4.336808689942018e-19` |
| minimum GLL `h`, `Qv`, `Qr` | `626.2467502261`, `0.5495801122`, `0` |
| minimum GLL `Qc` | `-0.0020388182` (small DG undershoot; bounded) |
| KE change over pilot | `-9.344753484%` |
| projected-enstrophy change | `-10.347116344%` |
| maximum high-wavenumber energy fraction | `2.1413853840e-10` |

At initialization, the refined mixed-state mass norm differs from the
16-by-16 value by `-3.2012493e-9` relative; KE and projected enstrophy differ
by `-7.2977704e-5` and `+2.7795253e-5`, respectively.  These are the expected
finite-element interpolation/resolution changes, while the physical UFL IC,
domain, and coefficients are identical.  Initial total water agrees to the
same mass-assembly roundoff scale.

At the same physical time the accepted 16-by-16 truth had reached
`3.070325863704023e-5`, or 30.7% of threshold.  The 64-by-64 value gained only
about 2.3% over the last 1,000 time units, attained its pilot maximum at
`t=7900`, and edged down to `2.1802209646077824e-5` at `t=8000`; it remains
78.2% below threshold.  Refinement materially changes the cloud maximum and
has not moved it toward rain.  This supports a combined B/C
diagnosis: the accepted physical configuration tends toward a no-rain regime,
and resolution/transport/diffusion affect the margin, but resolution alone is
not evidence of imminent activation.

The earlier `dt=50` engineering pilot completed 160 steps to the same physical
time, remained finite, conserved total water to `4.589888280961624e-15`
relative, and reached only `2.1837068623007184e-5` specific cloud water.  This
confirmed solver viability but not physical equivalence.

## 8. Runtime, memory, and storage

The matched-`dt=100`, output-every-step pilot provides the measured storage
scale: 81 saved states occupy 551,096,849 bytes, about 6.80 MB per saved state.
At 64-by-64 one flat state is 196,608 float64 coefficients (1,572,864 raw
bytes); an HDF5 checkpoint is approximately 5.2 MB.  Direct peak RSS was not
available inside the restricted execution environment, so no unsupported
memory number is claimed; the complete serial case and split caches did run
successfully on this Mac.

At output every `dt=100`, a 161-state `t=16000` segment is projected at about
1.095 GB (1.020 GiB).  A 321-state `t=32000` trajectory is about 2.184 GB
(2.034 GiB); 641 states would be about 4.361 GB (4.062 GiB).

The 80-step pilot took 157.629 seconds in repository metadata and 172.49
seconds end to end; its read-only post-prefix rain audit took another 133.915
seconds.  A simple fixed-setup/steady-step engineering fit projects the
160-step `t=16000` solver segment at about 240 seconds and the audit at roughly
3--4 minutes, or about 7--8 minutes total.  After setup, one physical hour
(3,600 model-time units, 36 steps) costs roughly 37 seconds of solver time;
from a cold process the first such interval is closer to two minutes.  A
`t=32000` solver-only trajectory is roughly 6--7 minutes plus its audit.  These
are estimates, not promises, but they show that serial 64-by-64 is practical
on this Mac.  Setup/JIT/form compilation and per-output projected-vorticity/
spectral diagnostics are included in the measured basis.

## 9. Restart and long-run strategy

`resolved_hidden_c0_driver` already writes atomic restart arrays, Firedrake
checkpoints, diagnostics, and spectra at every output and resumes an
interrupted matching configuration from its latest finite restart.  The
prepared first segment uses 160 steps (`t=16000`) and stores every step.

After that segment, the rain audit must be reviewed.  It is scientifically
acceptable only if it supplies a useful PRE_RAIN/ONSET/RAIN_ACTIVE record and
the sustained criterion has been approved.  If it remains dry, do **not**
silently lower `qprecip`, increase moisture, reduce diffusion, or relabel the
run.  An explicit next scientific decision is then required: continue the
unchanged physical case with a certified cross-segment continuation, or adopt
a separately justified rain-active initial/physical configuration.  The
present driver resumes interrupted fixed-duration runs but does not append a
new duration to a completed configuration fingerprint; that extension
contract should be implemented only after the first-segment decision.

## 10. Future ML truth contract

The production output must preserve every boundary state and all physics/time
metadata.  From these, the exact children-1..5 prefix can reconstruct `Y_k`,
and the certified local law supplies analytical `A`, analytical `R`, and the
full four-component tendency.  This supports the three planned rain-active
representations:

1. `A_theta` with analytical `R`;
2. structured `(A_theta,R_theta)` with the exact source map;
3. independent four-tendency black box.

The chronological training/model-selection/held-out partition remains
unfrozen until the truth contains sufficient PRE_RAIN, ONSET, and RAIN_ACTIVE
support.  It must be frozen using physical regime coverage before ML is fit,
not selected from later model performance.  Rain-active inputs (including
whether `Qr` is required) and nonzero output scales must also be frozen then;
the no-rain Test2A contracts cannot be copied automatically.

Problem-A conservation and thermodynamic coupling were exact by
representation, while Problem-B H1/H2/H5 exploited unphysical source
directions (systematic destruction, coherent creation, and destruction plus
spurious rain, respectively).  Test2B must therefore treat representation and
objective as interacting design choices; no conservation penalty or source
projection is added to the core black-box comparator here.

## 11. Prepared manual command (not executed)

This launches only the prepared first physical-refinement segment and its
read-only audit.  Because the current evidence does not yet guarantee rain,
it should be treated as a diagnostic production-candidate segment pending the
readiness caveat above.

```bash
cd /path/to/DIMSWE-collaborator
mkdir -p external-results/test2b-rain-active-truth
nohup caffeinate -i bash scripts/run_test2b_rain_truth_segment1.sh \
  > external-results/test2b-rain-active-truth/segment1-master.log 2>&1 &
```

The exact configuration validator fingerprint and source metadata hashes are
written by the config/audit commands.  Reissuing the same command after an
interruption resumes the matching segment; a changed configuration is refused.

## 12. Preparation status

No hours-long truth simulation and no ML/training job was launched by Codex.
Only bounded 64-by-64 pilots and read-only truth audits were run.

**STATUS: TEST2B_RAIN_TRUTH_NOT_READY_FOR_PRODUCTION.**  Numerical refinement
is prepared and stable, but rain activation under unchanged physics and the
sustained-rain scientific stopping criterion are not yet certified.
