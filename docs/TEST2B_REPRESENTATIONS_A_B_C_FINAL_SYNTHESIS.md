# Test2B rain-active Representations A/B/C: final matched synthesis

**Frozen synthesis date:** 2026-08-19

**Status:** all three representation ladders independently completed,
postprocessed, and frozen

**Scientific decomposition:**

- **A -> B:** remove the known analytical rain law while retaining the exact
  physical two-rate source manifold.
- **B -> C:** remove the source manifold and predict four moist tendencies
  independently.

No optimization, truth generation, parameter modification, source repair, or
model selection was performed for this synthesis.

## 1. Frozen evidence and matched contract

The authoritative evaluations are:

| representation | learned output | parameters | comparison artifact SHA256 |
|---|---|---:|---|
| A | `A_theta`; analytical state-dependent `R` | 1,281 | `75506a833862b69437a29a7a2b30e64c361d35b8560342bf8abe93302daf5b7f` |
| B | `(A_theta,R_theta)` with exact source map | 1,314 | `6044c0fbd42484e3bd6f0ec53bef91d9a871fa315fc83c193c10f1879813aadd` |
| C | independent `(S_t,Qv_t,Qc_t,Qr_t)_theta` | 1,380 | `8bc1d9fad90d1d5907c3ff8bc4a5e396ae09ce34f4d887d1f77ad429dfbba926` |

All use the same truth trajectory, local inputs `(h,S,Qv,Qc,B)`, float64
`32 x 32` hidden tanh architecture family, seed policy, six-child solver,
training support `0..80`, held-out mature-rain interval `81..160`, state
metric, nonoverlapping H2/H5 schedules, and optimizer family/budgets.  The
truth has meaningful onset at `t=5100` and final rain mass
`2.30733006980403e8`.

Representation A imposes

\[
s_A=h(\beta_2A_\theta,A_\theta,-A_\theta-R^*,R^*),
\]

where `R*` is evaluated analytically at the current deployed state.
Representation B imposes

\[
s_B=h(\beta_2A_\theta,A_\theta,-A_\theta-R_\theta,R_\theta).
\]

Both conserve water and preserve `S_t-beta2 Qv_t` algebraically.  B does not
impose the analytical threshold or `R>=0`.  Representation C predicts the
four-vector directly and imposes none of these properties.

The deployed `J_M2-X`, `J_H1`, `J_H2`, and `J_H5` losses use the same complete
state target, mixed mass metric, and denominators in all three
representations, so their scalar values are cross-representation comparable.
The representation-specific `J_M1` losses supervise different learned
quantities with different scaling and are **not** scalar-comparable.

The individual evidence and qualifications are documented in:

- `docs/TEST2B_REPRESENTATION_A_FINAL_SYNTHESIS.md`;
- `docs/TEST2B_REPRESENTATION_B_FINAL_SYNTHESIS.md`;
- `docs/TEST2B_REPRESENTATION_C_FINAL_SYNTHESIS.md`.

## 2. Matched deployed-objective results

The table gives the common state objective corresponding to each stage's own
deployment target.  Lower is better.

| stage artifact | A | B | C |
|---|---:|---:|---:|
| M1 evaluated under `J_M2-X` | `1.83511e-5` | `2.69831e-5` | `2.91056e-5` |
| warm M2-X under `J_M2-X` | `1.48758e-5` | `1.48142e-5` | **`1.40395e-5`** |
| H1 under `J_H1` | `3.00182e-5` | **`2.28200e-5`** | `3.42671e-5` |
| H2 under `J_H2` | `5.51872e-5` | **`3.99059e-5`** | `1.40039e-4` |
| H5 under `J_H5` | `1.10199e-4` | **`7.39776e-5`** | `7.03413e-4` |

C attains the lowest warm M2-X value, demonstrating that extra output freedom
can fit the fixed-X map.  That advantage does not survive deployment.  B gives
the lowest attained H1, H2, and H5 state objectives.  C's recursive objective
values are substantially worse even before autonomous extrapolation.

All three ladders show optimizer-basin sensitivity in the independent M2-X
fit.  The scientifically appropriate comparison retains both independent and
M1-warm-start branches; it does not interpret a poorer independent optimum as
an objective defect.

## 3. Matched autonomous state behavior

Every model starts from the same truth initial condition and advances 160
steps without reset.  The table reports `final / maximum / accumulated`; the
last column is accumulated error on held-out steps `81..160`.

| stage | representation | full state error | heldout accumulated |
|---|---|---|---:|
| M1 | A | `8.330e-6 / 9.880e-6 / 6.402e-6` | `6.863e-6` |
|  | B | `7.025e-3 / 7.025e-3 / 2.257e-3` | `3.202e-3` |
|  | C | `1.579e-5 / 1.879e-5 / 1.284e-5` | `1.282e-5` |
| warm M2-X | A | `8.025e-6 / 1.008e-5 / 6.376e-6` | `6.726e-6` |
|  | B | `8.968e-6 / 1.242e-5 / 7.619e-6` | `7.466e-6` |
|  | C | `1.239e-4 / 1.239e-4 / 7.377e-5` | `9.407e-5` |
| H1 | A | `4.934e-6 / 4.934e-6 / 3.010e-6` | `4.006e-6` |
|  | B | **`3.381e-6 / 3.889e-6 / 2.432e-6`** | **`3.265e-6`** |
|  | C | `1.815e-4 / 1.815e-4 / 9.750e-5` | `1.275e-4` |
| H2 | A | `4.935e-6 / 4.935e-6 / 3.009e-6` | `4.004e-6` |
|  | B | **`3.369e-6 / 3.875e-6 / 2.426e-6`** | **`3.255e-6`** |
|  | C | `3.959e-4 / 3.959e-4 / 1.866e-4` | `2.578e-4` |
| H5 | A | `4.967e-6 / 4.967e-6 / 3.011e-6` | `4.010e-6` |
|  | B | **`3.308e-6 / 3.817e-6 / 2.391e-6`** | **`3.205e-6`** |
|  | C | `1.389e-3 / 1.389e-3 / 7.182e-4` | `9.844e-4` |

The representation/objective interaction is decisive:

- A and B benefit strongly from H1.  B's H1/H2/H5 accumulated errors are
  about 19--21% below A's despite B having to learn rain.
- C does not obtain a solver-facing state benefit.  Its H1 accumulated error
  is 32.4 times A and 40.1 times B; the ratios grow to 62.0/76.9 at H2 and
  238.5/300.4 at H5.
- Within C, H1 lowers `J_H1` by 87.45% from C-M1 while autonomous accumulated
  error rises 659%.  H2 and H5 then lower their own targeted objectives while
  full and held-out errors worsen.  This is direct evidence of compensating,
  horizon-specific directions rather than a claim inferred from architecture
  alone.

These comparisons describe the tested continuation budgets.  H2/H5 each have
only 20 accepted iterations, so they do not establish what fully optimized
recursive objectives would do.

## 4. A -> B: the effect of learning rain

Learning `R` does not remove structural conservation: A and B both remain on
the exact two-rate source manifold and conserve total water at the host
numerical floor.  It removes three other pieces of prior knowledge: the exact
rain-rate magnitude, the precipitation threshold, and nonnegativity.

### 4.1 Constitutive and rain behavior

A recomputes the analytical law on every current model state.  Its H1/H2/H5
models retain the exact truth onset at `t=5100` and end with rain masses
`2.5485e8`, `2.5481e8`, and `2.5420e8`, about 10% high.  B predicts positive
rain source already at `t=0`, allows negative `R`, and its H1/H2/H5 final rain
masses are `1.5126e8`, `1.5147e8`, and `1.5027e8`, about 34--35% low.

B-M1 is the sharpest warning against equating a-priori rate fit with deployed
microphysics.  It fits `R` extremely accurately on truth states (training
active nRMSE `.00953`, heldout active `.02138`, correlations above `.9996`)
but its autonomous accumulated state error is `2.257e-3` and rain mass is
`6.8335e8`.  Small state departure places the learned two-rate model far off
its training distribution.  Analytical `R` in A prevents that failure mode.

### 4.2 State benefit and its cost

Once deployment-trained, B is better than A under the state metric: B-H1 is
19.21% lower in accumulated error than A-H1, and the advantage persists for
H2/H5.  The price is a poorer physical rain law—false pre-onset rain, negative
rate, incorrect onset, and about one-third too little final rain.  Exact
conservation alone therefore does not identify phase-conversion physics.

**A -> B conclusion:** learning `R` adds enough flexibility to improve the
tested deployment-state objective and trajectory norm after H1, but it loses
threshold, sign, onset, and phase-partition fidelity supplied by the analytical
law.  This is a cost/benefit trade, not a uniform improvement.

## 5. B -> C: the effect of removing source structure

Representation C's direct M1 fit approximately discovers the physical
manifold on training truth states: only 0.0748% of normalized source magnitude
is off manifold, and component nRMSE is below 0.6%.  This is meaningful
evidence that direct physical supervision regularizes the black-box source.
Its autonomous error is only about twice A-M1's, although it creates 0.01284%
net water and gives rain 17.9% low.

That discovered structure is not preserved by solver-facing optimization.
C-H1/H2/H5 off-manifold fractions on generated states are 2.26%, 3.26%, and
8.86%.  Their total-water drifts are `-0.356%`, `-1.132%`, and `-3.385%`.
The corresponding cumulative signed/absolute local water-defect ratios are
`-.684`, `-.951`, and `-.998`: these are coherent destruction directions, not
large local errors that happen to cancel.  Thermodynamic defect ratios also
approach one.  None of the three produces a positive rain-water mass; final
`Qr` is negative.

By contrast, A and B retain exact water and thermodynamic identities and
total-water conservation at roundoff for every network output.  C's additional
degrees of freedom thus lower warm M2-X slightly but make the deployed inverse
problem materially less identifiable.  The trained state loss can trade vapor,
cloud, rain, thermodynamics, and total water against one another outside the
physical two-rate manifold.

**B -> C conclusion:** removing source structure is decisively harmful in the
completed experiment.  It neither improves the recursive state objectives nor
the autonomous trajectory.  It creates systematic nonphysical source
directions whose magnitude and coherence grow with the tested horizon
continuations.

## 6. Rain, conservation, and phase partition summary

| stage | A rain behavior | B rain behavior | C rain behavior / water drift |
|---|---|---|---|
| M1 | onset `5100`; final `2.371e8` (+2.76%) | source at `0`; final `6.834e8` (+196%) | source at `0`; final `1.895e8` (-17.88%); water `+0.0128%` |
| warm M2-X | onset `5100`; final `2.276e8` (-1.34%) | source at `0`; final `1.426e8` (-38.2%) | no positive final rain; `Qr=-4.338e8`; water `-0.1496%` |
| H1 | onset `5100`; final `2.549e8` (+10.45%) | source at `0`; final `1.513e8` (-34.4%) | no positive rain mass; `Qr=-8.124e8`; water `-0.3563%` |
| H2 | onset `5100`; final `2.548e8` (+10.43%) | source at `0`; final `1.515e8` (-34.4%) | no positive rain mass; `Qr=-7.896e8`; water `-1.1316%` |
| H5 | onset `5100`; final `2.542e8` (+10.17%) | source at `0`; final `1.503e8` (-34.9%) | no positive rain mass; `Qr=-5.433e8`; water `-3.3850%` |

For C, a positive or negative `Qr` tendency is a component source diagnostic,
not a unique learned `R`; its `R_Qr=Qr_t/h` is labeled an effective rate.
Similarly, spurious rain redistribution is kept separate from total-water
creation/destruction.  The discrete state/source closure in the C audit proves
that the signed water drift is caused by the learned source rather than host
solver conservation noise.

## 7. Cost and budget accounting

| representation | accepted | objective / gradient evals | total wall (s) | hours | relative to A |
|---|---:|---:|---:|---:|---:|
| A | 30,040 | 62,997 / 30,046 | `103287.909` | `28.691` | baseline |
| B | 30,040 | 63,002 / 30,046 | `111428.787` | `30.952` | +7.882% |
| C | 30,040 | 62,676 / 30,046 | `83017.514` | `23.060` | -19.625% |

C is 25.50% cheaper than B in this serial experiment, but its lower wall cost
does not compensate for its physical and deployment failures.  These timings
are specific to one Mac, implementation, mesh, architecture family, and line
search history; they are not a scalability result.  Every fit ended at
`MAXITER`, and none is a demonstrated stationary optimum.

## 8. Hypotheses resolved by the data

1. **A gives the most faithful rain physics:** supported.  Analytical `R`
   preserves threshold, positivity, onset, and near-correct mass.
2. **B can improve state prediction while learning a poorer rain law:**
   supported for H1/H2/H5.  Conservation alone does not identify rain physics.
3. **C gains further state accuracy through nonphysical freedom:** not
   supported.  C uses nonphysical directions but is worse under both recursive
   objectives and autonomous deployment.
4. **Direct M1 acts as a source-manifold regularizer for C:** supported on the
   training distribution and approximately in deployment; not exact or fully
   held-out robust.
5. **Solver-facing objectives can weaken physical identifiability:** strongly
   supported in C, and supported for B's learned rain law.
6. **Good a-priori fitting plus conservation guarantees deployed
   microphysics:** refuted by B-M1.
7. **Removing structure makes recursive information more useful:** not
   supported under the tested 20-step continuations.  It opens more
   compensating directions, but those directions worsen deployment.

## 9. Frozen rain-active conclusions

The experiment does not support a one-dimensional slogan such as “more
physics” or “more solver awareness” is always better.  Representation and
objective interact:

- Analytical rain physics in A is the best protection for onset and phase
  partition.
- Learning `R` inside the exact structure (B) improves the deployed state norm
  after H1, but sacrifices the analytical rain law.
- Removing the structure (C) makes the inverse problem underconstrained.  The
  state objectives select off-manifold, nonconservative tendencies and fail in
  autonomous mature rain.
- Recursive gradients exist and are distinct, but their practical value
  depends on a representation that does not expose dominant unphysical
  compensation channels.

These are findings for the frozen Test2B trajectory and budgets, not universal
theorems.  Seed dependence, larger recursive budgets, explicit constrained or
penalized black-box ablations, other rain regimes, and broader spatial/temporal
generalization remain open.  No further learning campaign is started by this
synthesis.

**Final statuses:**

- `TEST2B_REPRESENTATION_C_FROZEN`
- `TEST2B_RAIN_ACTIVE_A_B_C_SYNTHESIS_COMPLETE`
