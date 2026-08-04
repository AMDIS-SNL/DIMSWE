# DIMSWE non-gating characterization

These observations describe the current implementation on the tiny serial
`tests/mtswe_small.cfg` problem.  They are not declarations of scientific
correctness and are deliberately separated from the permanent baseline
specifications.

Run them separately with:

```sh
python -m pytest -q tests/test_mtswe_characterization.py -m characterization
```

The original five procedures completed with finite results in the validated
local macOS environment on 2026-08-03.  They are marked `characterization`;
none is hidden with `xfail` or an unconditional skip.  The 2026-08-04
correction made configured and applied moist timesteps independent as
described below; the corrected procedure completed in the validated local
environment with the authoritative measurements recorded here.

## Limiter hook

- **Demonstrated behavior:** one active global `LieSplittingIntegrator` step
  called the instrumented `DG1LimiterTransport.post_step` zero times.
- **Code evidence:** the hook applies the vertex limiter in
  `dimswe/transport_operators.py:131-134`, and it is reachable through
  `AdvDensCF_H1_Dynamics.post_step` at `dimswe/dynamics.py:513-516`.
  `LieSplittingIntegrator.take_forward_step` advances all children at
  `dimswe/timestepping.py:528-537` but contains no `post_step` call.
- **Possible intended behavior:** a limited DG transport method would normally
  apply the limiter after a stage or completed transport step.
- **Unresolved intent:** placement after each stage, child, subcycle, or global
  step changes the numerical method.  An author must select that semantics;
  this milestone does not modify it.

## Hamiltonian topography

- **Demonstrated behavior:** with the nonzero TC5 mountain, the separately
  initialized dynamics topography had L2 norm `2.589916345847779e-07`, while
  `ThermalShallowWater_Hamiltonian_Base.bottom_topography` had norm exactly
  `0.0`.
- **Code evidence:** the Hamiltonian owns and can initialize its Function at
  `dimswe/hamiltonians.py:86-94`.  The active initialization path calls the
  variable and forcing initializers, then initializes only the dynamics-owned
  topography at `dimswe/dynamics.py:444-449`.
- **Possible intended behavior:** the Hamiltonian topographic energy and
  derivative may be intended to use the configured mountain.
- **Unresolved intent:** there are two topography Functions, and deciding
  whether to initialize, alias, or remove one affects the governing
  Hamiltonian.  Author confirmation is required.

## Moist conversion: configured versus applied `dt`

- **Measured characterization matrix:** the helper fixes one switch-safe
  active branch and accepts configured and applied timesteps independently.

  | Configured `dt` | Applied `dt` | Integrated vapour increment | Ratio to `(100,100)` |
  | ---: | ---: | ---: | ---: |
  | `100` | `100` | `-1.339285714285711e13` | `1.0` |
  | `100` | `50` | `-6.696428571428555e12` | `0.5` |
  | `50` | `100` | `-2.678571428571422e13` | `2.0` |

  These are explicitly non-gating measurements of the current implementation,
  not a scientific specification or production tolerance.
- **Code evidence:** `ThreeWayPhysics` stores configured `dt` and assigns both
  relaxation times from it at `dimswe/physics.py:19,29-30`; the conversion
  rates divide by those values at `dimswe/physics.py:85-90`, while Euler
  multiplies the tendency by its applied step at
  `dimswe/timestepping.py:346-369`.
- **Mathematical consequence:** on the fixed branch the pointwise rate has
  the form `R(state)/configured_dt`, while Euler applies
  `applied_dt*R(state)/configured_dt`.  Therefore the current increment scales
  as `applied_dt/configured_dt`.  Equality of the two values cancels the
  explicit factor, including the cloud cap; this explains the earlier coupled
  measurements but does not establish physical intent.
- **Unresolved intent:** the relaxation times may be physical parameters or
  timestep-dependent numerical parameters.  Changing this is moist-timestep
  semantics and requires an author decision.

## Isolated DG rain transport

- **Demonstrated behavior:** for a cosine `Qr` field and constant velocity, one
  isolated DG child changed `Qr`; initial L2 norm was
  `1.0152840849994279e6` and the change norm was `2.4318594664569247e2`.
- **Code evidence:** moist `Qv`, `Qc`, and `Qr` are placed in DG1 at
  `dimswe/variables.py:244-247`.  `DG1LimiterTransport.rhs` iterates all such
  densities and supplies volume and interior-facet fluxes at
  `dimswe/transport_operators.py:137-151`.
- **Possible intended behavior:** rain is plausibly intended to advect with
  the resolved velocity, but the missing boundary-term comment and inactive
  limiter hook make the complete transport design uncertain.
- **Unresolved intent:** this observation shows only that the present periodic
  serial operator is nontrivial.  It does not validate rain-transport physics,
  boundary behavior, or limiter placement.

## Hyperviscosity Fourier mode

- **Demonstrated behavior:** one isolated Euler hyperviscosity child mapped the
  measured mode amplitude from `9737.450795091365` to
  `9737.348077384857`, an amplification factor of
  `0.9999894512733702` (damping on this mesh and step).
- **Code evidence:** the diagnostic Laplacian solve is formed at
  `dimswe/dissipation.py:64-74`; the biharmonic weak term and coefficient
  `c0 * factor**s` are applied at `dimswe/dissipation.py:76-88`.
- **Possible intended behavior:** positive hyperviscosity is generally
  expected to damp resolved Fourier modes.
- **Unresolved intent:** this single discrete mode does not establish the
  intended spectrum, timestep stability envelope, mesh scaling, or whether
  all active fields should share the same operator.  Those are author choices.

The unexpectedly large reported modal coefficient is a projection/integration
quantity on the two-cell periodic coordinate representation; only the ratio
is used for this characterization.  No production tolerance or spectral
contract is inferred from it.
