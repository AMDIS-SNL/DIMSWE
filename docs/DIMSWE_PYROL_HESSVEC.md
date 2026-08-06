# DIMSWE production PyROL Hessian-vector interface

## Stage P1 scope

Stage P1 exposes only normalized scalar hyperviscosity `c0` through PyROL.
The implementation is `dimswe.mtswe_rol_adapter.ProductionMTSWEScalarC0Objective`.
It delegates to the accepted production MTSWE reduced-gradient and reduced-HVP
APIs and does not alter their mathematics.

This stage does not contain a Firedrake PyROL vector, initial-condition
optimization, a combined field/scalar vector, or a field Riesz conversion.
All PyROL inputs and outputs are installed one-element `NumPyVector` objects.

The legacy `dimswe.rol_adapter.ScalarC0Objective` remains the original
first-order adapter backed by `Lagrangian_ODEConstrainedOptimization`.  It is
not extended or replaced by Stage P1.

## Optional dependency boundary

`dimswe.mtswe_rol_adapter` imports PyROL and may be imported explicitly only
when `rol-python` is installed.  The module is not imported from
`dimswe/__init__.py`; ordinary `import dimswe` and non-PyROL tests therefore do
not acquire a PyROL dependency.

## Construction

The scalar production objective is constructed as

```python
ProductionMTSWEScalarC0Objective(
    timestepper,
    coefficient_template,
    fixed_initial_state,
    target,
    nsteps=1,                         # or 3
    t0=t0,
    dt=dt,
    c0_scale=0.07,
    gradient_zero_margin_tolerances={...},
    hvp_active_set_tolerances={...},
)
```

`nsteps` is restricted to the certified values 1 and 3.  `c0_scale` and `dt`
must be finite and positive.  The production coefficient order must be
exactly `[s,c0]`.

The objective owns deep copies of its coefficient template, fixed initial
state, and target.  It exclusively uses its timestepper and HVP helper;
sharing the same timestepper concurrently with another objective or forward
simulation is unsupported.

## Normalized scaling

For normalized control `z` and normalized direction `q_z`,

```text
c0       = d_c0*z
delta_c0 = d_c0*q_z
```

The production helper receives only those physical quantities.  The adapter
returns

```text
J_normalized(z) = J_physical(d_c0*z)
g_z             = d_c0*g_c0
h_z(q_z)        = d_c0*[H_physical(0,d_c0*q_z)]_c0.
```

Consequently, the scalar-scalar Hessian block carries `d_c0**2`.  No
normalized scalar is passed into the production helper.

The exact installed PyROL callbacks are

```python
value(self, x, tol)
gradient(self, g, x, tol)
hessVec(self, hv, v, x, tol)
update(self, x, *args)
```

Every vector is validated as a one-entry `pyrol.vectors.NumPyVector`.
Derivative output vectors may not be the same object as their inputs.

## Caching

Caching is bounded to two entries:

1. one current normalized point and its production gradient result;
2. one current normalized `(point,direction)` and its production HVP result.

The keys are owned Python floats and use exact equality.  A changed point
evicts both entries; a changed direction evicts only the HVP entry.  An HVP
result also contains the objective value and ordinary gradient, so it seeds
the current-point entry.  `update` clears both entries unconditionally, but
cache correctness does not depend on update notifications.

There is no accepted value-only production wrapper.  Stage P1 therefore
obtains `value` through the production reduced-gradient path.  This permits a
following gradient at the same point to reuse the result, at the cost of a
reverse solve for value-only line-search points.

## Ownership and restoration

Every production evaluation receives new owned copies of the fixed state and
target.  The HVP receives an owned zero initial-condition direction plus the
physical scalar direction.  Production results and scratch never alias `x`,
`g`, `v`, or `hv`.

Before setting physical `c0`, the adapter snapshots the coefficient Function
of every split child.  Its evaluation context resets production scratch,
installs an owned working coefficient, runs the certified API, restores every
child coefficient in a `finally` block, and resets scratch again.  Restoration
therefore also occurs after a production exception.  This relies on the
accepted result contract that reduced result objects own their trajectory and
reverse data independently of timestepper scratch.

## Active-set policy

The report contains one entry for each complete timestep and each of

```text
condensation
evaporation
evaporation_cap
rain
depth_denominator
```

Every entry records the timestep, switch, minimum absolute margin, configured
threshold, qualification result, and branch signature where applicable.
Thresholds are mappings by switch; no common tolerance is inferred for
quantities with different units.  Omitted mappings or omitted switch entries
default to the exact-boundary threshold `0.0`; callers requiring a stronger
separation must provide the five unit-appropriate HVP thresholds explicitly.
Timestep indices in reports are zero-based.

The three callback policies are deliberately different:

- `value` always returns the physically defined value and records a report;
  it does not require derivative qualification.
- `gradient` rejects only a zero margin or a margin at/below its configured
  machine-zero threshold.  Smaller margins relative to the stronger HVP
  threshold still return the current branchwise gradient.
- `hessVec` requires every margin to exceed its explicit HVP threshold and
  raises `MTSWEHVPActiveSetQualificationError` otherwise.

The qualification exceptions retain their report and identify every failing
timestep, switch, margin, and threshold.  The adapter never smooths,
regularizes, perturbs, or changes the production moist closure.

Centered derivative checks must additionally confirm that base, plus, and
minus branch signatures agree.  A positive base margin alone does not prove a
finite perturbation stayed on the same branch.

## Authoritative P1 validation

The authoritative first external run on 2026-08-06 reported:

```text
new scalar adapter:             31 passed, 1 failed, 2312 warnings
legacy adapter:                  6 passed, 594 warnings
accepted production MTSWE HVP: 22 passed, 29690 warnings
complete repository:           153 passed, 1 skipped, 1 xfailed,
                               1 failed, 54203 warnings
```

The sole failure was test-side construction of the explicit step vector:

```text
result = vector_double_t(len(values))
result[index] = float(value)

TypeError: 'pyrol.pyrol.std.vector_double_t' object does not support item
assignment
```

It occurred before `Objective.checkGradient`, `checkHessVec`, or
`checkHessSym` was called.  The installed binding docstrings confirm the
native `vector_double_t()` default constructor and `push_back(float)` method;
the test now uses that supported path and checks the exact native type, length,
and ordered values.  No adapter callback, scaling, cache, active-set, legacy,
or production-mathematics failure was observed.  PyROL derivative-utility
runtime certification, including proof that the utility invokes scalar
`hessVec`, remained pending until the focused test was rerun externally.

The test-only correction required no adapter or production mathematical
change.  The authoritative reruns then reported:

```text
focused scalar adapter:         33 passed, 2312 warnings in 373.40s
native PyROL utility test:       1 passed in 1.21s
legacy adapter:                  6 passed, 594 warnings in 73.03s
accepted production MTSWE HVP: 22 passed, 29690 warnings in 893.36s
complete repository:           155 passed, 1 skipped, 1 xfailed,
                               54213 warnings in 1567.67s (0:26:07)
```

The complete-suite log was closed and stable before inspection, contained no
failure or error section, and ended with the successful summary above.  The
existing optional-dependency skip and the existing
`ode_adjoint/test_optimize.py::test_optimize_params_plus_ic` expected failure
remain characterized; neither is part of the Stage P1 adapter interface.
Native `checkGradient`, `checkHessVec`, and `checkHessSym` execution is now
certified for the scalar test fixture, including the instrumented assertion
that scalar `hessVec` was called.

## Validation sequence

Run the focused scalar adapter tests first:

```text
python -m pytest -q tests/test_mtswe_rol_adapter.py
```

Then run the unchanged legacy adapter regression:

```text
python -m pytest -q tests/test_rol_adapter.py
```

Then run the accepted production MTSWE HVP regression:

```text
python -m pytest -q tests/test_production_mtswe_split_hvp.py
```

Only after all three pass, run the complete repository suite:

```text
python -m pytest -q
```

These are serial validation commands.  Stage P1 makes no MPI certification
claim.
