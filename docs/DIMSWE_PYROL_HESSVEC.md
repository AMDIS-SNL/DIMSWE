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
existing `tests/test_import.py::test_import_optional_plotting_module`
optional-dependency skip and the existing
`ode_adjoint/test_optimize.py::test_optimize_params_plus_ic` expected failure
remain characterized; neither is part of the Stage P1 adapter interface.
Native `checkGradient`, `checkHessVec`, and `checkHessSym` execution is now
certified for the scalar test fixture, including the instrumented assertion
that scalar `hessVec` was called.

## Stage P2 COMPLETE: physical initial-condition vector and objective

`MTSWEStateVector` owns a deep copy of one full mixed production state in the
deployed `(v,h,S,Qv,Qc,Qr)` space.  It implements the installed PyROL
`clone`, `set`, `plus`, `scale`, `dot`, `norm`, `axpy`, `zero`, `dual`,
`apply`, and `dimension` callbacks.  Clones own independent zero Functions;
no operation exports or stores a flattened NumPy coefficient vector.

The vector metric is exactly the already-certified production mixed L2
metric:

```text
u.dot(v) = helper.dual_pairing(helper.state_mass_map(u), v).
```

No nominally equivalent UFL form is reconstructed.  The vector stores primal
Riesz coordinates and is self-dual at the PyROL interface, matching the
installed binding's default self-dual contract.  `apply` therefore evaluates
the same certified metric pairing.  Exact cache comparisons use PETSc vector
equality between owned Function snapshots, not flattened coefficient arrays.

`ProductionMTSWEInitialConditionObjective` fixes physical `c0` and exposes the
physical initial state.  Its production gradient remains a Cofunction until a
single final call to `state_riesz_representative` copies primal coordinates
into the PyROL output.  Its HVP passes the physical field direction and
`delta_c0=0`; only the final returned Cofunction crosses the same explicit
Riesz boundary.  One- and three-step operation, bounded point/direction
caches, update invalidation, ownership, restoration, and active-set policies
match P1.

The focused P2 tests cover vector ownership/algebra, exact metric parity,
self-dual/apply behavior, rejection of the coefficient-vector metric,
one-/three-step production gradient and IC-only HVP parity, final Riesz
representatives, caches, nonmutation, exception restoration, active-set
qualification, native derivative utilities, and field Hessian symmetry.

## Stage P3 COMPLETE: combined physical-state and normalized-scalar control

`MTSWECombinedVector` owns independent `MTSWEStateVector` and one-entry
normalized scalar children.  It never flattens the blocks.  Its product
metric is

```text
((x,z),(q,r)) = helper.dual_pairing(helper.state_mass_map(x),q) + z*r.
```

All installed vector callbacks act componentwise, and both children remain
independent under construction and cloning.  Combined bound constraints are
not claimed by this stage.

For `ProductionMTSWECombinedObjective`,

```text
P(x0,z) = (x0,d_c0*z)
P*(lambda_x*,a) = (lambda_x*,d_c0*a)

grad_y J = P* grad_phys J
H_y q    = P* H_phys(Pq).
```

Thus a direction `(q_x,q_z)` enters production as `(q_x,d_c0*q_z)`.  The
returned field Cofunction receives exactly one final certified Riesz solve;
the returned physical scalar component is multiplied by `d_c0`.  The
scalar-scalar block carries `d_c0**2`, and each mixed field/scalar block
carries one `d_c0`.  Focused tests cover c0-only, IC-only, and combined
directions for one and three timesteps, both mixed blocks, product-metric
bilinear symmetry, caches, restoration, active sets, and instrumented native
PyROL derivative utilities.

## P2/P3 native `checkHessVec` diagnostic

The first authoritative external P2 and P3 runs stopped at their native
PyROL derivative-utility tests.  P2 reported 7 passes before the failure and
P3 reported 12.  Both failures returned the identical final
`checkHessVec` value `0.3998688789132763`; both `checkGradient` tables were
decreasing, and no deployed production-HVP test failed.

The installed ROL implementation was inspected at
`pyrol/include/ROL_ObjectiveDef.hpp` and `pyrol/include/ROL_Objective.hpp`.
For `checkHessVec(x, v, steps, False)`, each result row is exactly

```text
[step, norm(H*v), norm(FD gradient difference), norm(FD difference - H*v)].
```

The last entry is therefore an absolute dual-norm error, not a relative
error.  The selected overload is the documented explicit-step overload; it
does not normalize or otherwise transform `v`.  It clones the dual prototype
before using gradient, HVP, finite-difference, and work vectors.  The same
source shows that `checkGradient` evaluates `d.apply(g)`, while
`checkHessSym` evaluates `w.apply(Hv)` and `v.apply(Hw)`.  ROL's installed
`Vector` default explicitly permits `dual()` to return the current object.
Those source facts permit the proposed self-dual representation, but do not
alone certify the actual Firedrake-backed vector.

The next authoritative run failed earlier, in the added identity quadratic
evaluated at the large physical MTSWE state.  Both field and combined tests
reported the same minimum absolute error,
`0.0015918440945659738`, before either monkeypatched adapter check ran.  Thus
the earlier claim that centering only the fake production oracle completely
identified the root cause was too narrow.  The pure quadratic still asks ROL
to subtract large, nearly identical field vectors to recover an
`O(epsilon)` direction.  The r2 log recorded only the minimum, so it cannot
supply the three native rows retrospectively.

The revised field test now prints all physical-base native rows, `norm(x)`,
`norm(v)`, per-field coefficient maxima, successive error growth, and direct
vector-API finite-difference rows.  It requires each native row to agree with
its directly constructed `[norm(v), norm(fd), norm(fd-v)]` values and requires
the physical-base error to grow as the step decreases.  This physical-base
ladder is a conditioning characterization, not the strict vector-contract
gate.

The strict gates now evaluate the minimal quadratics
`Q(x)=0.5*x.dot(x)` and `Q(x,z)=0.5*(x.dot(x)+z*z)` at an owned zero field and
at `(zero field, z=0)`.  They retain steps `1e-3`, `1e-4`, and `1e-5`, the
absolute `1e-8` requirement, native `checkGradient`, `checkHessVec`, and
`checkHessSym`, explicit `hessVec` instrumentation, input nonmutation, and
dual-clone safety.  The combined zero-base gate independently isolates the
product-vector scalar block.

The monkeypatched adapter utilities likewise use numerically conditioned
synthetic points: a zero field for P2 and a zero field with modest normalized
scalar for P3.  Their translated quadratics are

```text
Q(x) = 0.5 * (x - x_ref).dot(x - x_ref),
gradient* = M * (x - x_ref),
H*v = M * v,
```

with `x_ref` equal to the synthetic utility point; the combined oracle also
centers its normalized scalar at that test's `z_ref`.  Translation does not
change the identity Hessian.  It preserves the fake production contract
(`Cofunction` gradient/HVP followed by exactly one adapter Riesz solve), the
native utilities, all explicit step values, the `1e-8` assertion, and the
HVP-call instrumentation.  A direct callback regression now records forward
and centered epsilon ladders at `1e-3`, `1e-4`, and `1e-5`, compares every
field, records the exact callback point and direction snapshots, checks cache
hits, and verifies input nonmutation.

Production-scale correctness remains separately covered, without weakening,
by direct one-/three-step production gradient and HVP parity, all direction
blocks, natural-pairing symmetry, nonmutation, and repeatability.  The strict
native finite-difference utility and production-oracle tests certify
different boundaries.

The authoritative r3 serial run confirmed that separation.  At the physical
base,

```text
norm(x) = 3.785903703396193e10
norm(v) = 7.273830663658165e6

epsilon   native/direct norm(FD-Hv)   epsilon * error
1e-3      1.5918440945659738e-3      about 1e-6
1e-4      1.3902243035109323e-2      about 1e-6
1e-5      1.1918356529583173e-1      about 1e-6
```

Native and direct vector-API rows agreed exactly.  The error grew by about
8.6--8.7 per decade while `epsilon * error` stayed near `1e-6`, confirming
subtraction/cancellation at the large physical state.  At zero base, the
strict field and combined native HVP ladders had absolute errors
`8.136640144695799e-10`, `2.467819228230368e-10`, and
`7.150985271159583e-10`; Hessian symmetry error was exactly `0.0`.

This certifies the field vector `dual`/`apply` contract and combined product
contract.  Production HVP mathematics was not implicated.  Strict native
finite-difference vector/callback checks use a conditioned zero/small
synthetic point, while physical-state correctness is certified by direct
production parity and natural pairings.

Only `tests/test_mtswe_rol_state_adapter.py`,
`tests/test_mtswe_rol_combined_adapter.py`, this document, and the P1/P2/P3
audit were changed for the correction.  No adapter callback, cache, vector
metric, Riesz boundary, normalized `c0` scaling, or accepted production
mathematics was changed.

## Authoritative P2/P3 serial certification

The final authoritative serial results were:

```text
native field/combined utilities:  2 passed, 56 warnings in 31.92s
P2 state adapter:                10 passed, 2046 warnings in 359.46s
P3 combined adapter:             14 passed, 4326 warnings in 427.68s
P1 scalar regression:            33 passed, 2312 warnings in 352.51s
legacy PyROL regression:          6 passed, 594 warnings in 68.06s
accepted production MTSWE HVP:   22 passed, 29690 warnings in 857.17s
complete repository:            179 passed, 1 skipped, 1 xfailed,
                                60449 warnings in 1659.94s
```

No `FAILED` or `ERROR` section occurred.  The skip and expected failure remain
the previously characterized repository cases.  Known warnings remain the
PyOP2/NumPy shape deprecation, FINAT quadrilateral DG-to-DQ warning, UFL
quadrature-metadata warning, SciPy L-BFGS-B `hessp`/`disp` notices, and PETSc
seeing pytest's `-q` option.

This certification is serial only.  It makes no MPI claim and adds no state
normalization, combined bound-constraint support, JAX, neural-network, or
checkpointing capability.
