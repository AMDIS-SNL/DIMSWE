# Test 2 moist-physics activity audit

## Purpose and scope

This is a quantitative prerequisite for Test 2, not a neural-model design or
training run.  It asks whether the selected double-vortex truth contains enough
deployed moist-physics activity to support a first learned-replacement
benchmark.  It reads only stored truth states 0 through 80 (times 0 through
8000 seconds), never advances the PDE, and never opens restart snapshots 81
through 160.  Those later states remain untouched future deployment data.

No architecture, features, normalization, network size, or final output
parameterization was selected by this audit.  Its externally executed result
has subsequently authorized the narrower Test 2A-1 design described in
`TEST2A_OPERATOR_LEARNING.md`.

## Accepted activity result and decision

The certified external audit examined all 331,776 training samples.  A was
active in every one of 81 states, had maximum absolute value
`1.08052350238e-7`, RMS `9.35488003107e-9`, negative fraction
`0.5944944782`, and positive fraction `0.4003213011`.  Fractions above
`1e-6` and `1e-3` of its global maximum were respectively `0.9580711082` and
`0.6440731096`.  Its exclusive-field global RMS Euler-increment ratio was
`4.87997980152e-4`.

R was exactly zero at every training sample.  The accepted classification is
therefore `A_ACTIVE_R_WEAK`.  This does not authorize replacing R by a
constant zero: an imperfect learned A can move an autonomous trajectory into
a state where the original rain law activates.  Test 2A learns A only and
retains the original deployed R evaluation.

## Certified deployed representation

The audit instantiates the existing
`dimswe.jax_moist_adapter.JAXMoistEulerPrimal`.  For each stored truth state it
uses the J1 interpolation and packing path:

```text
mixed production state
  -> broken CG3 carrier interpolation of h, S, Qv, Qc and B
  -> explicit cell_node_map packing
  -> [owned cells, 16] float64 array
  -> certified J1 JAX moist primal
  -> A, R and S/Qv/Qc/Qr source densities
```

This is the exact cell-local tensor 4-by-4 GLL representation used by the
deployed source form.  It is not a mixed coefficient vector, DG1 grid, cell
center, projection, or arbitrary interpolation.  J1 already certifies local
UFL/JAX parity at these points and the subsequent weak assembly and mixed mass
solve.  The production truth retains its UFL moist backend; this diagnostic
uses the certified JAX replica only to expose the same local values.

The selected 16-by-16 mesh has 256 quadrilateral cells, 16 GLL points per cell,
and 4,096 cell-local samples per state.  States 0 through 80 give 81 stored
states and 331,776 space-time samples.  Physical points shared by adjacent CG
cells appear repeatedly because the deployed broken carrier and local kernel
see those cell-local repetitions.  The audit does not deduplicate them.

## Quantitative output

For A and R separately, the JSON records signed extrema, mean, mean absolute,
RMS, standard deviation, exact-zero/positive/negative fractions, signed and
absolute percentiles at 0/1/5/25/50/75/95/99/100, and activity fractions for
`|rate|` above `1e-12`, `1e-9`, `1e-6`, and `1e-3` times the global maximum
absolute rate.  An exactly zero maximum has its own explicit contract.

Every state also records maximum absolute, mean absolute, RMS, exact-zero and
sign fractions, plus active fractions relative to `1e-6` and `1e-3` of the
global space-time maximum.  Four time plots show max/RMS and activity for A and
R.  Two additional scatter maps use the objectively selected times of maximum
domain RMS A and maximum domain RMS R.  Repeated cell-boundary samples remain
visible in the diagnostic representation.

The certified identities

```text
Qv_t =  h A
Qc_t = -h (A + R)
Qr_t =  h R
S_t  =  h beta2 A
```

are checked at every sample.  Each source has signed extrema, mean absolute,
RMS, and absolute percentiles.  The JSON separately reports `-hA` and `-hR`
cloud-water contributions and their RMS/mean-absolute ratios.

For each field, `dt` times the local GLL source is compared with the same truth
field on the same GLL grid using global and per-state RMS and maximum scales.
Zero truth scales are explicit.  A second per-state record uses the complete
certified weak source assembly and mixed mass solve, comparing the actual moist
Euler child update with the truth-field mass norm.  Thus raw nonzero rates are
distinguished from a measurable deployed state effect.

Input support includes only existing local moist inputs `h,S,Qv,Qc,B`, with
ranges and percentiles in the same deployed space-time representation.  It
does not propose extra features.

## Decision screen

The output classification is one of:

- `RICH_TWO_RATE_SIGNAL`
- `A_ACTIVE_R_WEAK`
- `R_ACTIVE_A_WEAK`
- `BOTH_WEAK_OR_DEGENERATE`
- `NEEDS_SCIENTIFIC_REVIEW`

This is a transparent degeneracy/numerical-effect screen, not a universal
scientific pass threshold.  A rate is screened active when all of the following
hold:

- its maximum absolute value is nonzero;
- at least `1e-4` of space-time samples exceed `1e-6` of its global maximum;
- at least 5 percent of stored states contain a value above that scale;
- its exclusive field increment (`Qv` for A, `Qr` for R) exceeds 100 float64
  eps relative to that truth-field RMS.

The JSON exposes every underlying value so the scientific decision can be
reviewed without rerunning Firedrake.  In particular, R degeneracy and tiny
moist increments must be reported even if another rate is active.

## External command

Run in the certified serial Firedrake/JAX environment:

```bash
cd /path/to/DIMSWE-collaborator

export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export TEST2_CACHE_ROOT="$(mktemp -d /tmp/dimswe-test2-moist.XXXXXX)"
export PYOP2_CACHE_DIR="$TEST2_CACHE_ROOT/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$TEST2_CACHE_ROOT/tsfc"
export XDG_CACHE_HOME="$TEST2_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$TEST2_CACHE_ROOT/matplotlib"

export TEST1B_TRUTH="$PWD/external-results/test1b-production/truth_c0_0.14"
export TEST1B_PLAN="$PWD/dimswe/configs/test1b_selected_plan.json"
export TEST2_ROOT="$PWD/external-results/test2-prep"

python -m dimswe.test2_moist_activity \
  --truth-run "$TEST1B_TRUTH" \
  --selected-plan "$TEST1B_PLAN" \
  --output "$TEST2_ROOT/doublevortex_training_moist_activity.json" \
  --plot-directory "$TEST2_ROOT/plots"
```

Expected generated files are the canonical JSON plus:

```text
plots/A_maximum_rms_vs_time.png
plots/R_maximum_rms_vs_time.png
plots/A_active_fraction_vs_time.png
plots/R_active_fraction_vs_time.png
plots/A_rms_selected_spatial_map.png
plots/R_rms_selected_spatial_map.png
```

These generated audit outputs remain untracked.  Review the quantitative
classification and source-increment evidence before authorizing any neural
architecture or training objective.
