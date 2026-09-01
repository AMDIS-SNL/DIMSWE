# Deployed Hybrid Dynamics — Draft Captions

## Shared convention

All event galleries use truth-defined reference columns at 0, 5000, 6100,
12000, and 16000 s: initial, last clearly pre-rain truth state, sustained-rain
truth reference, peak integrated truth rain-production time, and final state.
These event labels do not assert that a deployed model has the same event
timing.  Saturation departure, \(q_c\), and \(q_r\) are boundary-state
quantities; rates are evaluated at the corresponding model-generated pre-moist
state \(\hat Y_n=P(\hat X_n)\).  Gray contours are relative vorticity from the
same deployed model.  One normalization per variable is shared by truth and all
twelve deployed models over all 161 times; signed values are not clipped.

### Truth reference

**DoubleVortex truth.** Common-scale reference for saturation departure,
specific cloud water, specific rain water, and analytical rain-production rate.
The scale and vorticity levels are identical to every deployed-model gallery.

## Representation A

Representation A learns phase change \(A_\theta\); rain production remains the
analytical law \(R^*\) evaluated on the model-generated pre-moist state.

- **A — M1-Y.** Frozen M1-Y hybrid evolution under the shared event times and
  color normalization.  The learned phase-change law is sampled at its deployed
  pre-moist call site; the bottom row is analytical \(R^*(\hat Y_n)\).
- **A — H1.** Frozen one-step-trained hybrid evolution.  The columns remain
  truth-defined references and are not model-specific onset markers.
- **A — H2.** Frozen two-step continuation hybrid evolution.  No optimization
  or recursive checkpoint history was rerun to make this figure.
- **A — H5.** Frozen five-step continuation hybrid evolution, rendered from the
  validated spatial cache with the same limits as truth, M1-Y, H1, and H2.

Movie captions use the corresponding wording above and add: *All 161 boundary
states are shown at 10 fps in six panels: relative vorticity, saturation
departure, \(q_c\), \(q_r\), learned \(A\), and analytical \(R\).*

## Representation B

Representation B learns both phase change \(A_\theta\) and rain production
\(R_\theta\); the bottom row is therefore a deployed neural-network output.

- **B — M1-Y.** Frozen M1-Y hybrid evolution with learned rain production at
  the model-generated pre-moist state.  The signed common scale retains both
  positive and negative network rates.
- **B — H1.** Frozen one-step-trained hybrid evolution.  Signed learned
  rain-production structure is shown without thresholding or clipping.
- **B — H2.** Frozen two-step continuation hybrid evolution under the same
  physical units, vorticity levels, and common limits.
- **B — H5.** Frozen five-step continuation hybrid evolution.  Differences in
  rain amplitude and rate structure are displayed on exactly the truth/M1-Y
  scales.

Movie captions use the corresponding wording above and add: *All 161 boundary
states are shown at 10 fps; the bottom two panels are learned \(A_\theta\) and
learned \(R_\theta\) evaluated at \(\hat Y_n\).*

## Representation C

Representation C deploys four independently predicted moist-source
components.  For cross-representation visualization only, the two bottom rate
panels use the accepted scale-weighted projection onto the physical two-rate
source manifold.  The projected rates are diagnostic and are not substituted
into the trajectory.

- **C — M1-Y.** Frozen M1-Y hybrid evolution.  The bottom row is effective
  rain-production rate from the accepted physical projection of the four
  predicted sources.
- **C — H1.** Frozen one-step-trained hybrid evolution.  The shared signed
  scales expose negative cloud/rain water and negative effective rain
  production rather than clipping them.
- **C — H2.** Frozen two-step continuation hybrid evolution.  Effective \(A\)
  and \(R\) are diagnostic views of the unconstrained deployed source.
- **C — H5.** Frozen five-step continuation hybrid evolution.  The common
  normalization preserves the full unphysical signed range and permits direct
  comparison with truth and the other eleven deployed models.

Movie captions use the corresponding wording above and add: *All 161 boundary
states are shown at 10 fps; panels labeled effective \(A\) and effective \(R\)
use the accepted diagnostic two-rate projection, while the original four
source components drive the model.*

