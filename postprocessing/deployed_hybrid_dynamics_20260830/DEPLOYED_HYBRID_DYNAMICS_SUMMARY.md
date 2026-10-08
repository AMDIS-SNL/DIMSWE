# Deployed Hybrid Dynamics — Visual Scientific Summary

This summary is limited to features visible in the validated common-scale
galleries and final GIFs.  It does not infer causality from visual association.

## Representation A

All four methods retain the truth-like two-vortex skeleton and place the mature
rain bands on the same two counter-rotating outer arcs.  M1-Y, H1, H2, and H5
have very similar cloud morphology; their main visible differences are local
amplitude and late-time saturation/rate detail rather than displacement or a
different large-scale pattern.  M1-Y shows a somewhat more visible late-time
positive saturation feature between/near the cores.  H1, H2, and H5 are nearly
indistinguishable from one another in these fields and show slightly stronger
mature rain bands than the truth/M1-Y views on the common scale.  The
analytical rain law preserves narrow positive rain-production arcs and does not
introduce a separate learned sign pattern.

## Representation B

The vortex skeleton, cloud envelope, and spatial placement of the rain arcs
remain close to truth for all four methods.  M1-Y produces rain-water bands with
roughly the truth-like visual amplitude and extent.  H1, H2, and H5 retain the
correct arc morphology but the bands are visibly weaker, so their principal
state-field discrepancy is amplitude rather than gross location.

The learned rain-rate fields differ much more strongly than the state fields.
M1-Y has fine positive/negative structures around the vortices and mature rain
bands.  H1, H2, and H5 show broad negative interiors surrounded by thinner
positive structures, including activity in the truth-defined pre-rain columns.
Thus the poor local \(R\) recovery is spatially apparent even where cloud and
vortex morphology—and the previously reported global state errors—remain
small.  The galleries alone do not establish why those state errors remain
small.

## Representation C

M1-Y retains truth-like cloud and positive rain-band morphology despite a
strongly signed, spatially oscillatory effective rain-rate field.  This is a
visible example of state agreement coexisting with source-structure freedom;
the effective two-rate projection is diagnostic and does not replace the four
deployed source components.

H1, H2, and H5 show a qualitatively different water partition.  Negative rain
water appears around the vortex cores by the 5000 s truth pre-rain reference,
then grows into broad core-centered negative structures instead of the truth's
positive outer rain bands.  Negative cloud-water cores and increasingly strong
subsaturation accompany this pattern.  The effect becomes most pronounced for
H5, which supplies the global extrema that required common signed ranges of
approximately \(-0.035\) to \(0.11\) g kg\(^{-1}\) for \(q_c\), \(-80\) to
140 \(\mu\)g kg\(^{-1}\) for \(q_r\), and \(-22\) to 190 \(\mu\)g
kg\(^{-1}\) h\(^{-1}\) for effective \(R\).  H1 and H2 exhibit the same
failure pattern at smaller amplitude.  Relative-vorticity evolution remains
visually close to the common vortex skeleton even as the moisture partition
degrades.

## Visual-quality conclusion

All twelve galleries remain readable under one normalization per variable.
Signed symmetric-log normalization is needed only for \(A\) and \(R\); linear
zero-centered ranges are sufficient for the state fields.  No model-specific
autoscaling or negative-value clipping was used.  All twelve GIFs were
inspected through actual decoded frames at the five event times and are
legible with consistent labels and colors.

## Optional truth-minus-model maps for later review

If an error-map follow-up is desired, the most informative limited set is:

1. **Representation B — H5, 12000 s, learned \(R\):** isolates the sign and
   placement difference between the broad learned-rate structure and the
   truth's narrow positive peak-rain bands.
2. **Representation C — H5, 16000 s, \(q_r\):** shows directly where the
   large negative core-centered rain partition replaces the truth's positive
   outer arcs.
3. **Representation C — M1-Y, 12000 s, effective \(R\)** (optional third):
   contrasts the oscillatory projected rate with the truth rate despite the
   comparatively truth-like deployed rain-water morphology.

No error maps were generated in this task.

