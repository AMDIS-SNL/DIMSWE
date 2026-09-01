# Publication cleanup report

Status: **PASSED — VISUAL/EDITORIAL ONLY**

Eight main figures were regenerated from their existing plotted CSV files.
All scientific data/table/plot artifacts in the 63-file cleanup freeze remain
byte-for-byte identical. Figure JSON sidecars retain the same scientific
payload, model lists, state intervals, source paths, plotted-CSV hashes, and
row counts; only captions, terminology, units, editorial notes, and rendered
PDF/PNG hashes changed.

## Exact visual changes

- **ML-1:** iteration 0 is at the left edge using log10(iteration+1); axes say
  `iteration`; shorter objective-family headings; H2/H5 use categorical
  `initial`/`final` positions; legends appear once per column.
- **ML-2:** three-row spanning layout; short A/R/source-vector titles; all
  panels say `relative RMS error` and `iteration`; legends say `training error`
  and `evaluation error`; the selected-model labels are simplified.
- **ML-3:** short panel titles; X/Y categories are grouped under `Training` and
  `Evaluation`; legend entries are `M1-X model` and `M1-Y model`; axes say
  `relative RMS error`.
- **ML-4:** titles are `Rep. A/B/C`; model rows use the requested independent/
  warm labels; colorbars say `objective`; fitted cells retain red outlines and
  asterisks; representation-specific scales are unchanged.
- **ML-5:** connecting lines were removed; A/B/C are offset grouped points;
  titles and model labels were shortened; signed panels have prominent zero
  lines; the 2x3 layout remains legible and did not require splitting.
- **ML-6 A/B/C:** cloud/rain axes now use m^3 with the same 10^12/10^8 scaling
  as the ground-truth chronology; kinetic energy and projected relative-
  vorticity squared are named precisely; C rain water has a clear zero line;
  A/B water drift uses a linear roundoff-scale axis; C water drift uses a
  readable signed symlog axis; model colors are identical across A/B/C.

## Visual inspection

All eight 300-dpi main figures were opened and inspected at approximate report
width. No panel remains confusing. The first Representation C trajectory
render compressed its signed total-water-drift panel; widening the symlog
linear region and fixing the page margins resolved that problem without
changing any values.

## Safety

No metric was recomputed, no checkpoint or truth array was evaluated, and no
model/history/rollout was run. The authoritative repository and M1-Y workspace
retain their frozen branch, HEAD, status, and diff fingerprints.
