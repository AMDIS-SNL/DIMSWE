# Ground-truth figure package

This namespace was reconstructed fresh from the authoritative read-only DIMSWE repository. It contains no machine-learning training or learned-model trajectory analysis.

## Contents

- [Exact DoubleVortex specification](DOUBLEVORTEX_CASE_SPEC.md): exact equations and source-line provenance.
- [Physical analysis](PHYSICAL_ANALYSIS.md): verified Test 2A/Test 2B physics, conservation, and Test 2B chronology.
- [Figure plan](FIGURE_PLAN.md): compact plan checked before bulk rendering.
- [Caption drafts](FIGURE_CAPTIONS.md): report-ready captions.
- [Vortex tracker audit](VORTEX_CORE_TRACKING_AUDIT.md): cached-map proof that the raw jumps are adjacent-GLL argmax switches.
- The scripts namespace contains deterministic extraction, plotting, diagnostic-export, and movie programs.
- The output data namespace contains compact CSV time series, audits, metadata,
  and vortex-track data. The two full map NPZ files are external hashed
  artifacts.
- The output figures namespace contains 300 dpi PNG/PDF figures and JSON sidecars.
- Superseded standalone chronology/comparison files remain recoverable in the
  archaeological workspace and are not part of this collaborator package.
- Figure 5 and its CSV/NPZ track data are derived only from the cached Test 2B relative-vorticity maps.
- Movie metadata is retained; rendered frames and GIFs are external artifacts.

## Reproduction order

1. Run **extract_truth_maps.py** inside the existing authoritative Firedrake environment for each truth path. This is the only expensive scientific replay (approximately 5 minutes for Test 2B on the machine used here).
2. Run **export_temporal_diagnostics.py**.
3. Run **make_ground_truth_figures.py** with Matplotlib's Agg backend.
4. Run **render_truth_movies.py**; Pillow writes the GIFs without external video software.
5. Run **track_vortex_cores.py** on the Test 2B NPZ cache; it performs no Firedrake work or truth extraction.

The precise input paths, hashes, state indices, transformations, units, limits,
and producing-script hashes are recorded in the output JSON sidecars and in
`../../../../docs/provenance/EXTERNAL_ARTIFACTS.md`. ffmpeg was not installed and
was not added. Each movie sidecar includes an optional exact MP4 conversion
command for a later environment that already supplies it.

The raw truth data and authoritative source repository were never written. Display clipping applies only to small negative condensate undershoots; all raw derived maps remain in the NPZ caches.
