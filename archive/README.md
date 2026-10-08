# Archaeological material

Track 1's first pass preserved every scientifically relevant file at its
original path. The second-pass collaborator hygiene audit moved a small,
closed set of superseded development surfaces here so they no longer look like
production entry points:

- `development-history/firedrake_hvp_prototype/`: isolated precursor replaced
  by production split/HVP implementations and their regression suites;
- `development-history/test2a_residual_structure/`: one-off diagnostic whose
  frozen conclusion is already carried in the Test 2A record;
- `development-history/test2b_constrained_rain_variants/`: BTP/BTPL and B+
  launch sketches without a canonical completed campaign;
- `development-history/test2b_representation_a_interim/`: report superseded by
  the final Representation A synthesis; and
- `track1-forensic-tools/`: one-time preservation/disposition tooling.

These files are retained byte-for-byte except where an archive README says
otherwise. They are not imported, collected as tests, or advertised as normal
reproduction commands. Their original paths and hashes remain in
`docs/provenance/FROZEN_DIRTY_STATE_MANIFEST.tsv`; the permanent authoritative
checkout also retains the original layout.

Large, preparatory, superseded, and provenance-uncertain numerical outputs
remain at their original paths under `external-results/` and are classified in
`docs/provenance/DISPOSITION_MANIFEST.tsv`. No archive move is ever applied to
the authoritative archaeological checkout.
