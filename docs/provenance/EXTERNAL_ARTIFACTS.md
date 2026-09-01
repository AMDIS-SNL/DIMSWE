# Post-snapshot external artifact contract

These accepted artifacts are intentionally not stored in Git. Restore each file
at the repository-relative path below and verify both byte count and SHA-256
before running a consumer. A missing or mismatched file is an error; no workflow
may silently substitute another cache.

## M1-Y and quantitative postprocessing caches

| Expected repository-relative path | Bytes | SHA-256 | Producer | Consumer | Regenerable |
|---|---:|---|---|---|---|
| `external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz` | 187181823 | `6f16e6db2c6ebdbd8c00a23cdae9b5318355384723a2f1276b2ea93d95145668` | `dimswe.test2b_m1y_campaign prepare` | M1-Y certification/training; W4 fixed-array histories | Yes, from immutable truth with the recorded Firedrake environment; transfer preferred |
| `external-results/m1y-test2b-20260828/evaluation/m1y_heldout_data.npz` | 216940889 | `1ddfa2d2e28b6f8dc2a0fbe0a12d2fe7da42158745a70eb2e088706501c42d2f` | `dimswe.test2b_m1y_evaluation prepare-heldout` | matched M1-X/M1-Y evaluation and W4 Y* metrics | Yes, by analytical prefix replay; transfer preferred |
| `postprocessing/ml_results_20260829/data/heldout_x_test2b.npz` | 332789878 | `fd55559e2eb3277228099106c8043d3c1d11848a83c77efb87a3f4373f03274f` | `extract_heldout_x_test2b.py --scope heldout` | direct histories and objective-consistent evaluation | Yes, fixed extraction; Firedrake required |
| `postprocessing/ml_results_20260829/data/training_x_carriers_test2b.npz` | 29434314 | `0c6bf9378fd38eace300d0b2a6b8d6efdcfc865493a910b12e7b1a4a25a69bf4` | `extract_heldout_x_test2b.py --scope training-carriers` | fixed-array direct histories | Yes, fixed extraction; Firedrake required |

The M1-Y cache sidecars are versioned beside their expected paths. W4 cache
sidecars are versioned in `postprocessing/ml_results_20260829/data/`.

## Deployed-hybrid replay caches

All twelve files are produced by
`postprocessing/deployed_hybrid_dynamics_20260830/scripts/replay_spatial_maps.py`
from frozen checkpoints and immutable truth restart states. They are consumed
by `finalize_replays_and_limits.py` and `render_spatial_package.py`. They are
regenerable evaluation products, but regeneration requires the recorded
Firedrake/JAX environment and is not a training campaign.

| Expected path under `postprocessing/deployed_hybrid_dynamics_20260830/data/` | Bytes | SHA-256 |
|---|---:|---|
| `repA_H1_maps.npz` | 38584513 | `c1baf2f41a65c6a40ddda8e00623f38b49c012c98386c34f482a9f3aa9f59c4f` |
| `repA_H2_maps.npz` | 38609921 | `2df91af27e1fc7da5748b08de27aa6b9adebf076f1fea12695bcdc4236e00713` |
| `repA_H5_maps.npz` | 38604434 | `a60c6cd46d3f20884d4fe8bad116e066c5c379f3ee3313ae2f704a1705ecb837` |
| `repA_M1Y_maps.npz` | 38786074 | `e49787f46136f491604854e192883ad32aadd50d18f171f2e41fdb59ae195b92` |
| `repB_H1_maps.npz` | 50093664 | `ee8c627c9b3cd523ef42501ab02a035372f525fb0b5079f0de88bcae932bbf08` |
| `repB_H2_maps.npz` | 50091601 | `aea531dff007c4ce9546fca7c53c631fbf7409838f9e0868c9838589fd7126f1` |
| `repB_H5_maps.npz` | 50078774 | `ee60e2ad75a6576ab85376301c022f8ad10642a0b46451263ba8c8e8fbf6d34f` |
| `repB_M1Y_maps.npz` | 51132430 | `0f97d3265ab189d9f9f9f58eb5dc04b7f3917ede49602d0abd2f5936b1becda6` |
| `repC_H1_maps.npz` | 56685068 | `46eb8ed760cddffdc83d96b71f93e631d56089310c39cae8c9d65c44421ef427` |
| `repC_H2_maps.npz` | 56432616 | `14c310f4263936b3bd153d3a33828c447836a21e9779bcaa10a47ea5ce9886dd` |
| `repC_H5_maps.npz` | 56229644 | `0b69a4f12c895c77f06c3e7073af20bf0bf4dfb31ac796cf5458ab4a96b9b6ee` |
| `repC_M1Y_maps.npz` | 59926545 | `fb4e5e5a54e52fbec4982cb48f5c32b4422d6070f09c54341d24d7a41e12165f` |

## Deployed-hybrid movies

These GIFs are produced by `render_spatial_package.py --kind movie` from the
corresponding replay cache and common visual limits. They are consumed only for
visual inspection or optional `convert_gifs_to_mp4.sh` conversion. They are
fully regenerable from verified replay caches; JSON sidecars are versioned.

| Expected path under `postprocessing/deployed_hybrid_dynamics_20260830/movies/` | Bytes | SHA-256 |
|---|---:|---|
| `representation_A/movie_repA_H1.gif` | 7744201 | `da74c4332bab8a6ab3ad682bda8361f673d12ffd726c3b79182a04ccf14f58db` |
| `representation_A/movie_repA_H2.gif` | 7744719 | `dbeb4ec378be34ac4b4282a6c189f18f4c713b3cc148c254e75aaa7e65da5b15` |
| `representation_A/movie_repA_H5.gif` | 7744654 | `3bf783fbba5cee0905e7fcf101e2f163aa9c35cd4d8e08e7864618e1f039a444` |
| `representation_A/movie_repA_M1Y.gif` | 7787244 | `debb2dd5d5ec5e4eaf2d578ac14a14674816b1d559563ee90bd0611810a7d3e3` |
| `representation_B/movie_repB_H1.gif` | 8322536 | `540cdb88673883cd48782df61ee2f22aa4f398d8e13513ea2a29f0c69cfe0b75` |
| `representation_B/movie_repB_H2.gif` | 8323490 | `21ac2b898eeb5c47aa35d8c14a4228ce1a2b6e0ed988636bf7d33dcd3fe9c217` |
| `representation_B/movie_repB_H5.gif` | 8319685 | `e0c8e1ecf94d8c7642327096e3c265be127e05e8a45faf23923da3ff68267905` |
| `representation_B/movie_repB_M1Y.gif` | 8820326 | `f2af63fcf2fac851971316d48b5d68ab3a002fb18d782ff209d2b2a67c08d392` |
| `representation_C/movie_repC_H1.gif` | 8033141 | `b3086ec76cd4f4593e4d8141a57bfe0240778edd1b29bf8222b9ce28b38c8cdb` |
| `representation_C/movie_repC_H2.gif` | 8208650 | `ecd866ddd26bae8bb6d091b7c590d28056cfc30a4a8c9773100a52dc7741d2e8` |
| `representation_C/movie_repC_H5.gif` | 8495573 | `5f6c480ed9a65913e6edebc3f75db8764b76a8ec69fdd228261a1106c5a5ed38` |
| `representation_C/movie_repC_M1Y.gif` | 8832379 | `b46cf18daa26034feda26fdfce687217aaf24c3fde8eca39cc4d3e9f32678a8c` |

## Analytical ground-truth map/movie data

The map caches are produced by `extract_truth_maps.py` from accepted truth
runs and consumed by the W6 figure/movie/vortex scripts and the W5 replay
renderer. The GIFs are produced by `render_truth_movies.py`. All are
deterministically regenerable from accepted truth, but transfer is preferred
because map extraction replays the numerical fields.

| Expected repository-relative path | Bytes | SHA-256 |
|---|---:|---|
| `postprocessing/ground_truth_figures_20260829/outputs/ground_truth_figures_20260829/data/test2a_truth_maps.npz` | 2566710 | `0cbc5fac8a15c110701ba63e5051f214330532df7ef3435c784129ef405f36ea` |
| `postprocessing/ground_truth_figures_20260829/outputs/ground_truth_figures_20260829/data/test2b_truth_maps.npz` | 46967973 | `e9477a3ac4e54ebe4da73f9df5ffaafdf98c812afd7fe92b55820230db034de6` |
| `postprocessing/ground_truth_figures_20260829/outputs/ground_truth_figures_20260829/movies/test2a_truth_evolution.gif` | 5763885 | `bbf918632a5684b02ee7db79490f5c6dd43e40570f203c6748a3491a723fc0ab` |
| `postprocessing/ground_truth_figures_20260829/outputs/ground_truth_figures_20260829/movies/test2b_truth_evolution.gif` | 8714413 | `0ff5df5a97d5af5b938e9a8029bb9a40608d09f9bc0a71507ed31c268e61e38a` |

## Verification

On receipt:

```sh
shasum -a 256 /path/to/artifact
stat -f '%z %N' /path/to/artifact   # macOS
```

Compare to this table and to the versioned JSON sidecar. Do not edit a frozen
sidecar's historical absolute path; configure current roots through
`scripts/reproduction_environment.sh` and the optional variables documented
in `REPRODUCING_RESULTS.md`.
