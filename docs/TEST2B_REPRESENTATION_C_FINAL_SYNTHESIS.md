# Test2B Representation C: final scientific synthesis

**Frozen evaluation date:** 2026-08-19

**Status:** final evaluation of all six completed Representation-C fits

**Scope:** four independent learned moist tendencies; evaluation and
postprocessing only, with no optimization, truth generation, source projection,
or conservation repair

The authoritative machine-readable evaluation is
`external-results/test2b-rain-active-learning/production/representation-C/representation_c_final_comparison.json`
(SHA256
`8bc1d9fad90d1d5907c3ff8bc4a5e396ae09ce34f4d887d1f77ad429dfbba926`).
It was generated from immutable final parameter artifacts by
`dimswe.test2b_representation_c_postprocess`.  The file records
`status="complete"`, `evaluation_only=true`, `optimizer_instantiated=false`,
`truth_generated=false`, `projection_applied_to_trajectory=false`, and
`conservation_repair_applied=false`.  Its per-step and per-regime records are
authoritative where this document gives compact summaries.

## 1. Frozen Representation-C contract

The truth, input normalization, state metric, objective schedules, optimizer
conventions, and train/held-out partition are frozen in
`docs/TEST2B_RAIN_ACTIVE_LEARNING_PREPARATION.md`.  The local input is

\[
x=(h,S,Q_v,Q_c,B),
\]

and the float64, seed-0, `5 -> 32 -> 32 -> 4` tanh network has 1,380
parameters.  It directly predicts

\[
s_\theta=(S_t,Q_{v,t},Q_{c,t},Q_{r,t})_\theta.
\]

No analytical `A` or `R`, shared latent rate, positivity/threshold constraint,
water or thermodynamic projection, or correction is present.  The physical
two-rate source manifold is

\[
s(A,R)=hA(\beta_2,1,-1,0)^T+hR(0,0,-1,1)^T.
\]

The network is free to leave this manifold and to create or destroy water.
The frozen physical source scales, in `(S,Qv,Qc,Qr)` order, are
`(6.671477765500949e-3, 6.803353979030477e-5,
6.80335397581467e-5, 1.5076498196845062e-8)`; they condition the loss and
diagnostics but impose no source relation.
Training/model-selection support is truth states `0..80`; states `81..160`
are the frozen held-out mature-rain interval.  Truth first has meaningful rain
at step 51 (`t=5100`), has peak specific `Qc=1.0517909572531444e-4` at
step 89, and final rain mass `2.30733006980403e8`.

### 1.1 Frozen objectives

For source scales
`D_C=diag(sigma_S,sigma_Q,sigma_Q,sigma_Qr)` and the accepted four-field mass
metric, direct supervision is

\[
J_{M1}^{C}(\theta)=
\frac{\sum_{k=0}^{80}\|D_C^{-1}[s_\theta(X_k^*)-s^*(X_k^*)]\|_{M_4}^2}
     {\sum_{k=0}^{80}\|D_C^{-1}s^*(X_k^*)\|_{M_4}^2}.
\]

The fixed boundary-state deployed-map objective is

\[
J_{M2-X}(\theta)=
\frac{\sum_{k=0}^{80}\|G_4(X_k^*)[s_\theta(X_k^*)-s^*(X_k^*)]\|_M^2}
     {D_X},
\quad D_X=9.014768540958347\times10^{10}.
\]

With `P` denoting children 1--5 of the accepted six-child step and
`Y_k=P(X_k^*)`, the fixed/cacheable one-step objective is

\[
J_{H1}(\theta)=
\frac{\sum_{k=0}^{79}\|F_\theta(X_k^*)-X_{k+1}^*\|_M^2}{D_Y},
\quad D_Y=9.01944200525722\times10^{14}.
\]

For `H=2,5`, nonoverlapping windows start at `0,H,...,80-H` and

\[
J_H(\theta)=D_Y^{-1}\sum_k\sum_{j=1}^{H}
\|\widehat X_{k+j}-X_{k+j}^*\|_M^2,
\qquad \widehat X_{n+1}=F_\theta(\widehat X_n).
\]

Every target `1..80` appears once per dense horizon objective.  H1 is
truth-reset and offline/cacheable; H2 is the first recursive objective.  The
deployed `M2-X/H1/H2/H5` state losses and denominators are common to
Representations A, B, and C.  Their `M1` objectives are not cross-representation
comparable because their learned quantities and scaling differ.  Within C,
models are compared down an objective column rather than across columns.

## 2. Verified final artifacts and budgets

For every fit, `fit_result.json` and `fit_progress.json` agree exactly, report
`status=complete`, and agree with the final sidecar on representation, stage,
accepted iteration, and pytree fingerprint.  The fingerprint-validating loader
accepted every immutable `final_parameters.npz`.  Completion provenance does
not use the truncated campaign log from the refused accidental relaunch.

| model | accepted | value / gradient evals | wall (s) | termination | pytree SHA256 | NPZ SHA256 |
|---|---:|---:|---:|---|---|---|
| M1 | 10,000 | 20,746 / 10,001 | 18,969.169 | `MAXITER` | `8ad9c8017c9304827d8e9e73392c3a9503f7de05a38b86baf52f4351c35615e4` | `e13a7bd42f3f773d068a6cc0c1344a3a8db0caef74a269ce7980b0b50f91e3b0` |
| M2-X-independent | 10,000 | 20,935 / 10,001 | 30,660.714 | `MAXITER` | `4a51c90c2b1916da51affcec49f89461c391c7862300cb72a192ccf843244fc3` | `acf320fc0b546b906f21c6273e42e928326b30c4eddcbf7e0480f9348183a3cf` |
| M1-to-M2-X | 5,000 | 10,461 / 5,001 | 16,243.466 | `MAXITER` | `5b698bb87885a7faa9cc888de15ff75bac2661796c7bb9a31ddd788cd7270c11` | `cc4bb56d2959f8120191b1919354818322875780dc6f698cd24cb44832dae566` |
| H1 | 5,000 | 10,445 / 5,001 | 13,137.038 | `MAXITER` | `c44aa9e8811aacb423570a3bcb2f819462d882931b929a7bee442bb8b9a1d1c3` | `be3fac690063191ee78df7a8515b580d17842a34083835c73232ab1bb3995c6b` |
| H2 | 20 | 44 / 21 | 1,633.591 | `MAXITER` | `3ae5e9d6729977310e7e9d740490c223539aa514cf0e24e379adab25d7ca6d70` | `cce0fff0c49ad9577a5cfe6b1e8176eb45af8e1d1dbac6e98029d75e3dce1657` |
| H5 | 20 | 45 / 21 | 2,373.535 | `MAXITER` | `a9a95ed952ba410ce744c6fe1473c0c842193560d14b432ce4f8543c6ff40f70` | `ecb28ef27fbb4663970da2e4f72f9adbcf0c67ab966e80e65bb21f17ada63985` |

The total is 30,040 accepted iterations, 62,676 objective evaluations,
30,046 gradient evaluations, and `83017.51381379226 s` (`23.0604 h`).  All
fits are budget-limited; none is a demonstrated stationary optimum.  In
particular, H1 received 5,000 accepted steps whereas H2 and H5 received 20
each.

## 3. Exact common objective matrix

| network | `J_M1` | `J_M2-X` | `J_H1` | `J_H2` | `J_H5` |
|---|---:|---:|---:|---:|---:|
| M1 | **`3.8697519765561936e-5`** | `2.9105553417151760e-5` | `2.7297692708884850e-4` | `5.8734964336889430e-4` | `1.8760941709077540e-3` |
| M2-X-independent | `5.7563999043748800e-1` | `1.5389000360457850e-5` | `1.4091298515868874e-4` | `9.0675985465772190e-4` | `1.3137991667949330e-2` |
| M1-to-M2-X | `3.9879694705791080e-3` | **`1.4039537322282988e-5`** | `1.7998973063849757e-4` | `5.0130156769228440e-4` | `2.4027172684166994e-3` |
| H1 | `7.4372905560185325e-3` | `7.6904864814983470e-5` | **`3.4267131332575840e-5`** | `2.3343392561144002e-4` | `2.5645484461790870e-3` |
| H2 | `7.5273495849415680e-3` | `8.8855979764991000e-5` | `7.7326662637472000e-5` | **`1.4003948998326003e-4`** | `1.5276449175796880e-3` |
| H5 | `8.5366656460418720e-3` | `4.4673142300737970e-4` | `5.3708188894743480e-4` | `5.2141432437073300e-4` | **`7.0341324501142230e-4`** |

The best attained value in each fixed column is bold.  Key within-column
changes are:

| transition | `J_M1` | `J_M2-X` | `J_H1` | `J_H2` | `J_H5` |
|---|---:|---:|---:|---:|---:|
| M1 -> warm M2-X | +10,205.5% | -51.763% | -34.064% | -14.650% | +28.070% |
| warm M2-X -> H1 | +86.493% | +447.773% | -80.962% | -53.434% | +6.735% |
| M1 -> H1 | +19,119% | +164.227% | -87.447% | -60.256% | +36.696% |
| H1 -> H2 | +1.211% | +15.540% | +125.658% | -40.009% | -40.432% |
| H2 -> H5 | +13.409% | +402.759% | +594.562% | +272.334% | -53.954% |

The independent M2-X solution is 9.615% above the warm solution under the
same objective; equivalently, warm M2-X is 8.769% lower despite half the
accepted-iteration budget.  The independent solution is lower under H1 than
warm M2-X, but catastrophically worse under M1, H2, H5, and autonomous
deployment.  This is strong optimizer-basin sensitivity, not a property that
can be assigned to the M2-X objective alone.

## 4. Direct four-tendency accuracy and physical projection

The table reports fixed-truth normalized RMS error using the frozen component
scales.  It is compact; physical RMS, relative RMS, maximum error, signed bias,
correlation, and every regime are retained in the JSON.

| model | train `S/Qv/Qc/Qr` nRMSE | heldout `S/Qv/Qc/Qr` nRMSE | train / heldout off-manifold fraction | train projected `A/R` nRMSE | heldout projected `A/R` nRMSE |
|---|---|---|---:|---|---|
| M1 | `.00590/.00587/.00589/.00354` | `.03097/.03153/.03583/.01756` | `.000748 / .02032` | `.00640/.00357` | `.03442/.01789` |
| M2-X-independent | `.00448/.04315/.06221/1.31357` | `.05622/.11460/.16762/1.38597` | `.02398 / .12332` | `.03374/1.34260` | `.07982/1.42797` |
| M1-to-M2-X | `.00437/.02221/.01888/.10547` | `.04475/.06501/.07311/.28356` | `.00985 / .21784` | `.01447/.10581` | `.06247/.28167` |
| H1 | `.00902/.02870/.02818/.14376` | `.06129/.07587/.07893/.24776` | `.01494 / .16789` | `.01946/.14523` | `.07336/.24743` |
| H2 | `.00966/.03185/.02977/.14368` | `.05922/.08017/.07899/.24720` | `.01499 / .17907` | `.02214/.14515` | `.07406/.24688` |
| H5 | `.02124/.04167/.05199/.14416` | `.06227/.08893/.08759/.24642` | `.02107 / .20278` | `.03607/.14564` | `.08076/.24612` |

The projection is a mass-weighted least-squares projection in normalized
source coordinates.  Its basis and metric fingerprint
`ca9589c0fefd88a803be4ad3bdafa9053f730f19515a33afcd58b468598a38b3`
are stored in the result.  The output also reports the physical-unit,
mass-weighted residual without component scaling.  Projection is diagnostic
only and was never fed back into a rollout.

M1 is by far the best constitutive/source model.  On training truth states it
keeps only 0.0748% of predicted normalized source magnitude off the physical
manifold and fits all four components at about `0.35--0.59%` nRMSE.  Direct
supervision therefore acts as a strong empirical manifold regularizer even
though no manifold is imposed.  That discovery is incomplete out of sample:
the held-out off-manifold fraction rises to 2.03%, and the moisture-component
relative errors are large because the mature-rain truth tendency is small.

The fixed-X deployed objective is non-identifying.  Independent M2-X fits `S`
well but essentially reverses/misses the sparse rain source (`Qr` nRMSE
`1.31`, active false-negative rate one).  Warm-starting retains substantially
more structure, yet is much less physical than M1.  H1/H2/H5 do not recover a
unique physical pair: their componentwise rates disagree, and their projected
`R` errors remain roughly `0.145` on training support and `0.246--0.247`
held out.

## 5. Autonomous source structure and state evolution

All rollouts start from the same `X_0^*`, take 160 complete recursive steps,
and use only the neural four-vector at the current model state.  There is no
truth injection, reset, clipping, rain correction, or projection.

| model | mixed final / max / accumulated | heldout accumulated | off-manifold fraction | water-defect RMS / signed:abs | thermo-defect RMS / signed:abs |
|---|---|---:|---:|---|---|
| M1 | `1.579e-5 / 1.879e-5 / 1.284e-5` | `1.282e-5` | `.001930` | `1.875e-7 / +.1580` | `1.640e-5 / +.0799` |
| M2-X-independent | `5.644e-4 / 5.644e-4 / 3.334e-4` | `4.359e-4` | `.05418` | `2.478e-6 / +.3625` | `8.355e-4 / -.3372` |
| M1-to-M2-X | `1.239e-4 / 1.239e-4 / 7.377e-5` | `9.407e-5` | `.01514` | `5.728e-7 / -.4638` | `1.643e-4 / +.2010` |
| H1 | `1.815e-4 / 1.815e-4 / 9.750e-5` | `1.275e-4` | `.02262` | `8.940e-7 / -.6837` | `2.414e-4 / +.1624` |
| H2 | `3.959e-4 / 3.959e-4 / 1.866e-4` | `2.578e-4` | `.03261` | `1.323e-6 / -.9510` | `3.581e-4 / +.9254` |
| H5 | `1.389e-3 / 1.389e-3 / 7.182e-4` | `9.844e-4` | `.08864` | `3.277e-6 / -.9981` | `9.991e-4 / +.9894` |

M1 is the best C model by every full-trajectory state summary.  H1 lowers its
targeted one-step loss by 87.447% from M1 but increases autonomous accumulated
error by 659.4%.  Even relative to warm M2-X, H1 lowers `J_H1` by 80.962%
while autonomous accumulated error rises 32.17%.  The one-step state objective
therefore found compensating tendencies that do not remain valid recursively.

Under the tested 20-step continuations, H2 lowers `J_H2` by 40.009% from H1
but autonomous accumulated error rises 91.42%.  H5 lowers its targeted `J_H5`
by 53.954% from H2 while accumulated error rises another 284.82%.  Recursive
information is demonstrably different, but the realized continuations use it
to specialize the short-window objective rather than to improve 160-step
deployment.

## 6. Rain source, phase partition, and boundedness

`R_Qr=Q_{r,t}/h` is only an **effective Qr-based rain-rate diagnostic**; C has
no unique learned `R`.  Every C model emits negative rain tendency on part of
its deployed support.  M1 has a positive source at `t=0` and positive `Qr`
mass at `t=100`; it nevertheless gives the only qualitatively credible rain
partition, ending at `1.8948497697e8`, 17.875% below truth.  Its active-rate
nRMSE/correlation on its own active model states are about `.0358/.9979`.

Independent M2-X never produces a positive rain source and ends at
`Qr=-7.4093e9`.  Warm M2-X, H1, H2, and H5 can emit tiny positive source at
`t=0`, but their accumulated rain mass never becomes positive.  Their final
`Qr` masses are respectively `-4.3381e8`, `-8.1244e8`, `-7.8961e8`, and
`-5.4326e8`.  H1/H2/H5 keep model `Qc` below the analytical precipitation
threshold (maxima `9.534e-5`, `9.267e-5`, `8.429e-5`), so no physically
meaningful positive rain phase emerges.  Source onset and state rain-mass
onset are therefore explicitly different diagnostics.

All state coefficients remain finite.  `Qv` stays positive, but every model
has some negative `Qc` and `Qr` coefficients; the most negative `Qr` ranges
from `-2.94e-6` for M1 to roughly `-5.4e-5` for H5.  No clipping was used.
Flow diagnostics remain finite, but the independent, H1, and H2 models have
substantially larger KE/enstrophy mismatch than M1.  H5's final flow mismatch
is smaller than its peak mismatch and does not rescue its moisture/state
failure.

## 7. Signed total-water audit

The discrete source-map relation is numerically closed: for every model, the
difference between final state mass drift and the accumulated discrete source
defect is below `0.6` in an initial total-water mass of
`3.698687519349267e13`.  Host numerical error is therefore negligible here;
the drifts below are learned-source effects.

| model | final relative signed water drift | interpretation | cumulative local signed:abs |
|---|---:|---|---:|
| M1 | `+1.2840663e-4` (`+0.01284%`) | small net creation | `+.1580` |
| M2-X-independent | `+5.7679684e-3` (`+0.57680%`) | net creation | `+.3625` |
| M1-to-M2-X | `-1.4955744e-3` (`-0.14956%`) | net destruction | `-.4638` |
| H1 | `-3.5625792e-3` (`-0.35626%`) | net destruction | `-.6837` |
| H2 | `-1.1316100e-2` (`-1.13161%`) | strongly systematic destruction | `-.9510` |
| H5 | `-3.3849944e-2` (`-3.38499%`) | almost purely systematic destruction | `-.9981` |

Thus the large H2/H5 local defects do **not** cancel.  By H5, 99.806% of the
time-integrated absolute local water defect survives with the same negative
sign.  Thermodynamic defects likewise become coherent: the autonomous
signed/absolute ratio grows from `+.162` for H1 to `+.925` for H2 and `+.989`
for H5.  The longer-horizon optimization has selected systematic compensating
directions outside the physical source manifold.

## 8. Frozen Representation-C conclusions

Certified by this experiment:

1. Direct four-tendency M1 supervision can approximately discover the
   two-rate source structure on its training distribution without having it
   imposed.  It is the best C model physically and autonomously.
2. The fixed-X objective is strongly basin-sensitive.  M1 warm-starting
   improves its own objective but does not preserve M1-level constitutive
   fidelity.
3. H1 gives a very large in-objective one-step reduction without improving
   deployment.  It increases off-manifold, water, thermodynamic, and rain
   partition errors.
4. H2 and H5 contain genuine recursive information, but under the tested
   20-accepted-step continuations they further specialize their trained
   horizons while markedly worsening full deployment.
5. C's failure is not merely spurious redistribution into rain.  It includes
   signed creation/destruction of total water, culminating in 3.385% net
   destruction for H5.
6. Conservation and thermodynamic coupling are representation design choices,
   not properties reliably recovered by a short-horizon state objective in
   this experiment.

Limitations: all fits ended at `MAXITER`; H2/H5 budgets are tiny relative to
M1/M2-X/H1; this is one trajectory, resolution, architecture, and seed; the
held-out interval is temporal extrapolation from the same physical case; and
the projection diagnoses distance from the physical manifold but does not
identify a unique physical rate off it.  These qualifications prohibit claims
about mathematical optima or all black-box architectures.  They do not alter
the observed failure of the completed artifacts.

**Frozen status:** `TEST2B_REPRESENTATION_C_FROZEN`.
