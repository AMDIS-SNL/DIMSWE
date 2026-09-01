# Machine-Learning Results caption drafts

These captions use training/evaluation terminology while retaining the methodological qualifications needed to interpret each quantity.

## Tables

### Table 1 — Data and evaluation protocol

Data and evaluation protocol for Test 2A and Test 2B. Test 2A states 0--80 are training states; no evaluation or test set was defined, and states 81--160 were unused by the recorded learning studies. For Test 2B, states 0--80 are training states and states 81--160 are temporally adjacent evaluation states. The evaluation states did not influence stopping or model selection.

### Table 2 — Main Test 2B training-run contracts

Frozen contracts for M1-Y, H1/M2-Y, H2, and H5 in Representations A, B, and C. All networks use the production feature order (h,S,Qv,Qc,B), float64, and seed 0. H1, H2, and H5 are sequential continuations with unequal budgets, so objective and optimization history change together.

### Table 3 — Final local moist-physics accuracy at Y*

Final frozen-network errors at truth-derived pre-moist states Y*=P(X*) for Test 2B training states 0--80 and evaluation states 81--160. The evaluation states did not influence optimization. A, R, and source-component quantities remain separate because the representations learn different targets.

### Table 5 — Main rain-event and water-partition diagnostics

Test 2B rain-event and water-partition diagnostics for truth and the deployed M1-Y, H1, H2, and H5 models. Representation A uses analytical R on the model-generated state, Representation B learns R directly, and Representation C is reported through its effective Qr-source rain diagnostic. The A/B source identities are imposed by construction; their residuals are not evidence that conservation was learned.

### Supplementary table — Local moist-physics accuracy at deployed Yhat

Stored local-law errors on model-generated pre-moist states Yhat=P(Xhat) for M1-Y, H1, H2, and H5. Coverage is uniform across Representations A, B, and C, but no single cross-representation scalar is formed because the learned targets differ.

### Supplementary tables — Complete campaign results

The complete training contracts, X-based direct errors, cross-objective matrix, rain diagnostics, and Test 2A training-state results are retained unchanged in the supplementary table directory.

## Main figures

### ML-1 — Optimization of M1-Y, H1/M2-Y, H2, and H5

Optimization of the Test 2B training objectives for the four main-paper methods. M1-Y and H1/M2-Y are shown at saved iterations. H2 and H5 are shown only at their initial and final stage endpoints; intermediate recursive objective values were not reconstructed. Objective normalizations differ across representations.

### ML-2 — Training and evaluation of the fitted objectives

Training and evaluation histories of the fitted nonrecursive objectives. Solid curves show training states or windows; dashed curves with open markers show the later evaluation states or windows. M1-Y is evaluated on truth-derived pre-moist states Y*=P(X*). H1/M2-Y uses fixed one-step pairs from Y* to the corresponding next truth state. Evaluation values were calculated from saved networks and did not influence training.

### ML-3 — Local moist-physics accuracy at Y*

Final-model direct physical-law errors at truth-derived pre-moist states Y*=P(X*). Filled points show training states 0--80 and open points show evaluation states 81--160. Representation A reports A; Representation B reports A and both all-sample and truth-active R errors; Representation C reports the normalized source-vector error. Detailed C source-component and B activation metrics are retained in the supplementary data and main accuracy table.

### ML-4 — Deployed physical diagnostics

Physical diagnostics after deploying the four main-paper models in Test 2B. Points denote distinct frozen models and are not connected. Representations A and B impose the water and thermodynamic source identities algebraically; Representation C must approximate them. Moisture minima are finite-element coefficient minima rather than a pointwise positivity proof.

### ML-5A — Global trajectories, Representation A

Global Test 2B trajectories for Representation A and the four main-paper models. Integrated cloud and rain water use m^3 under the unit-reference-density convention. Kinetic energy is not total energy, and projected relative-vorticity squared is not potential enstrophy. Vertical guides mark first truth rain at 5100 s and peak integrated truth rain production at 12000 s.

### ML-5B — Global trajectories, Representation B

Global Test 2B trajectories for Representation B and the four main-paper models. Integrated cloud and rain water use m^3 under the unit-reference-density convention. Kinetic energy is not total energy, and projected relative-vorticity squared is not potential enstrophy. Vertical guides mark first truth rain at 5100 s and peak integrated truth rain production at 12000 s.

### ML-5C — Global trajectories, Representation C

Global Test 2B trajectories for Representation C and the four main-paper models. Integrated cloud and rain water use m^3 under the unit-reference-density convention. Kinetic energy is not total energy, and projected relative-vorticity squared is not potential enstrophy. Vertical guides mark first truth rain at 5100 s and peak integrated truth rain production at 12000 s.

## Supplement figures

### Supplementary — Complete Test 2B optimization histories

Optimization of the Test 2B training objectives. Markers show the iterations at which objective values are available. Direct and one-step histories use the iteration axis shown. H2 and H5 are displayed as paired initial and final values; their final values correspond to the 20-iteration optimization stages, and intermediate recursive values were not reconstructed. Objective normalizations differ across representations.

### Supplementary ML-1 — Test 2A training objectives

Test 2A training-objective histories available for the Representation A and C model ladders. All curves use training states 0--80; no Test 2A evaluation or test set is introduced.

### Supplementary ML-2 — Common-X direct physical-law histories

Common-X direct physical-law histories retained as an audit diagnostic for selected Test 2B models. Solid curves use states 0--80 and dashed curves with open markers use states 81--160; the later-state values were calculated from saved networks and did not influence optimization. Every curve evaluates the local physical law at timestep-boundary states X*, irrespective of the objective used to train the model. The objective-consistent training/evaluation comparison is the main ML-2 figure.

### Supplementary — Complete nonrecursive objective histories

Objective-consistent histories for the nonrecursive Test 2B objectives. Solid curves use the training portion and dashed curves with open markers use the later evaluation portion; the evaluation curves were calculated from saved networks after optimization and did not influence training. M1-X and both M2-X runs are evaluated on timestep-boundary states X*, whereas M1-Y is evaluated on Y*=P(X*) and H1 uses fixed one-step pairs from Y* to the corresponding next truth state. Each evaluation curve uses the same normalization, output scaling, weighting, and normalized-target-energy formula as its fitted objective. Objective normalizations differ across representations. H2 and H5 are omitted because their recursive evaluation histories were not computed.

### Supplementary ML-2 — Direct error for all Test 2B models

Direct physical-law error during optimization for all seven Test 2B models. Solid curves show training states 0--80 and dashed curves show evaluation states 81--160. Evaluation errors were computed from saved networks, did not influence optimization, and are evaluated on timestep-boundary states X*.

### Supplementary ML-3 — M1-X/M1-Y error histories

M1-X/M1-Y direct-error histories evaluated separately at X and Y. Training uses states 0--80 and evaluation uses states 81--160; evaluation states did not influence optimization. Each marker corresponds to an available saved network.

### Supplementary — Final M1-X/M1-Y cross-state comparison

Matched Test 2B comparison of the final M1-X and M1-Y models. X denotes a timestep-boundary state and Y=P(X) the state immediately before moist physics. Training uses states 0--80 and evaluation uses states 81--160; the evaluation states did not influence optimization. The two models otherwise share the architecture, initialization, normalization, weighting, optimizer, and budget.

### Supplementary ML-3 — Representation C source-component accuracy at Y*

Representation C source-component errors at truth-derived pre-moist states Y*. Filled points show training states 0--80 and open points show evaluation states 81--160. All values use the frozen carrier weighting and output scales.

### Supplementary ML-4 — Test 2A objective matrix

Test 2A objective matrix on training states. A red outline and asterisk mark the objective minimized for each row; other values evaluate that fixed model under another objective. M1-Y was not part of Test 2A.

### Supplementary — Test 2B objective matrix

Test 2B objective matrix for the final models. The red outline and asterisk mark the objective minimized for each row; all other populated cells evaluate that fixed model under a different objective. Each representation has an independent color scale, so colors should not be compared between panels.

### Supplementary — Complete deployed physical diagnostics

Physical diagnostics after deploying the 21 Test 2B models. Points identify separate trained models and are not connected as a continuous path. Representations A and B impose the water and thermodynamic source identities by construction; their roundoff-scale water drift is therefore not evidence that conservation was learned. Representation C does not impose those identities. Moisture minima are finite-element coefficient minima rather than a pointwise positivity proof.

### Supplementary ML-5 — Representation C source identities

Representation C source-identity defects after Test 2B deployment. The water and thermodynamic source identities are imposed by construction in Representations A and B and are therefore not shown as learned quantities.

### Supplementary — Prior Representation A trajectory subset

Global Test 2B trajectories for Representation A. Integrated cloud and rain water are shown in m^3 under the unit-reference-density convention. Kinetic energy is 0.5 integral h|v|^2 dA and is not total energy; projected relative-vorticity squared is 0.5 integral zeta_h^2 dA after CG(3) projection and is not potential enstrophy. Vertical guides mark first truth rain at 5100 s and peak integrated truth rain production at 12000 s. The last panel shows final and maximum state errors because a state-error time series is not available in the accepted records.

### Supplementary ML-6A — All Representation A trajectories

Global Test 2B trajectories for all seven Representation A models. Integrated cloud and rain water use m^3 under unit reference density. Kinetic energy is not total energy, and projected relative-vorticity squared is not potential enstrophy.

### Supplementary — Prior Representation B trajectory subset

Global Test 2B trajectories for Representation B. Integrated cloud and rain water are shown in m^3 under the unit-reference-density convention. Kinetic energy is 0.5 integral h|v|^2 dA and is not total energy; projected relative-vorticity squared is 0.5 integral zeta_h^2 dA after CG(3) projection and is not potential enstrophy. Vertical guides mark first truth rain at 5100 s and peak integrated truth rain production at 12000 s. The last panel shows final and maximum state errors because a state-error time series is not available in the accepted records.

### Supplementary ML-6B — All Representation B trajectories

Global Test 2B trajectories for all seven Representation B models. Integrated cloud and rain water use m^3 under unit reference density. Kinetic energy is not total energy, and projected relative-vorticity squared is not potential enstrophy.

### Supplementary — Prior Representation C trajectory subset

Global Test 2B trajectories for Representation C. Integrated cloud and rain water are shown in m^3 under the unit-reference-density convention. Kinetic energy is 0.5 integral h|v|^2 dA and is not total energy; projected relative-vorticity squared is 0.5 integral zeta_h^2 dA after CG(3) projection and is not potential enstrophy. Vertical guides mark first truth rain at 5100 s and peak integrated truth rain production at 12000 s. The last panel shows final and maximum state errors because a state-error time series is not available in the accepted records.

### Supplementary ML-6C — All Representation C trajectories

Global Test 2B trajectories for all seven Representation C models. Integrated cloud and rain water use m^3 under unit reference density. Kinetic energy is not total energy, and projected relative-vorticity squared is not potential enstrophy.

### Supplementary Test 2A deployment diagnostics

Test 2A deployed diagnostics over training states 0--80. The figure uses the available state-error endpoints, kinetic energy, and projected relative-vorticity-squared diagnostics for the A/C model ladders. States 81--160 are not evaluated.

