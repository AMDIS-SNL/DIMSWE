# Objective-consistent ML-2 replacement

## Outcome

The new main figure contains objective-consistent training/evaluation histories
for M1-X, M1-Y, independent M2-X, warm M2-X, and H1 in Representations A, B,
and C. H2/H5 are deliberately excluded because their evaluation histories are
recursive. The former common-X direct-law figure is retained in the supplement.

## State and window contract

| Objective | Training | Evaluation |
|---|---|---|
| M1-X | X* states 0--80 | X* states 81--160 |
| M1-Y | Y*=P(X*) states 0--80 | Y*=P(X*) states 81--160 |
| independent/warm M2-X | X* states 0--80 | X* states 81--160 |
| H1/M2-Y | Y* starts 0--79, next truth states 1--80 | Y* starts 81--159, next truth states 82--160 |

The H1 evaluation intentionally excludes the cross-boundary 80-to-81 window and
the final Y*_160 state, which has no X*_161 target in the frozen trajectory.

## Denominators and parity

M1 evaluation denominators already existed in the accepted direct/cross-state
artifacts. Support-specific evaluation analogues of the exact target-energy
formula were newly defined for M2-X and H1:

- M2-X: training 90147685409.583466; evaluation 14256196.634496419.
- H1 source-equivalent: training 90194420052.572205; evaluation 77838928.303702623.

The H1 source form is algebraically identical to the one-step state-increment
form because the common dt^2 factor cancels in numerator and denominator. All
15 final training objectives match the accepted values; the largest absolute
difference is 1.423e-19. All six M1 final evaluation
values with prior stored counterparts match exactly. No prior objective-consistent
M2-X/H1 evaluation endpoints existed, so their validation rests on checkpoint
hashes, the production fixed-map implementation, frozen normalizations/maps,
and exact state/window contracts rather than a historical-value parity check.

## Figure disposition

- Main: `figures/main/ML2_objective_training_evaluation_history_test2b.pdf` and `.png`.
- Supplement: `figures/supplement/ML2_common_x_direct_history_test2b.pdf` and `.png`.
- The supplementary CSV/PDF/PNG are byte-identical to the former main bundle.
- Visual inspection found no remaining confusing or occluded panel.

No training, optimizer, truth regeneration, timestep/prefix integration,
autonomous rollout, deployed simulation, or H2/H5 recursive history was run.
