# Test-2B M1-Y campaign specification

Campaign namespace: `m1y_test2b_20260828`.

This campaign adds one direct-regression control to the frozen Test-2B
learned-physics study.  Historical M1-X evaluates both the five production
inputs and analytical targets at the timestep-boundary truth state
`X_n*`.  M1-Y instead evaluates both at

`Y_n* = P(X_n*)`,

where `P` is the complete accepted DIMSWE split-timestep prefix before the
final moist Euler child.  M1-Y is offline: every `Y_n*` is precomputed with
the analytical model, independent of neural parameters; training neither
rolls out the neural model nor differentiates through `P`.

The only scientific change is X-state versus Y-state evaluation.  The frozen
feature order remains `(h, S, Qv, Qc, B)`.  The normalization remains the
historical normalization fitted on X-state support 0..80.  Architectures,
output scales, objective definitions, full-batch carrier-mass weighting,
seed-0 initialization, float64 arithmetic, PyROL line-search L-BFGS settings,
10,000-iteration cap, and checkpoint conventions are matched separately for
Representations A, B, and C.

The training support is exactly the same 81 truth indices, 0..80 inclusive.
The final state is retained because direct M1-Y needs no `X_{n+1}*` target.
Held-out truth indices 81..160 are evaluation-only and cannot affect training
or model selection.

Representation targets use the accepted production definitions:

- A: `A*(Y)`
- B: `(A*(Y), R*(Y))`
- C: `h [beta2 A*(Y), A*(Y), -(A*(Y)+R*(Y)), R*(Y)]`

No F0/F1/F2 feature study, recursive objective, architecture change,
hyperparameter tuning, or analytical-law change is in scope.
