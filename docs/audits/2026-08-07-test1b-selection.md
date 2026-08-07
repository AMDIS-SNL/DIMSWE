# Test 1B scientific selection audit

External Test-1B-0 evidence selected `doublevortex`, 16-by-16, dt=100,
160 steps through t=16000, c0 truth 0.14, initial c0 0.07, s=3.2, UFL moist,
and seed zero.  Canonical truth must use output stride one and is not reused
from the output-stride-8 pilot.

The accepted pilot evidence and rejected 32-by-32 dt=400 instability are
recorded verbatim in `dimswe/configs/test1b_selected_plan.json`.  The rejected
run had exact Euler sigma `2.5734163936944783>2` and remains diagnostic-only.
No 64-by-64 run is requested.

Training states are 0 through 80 and training transition starts are 0 through
79.  Held-out target states are 81 through 160; transition starts are 80
through 159, with state 80 serving only as the trusted held-out initializer
after fitting.  Scan and fit CLI paths now load only states 0 through 80.

The corrected canonical solver comparison matches target coverage.  With
observation stride one, reset uses 16 non-overlapping five-step windows with
starts `0,5,...,75`; accumulated targets in each window are disjoint and their
union is states 1 through 80.  Rollout starts from state zero and uses all
accumulated prefixes 1 through 80.  It therefore has recursion depth 80, not
the superseded five-step prefix.  Every training target appears exactly once
under both modes; the intended difference is recursion depth five versus 80.

The former per-start initial-guess normalizer would have assigned different
denominators to the same target under reset and rollout.  The selected J4B plan
now explicitly uses `||X*_n||_M^2` for target `n` in both objectives.  The
metric is `integral inner(Y,Y) dx_GLL` on the full mixed state, with natural
unit block weighting and no extra field scaling.  Each target has nominal
outer weight `1/80`; its target-mass scaling is identical in the two modes.
The generic initial-guess-residual convention remains an opt-in ablation, and
the committed J4A implementation is unchanged.

The accumulated J4B adapter now retains each certified production split cache
once.  Canonical reset advances 16 independent five-step windows and canonical
rollout advances one 80-step trajectory, so either objective-only evaluation
has 80 forward steps.  Gradient recursion uses one reverse step per cached
forward step and adds every local target dual at its state.  HVP recursion uses
one tangent and one exact incremental-reverse step per forward step, including
the local quadratic-loss Hessian action.  The redundant prefix implementation
is retained only as a tiny algebraic equivalence oracle.  This removes
implementation replay from cost comparison without asserting equal total
optimization cost.

Gate 2 is now explicitly an objective-only scalar landscape sanity check at
the nine selected c0 values.  Each reset or rollout point therefore has 80
forward steps and zero reverse, tangent, and incremental-reverse steps.  The
generic scan API still exposes objective-plus-gradient and full exact-Hessian
policies, and derivative-level metadata is part of restart compatibility.
Gate 3 is unchanged and continues to use exact gradients and exact HVPs.

External diagnostic evidence retains one earlier exact truth-reset derivative
spot check at c0=0.04: `J=7.927002933447668e-11`,
`dJ/dc0=-1.70886319543831e-09`, and
`d2J/dc0^2=2.108131445611791e-08`.  It took approximately 656.5 seconds and
reported 160 forward, 80 reverse, 80 tangent, and 80 incremental-reverse
steps.  Manual interruption during the next point's active exact incremental
adjoint assembly was not a correctness failure.  This result is evidence only,
not part of the canonical objective-only scan.

`dimswe.selected_test1b` validates the plan and audits completed truth without
Firedrake or solver execution.  It requires all 161 state/output records,
selected metadata, finite states, positive h, and no late-time growth warning.
The inference evaluation now reports projected-enstrophy mismatch and separate
finite/growth status; fit results expose accepted-step and cost summaries.

No production truth, landscape, optimization, held-out deployment, or resolved
simulation was executed during this selection-preparation change.  No accepted
J1/J2/J3/J4A production mathematics was changed.

Validation results for the corrected coverage plan are recorded in the task
handoff; no production truth, scan, fit, held-out deployment, or resolved
simulation is part of this source audit.
