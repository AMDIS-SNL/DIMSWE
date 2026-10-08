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
Gate 3's derivative path is unchanged and continues to use exact gradients and
exact HVPs.

The completed external Gate-2 landscapes place all four sampled minima exactly
at c0=0.14.  Operator/apriori and deployed-discrete coincide exactly in this
scalar normalized case.  Representative c0=0.12 values are about `1.02e-2`
for either offline loss, `2.81e-12` for five-step truth reset, and `1.49e-10`
for rollout.  Rollout is substantially more sensitive than reset, but these
absolute magnitudes are definition-dependent and do not rank training quality.
The common Gate-3 bounded-Newton policy consequently uses only positive-scale-
homogeneous decisions: relative gradient reduction, relative parameter step,
relative curvature magnitude with an independent strict sign check, projected
bound stationarity, and Armijo sufficient decrease.

Gate 3 subsequently recovered c0 as `0.14`, `0.14`,
`0.13999999999997986`, and `0.13999999999998547` for operator/apriori,
deployed-discrete, truth-reset, and rollout.  Accepted steps were 1, 1, 4, and
5, with objective/gradient/HVP counts `3/2/1`, `3/2/1`, `9/5/4`, and
`11/6/5`.  Truth-reset and rollout wall times were 1721.56 and 2372.70 seconds.
The optimizer was DIMSWE's custom safeguarded bounded scalar Newton solver,
not SciPy or ROL.  No held-out state was loaded during any fit.

The prepared Gate-4 evaluator independently rejects incomplete or unsuccessful
fit records, reads recovered c0 from the accepted fit, and advances recursively
from trusted `X*_80` through state 160 with no truth reset.  Truth targets 81
through 160 enter only post-prediction metrics.  Passing Gate 4 certifies the
deterministic correctly specified workflow; it is not a generalization claim.

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
