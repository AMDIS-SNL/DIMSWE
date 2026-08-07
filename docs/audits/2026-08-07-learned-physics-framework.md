# 2026-08-07 learned-physics framework audit

## Repository gate

- Required branch: `dev/dimswe-learned-physics-framework`
- Required starting J3 checkpoint: `31850886a4e18c9dc2a14cb86976750fdf7f33eb`
- Starting branch and HEAD matched exactly.
- Starting index was empty.
- The only unrelated files were untracked `.DS_Store` and
  `docs/.DS_Store`; neither was touched.
- No remote operation, push, staging, or commit was performed.

## Implemented scope

The new `dimswe.learned_physics` package contains a Firedrake-free composition
boundary, generic float64 JAX-pytree algebra, four explicit training
objectives, immutable experiment/data/result records, NPZ plus canonical JSON
truth serialization, summary helpers, and extension contracts for learned
moist replacement and configurable misspecification correction.

`dimswe.hidden_c0` is a separately opt-in production adapter.  It constructs a
tiny serial MTSWE case, generates explicit truth, prepares the two offline and
two solver-in-loop c0 objectives, applies exact normalized gradient/HVP
scaling, performs deterministic bounded scalar optimization, returns dense
objective scans, and cross-evaluates every fitted parameter under common
deployment metrics.

This finalized case is **Test 1A**, the 2-by-2 hidden-c0
plumbing/integration benchmark.  It executes the real complete six-field
production split, but is not scientifically resolved flow and supports no
scientific comparison of training strategies.

## JAX x64 runtime contract

The learned-physics library requires `jax_enable_x64=True`, never silently
enables x64 at import, and rejects float32 execution.  Every relevant recipe
starts a fresh process with

```bash
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
```

An earlier process omitted `JAX_ENABLE_X64=True`, causing nine framework-test
precondition failures.  Those failures were solely environmental, not
mathematical.  The subsequent x64 run passed 12/12 framework tests.

The self-contained case configuration is
`dimswe/configs/hidden_c0_tiny.cfg`.  It preserves the production
`[RK4,Euler,SSPRK43,Euler]`, `[2,1,2,1]` graph and default UFL moist backend.

## Mathematical boundaries

The J4A source does not modify `dimswe/dissipation.py`,
`dimswe/timestepping.py`, `dimswe/mtswe_split_hvp.py`, the JAX moist kernel or
adapter, the PyROL adapters, or any J1/J2/J3 production test.

The operator/apriori observation is the production assembled weak
hyperviscosity RHS
before `M^-1` and Euler `dt`.  The discrete-offline observation is the actual
production child increment `dt*M^-1*b`.  Both are evaluated at externally
provided truth states and precomputed once; their optimization paths have zero
complete-solver calls.

Truth-reset observations each own their trusted window start.  Recursion is
limited to the configured internal window and the next window discards that
prediction.  Rollout observations all own the first trusted state and recurse
autonomously through their full prefix.  No truth state is injected inside a
rollout prefix.

Physical c0 alone enters the existing production helper.  Normalized
derivatives preserve exactly

```text
c0 = 0.07 z
g_z = 0.07 g_c0
H_z q_z = 0.07 H_c0(0.07 q_z).
```

No normalized scalar enters a production form.  Coefficient contexts snapshot
and restore all child coefficients, and all state/target inputs are owned
copies.

## Authoritative external validation and measured smoke result

The authoritative external results are:

- pure framework: `12 passed in 0.72s`;
- Test 1A hidden c0: `21 passed, 17508 warnings in 631.62s`;
- J3/J2 regression: `45 passed, 11477 warnings in 335.46s`;
- J1 plus production MTSWE HVP regression:
  `58 passed, 29908 warnings in 629.39s`; and
- complete repository: `293 passed, 1 skipped, 1 xfailed, 89516 warnings in
  2407.05s (0:40:07)` with no FAILED or ERROR section.

The complete-repository result certifies the J4A implementation before the
separate J4B-PREP additions.

A three-step production truth/objective run found strict minima at physical
`c0=0.14` in all modes.  At physical c0 values `0.112`, `0.14`, and `0.168`,
the objectives increased on both sides of truth.  Bounded normalized Newton
returned:

- operator/apriori offline: `0.14`, zero complete solver calls;
- deployed-discrete offline: `0.14`, zero complete solver calls;
- truth-reset: `0.14000000000000107`, 21 complete solver calls; and
- accumulated three-step rollout: `0.13999999999987023`, 60 complete solver
  calls.

These are controlled plumbing results, not scientific evidence about the
relative merit of training strategies.

## Test coverage authored

`tests/test_learned_physics_framework.py` covers arbitrary pytrees,
composition order, ownership, float64 enforcement, deterministic config
serialization, truth metadata roundtrip, mode dispatch, explicit mode keys,
no solver in operator/apriori mode, nonrecursive deployed-discrete truth-state use,
truth-reset restart behavior, and autonomous rollout recursion.

`tests/test_hidden_c0_benchmark.py` covers deterministic finite truth,
truth/guess separation, metadata, strict objective minima, centered gradients,
centered HVPs, recovery by all four modes, dense-scan minima, cross-evaluation,
field errors, finite deployment, no truth/input mutation, and repeatability.

## Extension audit

The Benchmark-2 contract fixes only that certified moist physics supplies
truth and an output map supplies replacement deployment.  The Benchmark-3
contract fixes only separate reference and configurable perturbed baseline
formulations plus a correction output map.  Features, architectures, network
library/size, normalization, direct/residual choice, rate/source
representation, and misspecification selection remain open.

## Source audit checklist

1. J3 checkpoint mathematics: unchanged.
2. Existing UFL backend: still default.
3. Existing JAX moist backend: still opt-in.
4. Learned-physics package and production adapter: opt-in.
5. Neural architecture in core infrastructure: none.
6. Parameter trees: arbitrary float64 JAX pytrees.
7. Four objectives: separate stable enum values and separate functions.
8. Discrete-offline: fixed truth-state map, no recursive propagation.
9. Truth-reset versus rollout: separate observation construction and dataflow.
10. Benchmark-1 truth versus learner: only physical c0 differs.
11. c0 scaling: unchanged at `0.07`.
12. Benchmark-2/3 scientific choice: no choice fixed.
13. MPI/accelerator/checkpointing: explicitly unsupported and unclaimed by J4A.
14. Production tests: none modified or weakened.

## External validation record

The exact reproducibility commands are maintained in
`docs/LEARNED_PHYSICS_EXPERIMENTS.md`.  The authoritative focused and complete
external executions are green; the complete suite is not to be rerun for
J4B-PREP inside Codex.
