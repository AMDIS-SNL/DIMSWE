# Test-2B M1-Y results

Status: **COMPLETE**. A, B, and C passed the implementation contract, trained
for the frozen 10,000 accepted-iteration budget, and were evaluated with the
same 160-step standard hybrid calculation used by the accepted Test-2B M1-X
campaign.

Historical M1 is unambiguously M1-X: its production data path packs features
and evaluates analytical targets directly at `X_n*`, without applying the
timestep prefix. New M1-Y instead uses `Y_n*=P(X_n*)` for both. The feature
contract is unchanged: `(h,S,Qv,Qc,B)` with historical X-support input
normalization and historical output scales. Both objectives use truth states
0..80 inclusive (81 states; 5,308,416 samples). States 81..160 are held out.

## Matched training

All networks use float64, seed-0 Glorot-uniform weights, zero biases, tanh
hidden layers, a linear output, deterministic full-batch carrier-mass
weighting, and PyROL line-search L-BFGS with memory 20, `gtol=1e-8`,
`stol=1e-12`, and a 10,000 accepted-iteration cap. Every objective value was
finite, every checkpoint is readable, and no run met the gradient tolerance
before reaching the frozen historical cap.

| Representation | Objective/state | Architecture (parameters) | accepted / objective / gradient evaluations | final normalized training loss | termination |
|---|---|---:|---:|---:|---|
| A | M1-X / X | 5-32-32-1 (1,281) | 10,000 / 20,983 / 10,001 | 2.9897337576e-5 | MAXITER |
| A | M1-Y / Y | 5-32-32-1 (1,281) | 10,000 / 20,918 / 10,001 | 2.4395096189e-5 | MAXITER |
| B | M1-X / X | 5-32-32-2 (1,314) | 10,000 / 20,964 / 10,001 | 4.0727693178e-5 | MAXITER |
| B | M1-Y / Y | 5-32-32-2 (1,314) | 10,000 / 20,858 / 10,001 | 3.2610435198e-5 | MAXITER |
| C | M1-X / X | 5-32-32-4 (1,380) | 10,000 / 20,746 / 10,001 | 3.8697519766e-5 | MAXITER |
| C | M1-Y / Y | 5-32-32-4 (1,380) | 10,000 / 20,785 / 10,001 | 3.8058600674e-5 | MAXITER |

## Direct physical-law fit

Each row below uses the objective's nominal evaluation state: X for M1-X and
Y for M1-Y. Values are carrier-mass-weighted relative RMS errors. The complete
machine record also contains physical RMS, maximum, bias, correlation,
activity, cross-state, and regime-resolved metrics.

| Representation/objective | training-support direct error | held-out direct error |
|---|---|---|
| A / M1-X | A: 5.467846e-3 | A: 2.233068 |
| A / M1-Y | A: 4.939139e-3 | A: 2.236155 |
| B / M1-X | A: 6.290982e-3; R all/active: 1.396697e-2 / 9.594171e-3 | A: 2.895953; R all/active: 2.070556e-2 / 1.894718e-2 |
| B / M1-Y | A: 5.341154e-3; R all/active: 2.410416e-2 / 1.627781e-2 | A: 2.927051; R all/active: 3.274393e-2 / 2.610757e-2 |
| C / M1-X | source S/Qv/Qc/Qr: 5.898359e-3 / 5.868244e-3 / 5.886693e-3 / 4.093956e-2 | source S/Qv/Qc/Qr: 2.309033 / 2.350506 / 2.671031 / 5.724901e-2 |
| C / M1-Y | source S/Qv/Qc/Qr: 5.141345e-3 / 5.148927e-3 / 5.147779e-3 / 6.857387e-2 | source S/Qv/Qc/Qr: 1.536951 / 1.514758 / 1.499877 / 8.219670e-2 |

The state shift improves the nominal training A error by 9.7% for A and
15.1% for B. For C it improves the dominant S, Qv, and Qc training components
by 12.3--12.8%. It does **not** uniformly improve direct fit: A's held-out
relative error is effectively unchanged, B's active-R error is 69.7% larger
on training support and 37.8% larger held out, and C's Qr source error is
67.5% larger on training support and 43.6% larger held out.

For B, the inactive-sample held-out R false-positive rate improves from
0.57094 to 0.21029, and both direct fits have zero false negatives on active
samples. Thus the larger R RMS error is not simply an activity-classification
failure.

## Objective and standard-hybrid comparison

The H1/H2/H5 entries are diagnostic objective evaluations only; none of those
objectives was retrained. Hybrid errors are the accepted mixed-state metrics
from the identical 160-step standard Test-2B evaluation.

| Representation/objective | J_H1 | J_H2 | J_H5 | hybrid accumulated | hybrid final | hybrid maximum | final Qr-mass error |
|---|---:|---:|---:|---:|---:|---:|---:|
| A / M1-X | 1.770084e-4 | 3.612619e-4 | 1.001694e-3 | 6.401951e-6 | 8.330284e-6 | 9.880423e-6 | +6.371306e6 |
| A / M1-Y | 1.675512e-5 | 2.603019e-5 | 4.013699e-5 | 5.868512e-6 | 9.057834e-6 | 1.140110e-5 | +1.842975e6 |
| B / M1-X | 2.232161e-4 | 4.697184e-4 | 1.402828e-3 | 2.257237e-3 | 7.024825e-3 | 7.024825e-3 | +4.526210e8 |
| B / M1-Y | 2.065229e-5 | 3.009204e-5 | 4.384985e-5 | 2.526357e-6 | 5.099172e-6 | 5.099365e-6 | +2.547830e7 |
| C / M1-X | 2.729769e-4 | 5.873496e-4 | 1.876094e-3 | 1.283846e-5 | 1.578803e-5 | 1.878710e-5 | -4.124803e7 |
| C / M1-Y | 2.221322e-5 | 3.652921e-5 | 6.189596e-5 | 4.876781e-6 | 1.210148e-5 | 1.210148e-5 | -6.592344e7 |

M1-Y reduces J_H1/J_H2/J_H5 by 90.5--96.0% for A, 90.7--96.9% for B,
and 91.9--96.7% for C. The standard-hybrid outcome is representation
dependent:

- A: accumulated error improves 8.3%, while final and maximum errors worsen
  8.7% and 15.4%. Its analytically supplied R remains accurate, and final
  Qr-mass error decreases from 6.37e6 to 1.84e6.
- B: accumulated, final, and maximum errors improve 99.888%, 99.927%, and
  99.927%. R error on held-out **model post-prefix states** falls from
  0.99735 to 0.02942, maximum learned R changes from 7.9581e-11 to
  5.1559e-11 (truth maximum 5.1791e-11), and final Qr-mass error drops from
  4.526e8 to 2.548e7. M1-Y has no active-R false negatives, but its pre-truth-
  onset trajectory false-positive fraction is still substantial and rises
  from 0.21725 to 0.27409; unconstrained R sign/onset behavior is therefore
  not solved.
- C: accumulated, final, and maximum state errors improve 62.0%, 23.4%, and
  35.6%. Its held-out model-state S/Qv/Qc errors improve modestly, but Qr
  source error and rain partition worsen: final Qr-mass error magnitude grows
  59.8%.

## Scientific interpretation

The controlled X-to-Y state-location change matters before recursive
training, most strongly for Representation B. B is the decisive example:
its nominal direct R regression is worse on truth support, yet its R law is
far more accurate on the states actually visited by the deployed split model,
removing the catastrophic rain/cloud drift in the historical M1-X hybrid.
This is consistent with a train/deploy timestep-location mismatch.

The result is not universal evidence that Y sampling always improves direct
interpolation or every hybrid diagnostic. A is mixed, and C improves state
trajectory error while worsening rain partition. Because all six fits reached
the same finite optimization cap and X/Y supports differ, optimization and
finite-support extrapolation remain possible contributors. The evidence
supports resolving timestep-location consistency before introducing recursive
training; it does not by itself establish a universal causal hierarchy among
objectives or representations.

The full numeric records are `M1Y_RESULTS.json`, `M1Y_RESULTS.csv`, and the
three representation-specific matched-evaluation JSON files listed in
`manifest.json`.
