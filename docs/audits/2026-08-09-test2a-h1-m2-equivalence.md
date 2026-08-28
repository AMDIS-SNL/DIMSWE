# Test 2A H=1 truth-reset / Method-2 equivalence audit

Date: 2026-08-09

This is an interpretation and derivative audit only. It performs no
optimization and reads no truth state after index 80.

## Result

The literal current objectives are **not proportional**. The precise
classification is:

> **Case C: fixed-state, target-backend, support, and normalization
> conventions differ.**

This does not make H=1 recursively solver-in-the-loop. At H=1, all work is
still an exactly cacheable fixed-state child-6 regression. When the fixed
post-prefix state, stored truth target, support, and H=1 weights are copied
literally, the independently assembled fixed-state objective reproduces the
trajectory objective to float64 accuracy. Genuine parameter-mediated
cross-time feedback first appears at H=2.

## Actual evaluation states and indexing

Current Method 2 uses 81 standalone JAX moist-child observations. It applies
the analytical moist Euler child directly to boundary truth states

\[
X_0^*,\ldots,X_{80}^*.
\]

The source is explicit in `TRAINING_STEPS=range(81)` and the call to
`helper.take_forward_step_cached(trajectory.states[step], ...)` in
`dimswe/test2a_discrete_offline.py`. Its cached form evaluates the same 81
feature blocks and exact weak/mass actions in
`dimswe/test2a_discrete_training.py`.

Literal H=1 truth reset instead uses 80 complete transitions. For each
\(k=0,\ldots,79\), it constructs

\[
Y_k=P(X_k^*),\qquad P=C_5\circ\cdots\circ C_1,
\]

then evaluates neural child 6 at \(Y_k\) and compares the result with stored
boundary state \(X_{k+1}^*\). Targets are exactly states 1 through 80. The
implementation precomputes `take_fixed_prefix_cached` and completes the step
with `take_forward_step_from_prefix` in `dimswe/test2a_trajectory.py` and
`dimswe/mtswe_split_hvp.py`.

The numerical parameter-independence check used seed 0 and matched-M1-200k
parameters at reset origins 0, 40, and 79. Every field after children 1--5
agreed bitwise between the two parameter probes and the fixed-prefix cache.
The completed neural steps differed in S, Qv, and Qc, confirming that theta
enters only child 6. Child time metadata is retained: the dry and DG half
steps use their production subcycle times, while child 6 receives \(t_n\)
and \(dt=100\).

Truth files are complete-step boundary states. The canonical truth metadata
records `moist_backend=ufl`, `dt=100`, and every-step output. The H=1 audit
uses only files for states 0 through 80.

## H=1 derivation

At a common post-prefix state \(Y_k\), define

\[
\delta A_k=A_\theta(Y_k)-A_*(Y_k),\qquad
\beta_2=gL.
\]

The neural and analytical JAX child evaluate the same original rain law at
the same \(Y_k\). Hence R cancels exactly. In mixed-field order
\((v,h,S,Q_v,Q_c,Q_r)\), the source-density defect is

\[
H(Y_k)\delta A_k=
(0,0,h\beta_2\delta A_k,h\delta A_k,-h\delta A_k,0).
\]

With exact production weak assembly \(W\), mixed mass matrix \(M\), and

\[
G(Y_k)=M^{-1}WH(Y_k),
\]

the tendency and Euler-state defects are

\[
\delta T_k=G(Y_k)\delta A_k,
\qquad
r_k^{state}=dt\,G(Y_k)\delta A_k.
\]

The current certification-only trajectory metric is

\[
\ell_k={1\over2}w_k
{\|\widehat X_{k+1}-X^*_{k+1}\|_M^2
 \over \|X^*_{k+1}\|_M^2}.
\]

If the target is the same analytical child-6 map, this becomes

\[
\ell_k={w_kdt^2\over2\|X^*_{k+1}\|_M^2}
\|G(Y_k)\delta A_k\|_M^2.
\]

Thus H=1 is a fixed quadratic deployed-map regression. No derivative passes
through children 1--5 because their input is fixed truth and theta enters
only child 6.

## Why the literal current objectives differ

The differences are:

1. Method 2 evaluates closure inputs at \(X_k^*\); H=1 evaluates at
   \(Y_k=P(X_k^*)\).
2. Method 2 uses 81 observations, 0 through 80; H=1 has 80 transitions,
   starts 0 through 79, and targets 1 through 80.
3. Method 2 compares mass-solved tendencies; H=1 state defects introduce a
   positive \(dt^2/2\) factor.
4. Method 2 divides the total squared defect by one global analytical-A
   tendency-energy norm. The certification H=1 loss divides each target by
   its own full-state mixed-mass norm. The resulting 80 coefficients range
   from `3.630703040849231e-18` to `3.631765965654123e-18`; they are close,
   but not identical.
5. Both use the production mixed mass metric. There are no additional field
   weights. Under the analytical H=1 defect, v, h, and Qr are structurally
   zero, while S, Qv, and Qc contribute.
6. The stored truth target comes from the UFL moist backend, while the
   post-prefix analytical control uses the deployed analytical JAX child.
   These are certified implementations of the same physics but are not
   bitwise identical on this trajectory. The maximum one-step discrepancy is
   `1.2452933152003626e-6` in relative mixed mass norm. Maximum coefficient
   differences are 0 for v, h, and Qr; `7.794655734141998e-2` for S; and
   `2.974003057111263e-4` for each of Qv and Qc. R is exactly zero at all 80
   audited post-prefix states, but the implementation still evaluated the
   original R law rather than hard-coding zero.

Against the exact stored target, the fixed-state tendency target is simply

\[
T_k^{stored}=(X^*_{k+1}-Y_k)/dt.
\]

Using this target and the literal H=1 coefficients reproduces H=1 exactly.
This identity is useful diagnostically but does not redefine either method.

## Five-probe objective and gradient evidence

For the table below, alpha minimizes
\(\|g_{H1}-\alpha g_{M2}\|_2\), and residual is the remaining norm divided by
\(\|g_{H1}\|_2\).

| probe | J_disc | J_H1 | J_H1/J_disc | cos(g_H1,g_disc) | alpha | residual |
|---|---:|---:|---:|---:|---:|---:|
| seed 0 | 1.202741373033232 | 1.707777488544651e-9 | 1.419904167957366e-9 | 0.9452858824937904 | 1.091765426396251e-9 | 0.3262431613964282 |
| matched M1 200k | 8.346864309047664e-4 | 1.046952564223935e-12 | 1.254306438274168e-9 | 0.8377954485314705 | 7.432501277665536e-10 | 0.5459842364207520 |
| matched M2 200k | 1.721966994676836e-3 | 2.365421926701255e-12 | 1.373674370074194e-9 | 0.4684538366475913 | 3.326194594683466e-8 | 0.8834879755436130 |
| M1 to M2 50k | 5.167359629570338e-4 | 7.346943755365289e-13 | 1.421798419703987e-9 | 0.5316264174083015 | 1.497175146130853e-7 | 0.8469789562401270 |
| seed deterministic +1e-3 | 1.204627243224091 | 1.710114855501763e-9 | 1.419621600890225e-9 | 0.9462092557822794 | 1.092712067012908e-9 | 0.3235553187199143 |

The ratio and gradient scaling are not constant. Literal H=1 and current
Method 2 therefore are not the same objective up to a scalar.

The all-parameter gradient norms \((\|g_{H1}\|,\|g_{disc}\|)\) were,
respectively:

- seed 0: `(2.669356514396434e-9, 2.311215364944117)`;
- matched M1: `(4.248232283994590e-10, 4.788629747740870e-1)`;
- matched M2: `(5.057869739159059e-10, 7.123391061845835e-3)`;
- fine-tuned: `(3.503397342582282e-10, 1.244008480108713e-3)`;
- perturbation: `(2.692658412688520e-9, 2.331646542268671)`.

## Post-prefix controls

The globally normalized analytical post-prefix objective is not proportional
to literal H=1 because it retains Method-2 global normalization and uses the
analytical JAX target. More decisively, even the analytical post-prefix
objective with H=1 per-target weights differs at trained parameters because
the literal target is stored UFL truth.

The stored-target post-prefix control matches literal H=1:

| probe | objective ratio H1/control | gradient cosine | alpha | residual |
|---|---:|---:|---:|---:|
| seed 0 | 0.9999999999999436 | 0.9999999999999998 | 1.000000000000029 | 9.26e-14 |
| matched M1 200k | 1.000000000000768 | 0.9999999999999996 | 0.999999999996880 | 1.03e-11 |
| matched M2 200k | 1.000000000001296 | 1.0000000000000002 | 0.999999999998887 | 1.48e-11 |
| M1 to M2 50k | 0.9999999999983393 | 1.0000000000000002 | 1.000000000004590 | 1.52e-11 |
| perturbation | 0.9999999999999906 | 1.0000000000000002 | 1.000000000000072 | 9.55e-14 |

This is the decisive evidence that H=1 adds no recursive solver feedback. It
is an exact fixed-state objective at \(Y_k\) with a stored-step target.

## H=2 contrast

At seed 0, a continuous accumulated H=2 objective gave
`2.431877752420198e-10`, while the corresponding two independent H=1 truth
resets gave `1.0490268350048921e-10`. Their gradient cosine was
`0.9967873319662804`, but the best-scaled nonproportional residual was
`0.08009378772129659`, well above roundoff.

After the first neural moist child,
\(\widehat X_{k+1}=\widehat X_{k+1}(\theta)\). On the second complete step,
that parameter-dependent state enters both dry RK4 children,
hyperviscosity, both DG SSPRK43 children, the next neural A evaluation, and
the original R law through its state dependence. H=2 is therefore the first
horizon at which recursive solver-mediated cross-time feedback necessarily
appears.

## Recommended taxonomy

Do not present literal H=1 as an independent recursive training philosophy.
A cleaner scientific taxonomy would distinguish:

- M1: local operator/a-priori A regression;
- M2-X: fixed boundary-state deployed-discrete regression at \(X_k^*\);
- optional M2-Y/H1: fixed post-prefix one-step regression at \(Y_k\), with
  the target backend and metric stated explicitly;
- recursive truth-reset/rollout training only for horizons H >= 2;
- a longer continuous rollout as the separate long-feedback method.

No production names or the canonical AMDIS research-plan source were changed
by this audit.

## Reproduction and files

Run with x64 and serial threading:

```bash
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
python -m dimswe.test2a_h1_m2_equivalence \
  --output external-results/test2a/equivalence-audit/h1_m2_equivalence.json
```

Tracked-preparation files added by this audit:

- `dimswe/test2a_h1_m2_equivalence.py`
- `tests/test_test2a_h1_m2_equivalence.py`
- `docs/audits/2026-08-09-test2a-h1-m2-equivalence.md`
- `docs/manifests/TEST2A_H1_M2_EQUIVALENCE_FILES.txt`

The generated JSON is external evidence and remains untracked with other
`external-results` data.
