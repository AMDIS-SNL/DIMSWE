# Test 2A-2: frozen neural-A embedding

## Scope

Test 2A-2 is an embedding and derivative-certification gate. It performs no
training and reads no truth state after 80. The original UFL moist child and
the analytical JAX moist child remain unchanged. The new path is available
only when the existing JAX backend is explicitly selected and a verified
`FrozenNeuralAMoistPhysics` provider is passed to the timestepper.

This gate does not authorize deployed-discrete, truth-reset, or autonomous
neural-parameter optimization. It also does not claim held-out performance.

## Frozen Test 2A-1 operator

The selected artifact is

```text
external-results/test2a/optimizer-study/continuation-m20-plus45000/
  continuation_final_parameters.npz
```

Its contract is:

```text
features                 h, S, Qv, Qc, B
network                  5 -> 32 -> 32 -> 1
activation               tanh
dtype                    float64
parameters               1,281
input normalization      states 0..80 mean / population standard deviation
output reconstruction    RMS_training(A) * normalized network output
```

No normalization is fitted during embedding. The degenerate B scale is read
from the training sidecar exactly as recorded. The accepted practical
operator endpoint has normalized MSE `0.004285912836972889`, relative RMS
error `0.0654668835135207`, and correlation `0.9978490330152804`. Its active
`1e-3` relative RMS error is `0.06441392158258892`; active sign accuracies are
`0.9557392085657594`, `0.997284347541086`, and `1.0` at the `1e-3`, `1e-2`,
and `1e-1` thresholds. ROL did not meet its nominal absolute gradient
tolerance. These values define a frozen practical baseline, not a claim of
mathematical stationarity.

The embedding configuration records and the loader verifies these SHA-256
fingerprints:

```text
parameter NPZ       bb9910c798428215d76a96a165a67a02d4b2e19aacb31299d521151bf7bb2cf6
parameter pytree    f7d2fd9577ba2a824d3df6c1f8c90a425e6964e1fb5962508315509b247bed56
parameter sidecar   c7763cf04f0db00c640be3a6fc861d28a07944d01e9d700700a3b70ce6e46371
source result       53f2b2d640482604bdeda3672821533f0b449904a0e24f37cd4b2a2477c36796
dataset metadata    0e3f354676dbe2321db928db7f5cc93b95fbee23d16756d8cbb4260f54610893
selected config     78e5ad8accc9a0b10a35a95c6968a6da638ce5937ca810a651b76b258aee79e4
```

The raw NPZ leaf names, shapes, and float64 dtypes are checked before the
existing pytree loader is called. Architecture, parameter count, feature
order, training-state range, normalization provenance, and accepted source
result are then checked independently. Flattened parameter order is never
guessed or reconstructed from a different convention.

## Hybrid deployed physics

At each existing cell-local tensor-product 4-by-4 GLL array, the opt-in local
provider computes

```text
A = A_theta(h, S, Qv, Qc, B)
R = R_original(h, S, Qv, Qc, B; deployed moist parameters)

T_Qv =  h A
T_Qc = -h (A + R)
T_Qr =  h R
T_S  =  h (g L) A.
```

The provider calls the unchanged `moist_rates_jax` to obtain `R_original` at
the current model state. It does not use the training observation that R was
zero, and it cannot hard-code R to zero. A synthetic certification state is
chosen with nonzero R and verifies exact equality to the analytical JAX rain
rate.

The algebraic source construction makes the following identities structural:

```text
T_Qv + T_Qc + T_Qr = 0
T_S - (g L) T_Qv = 0.
```

They are checked at the exact local GLL representation. Total-water weak
assembly and the constant-test entropy/water relation are also checked in the
Firedrake certification where the field spaces permit a common statement.
They are not penalties learned by the network.

## Exact deployed path

`JAXMoistEulerPrimal` continues to own every global operation. For neural A it
uses the same path as the analytical JAX child:

1. interpolate `h,S,Qv,Qc,B` to the broken-CG3 carrier;
2. pack by the existing cell node map as `(owned cells, 16 GLL points)` with
   the first physical coordinate varying fastest;
3. evaluate the frozen local JAX provider;
4. unpack the four source arrays into the same broken carrier;
5. weakly assemble against the production mixed test space with the
   production GLL measure;
6. apply the same complete mixed mass solve; and
7. perform `X_plus = X + dt * tendency`.

Shared physical points remain repeated in the cell-local representation.
Quadrature, local ordering, function spaces, assembly, and the mass solver are
unchanged. The pure-network certification evaluates all 331,776 state-0..80
samples and obtains bitwise-equal predictions, zero maximum absolute
difference, and zero relative L2 difference from the standalone frozen
Test-2A-1 model.

## Derivative path

Let `P` be the installed Firedrake interpolation/packing map, `W` the weak
source assembly, `M` the mixed mass matrix, and `s(PX,theta)` the four neural-A
/ original-R source arrays. The embedded child is exactly

```text
F(X,theta) = X + dt M^-1 W s(PX,theta).
```

The state and parameter tangents are

```text
D_X F dX         = dX + dt M^-1 W [s_X(PX,theta) P dX],
D_theta F dtheta =      dt M^-1 W [s_theta(PX,theta) dtheta].
```

JAX supplies local JVP, VJP, and differentiated-VJP actions. The surrounding
code reuses the certified J2 `P`, installed Firedrake `P*`, `W`, `W*`, mixed
mass solve, and dual pairings. The neural network remains local to JAX; no
global Firedrake operation moves into JAX and no dense Jacobian or Hessian is
formed. No intermediate Riesz map is inserted. Parameter VJPs retain the
original arbitrary pytree and cover all 1,281 parameters.

The joint differentiated VJP differentiates both returned pullbacks with
respect to state, parameter, and incoming source covector. It therefore
contains state-state, state-parameter, parameter-state, parameter-parameter,
and covector-variation effects needed by a later exact HVP composition. Test
2A-2 certifies this complete moist-child action; accumulation of neural-
parameter adjoints across multiple full-split steps remains Test 2A-3 work.

## Full-split compatibility

The only production switch added is an optional local provider carried by the
existing JAX moist wrapper. With no provider, behavior is the analytical JAX
path. With `moist_backend="ufl"`, supplying a provider is rejected. The
original UFL backend remains the default.

The production split helper passes the provider only to the moist child and
retains the exact forward order

```text
dry_rk4_0
dry_rk4_1
hyperviscosity_euler
dg_ssprk43_0
dg_ssprk43_1
moist_euler
```

and its exact reverse. A tiny external smoke test checks this ordering and one
finite neural-A split step. It is not a long learned-physics trajectory.

## Certification commands

Run from the repository root in the certified serial Firedrake/JAX
environment:

```bash
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export TEST2A2_CACHE_ROOT="$(mktemp -d /tmp/dimswe-test2a2.XXXXXX)"
export PYOP2_CACHE_DIR="$TEST2A2_CACHE_ROOT/pyop2"
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR="$TEST2A2_CACHE_ROOT/tsfc"
export XDG_CACHE_HOME="$TEST2A2_CACHE_ROOT/xdg"
export MPLCONFIGDIR="$TEST2A2_CACHE_ROOT/matplotlib"
export PYTHONPATH="$PWD"

# A. frozen-network parity plus child primal/assembly/Euler certification
python -m dimswe.test2a_embedded_moist certify-network \
  --embedding-configuration dimswe/configs/test2a_embedded_neural_a.json \
  --dataset external-results/test2a/dataset/doublevortex_A_operator.npz \
  --output external-results/test2a-2/pure_network_parity.json
python -m pytest -q tests/test_test2a_embedded_moist_firedrake.py \
  -k weak_assembly_mass_solve_and_euler

# B. complete child state/parameter derivative certification
python -m pytest -q tests/test_test2a_embedded_moist_firedrake.py \
  -k 'complete_state_jvp or complete_parameter_jvp or complete_joint_differentiated_vjp'

# C. opt-in and one-step complete-split smoke certification
python -m pytest -q tests/test_test2a_embedded_moist_firedrake.py \
  -k neural_mode_is_opt_in_and_complete_split_order_is_unchanged
```

The pure-JAX suite can be repeated independently with:

```bash
python -m pytest -q tests/test_test2a_embedded_moist.py \
  tests/test_jax_moist_local.py \
  tests/test_jax_moist_derivatives.py::TestPureJAXMoistDerivatives
```

Generated parity JSON belongs under `external-results/test2a-2` and remains
untracked.

## Boundary before Test 2A-3

External Firedrake results must certify the weak primal, complete state
JVP/VJP, all-parameter JVP/VJP, joint differentiated VJP, and tiny full-split
smoke test. Only then can Test 2A-3 define deployed-discrete neural-parameter
training and the multi-step parameter-adjoint accumulation required by later
truth-reset and rollout objectives. No Test-2 deployment state 81..160 is
accessed in this gate.
