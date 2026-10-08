# Test 2A M1-to-M2 fine-tuning diagnostic

## Scientific status

The matched seed-zero Method-1/Method-2 200k comparison is complete and is
not replaced or reinterpreted here. This is a secondary sequential-workflow
diagnostic:

1. initialize from the matched 200k operator-trained parameter pytree;
2. create a new PyROL/ROL process with empty L-BFGS secant history;
3. minimize the unchanged, production-oracle-certified deployed-discrete
   objective for at most 50,000 accepted iterations.

The initialization is
`external-results/test2a/fair-longfit/operator-seed0-m20-200k/final_parameters.npz`
with pytree SHA256
`f86ee79be3086028f21de10b947c0089147234f494c066f8bbbb2fffb3f8bef8`.
No Method-1 secant pairs or optimizer state are transferred.

## Gradient geometry

At the matched Method-1 point:

| quantity | value |
|---|---:|
| J_op | 0.000373006108792648 |
| J_disc | 0.0008346864309047664 |
| norm(g_op) | 0.00040517422069644637 |
| norm(g_disc) | 0.4788629747740869 |
| dot(g_op,g_disc) | 1.7974916511512145e-6 |
| cosine | 0.009264325751792556 |
| best alpha in `g_disc-alpha*g_op` | 10.94922224112387 |
| relative nonproportional residual | 0.9999570852133429 |

The matched Method-1 point is therefore not stationary under the
deployed-discrete objective, and the discrete gradient is almost completely
orthogonal to the remaining operator gradient.

At the matched Method-2 point:

| quantity | value |
|---|---:|
| J_op | 0.002489117530253537 |
| J_disc | 0.001721966994676836 |
| norm(g_op) | 0.6793855846246729 |
| norm(g_disc) | 0.007123391061845834 |
| dot(g_op,g_disc) | 0.0024090516590788776 |
| cosine | 0.4977863670189403 |
| best alpha in `g_disc-alpha*g_op` | 0.005219314389030463 |
| relative nonproportional residual | 0.8672996787789587 |

Both gradients use the frozen operator dataset and the exact certified
fixed-state deployed-discrete cache. Truth access is restricted to states
0..80.

## Optimizer and checkpoints

The diagnostic uses PyROL/ROL line-search L-BFGS, memory 20, exact gradients,
no HVP, gradient tolerance `1e-8`, step tolerance `1e-12`, and a 50,000
accepted-iteration cap. Parameter artifacts are saved at iterations 0, 1000,
5000, 10000, 25000, and 50000. Natural ROL termination is retained.

Progress and parameter artifacts are atomic. A parameter-only restart is
available, but a new process cannot restore the process-local L-BFGS secant
history. Such a restart is not the same continuous optimizer trajectory.

The 20-iteration nonscientific smoke reduced J_disc from
`0.0008346864309047664` to `0.000790187980446671` with 46 objective and 21
gradient evaluations in 1.587 seconds. It made no HVP calls. Linear scaling
of this short run gives about 3969 seconds (1.10 hours) for 50k, but this is
only an engineering estimate because line-search behavior may change.

## Postprocessing

After fitting, every available major checkpoint is evaluated under both
offline objectives and the established direct-A metrics. The existing
complete autonomous evaluator then runs independently from truth state 0 for
80 steps with no reset. Autonomous errors never influence fitting or stopping.
No state after 80 is loaded.

The report retains the historical practical and matched 200k reference
results verbatim. Its interpretation is limited to whether deployed-discrete
fine-tuning adds useful information after operator pretraining; it does not
claim that Method 2 beats Method 1 in the matched from-seed experiment.

## Manual launch

Run from a normal Terminal, not from Codex:

```bash
cd /path/to/DIMSWE-collaborator
mkdir -p external-results/test2a/m1-to-m2-finetune
nohup caffeinate -i bash scripts/run_test2a_m1_to_m2_finetune_50k.sh \
  > external-results/test2a/m1-to-m2-finetune/manual-launch.log 2>&1 &
echo $!
```

If a reviewed parameter checkpoint must be restarted, add `--resume`. This
starts a new empty L-BFGS history and must be reported as such.
