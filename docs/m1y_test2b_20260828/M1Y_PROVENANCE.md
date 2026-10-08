# Test-2B M1-Y provenance

This campaign was executed entirely in the isolated writable workspace. The
authoritative tree supplied code and frozen artifacts only through the
verified initial copy; it was never a training working directory or output
target.

- Authoritative repository (read-only):
  `/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-study-d0eb615`
- Isolated writable workspace:
  `/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-feature-study-20260828/m1y_test2b_20260828_workspace`
- Starting branch: `dev/dimswe-learned-physics-framework`
- Starting HEAD: `d2f5d66ecb5500aad24eca37280f8a52e22a250f`
- Starting full status fingerprint:
  `b4dbc351a8a4d4bcc18bb2b037888b69a38813a75fa7d85f1abb443cd1ebad22`
- Starting tracked-diff fingerprint:
  `ab006aabcaf02020390a5ecc16b1db557dcec1d52ff3b0680f2cfe9ca1e666fd`
- Copy verification: an `rsync -ani --delete` comparison returned no
  differences before campaign edits.

The authoritative tree was not used as a training output location.  All new
generated artifacts are rooted at `external-results/m1y-test2b-20260828/` in
the isolated workspace.

## Immutable Test-2B inputs

Before prefix replay, all 161 truth restart arrays were checked against the
accepted truth manifest. The complete inventory contains 253,251,712 bytes;
its ordered inventory fingerprint is
`ef90f628dfe0b482a9549c055936c670175e82d1bc9d7f46ee7ff71f89a85ef6`.
All 161 entries passed their individual hashes.

| Artifact | SHA-256 |
|---|---|
| historical fixed learning data | `6e159015234fd94881b0b97888b7481eb049a02dcd96c571078920c0bedc901c` |
| historical fixed-data sidecar | `efeba6e0f80063a79ff47209aa1b8973565386e4d4e8844b4b4c2fb0281f0cae` |
| truth manifest | `746ae7020093261a4c5292fb37f61c8b12b0c825d5e5fa76a560fc830f37fe40` |
| learning-support audit | `e83edbfe67e90e04afd78b40249c04344948849a141fbb558c5eb3230b9d383d` |
| truth metadata | `744e728934c1e97a67a7bab232993c7444980b1d235b995a5887c7eeff5d1406` |
| rain-activity audit | `0302d1cb3808e9543986665eaa05aa3bcd49b1ab70326c2cca9a4d5dc1861b5d` |
| new M1-Y training preparation | `6f16e6db2c6ebdbd8c00a23cdae9b5318355384723a2f1276b2ea93d95145668` |
| new M1-Y held-out preparation | `1ddfa2d2e28b6f8dc2a0fbe0a12d2fe7da42158745a70eb2e088706501c42d2f` |

The frozen campaign configuration SHA-256 is
`d9836c3d28d7ab192f04da6fe6eab75018173c808296c262d1f20a95a55006d5`.
The pre-training validation SHA-256 is
`bca0fc0c33108aafb4785ee4ae610013d83abc1c89ac723d1cd709f88d9f37b7`.

## Checkpoints

| Representation | checkpoint file SHA-256 | canonical parameter-pytree SHA-256 |
|---|---|---|
| A | `a7f98ffe462a89a00d9d2dedd4efb4531cd0feea6e1d64de0b5359bae5e31db4` | `ed838229022ef2ac38a29bb88c8e98f2c173b89e0573088036ad301f76f4e158` |
| B | `3a829ffff2c2ee96946c197e18a7bd7d7495c255212c61bec1b3d24d788a71ee` | `1adbd173137cbf10928ee4b25865258fc97f74b3c2c4c0a09cfd8629020566ab` |
| C | `169eadab27a39916e37a875c27708873595f133cb4ea39e9f23eab14e86d49ee` | `c8da7ea375ced0f692210f97097e069f9d5757b6bead22b7a4a10d9a265d8a65` |

## Exact campaign argv

The scientific commands embed their exact `sys.argv` in their output
sidecars. They were run with
`/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/python` from
the isolated workspace.

```text
/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/python -m dimswe.test2b_m1y_campaign verify-immutable-inputs --configuration dimswe/configs/test2b_m1y_20260828.json --output external-results/m1y-test2b-20260828/preparation/immutable_inputs.json

/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/python -m dimswe.test2b_m1y_campaign prepare --configuration dimswe/configs/test2b_m1y_20260828.json --immutable-manifest external-results/m1y-test2b-20260828/preparation/immutable_inputs.json --output external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz

/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/python -m dimswe.test2b_m1y_campaign validate --configuration dimswe/configs/test2b_m1y_20260828.json --preparation external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz --output external-results/m1y-test2b-20260828/preparation/pretraining_validation.json

/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/python -m dimswe.test2b_m1y_campaign certify-objectives --configuration dimswe/configs/test2b_m1y_20260828.json --preparation external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz --validation external-results/m1y-test2b-20260828/preparation/pretraining_validation.json --output external-results/m1y-test2b-20260828/preparation/objective_certification.json

/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/python -m dimswe.test2b_m1y_campaign train --configuration dimswe/configs/test2b_m1y_20260828.json --preparation external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz --validation external-results/m1y-test2b-20260828/preparation/pretraining_validation.json --representation A --output-directory external-results/m1y-test2b-20260828/production/representation-A/m1y-seed0-m20-10k

/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/python -m dimswe.test2b_m1y_campaign train --configuration dimswe/configs/test2b_m1y_20260828.json --preparation external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz --validation external-results/m1y-test2b-20260828/preparation/pretraining_validation.json --representation B --output-directory external-results/m1y-test2b-20260828/production/representation-B/m1y-seed0-m20-10k

/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/python -m dimswe.test2b_m1y_campaign train --configuration dimswe/configs/test2b_m1y_20260828.json --preparation external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz --validation external-results/m1y-test2b-20260828/preparation/pretraining_validation.json --representation C --output-directory external-results/m1y-test2b-20260828/production/representation-C/m1y-seed0-m20-10k

/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/python -m dimswe.test2b_m1y_evaluation prepare-heldout --configuration dimswe/configs/test2b_m1y_20260828.json --preparation external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz --output external-results/m1y-test2b-20260828/evaluation/m1y_heldout_data.npz

/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/python -m dimswe.test2b_m1y_evaluation evaluate --configuration dimswe/configs/test2b_m1y_20260828.json --preparation external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz --heldout external-results/m1y-test2b-20260828/evaluation/m1y_heldout_data.npz --representation A --output external-results/m1y-test2b-20260828/evaluation/representation_A_matched.json

/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/python -m dimswe.test2b_m1y_evaluation evaluate --configuration dimswe/configs/test2b_m1y_20260828.json --preparation external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz --heldout external-results/m1y-test2b-20260828/evaluation/m1y_heldout_data.npz --representation B --output external-results/m1y-test2b-20260828/evaluation/representation_B_matched.json

/Users/arjunsharma/venvs/dimswe-firedrake-2026.4.1-py312/bin/python -m dimswe.test2b_m1y_evaluation evaluate --configuration dimswe/configs/test2b_m1y_20260828.json --preparation external-results/m1y-test2b-20260828/preparation/m1y_learning_data.npz --heldout external-results/m1y-test2b-20260828/evaluation/m1y_heldout_data.npz --representation C --output external-results/m1y-test2b-20260828/evaluation/representation_C_matched.json

/opt/homebrew/Caskroom/miniforge/base/bin/python3 -m dimswe.test2b_m1y_report --output-directory docs/m1y_test2b_20260828
```

The matched-evaluation records explicitly state `optimizer_instantiated=false`
and `truth_generated=false`: evaluation neither retrained a model nor produced
a replacement truth trajectory. Historical M1-X direct metrics were
recomputed only for parity and agreed exactly with the frozen accepted
post-processing records.

The final read-only authoritative-repository recheck is recorded separately
in `authoritative_final_recheck.json` and linked by `manifest.json`.
