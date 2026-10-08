# Track 1 second-pass hygiene verification

Date: 2026-08-28 (America/Denver)

## Scope and commits

The collaborator cleanup began at
`10a0482a16cb190c2012550a40a72af53893abfe`. It operated only in
`/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-collaborator-track1-20260828`.
The authoritative archaeological checkout was read-only.

The reviewable implementation commits before this verification record are:

1. `245ba43` — `chore(dimswe): archive superseded development surfaces`;
2. `c820b75` — `chore(dimswe): make reproduction entry points portable`; and
3. `da21c95` — `chore(dimswe): remove verified unused imports`.

The first commit uses 100%-similarity Git renames for the closed development
surfaces. The 12 destinations in `SECOND_PASS_DISPOSITION.tsv` all match their
recorded source SHA-256. No candidate was permanently discarded. The second
commit changes shell/path plumbing only. The third removes 24 verified-unused
import lines from 14 post-Chris modules; it changes no executable statement,
interface, equation, objective, network, optimizer setting, or configuration.

## Static and integrity checks actually run

- `ast.parse` accepted all 175 tracked Python files.
- All 165 tracked JSON files parsed successfully.
- `bash -n` accepted all 17 tracked scripts under `scripts/`.
- `git diff --check` reported no whitespace errors.
- All 12 archived files matched the SHA-256 recorded before their move.
- The shared reproduction-environment helper resolved this repository and the
  selected Python executable successfully without a machine-local default.
- Active source, configs, and runners contain no literal `/Users` or `/home`
  path. Eighty-four frozen result JSON files retain historical absolute output
  paths as immutable result provenance; chain-of-custody documents and
  byte-preserved archive files also intentionally retain historical paths.
- Eight `/tmp/...-no-output` strings in case builders are non-writing
  configuration sentinels, and ten `/tmp/...` runner templates are passed to
  `mktemp -d` for ephemeral caches. They are not accepted result locations and
  were retained as intentional scratch behavior.
- No active source imports an archived module. References from primary
  documentation now identify those paths explicitly as archived history.
- A conservative import/name scan found only unused imports in the original
  upstream `dimswe/timestepping.py` import block; those predate the Chris
  baseline and were not changed.
- A repository-local import check reported the legacy
  `dimswe/run_model_set.py -> dimswe.model` edge. Both the file and import
  predate `d0eb61598a2cb1049628c3cc054ab9a1f3143bf6`, so this pass retained it
  rather than guessing at an upstream repair.

`ruff`, `pyflakes`, and `shellcheck` are not installed in the available
environment. They were not installed for this audit.

## Tests actually run by Codex

The environment-independent subset was run without pytest's cache provider and
with bytecode generation disabled:

```text
python -m pytest -q -p no:cacheprovider \
  tests/test_resolved_hidden_c0_prep.py \
  tests/test_selected_test1b_plan.py \
  tests/test_test2b_rain_truth.py \
  tests/test_test2a_backend_offset_audit.py
42 passed in 0.61s
```

An attempted selected JAX test collection aborted with process status 134 while
`jaxlib` initialized its CPU backend, before a test ran. This is not recorded as
a test failure or pass, and the attempt was not repeated. A separate attempt to
collect `tests/test_test2a_problem_b_signed_water_budget.py` stopped with
`ModuleNotFoundError: firedrake`; no test from that file ran. Firedrake and
PyROL are absent from the available interpreter, so their runtime suites were
not run. JAX is installed but is not usable in this process environment.

The accepted H1/M2-Y regression remains the manual fact supplied by Arjun
Sharma, not a Codex test result:

```text
python -m pytest -q tests/test_test2a_h1_m2_equivalence.py
5 passed in 6.62 s
```

Arjun ran it in the recorded Firedrake environment on 2026-08-28.

No training, truth generation, scientific campaign, optimizer run, or B+
experiment was launched.

## Authoritative checkout invariant

Read-only checks use `GIT_OPTIONAL_LOCKS=0`. The expected unchanged values are:

- branch `dev/dimswe-learned-physics-framework`;
- HEAD `d2f5d66ecb5500aad24eca37280f8a52e22a250f`;
- short-status SHA-256
  `f6aad727ab0001a0976827a41bce322cf325ae0757a50bd4153aa7461754e076`;
- porcelain-v2 SHA-256
  `174a8eb8e7b8fb068b0935f2bcfb0b94d2cc6c8e2e7d2acc5fc3e5c785c4489d`;
  and
- linked-worktree `.git` pointer SHA-256
  `63137c597f563232672c667b32f4ae973c3adfd83f31abe00764f7e7d238b00b`.

These fingerprints are rechecked after the final collaborator documentation
commit; their presence here does not imply any write to the authoritative
checkout.
