# Final Track 1 verification

Date: 2026-08-28 (America/Denver)

## Collaborator repository

- path:
  `/Users/arjunsharma/Documents/SandiaProjects/4-LDRDAMDIS/DIMSWE-collaborator-track1-20260828`
- branch: `collaborator/track1-closeout`
- Git directory and common directory: `.git` inside the collaborator root
- baseline `d0eb61598a2cb1049628c3cc054ab9a1f3143bf6`: present and ancestral
- fetch provenance: `https://github.com/AMDIS-SNL/DIMSWE.git`
- push target: `DISABLE_PUSH`
- object alternates: absent
- `git fsck --full --strict`: no corruption; only unreferenced dangling
  commits/blobs were reported

The reviewable commits made before this final verification record were:

1. `69aa2ae` — `feat(dimswe): preserve accepted learned-physics state`
2. `18e2b2b` — `data(dimswe): retain compact canonical experiment artifacts`
3. `1dfb9ef` — `docs(dimswe): add Track 1 handoff and provenance`

## Evidence integrity after cleanup

The disposition table identifies 6,872 retained evidence files and 76 generated
files selected for exclusion. Every retained path was rehashed after cleanup:

```text
checked 6872
SHA-256 matches 6872
failures 0
```

No `.DS_Store`, `.pyc`, or `.pyo` file remains in the collaborator tree outside
Git metadata. No scientific evidence file was removed. Large local-only outputs
remain under their original `external-results/` paths.

## Authoritative checkout unchanged

The read-only check used `GIT_OPTIONAL_LOCKS=0`. Final values were:

- branch: `dev/dimswe-learned-physics-framework`
- HEAD: `d2f5d66ecb5500aad24eca37280f8a52e22a250f`
- `git status --short` SHA-256:
  `f6aad727ab0001a0976827a41bce322cf325ae0757a50bd4153aa7461754e076`
- `git status --porcelain=v2 -uall` SHA-256:
  `174a8eb8e7b8fb068b0935f2bcfb0b94d2cc6c8e2e7d2acc5fc3e5c785c4489d`
- `.git` pointer SHA-256:
  `63137c597f563232672c667b32f4ae973c3adfd83f31abe00764f7e7d238b00b`
- fetch: GitHub AMDIS-SNL/DIMSWE; push: `DISABLE_PUSH`
- collaborator branch present in authoritative/common Git metadata: no

These values exactly match the frozen gate.

## Checks actually run by Codex

- all repository JSON files parsed with `jq empty`;
- all retained shell campaign scripts passed `bash -n`;
- all Python files under `dimswe/`, `tests/`, and `scripts/` passed static
  `ast.parse` syntax parsing;
- staged and final changes passed `git diff --check`;
- selected artifact paths all existed and none exceeded 100 MB;
- critical manifests and A/B/C comparison artifacts were SHA-256 checked;
- independent Git location, remote policy, ancestry, absence of alternates,
  repository integrity, and clean final status were checked.

Codex did not run pytest, execute the DIMSWE solver, install dependencies,
launch training, regenerate truth, or run B+ during Track 1.

The known H1/M2-Y test fact remains a manual result supplied by Arjun Sharma:

```text
python -m pytest -q tests/test_test2a_h1_m2_equivalence.py
5 passed in 6.62 s
```

It was run in the recorded Firedrake environment on 2026-08-28, not by Codex.
