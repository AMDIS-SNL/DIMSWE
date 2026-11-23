# Plan: Sphinx documentation for DIMSWE

## What the repo has now

- **API surface.** `dimswe/` has about 40 modules (~14.5k lines) plus `dimswe/plotting/`. `dimswe/__init__.py` re-exports most modules with `from .x import *`.
- **Docstring coverage is uneven.**
  - The newer elliptic and optimization modules have Google-style docstrings with `Args:` / `Returns:`: `elliptic*.py`, `optimize.py`, `timestepping.py`, `meshes.py`, `assembly.py`, `boundary_conditions.py`.
  - The core geometric-mechanics modules have none: `poisson_brackets.py`, `metric_brackets.py`, `hamiltonians.py`, `entropies.py`, `dynamics.py`, `dissipation.py`, `initial_conditions.py`, `operators.py`, `parameters.py`, `output.py`, `diagnostics.py`, `statistics.py`, `logger.py`.
- **Existing prose:**
  - `README.md` (design overview)
  - `workflows/README.md` (long, with anchors)
  - `docs/hessian_vector_products.md` and `docs/hvp_performance_plan.md`
  - `docs/hodge_laplacian.tex` (custom macros such as `\dd`, `\cod`, `\hodge`)
  - `TODO-elliptic.md`, `TODO-hvp.md` and `HANDOFF-hvp-opt.md` (planning notes, not published)
  - Commented `.cfg` files that are effectively the config reference
- **Constraints:**
  - Firedrake can't be pip-installed, so autodoc must either run in the Firedrake container CI already uses or mock Firedrake.
  - Per AGENTS.md, nothing can be built locally; only `py_compile` can be run.
- **Two repo issues to fix first:**
  - `.gitignore` contains unresolved merge-conflict markers (`<<<<<<< HEAD` … `>>>>>>> main`).
  - `auto-test.yml` uses `paths-ignore: 'doc/**'`, but the directory is `docs/`.

## Proposed design

**Tooling.** Sphinx with:
- `sphinx.ext.autodoc` and `autosummary` for the API pages
- `napoleon` to read the existing Google-style docstrings
- `mathjax`, `viewcode` and `intersphinx` (linking to Python, NumPy, SciPy, Firedrake, UFL and PETSc4py)
- `myst-parser`, so the existing Markdown docs are included rather than rewritten
- `sphinx-copybutton`

Theme: `furo` (recommended: simple, good math rendering) or `pydata-sphinx-theme`.

**Layout.** The existing Markdown notes stay in `docs/`; Sphinx goes in a subfolder so it doesn't mix with them:
```
docs/
  sphinx/                    (or docs/source + docs/Makefile)
    conf.py
    index.md
    getting_started.md       install (Firedrake env), run_model.py / plot_model.py, run the tests
    theory/
      geometric_mechanics.md Poisson/metric brackets, Hamiltonians, entropies, the CF/LP/H1 formulations
      hodge_laplacian.md     converted from hodge_laplacian.tex (replaces it; .tex and PDF are deleted)
      hessian_vector_products.md   included from docs/ via MyST {include}
    user_guide/
      configuration.md       one section per cfg block: mesh, model, spatial-discretization, timestepping, elliptic, ...
      time_stepping.md       RK and LieSplit schemes; adjoint, TLM and SOA steps
      forcings.md            physics, DG1 transport, hyperviscosity
      elliptic.md            families, boundary conditions, harmonic forms, solvers, eigenproblems
      optimization.md        scipy methods, gradients and HVPs
      workflows.md           includes workflows/README.md
    api/
      index.rst              grouped autosummary: core model, discretization, time stepping,
                             optimization, elliptic, I/O and diagnostics, plotting, workflows
    developer/
      contributing.md        AGENTS.md conventions, docstring style, how to build the docs
```

**`conf.py` details:**
- Add the repo root to `sys.path`.
- Use `autosummary_generate = True`, generating one page per module (not from the package's `import *` re-exports, which would duplicate everything).
- Set `autodoc_mock_imports = ["firedrake", "ufl", "petsc4py", "pyop2", "FIAT", "finat", "slepc4py", "h5py"]` behind an env flag. Mocking allows quick builds without Firedrake; the CI build runs unmocked in the Firedrake container.
- Port the LaTeX macros from `hodge_laplacian.tex` into `mathjax3_config` so the theory pages can use `\dd`, `\cod` and the rest.
- Include `workflows/` in the API docs. It's a package with `__init__.py`.
- Leave `hamiltonian_neural_odes/` and `ode_adjoint/` out, or put them in a short "legacy" section.
- Exclude the planning notes with `exclude_patterns` (`TODO-*.md`, `HANDOFF-*.md`), and strip any README or workflow-README links to them when the files are included.

## Steps

1. **Housekeeping.** Resolve the `.gitignore` conflict and add `docs/_build/` to it. Fix the CI `paths-ignore` to `docs/**`.
2. **Scaffold.** Add `conf.py`, `index.md`, a `Makefile`, and a `docs` optional dependency in `pyproject.toml` (sphinx, furo, myst-parser, sphinx-copybutton).
3. **API reference.** Write `api/index.rst` with grouped autosummary tables and a custom template that shows members and inheritance. Inheritance matters here because of hierarchies like `BracketBase` → `PoissonBracket` → the concrete brackets.
4. **Bring in existing prose.** Make README sections the overview and getting-started pages, and include `hessian_vector_products.md` and `workflows/README.md` with MyST.
5. **Convert the Hodge Laplacian notes.** Fully convert `hodge_laplacian.tex` to `theory/hodge_laplacian.md`: a first pass with pandoc, then manual cleanup.
   - Turn sections into headings, equations into `$$ ... $$` / `{math}` blocks with labels, and `\ref`/`\eqref` into MyST cross-references.
   - Turn theorem/definition environments into admonitions, and `booktabs` tables into Markdown/list-tables.
   - Rely on the macros ported into `mathjax3_config`.
   - From then on the web page is the only maintained version. Once the conversion has been checked against the PDF, delete `docs/hodge_laplacian.tex`, `docs/hodge_laplacian.pdf`, and the LaTeX build artifacts (`.aux`, `.log`, `.out`, `.toc`). No PDF download is offered.
   - Point the existing references at the web page instead of `docs/hodge_laplacian.tex`: `README.md:37`, `elliptic.cfg:1` and `workflows/README.md:583`.
   - Remove the LaTeX entries from `.gitignore` (`*.aux`, `*.log`, `*.out`, `*.toc`, `docs/*.pdf`) while resolving the merge conflict in step 1, unless another LaTeX source still needs them.
6. **Configuration reference.** Document each cfg block with its keys, types, defaults and allowed values, using the comments in the root `.cfg` files and the code that reads them. This is probably the most useful new page for users.
7. **Full docstring backfill.** Write a module docstring and Google-style docstrings (`Args:` / `Returns:` / `Raises:`) for every public and nontrivial private function, class and method in `dimswe/`, `dimswe/plotting/` and `workflows/`. AGENTS.md requires this anyway. Put the math in `.. math::` blocks, and add a package docstring for `dimswe`.
   Work in this order, one module per change, checking each with `py_compile`:
   1. Core geometric mechanics: `operators.py`, `poisson_brackets.py`, `metric_brackets.py`, `hamiltonians.py`, `entropies.py`, `dynamics.py`, `dissipation.py`, `models.py`, `variables.py`.
   2. Discretization and physics: `transport_operators.py`, `ufl_helpers.py`, `initial_conditions.py`, `physics.py`, `positivity.py`, `topology.py`, `bernstein.py`.
   3. Infrastructure: `parameters.py`, `output.py`, `diagnostics.py`, `statistics.py`, `logger.py`, `numpy_helpers.py`, `profiling.py`.
   4. Plotting: `dimswe/plotting/*`.
   5. Workflows: `workflows/*.py`.
   6. Completing the partially documented modules: `timestepping.py`, `optimize.py`, `assembly.py`, `meshes.py` and the `elliptic*` modules.

   Only docstrings and comments change; no behavior changes. Where the code has questions (for example the `#WHY IS SV LIE DERIVATIVE FAILING?` notes), leave them as they are and record them rather than "fixing" them.
8. **CI.** Add `.github/workflows/docs.yml`, running in the same Firedrake container as the tests:
   - Run `sphinx-build -W --keep-going -b html` so warnings fail the build.
   - On pull requests: build only, and upload the HTML as an artifact for review.
   - On pushes to `main`: build, then deploy with `actions/upload-pages-artifact` + `actions/deploy-pages`, with `permissions: pages: write, id-token: write` and a `github-pages` environment.
   - One-time manual setup: in the repository settings, set Pages → Source to "GitHub Actions".
   - Run `-b linkcheck` on a weekly schedule, not on every push.
9. **Tests.** AGENTS.md asks for an integration test for new pipeline paths, so add `tests/integration/test_docs_build.py`. It builds the docs with mocked imports and is skipped if Sphinx isn't installed, which catches broken autodoc imports and cross-references.

## Decisions

- **Hosting.** GitHub Pages, deployed from CI (step 8). No Read the Docs.
- **Docstring backfill.** Full backfill of every module in `dimswe/` and `workflows/` is in scope (step 7), not a follow-up.
- **Hodge Laplacian notes.** Fully converted to a web page (step 5), which becomes the only maintained version. The `.tex`, the PDF and the LaTeX build artifacts are dropped.
- **TODO and HANDOFF files.** Kept out of the documentation: `TODO-*.md` and `HANDOFF-*.md` are not included or linked, and `conf.py` lists them in `exclude_patterns`.

## Verification note

Nothing can be built locally under the AGENTS.md rules, so the first real Sphinx build happens in CI. Locally only `py_compile` can check the new Python.
