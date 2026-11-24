# Getting started

## Installation

DIMSWE is built on [Firedrake](https://www.firedrakeproject.org/), which cannot be installed from PyPI on
its own. Follow the [Firedrake installation instructions](https://www.firedrakeproject.org/install.html)
first and activate the resulting environment. The shell wrappers in the repository (`run_model.sh`,
`plot_model.sh`, `workflows/*.sh`) assume it lives in `~/venv-firedrake`.

The scripts run directly from the repository root, so installing DIMSWE itself is optional. To make the
`dimswe` package importable from anywhere, install it in editable mode inside the Firedrake environment:

```bash
pip install -e .
```

This also installs the remaining runtime dependencies (NumPy, SciPy, Matplotlib, h5py and PyYAML). To
build this documentation, add the `docs` extra (`pip install -e ".[docs]"`) and run `make -C docs html`;
the result is in `docs/_build/html`.

## Running a model

Every run is described by a YAML configuration file (cfg). Run one from the repository root with:

```bash
python run_model.py mtswe-cf.cfg
```

or in parallel with MPI:

```bash
mpiexec -n 4 python run_model.py mtswe-cf.cfg
```

The repository root contains example cfgs:

| cfg | Model |
| --- | --- |
| `mtswe-cf.cfg`, `mtswe-cf-physics.cfg` | Moist thermal shallow water, CF formulation, without / with three-way physics |
| `mtswe-lp.cfg`, `mtswe-lp-physics.cfg` | Moist thermal shallow water, LP formulation, without / with three-way physics |
| `mtswe-h1.cfg`, `mtswe-h1-physics.cfg` | Moist thermal shallow water, H1 formulation, without / with three-way physics |
| `elliptic.cfg` | Hodge Laplacian problem for a k-form |
| `elliptic-eigen.cfg` | Hodge Laplacian eigenproblem |

Output is written to the current directory, named by `output.outfile_name` in the cfg: a checkpoint file
`<outfile_name>.h5` and a VTK series `<outfile_name>.pvd`, which can be opened in
[ParaView](https://www.paraview.org/). `output.output_freq` and `output.stat_freq` set how often fields and
statistics are written.

`run_model.sh <cfg> <nprocs>` does the same in parallel with PETSc's `-log_view` profiling and turns the
result into `flamegraph.svg`.

## Plotting results

After a run, plot it with the same cfg:

```bash
python plot_model.py mtswe-cf.cfg
```

The plots (PNG images and WEBP movies) are written to the current directory. `plot_model.py` also accepts
the `summary.yaml` of an elliptic convergence sweep (`workflows/elliptic_convergence.py`) and plots the
observed convergence rates. Sweeps, batch runs, comparison plots and parameter optimization are covered in
{doc}`user_guide/workflows`.

## Running the tests

% The rest of this section is the "Running the tests" section of the repository README, included from
% its docs-tests-start marker to the end of the file. Edit the README, not this file.

```{include} ../../README.md
:start-after: <!-- docs-tests-start: included in the Sphinx docs (docs/sphinx/getting_started.md) -->
```

## Next steps

- {doc}`overview`: the formulations, forcings, time integrators and optimization methods.
- {doc}`user_guide/workflows`: parameter sweeps, batch runs, comparisons and optimization.
- {doc}`theory/hessian_vector_products`: how gradients and Hessian-vector products are computed.
- {doc}`api/index`: the full API reference.
