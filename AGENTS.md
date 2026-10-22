# Repository Guidelines

## Project Structure & Module Organization
- `dimswe/` contains the main Python package.
  - `dimswe/plotting` contains the plotting routines.
- `hamiltonian_neural_odes/` contains some old code for testing structure-preserving time integrators for Hamiltonian ODEs
- `ode_adjoint/` contains some old code for testing ODE-constrainted optimization using adjoints
- `tests/` is split into `tests/unit/` and `tests/integration/`. 
- `runs/` contains data from various simulations, and is not part of the GitHub repository

## Build, Test, and Development Commands
- Only run py_compile, installation and testing is not possible.

## Coding Style & Naming Conventions
- Use Python conventions: 4-space indentation, snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants.
- Keep module names lowercase with underscores (for example, `metric_brackets.py`).
- Favor small, composable functions in `dimswe`
- Match existing typing and dataclass/pydantic patterns when extending configuration or model APIs.
- Every method must have good comments and doc strings
- Add useful comments for nontrivial code segments.

## Testing Guidelines
- Add unit tests for isolated logic and integration tests for cross-module workflows/CLI behavior.
- Name tests as `test_<feature>.py` and test functions as `test_<behavior>()`.
- Place shared fixtures in `tests/conftest.py`.
- For new CLI or pipeline paths, include at least one integration test under `tests/integration/`.

## Commit & Pull Request Guidelines
- Do not commit pull or push.

## Configuration & Safety Notes
- Avoid committing large generated artifacts, model checkpoints, or local caches.
- Prefer deterministic settings (explicit seeds) in training/data-gen code when adding tests.
