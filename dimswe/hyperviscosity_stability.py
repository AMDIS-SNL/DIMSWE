"""Exact discrete stability audit for the deployed hyperviscosity Euler child.

The production forms are not modified.  For each active field they implement

    M Q = -K x,
    M F = c0 r**s K Q,
    x_plus = x + dt F,

because :class:`GeneralRK` negates ``model.rhs`` when it forms the stage
right-hand side.  Eliminating ``Q`` and ``F`` gives

    x_plus = (I - dt c0 H) x,
    H = r**s (M**-1 K)**2,
    r = max(dx/order, dy/order).

If ``K phi = mu M phi``, then ``lambda(H)=r**s mu**2``.  Explicit Euler is
non-growing on this positive-semidefinite operator exactly when
``dt*c0*lambda_max <= 2``.

Production matrices remain PETSc AIJ/sparse.  In serial, their CSR data are
passed to symmetric ARPACK Lanczos after the exact GLL diagonal-mass similarity
transform ``B=M**-1/2 K M**-1/2``.  A residual is reported for the largest
Ritz pair and the maximum absolute row sum supplies an independent certified
upper bound.  Dense conversion is forbidden except in the explicitly bounded
tiny-oracle verification routine.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from firedrake import TrialFunction, TestFunction, assemble, grad, inner
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .logger import EmptyLogger
from .models import get_model
from .resolved_hidden_c0 import ResolvedPilotConfiguration, write_json_record
from .resolved_hidden_c0_driver import (
    build_resolved_hidden_c0_case,
    resolved_hidden_c0_parameters,
)


ACTIVE_FIELDS = ("v", "h", "S")
FORMAT_VERSION = 1


def _positive_integer(name, value):
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_float(name, value, *, positive=False):
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class SpectralSolverConfiguration:
    tolerance: float = 1.0e-10
    max_iterations: int = 20000
    seed: int = 1729
    mass_diagonal_tolerance: float = 1.0e-13
    symmetry_tolerance: float = 1.0e-12

    def __post_init__(self):
        _finite_float("tolerance", self.tolerance, positive=True)
        _positive_integer("max_iterations", self.max_iterations)
        if not isinstance(self.seed, Integral) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        _finite_float(
            "mass_diagonal_tolerance",
            self.mass_diagonal_tolerance,
            positive=True,
        )
        _finite_float("symmetry_tolerance", self.symmetry_tolerance, positive=True)


@dataclass(frozen=True)
class FieldEigenvalueEstimate:
    field: str
    space: str
    value_size: int
    dofs: int
    mass_nonzeros: int
    stiffness_nonzeros: int
    mass_diagonal_min: float
    mass_diagonal_max: float
    relative_mass_offdiagonal_max: float
    relative_stiffness_symmetry_defect: float
    r: float
    s: float
    laplacian_mu_max: float
    laplacian_mu_residual_relative: float
    laplacian_mu_certified_upper_bound: float
    lambda_max: float
    lambda_max_certified_upper_bound: float
    eigensolve_wall_time_seconds: float
    dense_laplacian_mu_max: float | None = None
    dense_lambda_max: float | None = None
    dense_relative_error: float | None = None


@dataclass(frozen=True)
class EulerStabilityRow:
    case: str
    nx: int
    ny: int
    field: str
    space: str
    dofs: int
    dt: float
    c0: float
    s: float
    r: float
    laplacian_mu_max: float
    laplacian_mu_residual_relative: float
    lambda_max: float
    lambda_max_certified_upper_bound: float
    sigma: float
    sigma_certified_upper_bound: float
    largest_mode_amplification: float
    euler_amplification_bound: float
    dt_max: float
    dt_max_certified_conservative: float
    safety_factor: float
    recommended_dt: float
    ritz_detects_instability: bool
    upper_bound_certifies_stability: bool
    stability_classification: str


@dataclass(frozen=True)
class TinyOracleVerification:
    case: str
    nx: int
    ny: int
    dt: float
    c0: float
    s: float
    field_comparisons: tuple[dict[str, Any], ...]
    deployed_child_relative_errors: dict[str, float]
    inactive_field_relative_errors: dict[str, float]
    passed: bool


@dataclass(frozen=True)
class _SparseFieldOperator:
    field: str
    space_description: str
    value_size: int
    mass: Any
    stiffness: Any
    mass_diagonal: np.ndarray
    symmetric_laplacian: Any
    relative_mass_offdiagonal_max: float
    relative_stiffness_symmetry_defect: float


def _max_abs_sparse(matrix) -> float:
    return 0.0 if matrix.nnz == 0 else float(np.max(np.abs(matrix.data)))


def _petsc_csr(form):
    firedrake_matrix = assemble(form, mat_type="aij")
    petsc_matrix = firedrake_matrix.petscmat
    if petsc_matrix.comm.getSize() != 1:
        raise RuntimeError("hyperviscosity stability audit is serial-only")
    row_pointer, columns, values = petsc_matrix.getValuesCSR()
    result = sparse.csr_matrix(
        (
            np.asarray(values, dtype=np.float64).copy(),
            np.asarray(columns, dtype=np.int32).copy(),
            np.asarray(row_pointer, dtype=np.int32).copy(),
        ),
        shape=petsc_matrix.getSize(),
    )
    result.sum_duplicates()
    result.sort_indices()
    return result


def _field_space_map(model):
    variables = model.dynamics.variableset
    spaces = {
        name: space
        for name, space in zip(variables.varlist, variables.spacelist)
        if name in ACTIVE_FIELDS
    }
    if tuple(spaces) != ACTIVE_FIELDS:
        raise RuntimeError(
            f"production hyperviscosity active fields changed: {tuple(spaces)}"
        )
    matching = [
        term
        for term in model.dynamics.forcing_terms
        if term.name == "hyperviscosity"
    ]
    if len(matching) != 1 or tuple(matching[0].varlist) != ACTIVE_FIELDS:
        raise RuntimeError("production hyperviscosity field contract changed")
    for name, expected in zip(matching[0].varlist, matching[0].spacelist):
        if spaces[name] != expected:
            raise RuntimeError(f"production hyperviscosity space changed for {name}")
    return spaces, matching[0]


def _assemble_sparse_field_operator(model, field, space, configuration):
    trial = TrialFunction(space)
    test = TestFunction(space)
    mass = _petsc_csr(inner(test, trial) * model.spaces.dx)
    stiffness = _petsc_csr(inner(grad(test), grad(trial)) * model.spaces.dx)
    mass_diagonal = np.asarray(mass.diagonal(), dtype=np.float64)
    if np.any(~np.isfinite(mass_diagonal)) or np.any(mass_diagonal <= 0.0):
        raise RuntimeError(f"{field} mass diagonal is not finite positive")
    offdiagonal = mass - sparse.diags(mass_diagonal, format="csr")
    mass_scale = max(float(np.max(np.abs(mass_diagonal))), np.finfo(float).tiny)
    relative_mass_offdiagonal = _max_abs_sparse(offdiagonal) / mass_scale
    if relative_mass_offdiagonal > configuration.mass_diagonal_tolerance:
        raise RuntimeError(
            f"{field} deployed GLL mass is not diagonal: relative offdiagonal "
            f"{relative_mass_offdiagonal}"
        )
    stiffness_scale = max(_max_abs_sparse(stiffness), np.finfo(float).tiny)
    relative_symmetry = _max_abs_sparse(stiffness - stiffness.transpose()) / (
        stiffness_scale
    )
    if relative_symmetry > configuration.symmetry_tolerance:
        raise RuntimeError(
            f"{field} stiffness is not symmetric: relative defect "
            f"{relative_symmetry}"
        )
    inverse_sqrt_mass = sparse.diags(
        1.0 / np.sqrt(mass_diagonal), format="csr"
    )
    symmetric_laplacian = (
        inverse_sqrt_mass @ stiffness @ inverse_sqrt_mass
    ).tocsr()
    value_shape = space.value_shape
    value_size = int(np.prod(value_shape)) if value_shape else 1
    return _SparseFieldOperator(
        field=field,
        space_description=str(space.ufl_element()),
        value_size=value_size,
        mass=mass,
        stiffness=stiffness,
        mass_diagonal=mass_diagonal,
        symmetric_laplacian=symmetric_laplacian,
        relative_mass_offdiagonal_max=relative_mass_offdiagonal,
        relative_stiffness_symmetry_defect=relative_symmetry,
    )


def _largest_eigenvalue(
    operator,
    configuration,
    *,
    dense_oracle_max_dofs=None,
):
    matrix = operator.symmetric_laplacian
    size = matrix.shape[0]
    if size < 2:
        raise RuntimeError("spectral audit requires at least two degrees of freedom")
    random = np.random.default_rng(configuration.seed)
    initial = random.standard_normal(size)
    started = perf_counter()
    values, vectors = eigsh(
        matrix,
        k=1,
        which="LA",
        v0=initial,
        tol=configuration.tolerance,
        maxiter=configuration.max_iterations,
        return_eigenvectors=True,
    )
    elapsed = perf_counter() - started
    mu = float(values[-1])
    vector = np.asarray(vectors[:, -1], dtype=np.float64)
    residual = matrix @ vector - mu * vector
    relative_residual = float(
        np.linalg.norm(residual)
        / max(abs(mu) * np.linalg.norm(vector), np.finfo(float).tiny)
    )
    row_sum_upper = float(
        np.max(np.asarray(np.abs(matrix).sum(axis=1)).ravel())
    )
    row_sum_upper = max(row_sum_upper, abs(mu))
    dense_mu = None
    if dense_oracle_max_dofs is not None:
        maximum = _positive_integer(
            "dense_oracle_max_dofs", dense_oracle_max_dofs
        )
        if size > maximum:
            raise RuntimeError(
                f"dense oracle refuses {size} dofs above explicit limit {maximum}"
            )
        dense_mu = float(np.linalg.eigvalsh(matrix.toarray())[-1])
    return mu, relative_residual, row_sum_upper, elapsed, dense_mu


def _build_lightweight_model(configuration):
    parameters = resolved_hidden_c0_parameters(configuration)
    model = get_model(parameters, EmptyLogger(), has_dynamics_statistics=False)
    if model.mesh.comm.size != 1:
        raise RuntimeError("hyperviscosity stability audit is serial-only")
    return model


def estimate_field_eigenvalues(
    configuration: ResolvedPilotConfiguration,
    solver_configuration=SpectralSolverConfiguration(),
    *,
    dense_oracle_max_dofs=None,
    model=None,
):
    """Estimate every active-field lambda_max without dense production matrices."""
    if not isinstance(configuration, ResolvedPilotConfiguration):
        raise TypeError("configuration must be ResolvedPilotConfiguration")
    if not isinstance(solver_configuration, SpectralSolverConfiguration):
        raise TypeError("solver_configuration must be SpectralSolverConfiguration")
    owned_model = model or _build_lightweight_model(configuration)
    spaces, hyperviscosity = _field_space_map(owned_model)
    exact_r = float(hyperviscosity.factor)
    expected_r = float(
        max(
            owned_model.spaces.mesh.dx / owned_model.spaces.order,
            owned_model.spaces.mesh.dy / owned_model.spaces.order,
        )
    )
    if exact_r != expected_r:
        raise RuntimeError("production hyperviscosity r scaling changed")
    cache = {}
    results = []
    for field in ACTIVE_FIELDS:
        space = spaces[field]
        identity = id(space)
        if identity not in cache:
            operator = _assemble_sparse_field_operator(
                owned_model, field, space, solver_configuration
            )
            spectral = _largest_eigenvalue(
                operator,
                solver_configuration,
                dense_oracle_max_dofs=dense_oracle_max_dofs,
            )
            cache[identity] = (operator, spectral)
        operator, spectral = cache[identity]
        mu, residual, upper, elapsed, dense_mu = spectral
        lambda_max = exact_r**configuration.s * mu * mu
        lambda_upper = exact_r**configuration.s * upper * upper
        dense_lambda = (
            None
            if dense_mu is None
            else exact_r**configuration.s * dense_mu * dense_mu
        )
        dense_relative = (
            None
            if dense_mu is None
            else abs(mu - dense_mu) / max(abs(dense_mu), np.finfo(float).tiny)
        )
        results.append(
            FieldEigenvalueEstimate(
                field=field,
                space=operator.space_description,
                value_size=operator.value_size,
                dofs=operator.symmetric_laplacian.shape[0],
                mass_nonzeros=operator.mass.nnz,
                stiffness_nonzeros=operator.stiffness.nnz,
                mass_diagonal_min=float(np.min(operator.mass_diagonal)),
                mass_diagonal_max=float(np.max(operator.mass_diagonal)),
                relative_mass_offdiagonal_max=(
                    operator.relative_mass_offdiagonal_max
                ),
                relative_stiffness_symmetry_defect=(
                    operator.relative_stiffness_symmetry_defect
                ),
                r=exact_r,
                s=configuration.s,
                laplacian_mu_max=mu,
                laplacian_mu_residual_relative=residual,
                laplacian_mu_certified_upper_bound=upper,
                lambda_max=lambda_max,
                lambda_max_certified_upper_bound=lambda_upper,
                eigensolve_wall_time_seconds=elapsed,
                dense_laplacian_mu_max=dense_mu,
                dense_lambda_max=dense_lambda,
                dense_relative_error=dense_relative,
            )
        )
    return tuple(results)


def stability_rows(
    case,
    nx,
    ny,
    dt,
    c0_values,
    estimates,
    *,
    safety_factor=0.8,
):
    """Apply the exact Euler interval [-2,0] to field eigenvalue estimates."""
    step = _finite_float("dt", dt, positive=True)
    safety = _finite_float("safety_factor", safety_factor, positive=True)
    if safety >= 1.0:
        raise ValueError("safety_factor must be below one")
    rows = []
    for estimate in estimates:
        for candidate in c0_values:
            c0 = _finite_float("c0", candidate, positive=True)
            sigma = step * c0 * estimate.lambda_max
            sigma_upper = (
                step * c0 * estimate.lambda_max_certified_upper_bound
            )
            high_amplification = abs(1.0 - sigma)
            amplification_bound = max(1.0, abs(1.0 - sigma_upper))
            dt_max = 2.0 / (c0 * estimate.lambda_max)
            conservative = 2.0 / (
                c0 * estimate.lambda_max_certified_upper_bound
            )
            ritz_unstable = bool(sigma > 2.0)
            upper_stable = bool(sigma_upper <= 2.0)
            if ritz_unstable:
                classification = "unstable: Ritz mode lies outside Euler interval"
            elif upper_stable:
                classification = "stable: certified upper bound lies inside Euler interval"
            else:
                classification = (
                    "inconclusive bound: converged Ritz estimate is stable but "
                    "row-sum upper bound crosses Euler limit"
                )
            rows.append(
                EulerStabilityRow(
                    case=str(case),
                    nx=int(nx),
                    ny=int(ny),
                    field=estimate.field,
                    space=estimate.space,
                    dofs=estimate.dofs,
                    dt=step,
                    c0=c0,
                    s=estimate.s,
                    r=estimate.r,
                    laplacian_mu_max=estimate.laplacian_mu_max,
                    laplacian_mu_residual_relative=(
                        estimate.laplacian_mu_residual_relative
                    ),
                    lambda_max=estimate.lambda_max,
                    lambda_max_certified_upper_bound=(
                        estimate.lambda_max_certified_upper_bound
                    ),
                    sigma=sigma,
                    sigma_certified_upper_bound=sigma_upper,
                    largest_mode_amplification=high_amplification,
                    euler_amplification_bound=amplification_bound,
                    dt_max=dt_max,
                    dt_max_certified_conservative=conservative,
                    safety_factor=safety,
                    recommended_dt=safety * conservative,
                    ritz_detects_instability=ritz_unstable,
                    upper_bound_certifies_stability=upper_stable,
                    stability_classification=classification,
                )
            )
    return tuple(rows)


def build_stability_table(
    grids,
    c0_values,
    *,
    case="doublevortex",
    dt=400.0,
    s=3.2,
    safety_factor=0.8,
    solver_configuration=SpectralSolverConfiguration(),
):
    """Build a serial sparse stability table; this advances no model state."""
    candidates = tuple(c0_values)
    if not candidates:
        raise ValueError("c0_values must not be empty")
    rows = []
    spectra = []
    for nx, ny in grids:
        configuration = ResolvedPilotConfiguration(
            case=case,
            nx=_positive_integer("nx", nx),
            ny=_positive_integer("ny", ny),
            dt=dt,
            nsteps=1,
            output_stride=1,
            c0=float(candidates[0]),
            s=s,
            output_directory="unused-hyperviscosity-stability-diagnostic",
        )
        estimates = estimate_field_eigenvalues(
            configuration, solver_configuration
        )
        spectra.extend(
            {
                "case": case,
                "nx": int(nx),
                "ny": int(ny),
                **asdict(estimate),
            }
            for estimate in estimates
        )
        rows.extend(
            stability_rows(
                case,
                nx,
                ny,
                dt,
                candidates,
                estimates,
                safety_factor=safety_factor,
            )
        )
    return {
        "format_version": FORMAT_VERSION,
        "diagnostic": "exact deployed hyperviscosity Euler stability",
        "advances_model_state": False,
        "operator": {
            "auxiliary": "M Q = -K x",
            "stage": "M F = c0 r^s K Q",
            "update": "x_plus = x + dt F = (I-dt*c0*H)x",
            "H": "r^s (M^-1 K)^2",
            "r": "max(mesh.dx/order, mesh.dy/order)",
            "active_fields": ACTIVE_FIELDS,
            "inactive_fields": ("Qv", "Qc", "Qr"),
            "euler_condition": "dt*c0*lambda_max <= 2",
        },
        "spectral_method": {
            "production_storage": "PETSc AIJ converted to serial sparse CSR",
            "similarity_transform": "B=M^-1/2 K M^-1/2",
            "solver": "scipy.sparse.linalg.eigsh symmetric ARPACK Lanczos",
            "residual": "||Bv-mu*v||/(|mu|*||v||)",
            "certified_upper_bound": "maximum absolute row sum of symmetric B",
            "dense_production_matrices": False,
            "solver_configuration": asdict(solver_configuration),
        },
        "safety_policy": {
            "factor": float(safety_factor),
            "recommended_dt": (
                "safety_factor times dt_max from the certified row-sum "
                "upper bound"
            ),
        },
        "field_spectra": tuple(spectra),
        "rows": tuple(asdict(row) for row in rows),
    }


def verify_tiny_dense_oracle(
    *,
    case="doublevortex",
    nx=2,
    ny=2,
    dt=100.0,
    c0=0.14,
    s=3.2,
    solver_configuration=SpectralSolverConfiguration(),
    dense_oracle_max_dofs=512,
    relative_tolerance=2.0e-10,
):
    """Check sparse Lanczos and the actual child action on an explicitly tiny case."""
    tolerance = _finite_float(
        "relative_tolerance", relative_tolerance, positive=True
    )
    configuration = ResolvedPilotConfiguration(
        case=case,
        nx=nx,
        ny=ny,
        dt=dt,
        nsteps=1,
        output_stride=1,
        c0=c0,
        s=s,
        output_directory="unused-tiny-hyperviscosity-oracle",
    )
    production_case = build_resolved_hidden_c0_case(configuration)
    estimates = estimate_field_eigenvalues(
        configuration,
        solver_configuration,
        dense_oracle_max_dofs=dense_oracle_max_dofs,
        model=production_case.model,
    )
    spaces, hyperviscosity = _field_space_map(production_case.model)
    operators = {
        field: _assemble_sparse_field_operator(
            production_case.model,
            field,
            spaces[field],
            solver_configuration,
        )
        for field in ACTIVE_FIELDS
    }
    state = production_case.new_state("hyperviscosity_tiny_oracle_state")
    state.assign(0)
    random = np.random.default_rng(solver_configuration.seed)
    for index, field in enumerate(ACTIVE_FIELDS):
        with state.sub(index).dat.vec as vector:
            vector.array[:] = random.standard_normal(vector.getSize())
    input_values = {
        field: _subfunction_values(state.sub(index))
        for index, field in enumerate(ACTIVE_FIELDS)
    }
    with production_case.physical_c0(c0):
        deployed = production_case.helper.hyper_helper.take_forward_step_cached(
            state, production_case.t0, dt
        ).state_out
    deployed_errors = {}
    for index, field in enumerate(ACTIVE_FIELDS):
        operator = operators[field]
        values = input_values[field]
        first = operator.stiffness @ values / operator.mass_diagonal
        second = operator.stiffness @ first / operator.mass_diagonal
        predicted = values - dt * c0 * hyperviscosity.factor**s * second
        actual = _subfunction_values(deployed.sub(index))
        deployed_errors[field] = float(
            np.linalg.norm(predicted - actual)
            / max(np.linalg.norm(actual), np.finfo(float).tiny)
        )
    inactive_errors = {}
    for index, field in enumerate(("Qv", "Qc", "Qr"), start=3):
        initial = _subfunction_values(state.sub(index))
        actual = _subfunction_values(deployed.sub(index))
        inactive_errors[field] = float(
            np.linalg.norm(actual - initial)
            / max(np.linalg.norm(initial), 1.0)
        )
    comparisons = tuple(
        {
            "field": estimate.field,
            "dofs": estimate.dofs,
            "sparse_laplacian_mu_max": estimate.laplacian_mu_max,
            "dense_laplacian_mu_max": estimate.dense_laplacian_mu_max,
            "sparse_lambda_max": estimate.lambda_max,
            "dense_lambda_max": estimate.dense_lambda_max,
            "relative_error": estimate.dense_relative_error,
            "passed": bool(estimate.dense_relative_error <= tolerance),
        }
        for estimate in estimates
    )
    passed = (
        all(item["passed"] for item in comparisons)
        and all(value <= tolerance for value in deployed_errors.values())
        and all(value <= tolerance for value in inactive_errors.values())
    )
    return TinyOracleVerification(
        case=case,
        nx=int(nx),
        ny=int(ny),
        dt=float(dt),
        c0=float(c0),
        s=float(s),
        field_comparisons=comparisons,
        deployed_child_relative_errors=deployed_errors,
        inactive_field_relative_errors=inactive_errors,
        passed=bool(passed),
    )


def _subfunction_values(function):
    with function.dat.vec_ro as vector:
        return np.asarray(vector.array_r, dtype=np.float64).copy()


def _grid(value):
    pieces = value.lower().split("x")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("grid must have form NXxNY")
    try:
        return _positive_integer("nx", int(pieces[0])), _positive_integer(
            "ny", int(pieces[1])
        )
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _write_csv(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    records = tuple(rows)
    if not records:
        raise ValueError("CSV stability table is empty")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(destination)


def _parser():
    parser = argparse.ArgumentParser(
        description="Sparse exact-child hyperviscosity Euler stability audit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    table = subparsers.add_parser("table")
    table.add_argument("--grid", nargs="+", type=_grid, required=True)
    table.add_argument("--case", default="doublevortex")
    table.add_argument("--dt", type=float, required=True)
    table.add_argument("--c0-values", nargs="+", type=float, required=True)
    table.add_argument("--s", type=float, default=3.2)
    table.add_argument("--safety-factor", type=float, default=0.8)
    table.add_argument("--tolerance", type=float, default=1.0e-10)
    table.add_argument("--max-iterations", type=int, default=20000)
    table.add_argument("--seed", type=int, default=1729)
    table.add_argument("--output", required=True)
    table.add_argument("--csv")
    verify = subparsers.add_parser("verify-tiny")
    verify.add_argument("--grid", type=_grid, default=(2, 2))
    verify.add_argument("--case", default="doublevortex")
    verify.add_argument("--dt", type=float, default=100.0)
    verify.add_argument("--c0", type=float, default=0.14)
    verify.add_argument("--s", type=float, default=3.2)
    verify.add_argument("--tolerance", type=float, default=1.0e-10)
    verify.add_argument("--max-iterations", type=int, default=20000)
    verify.add_argument("--seed", type=int, default=1729)
    verify.add_argument("--dense-oracle-max-dofs", type=int, default=512)
    verify.add_argument("--relative-tolerance", type=float, default=2.0e-10)
    verify.add_argument("--output", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    solver = SpectralSolverConfiguration(
        tolerance=arguments.tolerance,
        max_iterations=arguments.max_iterations,
        seed=arguments.seed,
    )
    if arguments.command == "verify-tiny":
        result = verify_tiny_dense_oracle(
            case=arguments.case,
            nx=arguments.grid[0],
            ny=arguments.grid[1],
            dt=arguments.dt,
            c0=arguments.c0,
            s=arguments.s,
            solver_configuration=solver,
            dense_oracle_max_dofs=arguments.dense_oracle_max_dofs,
            relative_tolerance=arguments.relative_tolerance,
        )
        write_json_record(arguments.output, asdict(result))
        print(json.dumps({"output": str(Path(arguments.output).resolve()), "passed": result.passed}))
        return 0 if result.passed else 2
    result = build_stability_table(
        arguments.grid,
        arguments.c0_values,
        case=arguments.case,
        dt=arguments.dt,
        s=arguments.s,
        safety_factor=arguments.safety_factor,
        solver_configuration=solver,
    )
    write_json_record(arguments.output, result)
    if arguments.csv:
        _write_csv(arguments.csv, result["rows"])
    print(json.dumps({"output": str(Path(arguments.output).resolve()), "rows": len(result["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ACTIVE_FIELDS",
    "EulerStabilityRow",
    "FieldEigenvalueEstimate",
    "SpectralSolverConfiguration",
    "TinyOracleVerification",
    "build_stability_table",
    "estimate_field_eigenvalues",
    "stability_rows",
    "verify_tiny_dense_oracle",
)
