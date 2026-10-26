"""Firedrake-side helpers shared by the optimization and validation workflows.

Everything here imports Firedrake lazily so the pure-Python parts of the workflows (spec
validation, dry runs, tests) work without a Firedrake environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from workflows.common import deep_copy_parameters, get_by_dotted_path, set_by_dotted_path

# x-space variable used by run_model's Output for the prognostic state in the checkpoint
TRUTH_STATE_NAME = "xn0"
# name of the optimized initial state stored in optimal_ic.h5 for the ic-field target
OPTIMAL_IC_NAME = "x0_opt"


def optimization_solver_parameters() -> dict[str, Any]:
    """Return solver parameters with direct solves for the explicit RK and adjoint systems.

    This mirrors ic_optimize.py/coeff_optimize.py: LU solves make the forward and adjoint steps
    consistent to round-off, which keeps the adjoint gradients accurate.
    """
    from dimswe.parameters import overall_solver_parameters

    solver_parameters = overall_solver_parameters.copy()
    direct = {"ksp_type": "preonly", "pc_type": "lu"}
    for key in ["erkstage-f", "erkstage-aux", "erkstage-mu", "erkstage-muaux", "erk-dlambda", "erk-grad"]:
        solver_parameters[key] = direct
    return solver_parameters


def build_model(parameters: dict[str, Any], mesh: Any, with_statistics: bool = False) -> Any:
    """Build a model on an existing mesh, without logging output, and initialize its fixed data.

    model.initialize sets the time-independent data the dynamics depends on, such as the
    Coriolis parameter, the bottom topography and forcing-term data (the topography used by
    threewayphysics), exactly as run_model does for the truth run. Without it these Functions keep
    their zero defaults and the model being optimized is not the model that made the truth. The
    analytic initial state written to a scratch variable is not used: optimizations start from
    truth states or from their own initial guess.
    """
    from dimswe.logger import EmptyLogger
    from dimswe.models import get_model

    model = get_model(parameters, EmptyLogger(), has_dynamics_statistics=with_statistics, mesh=mesh)
    _, scratch_sub, _ = model.get_full_var("initialize_scratch", split_x_and_aux=True)
    model.initialize(scratch_sub, model.get_t_var())
    return model


def copy_function_data(destination: Any, source: Any, meshes_verified: bool = False) -> None:
    """Copy a Function loaded from a checkpoint onto a model Function on the same mesh.

    A loaded Function lives in a function space rebuilt from the checkpoint, which is a different
    Python object from the model's space. When both components use the same element on the same
    mesh, their dofs are numbered identically and are copied directly; otherwise the component is
    interpolated, which is exact between equivalent elements on one mesh.

    meshes_verified: set when the caller has checked (with meshes_match) that a different mesh
    object has identical coordinates, so dofs of identical elements can be copied directly.
    """
    destination_parts = destination.subfunctions
    source_parts = source.subfunctions
    if len(destination_parts) != len(source_parts):
        raise ValueError(
            f"Cannot copy {source.name()} into {destination.name()}: "
            f"{len(source_parts)} vs {len(destination_parts)} components. "
            "Optimization and truth must use the same model and discretization."
        )
    for destination_part, source_part in zip(destination_parts, source_parts, strict=True):
        destination_space = destination_part.function_space()
        source_space = source_part.function_space()
        same_element = destination_space.ufl_element() == source_space.ufl_element()
        same_mesh = destination_space.mesh() is source_space.mesh()
        if same_element and (same_mesh or meshes_verified):
            if destination_part.dat.data_ro.shape != source_part.dat.data_ro.shape:
                raise ValueError(f"Cannot copy {source.name()}: dof layouts differ")
            destination_part.dat.data[:] = source_part.dat.data_ro
        elif same_mesh:
            destination_part.interpolate(source_part)
        else:
            raise ValueError(f"Cannot copy {source.name()}: functions live on different meshes")


def load_truth_run(
    checkpoint: str | Path,
    output_indices: list[int],
    names: list[str] | None = None,
) -> tuple[Any, dict[str, dict[int, Any]]]:
    """Load a truth run's mesh and the named functions at the given output indices.

    The mesh and functions are loaded through one CheckpointFile handle, as Firedrake requires.
    Returns (mesh, {name: {output_index: Function}}); the Functions live on spaces rebuilt from the
    checkpoint, so use copy_function_data to move them onto a model's spaces.
    """
    from firedrake import CheckpointFile

    names = [TRUTH_STATE_NAME] if names is None else names
    loaded: dict[str, dict[int, Any]] = {name: {} for name in names}
    with CheckpointFile(str(checkpoint), "r") as checkpoint_file:
        mesh = checkpoint_file.load_mesh()
        for name in names:
            for index in sorted(set(output_indices)):
                loaded[name][index] = checkpoint_file.load_function(mesh, name, idx=index)
    return mesh, loaded


def truth_states_on_model(
    model: Any,
    truth_mesh: Any,
    loaded_states: dict[int, Any],
    prefix: str,
) -> dict[int, list[Any]]:
    """Copy loaded truth states onto the model's x space as [x, None] state lists.

    The truth mesh must either be the model mesh, or have identical coordinates (same
    distribution and numbering), which is checked; this is what allows several truth runs from the
    same sweep to be combined in one joint optimization.
    """
    verified = False
    if truth_mesh is not model.mesh:
        if not meshes_match(truth_mesh, model.mesh):
            raise ValueError(
                "Truth runs do not share the same mesh layout; joint optimization needs runs with the "
                "same mesh, discretization and MPI rank count"
            )
        verified = True
    states: dict[int, list[Any]] = {}
    for index, loaded in sorted(loaded_states.items()):
        state, _, _ = model.get_x_var(f"{prefix}_{index}")
        copy_function_data(state, loaded, meshes_verified=verified)
        states[index] = [state, None]
    return states


def output_time(parameters: dict[str, Any], model: Any, output_index: int) -> float:
    """Return the model time of an output index of a run."""
    step = output_index * parameters["output"]["output_freq"]
    return float(model.initcond.get_t0()) + step * parameters["timestepping"]["dt"]


def coefficient_paths(model: Any) -> list[str]:
    """Return the dotted cfg path of every model coefficient, in coefficient-vector order.

    The order matches how the dynamics builds its mixed coefficient space: forcing terms in order,
    and within a term the order of term.get_coeff(). All coefficients are R-space scalars.
    """
    paths = []
    for term in model.dynamics.forcing_terms:
        if term.has_coeff():
            for name, _ in term.get_coeff():
                paths.append(f"{term.name}.{name}")
    if len(paths) != model.get_coeff_size():
        raise ValueError(
            f"Expected {model.get_coeff_size()} scalar coefficients, found paths {paths}; "
            "only R-space coefficients are supported"
        )
    return paths


def enable_coefficients(parameters: dict[str, Any], dotted_paths: list[str]) -> None:
    """Set treat_as_coeffs on every forcing-term section containing an optimized coefficient."""
    for path in dotted_paths:
        section = path.split(".")[0]
        if not isinstance(parameters.get(section), dict) or "treat_as_coeffs" not in parameters[section]:
            raise ValueError(f"'{path}' is not a coefficient of a forcing term with treat_as_coeffs")
        if section not in parameters["model"].get("forcing_terms", []):
            raise ValueError(f"'{path}' belongs to forcing term '{section}', which is not active in model.forcing_terms")
        parameters[section]["treat_as_coeffs"] = True


def parameters_with_values(parameters: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    """Return a copy of parameters with dotted-path values replaced."""
    updated = deep_copy_parameters(parameters)
    for path, value in values.items():
        get_by_dotted_path(updated, path)  # raises on unknown paths
        set_by_dotted_path(updated, path, float(value))
    return updated


def variable_weights(model: Any, states: list[list[Any]]) -> dict[str, float]:
    """Return per-variable weights 1/max_k ||d_k,var||^2 over the given truth states.

    Normalizing by the truth magnitude stops large-magnitude variables (such as h ~ 750 m) from
    dominating the loss. Variables that are zero in every state get weight 1.
    """
    from firedrake import assemble, inner

    weights = {}
    for index, var in enumerate(model.get_x_var_list()):
        largest = 0.0
        for state in states:
            component = state[0].subfunctions[index]
            largest = max(largest, float(assemble(inner(component, component) * model.spaces.dx)))
        weights[var] = 1.0 / largest if largest > 0 else 1.0
    return weights


def save_state(path: str | Path, mesh: Any, state: Any, name: str) -> None:
    """Save a state Function and its mesh to a new checkpoint file."""
    from firedrake import CheckpointFile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with CheckpointFile(str(path), "w") as checkpoint_file:
        checkpoint_file.save_mesh(mesh)
        state.rename(name)
        checkpoint_file.save_function(state)


def meshes_match(mesh_a: Any, mesh_b: Any) -> bool:
    """Return whether two meshes have identical local coordinate dofs on every rank."""
    import numpy as np

    coordinates_a = mesh_a.coordinates.dat.data_ro
    coordinates_b = mesh_b.coordinates.dat.data_ro
    local = coordinates_a.shape == coordinates_b.shape and bool(np.allclose(coordinates_a, coordinates_b))
    return bool(mesh_a.comm.allreduce(local, op=_mpi_land()))


def _mpi_land() -> Any:
    """Return the MPI logical-and reduction."""
    from mpi4py import MPI

    return MPI.LAND


def load_state_into(path: str | Path, name: str, destination: Any) -> None:
    """Load a state saved with save_state into a Function on a matching mesh.

    The checkpoint's mesh is a different object from the destination mesh, so dof values are
    copied after checking that the meshes have identical coordinates and that the elements match.
    This holds when the state is loaded with the same MPI size that saved it.
    """
    from firedrake import CheckpointFile

    with CheckpointFile(str(path), "r") as checkpoint_file:
        mesh = checkpoint_file.load_mesh()
        if not meshes_match(mesh, destination.function_space().mesh()):
            raise ValueError(
                f"The mesh in {path} does not match the truth mesh; load it with the same number "
                "of MPI ranks used to save it"
            )
        loaded = checkpoint_file.load_function(mesh, name)
        for destination_part, source_part in zip(destination.subfunctions, loaded.subfunctions, strict=True):
            if destination_part.function_space().ufl_element() != source_part.function_space().ufl_element():
                raise ValueError(f"Element mismatch when loading {name} from {path}")
            destination_part.dat.data[:] = source_part.dat.data_ro


def project_initial_state(model: Any, parameters: dict[str, Any], name: str) -> Any:
    """Project the analytic initial condition described by parameters onto the model x space.

    This repeats the prognostic part of model.initialize (each x variable is projected from its
    initial-condition expression) without touching the forcing-term data of the model, so it can
    be used to build an initial guess from perturbed initial-condition parameters.
    """
    varexpr = model.get_initial_expressions(parameters, model.get_t_var())
    state, state_sub, _ = model.get_x_var(name)
    for var in model.get_x_var_list():
        expr = varexpr.get(var, 0)
        if not (isinstance(expr, (int, float)) and expr == 0):
            state_sub[var].project(expr)
    return state


def per_variable_scaling(model: Any, state: Any) -> Any:
    """Return a flattened array that scales every dof by the max magnitude of its variable.

    Used as a diagonal preconditioner for full-field initial-condition optimization, so variables
    of very different size (h ~ 750 m, v ~ 10 m/s) are equally well resolved by the optimizer.
    Collective: the maxima are reduced over MPI ranks.
    """
    import numpy as np
    from mpi4py import MPI

    blocks = []
    for component in state.subfunctions:
        data = component.dat.data_ro
        local = float(np.max(np.abs(data))) if data.size else 0.0
        largest = model.mesh.comm.allreduce(local, op=MPI.MAX)
        blocks.append(np.full(data.size, largest if largest > 0 else 1.0))
    return np.hstack(blocks)
