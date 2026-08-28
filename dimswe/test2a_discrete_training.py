"""Test 2A-3C exact cached deployed-discrete offline training.

The production Test-2A-3A objective remains the certification oracle.  This
module exploits only fixed-truth-state algebra: sparse weak matrices, diagonal
GLL mass blocks, packed h values, analytical targets, and normalization are
cached once.  No dense G_k or K_k matrix and no recursive trajectory is used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
from jax.flatten_util import ravel_pytree
import numpy as np

from .learned_physics.parameters import tree_axpy, tree_norm
from .resolved_hidden_c0 import read_json_record, write_json_record
from .test2a_discrete_offline import (
    _deterministic_parameter_vectors,
    load_discrete_offline_configuration,
    objective_gradient_comparison,
    prepare_production_problem,
)
from .test2a_embedded_moist import parameter_pytree_sha256
from .test2a_operator import (
    DenseMLP,
    initialize_mlp,
    load_mlp_parameters,
    load_operator_dataset,
    load_selected_configuration,
    mlp_configuration_from_record,
    normalization_from_record,
    operator_metrics,
    physical_predictions,
    save_mlp_parameters_atomic,
)
from .test2a_pyrol import JAXPytreeObjective, build_test2a_lbfgs_parameters


TRAINING_STEPS = tuple(range(81))


def _file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def load_discrete_training_configuration(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    allowed_stages = (
        "Test 2A-3C deployed-discrete offline 50k training",
        "Test 2A fair deployed-discrete seed-0 m20 long fit",
        "Test 2A M1-to-M2 deployed-discrete fine-tune",
    )
    if record.get("benchmark_stage") not in allowed_stages:
        raise ValueError("not a selected Test 2A-3C configuration")
    if record["truth_state_indices"] != [0, 80] or not record[
        "states_after_80_forbidden"
    ]:
        raise ValueError("Test 2A-3C may use only states 0..80")
    if record["recursive_model_state_propagation"] is not False:
        raise ValueError("deployed-discrete offline training cannot recurse")
    optimizer = record["optimizer"]
    if (
        optimizer["library"] != "PyROL/ROL"
        or optimizer["method"] != "line-search L-BFGS"
        or int(optimizer["maximum_secant_storage"]) != 20
        or int(optimizer["accepted_iteration_limit"]) < 1
        or float(optimizer["gradient_tolerance"]) != 1.0e-8
        or float(optimizer["step_tolerance"]) != 1.0e-12
        or optimizer["production_HVP"] is not False
    ):
        raise ValueError("Test 2A-3C optimizer contract changed")
    checkpoints = tuple(int(value) for value in record["checkpoint_accepted_iterations"])
    if (
        checkpoints != tuple(sorted(set(checkpoints)))
        or not checkpoints
        or checkpoints[-1] != int(optimizer["accepted_iteration_limit"])
    ):
        raise ValueError("invalid Test 2A-3C checkpoint schedule")
    initialization = record["initialization"]
    if record["benchmark_stage"] == "Test 2A M1-to-M2 deployed-discrete fine-tune":
        if (
            initialization.get("kind") != "operator_200k_warm_start"
            or initialization.get("operator_pretraining") is not True
            or initialization.get("source_optimizer_secant_history_reused") is not False
            or initialization.get("parameter_pytree_sha256")
            != "f86ee79be3086028f21de10b947c0089147234f494c066f8bbbb2fffb3f8bef8"
        ):
            raise ValueError("invalid Test 2A M1-to-M2 warm-start contract")
        if (
            float(initialization.get("initial_J_op")) != 0.000373006108792648
            or float(initialization.get("initial_J_disc"))
            != 0.0008346864309047664
        ):
            raise ValueError("matched Method-1 warm-start objectives changed")
    elif initialization["operator_pretraining"] is not False:
        raise ValueError("canonical Test 2A-3C cannot use operator pretraining")
    direct = record["direct_production_method2_baseline"]
    if (
        direct["parameter_pytree_sha256"]
        != "4db13a84f52d6fcf66f0345ce5c677aff2b46a84350228a78bc05b073ff6b12a"
        or float(direct["accepted_J_disc"]) != 0.0017427829635521567
        or int(direct["accepted_iterations"]) != 50000
    ):
        raise ValueError("accepted direct-production Method-2 baseline changed")
    return record


def load_training_initial_parameters(training, model_configuration):
    """Load and fingerprint the explicitly configured optimizer start.

    A warm start is a new optimizer initialization.  This function loads only
    the parameter pytree; no PyROL/ROL secant state is read or transferred.
    """

    initialization = training["initialization"]
    expected = initialization["parameter_pytree_sha256"]
    if initialization.get("operator_pretraining", False):
        parameters, artifact_configuration = load_mlp_parameters(
            initialization["source_parameter_file"]
        )
        if artifact_configuration != model_configuration:
            raise ValueError("warm-start architecture changed")
    else:
        parameters = initialize_mlp(model_configuration)
    if parameter_pytree_sha256(parameters) != expected:
        raise ValueError("configured initial parameter fingerprint changed")
    return parameters


@dataclass(frozen=True)
class FixedDiscreteCache:
    normalized_features: np.ndarray
    normalized_targets: np.ndarray
    h: np.ndarray
    beta2: float
    output_scale: float
    w_s_data: np.ndarray
    w_s_indices: np.ndarray
    w_s_shape: tuple[int, int]
    w_q_data: np.ndarray
    w_q_indices: np.ndarray
    w_q_shape: tuple[int, int]
    mass_inverse_s_x: np.ndarray
    mass_inverse_s_y: np.ndarray
    mass_s_grid_order: np.ndarray
    mass_s_grid_shape: tuple[int, int]
    mass_inverse_q_data: np.ndarray
    mass_inverse_q_indices: np.ndarray
    mass_inverse_q_shape: tuple[int, int]
    normalizer: float
    metadata: dict

    def __post_init__(self):
        features = np.array(self.normalized_features, dtype=np.float64, copy=True)
        targets = np.array(self.normalized_targets, dtype=np.float64, copy=True)
        h = np.array(self.h, dtype=np.float64, copy=True)
        if features.ndim != 2 or features.shape[1] != 5:
            raise ValueError("fixed cache features must have five columns")
        if h.ndim != 2 or features.shape[0] != h.size:
            raise ValueError("fixed cache h shape must partition all samples")
        if targets.size != features.shape[0]:
            raise ValueError("fixed cache targets must cover all samples")
        targets = targets.reshape(h.shape)
        if self.w_s_shape[1] != h.shape[1] or self.w_q_shape[1] != h.shape[1]:
            raise ValueError("weak matrices must consume one state's packed samples")
        for name, value in (
            ("normalized_features", features),
            ("normalized_targets", targets),
            ("h", h),
            ("w_s_data", self.w_s_data),
            ("w_q_data", self.w_q_data),
            ("mass_inverse_s_x", self.mass_inverse_s_x),
            ("mass_inverse_s_y", self.mass_inverse_s_y),
            ("mass_s_grid_order", self.mass_s_grid_order),
            ("mass_inverse_q_data", self.mass_inverse_q_data),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        if int(np.prod(self.mass_s_grid_shape)) != self.w_s_shape[0] or (
            self.mass_inverse_q_shape[0] != self.w_q_shape[0]
        ):
            raise ValueError("cached mass inverses and weak rows differ")
        if np.unique(np.asarray(self.mass_s_grid_order)).size != self.w_s_shape[0]:
            raise ValueError("S mass grid order must be a permutation")
        if not np.isfinite(self.normalizer) or self.normalizer <= 0.0:
            raise ValueError("cached objective normalizer must be positive")
        for name, value in (
            ("normalized_features", features),
            ("normalized_targets", targets),
            ("h", h),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def sample_count(self):
        return int(self.normalized_features.shape[0])


def save_fixed_cache(path, cache):
    destination = Path(path)
    metadata_path = destination.with_suffix(".json")
    temporary = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite a fixed-operator cache")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                normalized_features=cache.normalized_features,
                normalized_targets=cache.normalized_targets,
                h=cache.h,
                beta2=np.float64(cache.beta2),
                output_scale=np.float64(cache.output_scale),
                w_s_data=np.asarray(cache.w_s_data, dtype=np.float64),
                w_s_indices=np.asarray(cache.w_s_indices, dtype=np.int32),
                w_s_shape=np.asarray(cache.w_s_shape, dtype=np.int64),
                w_q_data=np.asarray(cache.w_q_data, dtype=np.float64),
                w_q_indices=np.asarray(cache.w_q_indices, dtype=np.int32),
                w_q_shape=np.asarray(cache.w_q_shape, dtype=np.int64),
                mass_inverse_s_x=np.asarray(
                    cache.mass_inverse_s_x, dtype=np.float64
                ),
                mass_inverse_s_y=np.asarray(
                    cache.mass_inverse_s_y, dtype=np.float64
                ),
                mass_s_grid_order=np.asarray(
                    cache.mass_s_grid_order, dtype=np.int32
                ),
                mass_s_grid_shape=np.asarray(
                    cache.mass_s_grid_shape, dtype=np.int64
                ),
                mass_inverse_q_data=np.asarray(
                    cache.mass_inverse_q_data, dtype=np.float64
                ),
                mass_inverse_q_indices=np.asarray(
                    cache.mass_inverse_q_indices, dtype=np.int32
                ),
                mass_inverse_q_shape=np.asarray(
                    cache.mass_inverse_q_shape, dtype=np.int64
                ),
                normalizer=np.float64(cache.normalizer),
            )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    metadata = dict(cache.metadata)
    metadata.update(
        {
            "format_version": 1,
            "cache_file": destination.name,
            "cache_npz_sha256": _file_sha256(destination),
            "truth_state_indices": [0, 80],
            "states_after_80_accessed": False,
            "recursive_model_state_propagation": False,
            "cache_npz_bytes": int(destination.stat().st_size),
            "acceptance_marker": "this JSON is written only after the NPZ is complete",
        }
    )
    # The metadata is the acceptance marker and is committed atomically last.
    write_json_record(metadata_path, metadata)
    return destination


def load_fixed_cache(path, *, require_canonical=True):
    source = Path(path)
    metadata_path = source.with_suffix(".json")
    if not source.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "a complete fixed cache requires both the NPZ and its JSON acceptance "
            "marker; rerun prepare-cache"
        )
    metadata = read_json_record(metadata_path)
    if _file_sha256(source) != metadata["cache_npz_sha256"]:
        raise ValueError("fixed-operator cache fingerprint mismatch")
    if metadata.get("production_oracle_certified") is not True:
        raise ValueError("fixed-operator cache lacks production-oracle certification")
    if metadata.get("truth_state_indices") != [0, 80] or metadata.get(
        "states_after_80_accessed", True
    ):
        raise ValueError("fixed-operator cache violates truth support")
    with np.load(source, allow_pickle=False) as archive:
        cache = FixedDiscreteCache(
            normalized_features=np.array(archive["normalized_features"], copy=True),
            normalized_targets=np.array(archive["normalized_targets"], copy=True),
            h=np.array(archive["h"], copy=True),
            beta2=float(archive["beta2"]),
            output_scale=float(archive["output_scale"]),
            w_s_data=np.array(archive["w_s_data"], copy=True),
            w_s_indices=np.array(archive["w_s_indices"], copy=True),
            w_s_shape=tuple(int(value) for value in archive["w_s_shape"]),
            w_q_data=np.array(archive["w_q_data"], copy=True),
            w_q_indices=np.array(archive["w_q_indices"], copy=True),
            w_q_shape=tuple(int(value) for value in archive["w_q_shape"]),
            mass_inverse_s_x=np.array(archive["mass_inverse_s_x"], copy=True),
            mass_inverse_s_y=np.array(archive["mass_inverse_s_y"], copy=True),
            mass_s_grid_order=np.array(archive["mass_s_grid_order"], copy=True),
            mass_s_grid_shape=tuple(int(value) for value in archive["mass_s_grid_shape"]),
            mass_inverse_q_data=np.array(archive["mass_inverse_q_data"], copy=True),
            mass_inverse_q_indices=np.array(archive["mass_inverse_q_indices"], copy=True),
            mass_inverse_q_shape=tuple(int(value) for value in archive["mass_inverse_q_shape"]),
            normalizer=float(archive["normalizer"]),
            metadata=metadata,
        )
    if require_canonical and (
        cache.sample_count != 331_776 or cache.h.shape != (81, 4096)
    ):
        raise ValueError("production fixed cache sample accounting changed")
    return cache


def validate_resume_record(record, configuration_sha256, cache_npz_sha256):
    if record.get("status") != "in_progress":
        raise ValueError("Test 2A-3C progress is not resumable")
    if record.get("configuration_sha256") != configuration_sha256:
        raise ValueError("resume configuration changed")
    if record.get("cache_npz_sha256") != cache_npz_sha256:
        raise ValueError("resume fixed cache changed")
    iteration = int(record.get("last_checkpoint_accepted_iteration", -1))
    if iteration < 0:
        raise ValueError("resume record lacks a completed parameter checkpoint")
    return iteration


class FastFixedDiscreteObjective:
    """Exact sparse G_k weighting on all fixed states in one JAX graph."""

    def __init__(self, cache, model_configuration, *, use_jit=True):
        self.cache = cache
        self.model = DenseMLP(model_configuration)
        self.features = jnp.asarray(cache.normalized_features, dtype=jnp.float64)
        self.targets = jnp.asarray(cache.normalized_targets, dtype=jnp.float64)
        self.h = jnp.asarray(cache.h, dtype=jnp.float64)
        self.w_s = jsparse.BCOO(
            (
                jnp.asarray(cache.w_s_data, dtype=jnp.float64),
                jnp.asarray(cache.w_s_indices, dtype=jnp.int32),
            ),
            shape=cache.w_s_shape,
        )
        self.w_q = jsparse.BCOO(
            (
                jnp.asarray(cache.w_q_data, dtype=jnp.float64),
                jnp.asarray(cache.w_q_indices, dtype=jnp.int32),
            ),
            shape=cache.w_q_shape,
        )
        self.mass_inverse_s_x = jnp.asarray(cache.mass_inverse_s_x, dtype=jnp.float64)
        self.mass_inverse_s_y = jnp.asarray(cache.mass_inverse_s_y, dtype=jnp.float64)
        self.mass_s_grid_order = jnp.asarray(cache.mass_s_grid_order, dtype=jnp.int32)
        self.mass_s_grid_shape = tuple(int(value) for value in cache.mass_s_grid_shape)
        self.mass_inverse_q = jsparse.BCOO(
            (
                jnp.asarray(cache.mass_inverse_q_data, dtype=jnp.float64),
                jnp.asarray(cache.mass_inverse_q_indices, dtype=jnp.int32),
            ),
            shape=cache.mass_inverse_q_shape,
        )

        def objectives(parameters):
            prediction = self.model(parameters, self.features).reshape(self.h.shape)
            error = float(cache.output_scale) * (prediction - self.targets)
            source_q = self.h * error
            source_s = float(cache.beta2) * source_q
            weak_s = (self.w_s @ source_s.T).T
            weak_q = (self.w_q @ source_q.T).T
            weak_s_grid = weak_s[:, self.mass_s_grid_order].reshape(
                (self.h.shape[0], *self.mass_s_grid_shape)
            )
            riesz_s_grid = jnp.einsum(
                "ij,bjk,lk->bil",
                self.mass_inverse_s_y,
                weak_s_grid,
                self.mass_inverse_s_x,
            )
            riesz_q = (self.mass_inverse_q @ weak_q.T).T
            numerator = jnp.sum(weak_s_grid * riesz_s_grid)
            numerator = numerator + 2.0 * jnp.sum(weak_q * riesz_q)
            discrete = numerator / float(cache.normalizer)
            operator = jnp.mean((prediction - self.targets) ** 2)
            return discrete, operator

        value_gradient = jax.value_and_grad(lambda parameters: objectives(parameters)[0])
        gradient = jax.grad(lambda parameters: objectives(parameters)[0])

        def hess_vec(parameters, direction):
            return jax.jvp(gradient, (parameters,), (direction,))[1]

        self._objectives = jax.jit(objectives) if use_jit else objectives
        self._value_gradient = jax.jit(value_gradient) if use_jit else value_gradient
        self._hess_vec = jax.jit(hess_vec) if use_jit else hess_vec

    def objectives(self, parameters):
        discrete, operator = self._objectives(parameters)
        return float(discrete), float(operator)

    def value(self, parameters):
        return self.objectives(parameters)[0]

    def jax_value(self, parameters):
        """Return the scalar without host conversion for PyROL autodiff."""
        return self._objectives(parameters)[0]

    def value_and_gradient(self, parameters):
        value, gradient = self._value_gradient(parameters)
        return float(value), gradient

    def gradient(self, parameters):
        return self.value_and_gradient(parameters)[1]

    def hess_vec(self, parameters, direction):
        return self._hess_vec(parameters, direction)


def _scipy_csr_from_petsc(matrix):
    from scipy.sparse import csr_matrix

    petsc = matrix.petscmat
    row_pointer, columns, values = petsc.getValuesCSR()
    return csr_matrix(
        (np.asarray(values), np.asarray(columns), np.asarray(row_pointer)),
        shape=petsc.getSize(),
    )


def _sparse_local_inverse(matrix, *, residual_tolerance, maximum_component_size):
    from scipy.sparse import block_diag, csr_matrix, identity
    from scipy.sparse.csgraph import connected_components

    graph = matrix.copy()
    graph.data = np.ones_like(graph.data)
    component_count, labels = connected_components(
        graph, directed=False, connection="weak"
    )
    components = [np.flatnonzero(labels == index) for index in range(component_count)]
    maximum = max(int(component.size) for component in components)
    if maximum > int(maximum_component_size):
        raise RuntimeError(
            f"mass inverse component size {maximum} exceeds certified local bound"
        )
    order = np.concatenate(components)
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)
    blocks = []
    for component in components:
        dense = matrix[component][:, component].toarray()
        blocks.append(csr_matrix(np.linalg.inv(dense)))
    ordered_inverse = block_diag(blocks, format="csr")
    inverse = ordered_inverse[inverse_order][:, inverse_order].tocsr()
    residual = matrix @ inverse - identity(matrix.shape[0], format="csr")
    maximum_residual = 0.0 if residual.nnz == 0 else float(np.max(np.abs(residual.data)))
    symmetry_residual = inverse - inverse.T
    maximum_symmetry_residual = (
        0.0
        if symmetry_residual.nnz == 0
        else float(np.max(np.abs(symmetry_residual.data)))
    )
    if maximum_residual > float(residual_tolerance):
        raise RuntimeError(
            f"local sparse mass inverse residual {maximum_residual} exceeds tolerance"
        )
    if maximum_symmetry_residual > float(residual_tolerance):
        raise RuntimeError(
            "local sparse mass inverse transpose action is not symmetric within "
            "the certification tolerance"
        )
    return inverse, {
        "connected_component_count": int(component_count),
        "maximum_component_size": maximum,
        "inverse_nonzeros": int(inverse.nnz),
        "maximum_absolute_M_Minv_minus_I": maximum_residual,
        "maximum_absolute_Minv_minus_Minv_transpose": (
            maximum_symmetry_residual
        ),
        "transpose_action_certified": True,
    }


def _clustered_ranks(values, expected_count, *, name):
    """Return stable ranks for nearly identical geometric coordinates."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    ordering = np.argsort(values, kind="stable")
    scale = max(float(np.max(np.abs(values))), float(np.ptp(values)), 1.0)
    tolerance = 512.0 * np.finfo(np.float64).eps * scale
    ranks = np.empty(values.size, dtype=np.int32)
    cluster = -1
    representative = None
    for index in ordering:
        value = float(values[index])
        if representative is None or abs(value - representative) > tolerance:
            cluster += 1
            representative = value
        ranks[index] = cluster
    if cluster + 1 != int(expected_count):
        raise RuntimeError(
            f"{name} has {cluster + 1} coordinate clusters, expected "
            f"{int(expected_count)}"
        )
    return ranks


def _periodic_tensor_grid_order(
    space,
    carrier_space,
    carrier_cell_nodes,
    *,
    periodic_cell_shape,
):
    """Build the periodic CG tensor permutation from cell topology.

    Cartesian coordinates interpolated into a periodic CG space are not nodal
    coordinates at seam-identified DOFs.  Broken carrier coordinates remain
    cell-local, so they identify cell ranks and local tensor-node ranks without
    asking a periodic global DOF to own two physical coordinates.
    """
    from firedrake import Function, SpatialCoordinate

    cells_x, cells_y = (int(value) for value in periodic_cell_shape)
    if cells_x < 1 or cells_y < 1:
        raise ValueError("periodic cell counts must be positive")
    global_cell_nodes = np.asarray(space.cell_node_map().values, dtype=np.int64)
    carrier_cell_nodes = np.asarray(carrier_cell_nodes, dtype=np.int64)
    if global_cell_nodes.shape != carrier_cell_nodes.shape:
        raise RuntimeError("CG and broken-carrier cell-node layouts differ")
    coordinates = SpatialCoordinate(space.mesh())
    carrier_coordinates = np.column_stack(
        tuple(
            np.asarray(
                Function(carrier_space).interpolate(component).dat.data_ro,
                dtype=np.float64,
            )
            for component in coordinates
        )
    )
    local_coordinates = carrier_coordinates[carrier_cell_nodes]
    if local_coordinates.shape[-1] != 2:
        raise ValueError("periodic tensor topology requires two dimensions")
    cell_centres = np.mean(local_coordinates, axis=1)
    cell_x = _clustered_ranks(
        cell_centres[:, 0], cells_x, name="periodic x-cell topology"
    )
    cell_y = _clustered_ranks(
        cell_centres[:, 1], cells_y, name="periodic y-cell topology"
    )
    local_count = global_cell_nodes.shape[1]
    local_axis_count = int(round(np.sqrt(local_count)))
    if local_axis_count * local_axis_count != local_count:
        raise RuntimeError("CG cell nodes do not form a square tensor element")
    degree = local_axis_count - 1
    element_degree = space.ufl_element().degree()
    if isinstance(element_degree, tuple):
        expected_degree = tuple(int(value) for value in element_degree)
    else:
        expected_degree = (int(element_degree), int(element_degree))
    if expected_degree != (degree, degree):
        raise RuntimeError("CG topology degree and UFL element degree differ")
    grid_x = cells_x * degree
    grid_y = cells_y * degree
    if grid_x * grid_y != space.dim():
        raise RuntimeError(
            "periodic CG topology size does not equal the global space dimension"
        )
    global_keys = np.full(space.dim(), -1, dtype=np.int64)
    occurrence_count = np.zeros(space.dim(), dtype=np.int64)
    for cell in range(global_cell_nodes.shape[0]):
        local_x = _clustered_ranks(
            local_coordinates[cell, :, 0],
            local_axis_count,
            name=f"cell {cell} local x topology",
        )
        local_y = _clustered_ranks(
            local_coordinates[cell, :, 1],
            local_axis_count,
            name=f"cell {cell} local y topology",
        )
        keys = (
            ((cell_y[cell] * degree + local_y) % grid_y) * grid_x
            + ((cell_x[cell] * degree + local_x) % grid_x)
        )
        for global_dof, key in zip(global_cell_nodes[cell], keys):
            previous = global_keys[global_dof]
            if previous not in (-1, key):
                raise RuntimeError(
                    "periodic seam occurrences assign conflicting tensor indices"
                )
            global_keys[global_dof] = key
            occurrence_count[global_dof] += 1
    if np.any(global_keys < 0) or np.any(occurrence_count < 1):
        raise RuntimeError("periodic tensor topology leaves CG DOFs unassigned")
    if not np.array_equal(np.sort(global_keys), np.arange(space.dim())):
        raise RuntimeError("periodic tensor topology is not a complete bijection")
    global_coordinates = np.column_stack(
        tuple(
            np.asarray(
                Function(space).interpolate(component).dat.data_ro,
                dtype=np.float64,
            )
            for component in coordinates
        )
    )
    global_coordinate_counts = [
        int(np.unique(global_coordinates[:, axis]).size) for axis in (0, 1)
    ]
    return np.argsort(global_keys).astype(np.int32), (grid_y, grid_x), {
        "indexing": "cell-topology-aware periodic tensor ordering",
        "periodic_cell_shape": [cells_y, cells_x],
        "periodic_grid_shape": [grid_y, grid_x],
        "local_tensor_axis_nodes": local_axis_count,
        "polynomial_degree": degree,
        "minimum_global_dof_occurrences": int(np.min(occurrence_count)),
        "maximum_global_dof_occurrences": int(np.max(occurrence_count)),
        "interpolated_global_coordinate_unique_counts": global_coordinate_counts,
        "interpolated_coordinate_cartesian_product": int(
            np.prod(global_coordinate_counts)
        ),
        "global_space_dimension": int(space.dim()),
        "interpolated_coordinates_form_complete_grid": bool(
            np.prod(global_coordinate_counts) == space.dim()
        ),
        "diagnosis": (
            "periodic seam-identified CG coordinate interpolation is not a "
            "global nodal-coordinate map; cell topology supplies the bijection"
        ),
    }


def _coordinate_tensor_grid_order(space):
    """Nonperiodic coordinate ordering retained for tiny oracle cases."""
    from firedrake import Function, SpatialCoordinate

    physical_coordinates = SpatialCoordinate(space.mesh())
    coordinates = np.column_stack(
        tuple(
            np.asarray(
                Function(space).interpolate(component).dat.data_ro,
                dtype=np.float64,
            )
            for component in physical_coordinates
        )
    )
    if coordinates.shape[1] != 2:
        raise ValueError("tensor mass factorization requires two-dimensional coordinates")
    order = np.lexsort((coordinates[:, 0], coordinates[:, 1])).astype(np.int32)
    x_values = np.unique(coordinates[:, 0])
    y_values = np.unique(coordinates[:, 1])
    nx = int(x_values.size)
    ny = int(y_values.size)
    if nx * ny != space.dim():
        raise RuntimeError("nonperiodic CG mass DOFs do not form a tensor grid")
    ordered_coordinates = coordinates[order].reshape(ny, nx, 2)
    if not (
        np.allclose(ordered_coordinates[:, :, 0], x_values[None, :], rtol=0.0, atol=0.0)
        and np.allclose(ordered_coordinates[:, :, 1], y_values[:, None], rtol=0.0, atol=0.0)
    ):
        raise RuntimeError("CG mass coordinate order is not tensor-product lexicographic")
    return order, (ny, nx), {
        "indexing": "nonperiodic global nodal-coordinate ordering",
        "grid_shape": [ny, nx],
    }


def _tensor_mass_inverse(
    matrix,
    space,
    *,
    factorization_tolerance,
    carrier_space=None,
    carrier_cell_nodes=None,
    periodic_cell_shape=None,
):
    from scipy.sparse import csr_matrix, kron

    if periodic_cell_shape is None:
        order, (ny, nx), topology_audit = _coordinate_tensor_grid_order(space)
    else:
        if carrier_space is None or carrier_cell_nodes is None:
            raise ValueError("periodic tensor indexing requires broken-carrier topology")
        order, (ny, nx), topology_audit = _periodic_tensor_grid_order(
            space,
            carrier_space,
            carrier_cell_nodes,
            periodic_cell_shape=periodic_cell_shape,
        )
    ordered = matrix[order][:, order].tocsr()
    factor_x = ordered[:nx, :nx].toarray()
    y_indices = np.arange(0, nx * ny, nx)
    factor_y_raw = ordered[y_indices][:, y_indices].toarray()
    pivot = float(ordered[0, 0])
    if pivot == 0.0:
        raise RuntimeError("tensor mass factorization has a zero pivot")
    factor_y = factor_y_raw / pivot
    reconstructed = kron(
        csr_matrix(factor_y), csr_matrix(factor_x), format="csr"
    )
    residual = ordered - reconstructed
    maximum_matrix = float(np.max(np.abs(ordered.data)))
    maximum_residual = 0.0 if residual.nnz == 0 else float(np.max(np.abs(residual.data)))
    relative = maximum_residual / max(maximum_matrix, np.finfo(np.float64).tiny)
    if relative > float(factorization_tolerance):
        raise RuntimeError(
            f"CG mass tensor factorization residual {relative} exceeds tolerance"
        )
    inverse_x = np.linalg.inv(factor_x)
    inverse_y = np.linalg.inv(factor_y)
    inverse_residual = max(
        float(np.max(np.abs(factor_x @ inverse_x - np.eye(nx)))),
        float(np.max(np.abs(factor_y @ inverse_y - np.eye(ny)))),
    )
    return {
        "inverse_x": inverse_x,
        "inverse_y": inverse_y,
        "factor_x": factor_x,
        "factor_y": factor_y,
        "grid_order": order,
        "grid_shape": (ny, nx),
    }, {
        **topology_audit,
        "grid_shape": [ny, nx],
        "tensor_factorization_relative_residual": relative,
        "maximum_factor_inverse_residual": inverse_residual,
        "dense_global_mass_inverse_formed": False,
    }


def _tensor_operator_action(vector, tensor, *, inverse, transpose=False):
    order = np.asarray(tensor["grid_order"], dtype=np.int64)
    ny, nx = (int(value) for value in tensor["grid_shape"])
    grid = np.asarray(vector, dtype=np.float64)[order].reshape(ny, nx)
    prefix = "inverse_" if inverse else "factor_"
    left = np.asarray(tensor[prefix + "y"], dtype=np.float64)
    right = np.asarray(tensor[prefix + "x"], dtype=np.float64)
    if transpose:
        result_grid = left.T @ grid @ right
    else:
        result_grid = left @ grid @ right.T
    result = np.empty(order.size, dtype=np.float64)
    result[order] = result_grid.reshape(-1)
    return result


def _vector_error_record(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    difference = actual - expected
    absolute = float(np.linalg.norm(difference))
    reference = float(np.linalg.norm(expected))
    return {
        "absolute_l2_error": absolute,
        "relative_l2_error": absolute
        / max(reference, np.finfo(np.float64).tiny),
        "maximum_absolute_component_error": float(np.max(np.abs(difference))),
        "reference_l2_norm": reference,
    }


def _certify_tensor_mass_actions(
    firedrake_mass,
    mass_csr,
    tensor,
    *,
    relative_tolerance,
):
    """Certify tensor mass and inverse actions against assembled PETSc data."""
    from petsc4py import PETSc

    petsc_matrix = firedrake_mass.petscmat
    solver = PETSc.KSP().create(comm=petsc_matrix.comm)
    solver.setOperators(petsc_matrix)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.setUp()
    probes = (
        np.linspace(-1.0, 1.0, mass_csr.shape[0], dtype=np.float64),
        np.random.default_rng(20260808).normal(size=mass_csr.shape[0]),
        np.sin(np.arange(mass_csr.shape[0], dtype=np.float64) * 0.37),
    )
    records = []
    for index, probe in enumerate(probes):
        mass_reference = np.asarray(mass_csr @ probe, dtype=np.float64)
        mass_cached = _tensor_operator_action(probe, tensor, inverse=False)
        right_hand_side = petsc_matrix.createVecRight()
        solution = petsc_matrix.createVecLeft()
        transpose_right_hand_side = petsc_matrix.createVecLeft()
        transpose_solution = petsc_matrix.createVecRight()
        right_hand_side.array[:] = probe
        transpose_right_hand_side.array[:] = probe
        solver.solve(right_hand_side, solution)
        solver.solveTranspose(transpose_right_hand_side, transpose_solution)
        inverse_cached = _tensor_operator_action(probe, tensor, inverse=True)
        transpose_cached = _tensor_operator_action(
            probe, tensor, inverse=True, transpose=True
        )
        record = {
            "probe": index,
            "mass_action": _vector_error_record(mass_cached, mass_reference),
            "inverse_action_vs_PETSc_LU": _vector_error_record(
                inverse_cached, solution.array_r
            ),
            "transpose_inverse_action_vs_PETSc_LU": _vector_error_record(
                transpose_cached, transpose_solution.array_r
            ),
        }
        records.append(record)
    maximum_relative = max(
        entry[comparison]["relative_l2_error"]
        for entry in records
        for comparison in (
            "mass_action",
            "inverse_action_vs_PETSc_LU",
            "transpose_inverse_action_vs_PETSc_LU",
        )
    )
    if maximum_relative > float(relative_tolerance):
        raise RuntimeError(
            f"cached tensor mass action relative error {maximum_relative} "
            f"exceeds tolerance {float(relative_tolerance)}"
        )
    return {
        "production_reference": "assembled Firedrake matrix and PETSc preonly/LU",
        "probe_count": len(records),
        "relative_tolerance": float(relative_tolerance),
        "maximum_relative_l2_error": maximum_relative,
        "probes": records,
        "transpose_action_certified": True,
    }


def _matrix_cache_components(
    prepared,
    residual_tolerance,
    maximum_component_size,
    tensor_factorization_tolerance,
    *,
    periodic_cell_shape=None,
):
    from firedrake import TestFunction, TrialFunction, assemble, inner

    helper = prepared.objective.operations.helper
    carrier = helper.carrier_space
    field_spaces = {
        "S": helper.model.spaces.CG,
        "Q": helper.state_space.sub(3),
    }
    results = {}
    audits = {}
    packed_columns = np.asarray(helper.layout.cell_nodes, dtype=np.int64).reshape(-1)
    if np.unique(packed_columns).size != carrier.dim():
        raise ValueError("broken-GLL packed ordering is not a carrier permutation")
    for name, space in field_spaces.items():
        weak = assemble(
            inner(TestFunction(space), TrialFunction(carrier))
            * helper.model.spaces.dx,
            mat_type="aij",
            form_compiler_parameters={"mode": "vanilla"},
        )
        mass = assemble(
            inner(TestFunction(space), TrialFunction(space))
            * helper.model.spaces.dx,
            mat_type="aij",
        )
        weak_csr = _scipy_csr_from_petsc(weak)[:, packed_columns].tocsr()
        mass_csr = _scipy_csr_from_petsc(mass).tocsr()
        coo = weak_csr.tocoo()
        results[name] = {
            "data": np.asarray(coo.data, dtype=np.float64),
            "indices": np.column_stack((coo.row, coo.col)).astype(np.int32),
            "shape": tuple(int(value) for value in coo.shape),
        }
        base_audit = {
            "shape": list(mass_csr.shape),
            "weak_shape": list(weak_csr.shape),
            "weak_nonzeros": int(weak_csr.nnz),
        }
        if name == "S":
            tensor, inverse_audit = _tensor_mass_inverse(
                mass_csr,
                space,
                factorization_tolerance=tensor_factorization_tolerance,
                carrier_space=carrier,
                carrier_cell_nodes=helper.layout.cell_nodes,
                periodic_cell_shape=periodic_cell_shape,
            )
            action_audit = _certify_tensor_mass_actions(
                mass,
                mass_csr,
                tensor,
                relative_tolerance=residual_tolerance,
            )
            results[name].update(tensor)
            inverse_audit = {**inverse_audit, "action_certification": action_audit}
        else:
            mass_inverse, inverse_audit = _sparse_local_inverse(
                mass_csr,
                residual_tolerance=residual_tolerance,
                maximum_component_size=maximum_component_size,
            )
            inverse_coo = mass_inverse.tocoo()
            results[name].update(
                {
                    "mass_inverse_data": np.asarray(
                        inverse_coo.data, dtype=np.float64
                    ),
                    "mass_inverse_indices": np.column_stack(
                        (inverse_coo.row, inverse_coo.col)
                    ).astype(np.int32),
                    "mass_inverse_shape": tuple(
                        int(value) for value in inverse_coo.shape
                    ),
                }
            )
        audits[name] = {**base_audit, **inverse_audit}
    return results, audits


def _tree_relative_error(actual, expected):
    difference = jax.tree.map(lambda left, right: left - right, actual, expected)
    return float(tree_norm(difference)) / max(
        float(tree_norm(expected)), np.finfo(np.float64).tiny
    )


def _gradient_parity_record(fast_gradient, production_gradient, direction):
    fast_flat, _ = ravel_pytree(fast_gradient)
    production_flat, _ = ravel_pytree(production_gradient)
    direction_flat, _ = ravel_pytree(direction)
    fast_values = np.asarray(fast_flat, dtype=np.float64)
    production_values = np.asarray(production_flat, dtype=np.float64)
    direction_values = np.asarray(direction_flat, dtype=np.float64)
    difference = fast_values - production_values
    absolute = float(np.linalg.norm(difference))
    production_norm = float(np.linalg.norm(production_values))
    fast_norm = float(np.linalg.norm(fast_values))
    denominator = fast_norm * production_norm
    cosine = None
    if denominator != 0.0:
        cosine = float(
            np.clip(
                np.dot(fast_values, production_values) / denominator,
                -1.0,
                1.0,
            )
        )
    fast_directional = float(np.dot(fast_values, direction_values))
    production_directional = float(
        np.dot(production_values, direction_values)
    )
    return {
        "absolute_l2_error": absolute,
        "relative_l2_error": absolute
        / max(production_norm, np.finfo(np.float64).tiny),
        "maximum_absolute_component_error": float(np.max(np.abs(difference))),
        "fast_l2_norm": fast_norm,
        "production_l2_norm": production_norm,
        "cosine_similarity": cosine,
        "fast_directional_derivative": fast_directional,
        "production_directional_derivative": production_directional,
        "directional_derivative_absolute_difference": abs(
            fast_directional - production_directional
        ),
    }


def prepare_and_certify_cache(training_configuration_path, cache_path):
    construction_started = perf_counter()
    training = load_discrete_training_configuration(training_configuration_path)
    source_path = training["source_objective_configuration"]
    load_discrete_offline_configuration(source_path)
    prepared = prepare_production_problem(source_path)
    dataset, dataset_metadata = load_operator_dataset(training["operator_dataset"])
    normalization = normalization_from_record(dataset_metadata["normalization"])
    truth_mesh = prepared.truth_metadata["mesh"]
    if prepared.truth_metadata["domain"]["type"] != "doubly_periodic_rectangle":
        raise ValueError("canonical Test 2A-3C cache requires the periodic truth mesh")
    periodic_cell_shape = (int(truth_mesh["nx"]), int(truth_mesh["ny"]))
    matrices, mass_audit = _matrix_cache_components(
        prepared,
        training["cache_certification"]["mass_inverse_residual_tolerance"],
        training["cache_certification"]["maximum_local_DG_mass_component_size"],
        training["cache_certification"][
            "tensor_mass_factorization_relative_tolerance"
        ],
        periodic_cell_shape=periodic_cell_shape,
    )
    h = np.asarray(dataset.features[:, 0], dtype=np.float64).reshape(81, 4096)
    first_parameters = prepared.objective.observations[0].payload.moist_parameters
    beta2 = float(first_parameters["g"] * first_parameters["L"])
    preliminary = FixedDiscreteCache(
        normalized_features=np.asarray(
            normalization.normalize_features(dataset.features), dtype=np.float64
        ),
        normalized_targets=np.asarray(
            normalization.normalize_a(dataset.targets), dtype=np.float64
        ),
        h=h,
        beta2=beta2,
        output_scale=normalization.output_scale,
        w_s_data=matrices["S"]["data"],
        w_s_indices=matrices["S"]["indices"],
        w_s_shape=matrices["S"]["shape"],
        w_q_data=matrices["Q"]["data"],
        w_q_indices=matrices["Q"]["indices"],
        w_q_shape=matrices["Q"]["shape"],
        mass_inverse_s_x=matrices["S"]["inverse_x"],
        mass_inverse_s_y=matrices["S"]["inverse_y"],
        mass_s_grid_order=matrices["S"]["grid_order"],
        mass_s_grid_shape=matrices["S"]["grid_shape"],
        mass_inverse_q_data=matrices["Q"]["mass_inverse_data"],
        mass_inverse_q_indices=matrices["Q"]["mass_inverse_indices"],
        mass_inverse_q_shape=matrices["Q"]["mass_inverse_shape"],
        normalizer=prepared.objective.normalizer,
        metadata={},
    )
    fast = FastFixedDiscreteObjective(
        preliminary, prepared.model_configuration, use_jit=True
    )
    vectors, _ = _deterministic_parameter_vectors(prepared)
    direct_parameters, direct_configuration = load_mlp_parameters(
        training["direct_production_method2_baseline"]["parameter_file"]
    )
    if direct_configuration != prepared.model_configuration:
        raise ValueError("direct-production Method-2 artifact architecture changed")
    direct_fingerprint = parameter_pytree_sha256(direct_parameters)
    if direct_fingerprint != training["direct_production_method2_baseline"][
        "parameter_pytree_sha256"
    ]:
        raise ValueError("direct-production Method-2 artifact fingerprint changed")
    vectors["direct_production_method2"] = direct_parameters
    initial_flat, unravel = ravel_pytree(prepared.initial_parameters)
    direction_flat = np.linspace(
        -0.7, 0.9, int(initial_flat.size), dtype=np.float64
    )
    direction_flat /= np.linalg.norm(direction_flat)
    certification_direction = unravel(
        jnp.asarray(direction_flat, dtype=jnp.float64)
    )
    certifications = []
    value_tolerance = float(
        training["cache_certification"]["value_relative_tolerance"]
    )
    gradient_tolerance = float(
        training["cache_certification"]["gradient_relative_tolerance"]
    )
    for name, parameters in vectors.items():
        production_value, production_gradient = prepared.objective.value_and_gradient(
            parameters
        )
        fast_value, fast_gradient = fast.value_and_gradient(parameters)
        value_relative = abs(fast_value - production_value) / max(
            abs(production_value), np.finfo(np.float64).tiny
        )
        gradient_relative = _tree_relative_error(fast_gradient, production_gradient)
        gradient_parity = _gradient_parity_record(
            fast_gradient, production_gradient, certification_direction
        )
        epsilon = 2.0e-6
        fast_centered_directional = (
            fast.value(tree_axpy(parameters, epsilon, certification_direction))
            - fast.value(tree_axpy(parameters, -epsilon, certification_direction))
        ) / (2.0 * epsilon)
        passed = value_relative <= value_tolerance and gradient_relative <= gradient_tolerance
        certifications.append(
            {
                "name": name,
                "parameter_pytree_sha256": parameter_pytree_sha256(parameters),
                "production_value": production_value,
                "cached_value": fast_value,
                "value_absolute_error": abs(fast_value - production_value),
                "value_relative_error": value_relative,
                "gradient_relative_error": gradient_relative,
                "gradient_parity": gradient_parity,
                "fast_centered_directional_derivative": fast_centered_directional,
                "fast_reverse_vs_centered_directional_absolute_difference": abs(
                    gradient_parity["fast_directional_derivative"]
                    - fast_centered_directional
                ),
                "passed": passed,
            }
        )
    if not all(record["passed"] for record in certifications):
        raise RuntimeError("fixed sparse objective failed production-oracle certification")
    frozen_record = next(
        record for record in certifications if record["name"] == "frozen_operator_trained"
    )
    accepted = training["operator_baseline"]
    if not np.isclose(
        frozen_record["production_value"],
        accepted["accepted_J_disc"],
        rtol=0.0,
        atol=value_tolerance * abs(accepted["accepted_J_disc"]),
    ):
        raise RuntimeError("production oracle no longer reproduces accepted theta_op J_disc")
    frozen_fast_operator = fast.objectives(prepared.frozen_operator_parameters)[1]
    if not np.isclose(
        frozen_fast_operator,
        accepted["accepted_J_op"],
        rtol=0.0,
        atol=value_tolerance * abs(accepted["accepted_J_op"]),
    ):
        raise RuntimeError("fixed cache no longer reproduces accepted theta_op J_op")
    direct_record = next(
        record
        for record in certifications
        if record["name"] == "direct_production_method2"
    )
    direct_accepted = training["direct_production_method2_baseline"]
    if not np.isclose(
        direct_record["production_value"],
        direct_accepted["accepted_J_disc"],
        rtol=0.0,
        atol=value_tolerance * abs(direct_accepted["accepted_J_disc"]),
    ):
        raise RuntimeError("production oracle no longer reproduces theta_disc J_disc")
    direct_fast_operator = fast.objectives(direct_parameters)[1]
    if not np.isclose(
        direct_fast_operator,
        direct_accepted["accepted_J_op"],
        rtol=0.0,
        atol=value_tolerance * abs(direct_accepted["accepted_J_op"]),
    ):
        raise RuntimeError("fixed cache no longer reproduces theta_disc J_op")
    metadata = {
        "benchmark_stage": "Test 2A-3C exact fixed-state operator cache",
        "training_configuration_sha256": _canonical_json_sha256(training),
        "source_objective_configuration": source_path,
        "operator_dataset": training["operator_dataset"],
        "operator_dataset_content_sha256": dataset_metadata["sha256_float64_content"],
        "normalization_refitted": False,
        "sample_count": dataset.sample_count,
        "mass_inverse_audit": mass_audit,
        "production_oracle_certified": True,
        "oracle_certifications": certifications,
        "cross_objective_direct_production_method2": {
            "J_op": direct_fast_operator,
            "J_disc": direct_record["production_value"],
            "parameter_pytree_sha256": direct_fingerprint,
        },
        "fixed_cache_contents": [
            "normalized truth-state features",
            "analytical A targets",
            "truth-state h factors",
            "beta2",
            "sparse production weak matrices",
            "exact tensor factors of the global CG mass inverse",
            "exact sparse inverses of local production DG mass blocks",
            "global objective normalizer",
        ],
        "dense_G_or_K_formed": False,
        "original_R_elimination": (
            "exact cancellation only: target and prediction evaluate the same "
            "parameter-independent original R at each fixed trusted state"
        ),
        "cache_construction_wall_seconds_before_serialization": float(
            perf_counter() - construction_started
        ),
        "hot_loop_firedrake_or_PETSc_actions": 0,
    }
    certified = FixedDiscreteCache(
        **{
            key: getattr(preliminary, key)
            for key in preliminary.__dataclass_fields__
            if key != "metadata"
        },
        metadata=metadata,
    )
    save_fixed_cache(cache_path, certified)
    return metadata


def benchmark_fixed_cache(
    configuration_path,
    cache_path,
    output,
    repeats=5,
    production_repeats=1,
):
    training = load_discrete_training_configuration(configuration_path)
    cache = load_fixed_cache(cache_path)
    selected = load_selected_configuration(training["selected_operator_configuration"])
    model_configuration = mlp_configuration_from_record(selected["model"])
    parameters = initialize_mlp(model_configuration)
    objective = FastFixedDiscreteObjective(cache, model_configuration, use_jit=True)
    started = perf_counter()
    objective.value(parameters)
    first_value = perf_counter() - started
    started = perf_counter()
    objective.value_and_gradient(parameters)
    first_gradient = perf_counter() - started
    value_times = []
    gradient_times = []
    for _ in range(int(repeats)):
        started = perf_counter()
        objective.value(parameters)
        value_times.append(perf_counter() - started)
        started = perf_counter()
        objective.value_and_gradient(parameters)
        gradient_times.append(perf_counter() - started)
    steady_value = float(np.median(value_times))
    steady_gradient = float(np.median(gradient_times))
    production = prepare_production_problem(training["source_objective_configuration"])
    probe_flat, probe_unravel = ravel_pytree(parameters)
    direction = np.linspace(-0.5, 0.8, int(probe_flat.size), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    production_value_times = []
    production_cached_gradient_times = []
    production_combined_times = []
    for index in range(int(production_repeats)):
        value_parameters = probe_unravel(
            probe_flat + (index + 1) * 1.0e-7 * direction
        )
        started = perf_counter()
        production.objective.value(value_parameters)
        production_value_times.append(perf_counter() - started)
        started = perf_counter()
        production.objective.gradient(value_parameters)
        production_cached_gradient_times.append(perf_counter() - started)
        combined_parameters = probe_unravel(
            probe_flat - (index + 1) * 1.0e-7 * direction
        )
        started = perf_counter()
        production.objective.value_and_gradient(combined_parameters)
        production_combined_times.append(perf_counter() - started)
    production_value = float(np.median(production_value_times))
    production_cached_gradient = float(
        np.median(production_cached_gradient_times)
    )
    production_combined = float(np.median(production_combined_times))
    reference = training["direct_production_method2_baseline"]
    reference_evaluations = (
        int(reference["objective_evaluations"]),
        int(reference["gradient_evaluations"]),
    )
    estimated_reference_work = (
        reference_evaluations[0] * steady_value
        + reference_evaluations[1] * steady_gradient
    )
    cache_array_bytes = sum(
        int(value.nbytes)
        for value in (
            cache.normalized_features,
            cache.normalized_targets,
            cache.h,
            cache.w_s_data,
            cache.w_s_indices,
            cache.w_q_data,
            cache.w_q_indices,
            cache.mass_inverse_s_x,
            cache.mass_inverse_s_y,
            cache.mass_s_grid_order,
            cache.mass_inverse_q_data,
            cache.mass_inverse_q_indices,
        )
    )
    result = {
        "status": "complete",
        "cache_construction_wall_seconds": cache.metadata[
            "cache_construction_wall_seconds_before_serialization"
        ],
        "cache_memory": {
            "uncompressed_array_bytes": cache_array_bytes,
            "npz_bytes": int(Path(cache_path).stat().st_size),
        },
        "first_JIT_value_wall_seconds": first_value,
        "first_JIT_value_and_gradient_wall_seconds": first_gradient,
        "steady_value_median_wall_seconds": steady_value,
        "steady_value_and_gradient_median_wall_seconds": steady_gradient,
        "repeats": int(repeats),
        "production_objective_median_wall_seconds": production_value,
        "production_gradient_after_cached_value_median_wall_seconds": (
            production_cached_gradient
        ),
        "production_fresh_value_and_gradient_median_wall_seconds": (
            production_combined
        ),
        "production_repeats": int(production_repeats),
        "objective_speedup": production_value / steady_value,
        "fresh_value_and_gradient_speedup": production_combined / steady_gradient,
        "estimated_seconds_for_reference_50k_evaluation_counts": (
            estimated_reference_work
        ),
        "reference_50k_evaluation_counts": {
            "objective": reference_evaluations[0],
            "gradient": reference_evaluations[1],
        },
        "estimate_excludes_checkpoint_diagnostics": True,
        "steady_hot_loop_firedrake_or_PETSc_solves": 0,
        "interpretation": (
            "timing estimate uses the accepted direct-production evaluation "
            "counts; actual accelerated ROL counts and wall time remain authoritative"
        ),
    }
    write_json_record(output, result)
    return result


class CompactCheckpointObjective(JAXPytreeObjective):
    """ROL objective retaining only two controls and scheduled checkpoints."""

    def __init__(self, *args, accepted_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.accepted_callback = accepted_callback
        self.accepted_update_count = 0
        self.previous_accepted_control = None
        self.current_accepted_control = None

    def _pending_record(self, flat):
        del flat
        return None

    def update(self, control, *args):
        update_type = str(args[0]) if args else "unspecified"
        if "Initial" not in update_type and "Accept" not in update_type:
            return
        values = np.asarray(
            self._flat_from_vector(control, "control"), dtype=np.float64
        ).copy()
        self.previous_accepted_control = self.current_accepted_control
        self.current_accepted_control = values
        local_index = self.accepted_update_count
        self.accepted_update_count += 1
        self.accepted_iteration_history.append(
            {
                "local_accepted_iteration": local_index,
                "rol_update_type": update_type,
                "rol_iteration_argument": int(args[1]) if len(args) > 1 else -1,
            }
        )
        if self.accepted_callback is not None:
            self.accepted_callback(control, local_index, self)

    def hessVec(self, output, direction, control, tolerance):
        del output, direction, control, tolerance
        self.hvp_evaluations += 1
        raise RuntimeError("Test 2A-3C L-BFGS must not request an HVP")


def _checkpoint_diagnostics(
    parameters, fast, model, normalization, dataset, gradient, iteration
):
    discrete, operator = fast.objectives(parameters)
    predictions = physical_predictions(
        parameters, model, normalization, dataset.features
    )
    return {
        "accepted_iteration": int(iteration),
        "J_disc": discrete,
        "J_op": operator,
        "gradient_norm_J_disc": float(tree_norm(gradient)),
        "physical_A_metrics": operator_metrics(predictions, dataset.targets),
        "parameter_pytree_sha256": parameter_pytree_sha256(parameters),
    }


def train_discrete_50k(
    configuration_path,
    cache_path,
    output_directory,
    *,
    resume=False,
):
    from pyrol import Problem, Solver

    training = load_discrete_training_configuration(configuration_path)
    cache = load_fixed_cache(cache_path)
    compatible_cache_configuration = training.get(
        "compatible_cache_training_configuration_sha256",
        _canonical_json_sha256(training),
    )
    if cache.metadata["training_configuration_sha256"] != compatible_cache_configuration:
        raise ValueError("fixed cache and training configuration are incompatible")
    output_root = Path(output_directory)
    result_path = output_root / "fit_result.json"
    progress_path = output_root / "fit_progress.json"
    if result_path.exists():
        raise FileExistsError("refusing to overwrite a complete Test 2A-3C fit")
    output_root.mkdir(parents=True, exist_ok=True)
    selected = load_selected_configuration(training["selected_operator_configuration"])
    model_configuration = mlp_configuration_from_record(selected["model"])
    dataset, dataset_metadata = load_operator_dataset(training["operator_dataset"])
    normalization = normalization_from_record(dataset_metadata["normalization"])
    model = DenseMLP(model_configuration)
    initial = load_training_initial_parameters(training, model_configuration)
    offset = 0
    previous_checkpoints = []
    cumulative = {
        "objective": 0,
        "gradient": 0,
        "HVP": 0,
        "wall_seconds": 0.0,
        "monitor_objective": 0,
        "monitor_gradient": 0,
        "monitor_physical_metrics": 0,
    }
    start_parameters = initial
    if resume:
        if not progress_path.exists():
            raise FileNotFoundError("no Test 2A-3C progress record to resume")
        progress = read_json_record(progress_path)
        offset = validate_resume_record(
            progress,
            _canonical_json_sha256(training),
            cache.metadata["cache_npz_sha256"],
        )
        start_parameters, checkpoint_configuration = load_mlp_parameters(
            progress["last_checkpoint_parameter_file"]
        )
        if checkpoint_configuration != model_configuration:
            raise ValueError("resume checkpoint architecture changed")
        if parameter_pytree_sha256(start_parameters) != progress[
            "last_checkpoint_parameter_pytree_sha256"
        ]:
            raise ValueError("resume checkpoint pytree fingerprint changed")
        if _file_sha256(progress["last_checkpoint_parameter_file"]) != progress[
            "last_checkpoint_parameter_npz_sha256"
        ]:
            raise ValueError("resume checkpoint file fingerprint changed")
        previous_checkpoints = list(progress.get("checkpoint_diagnostics", []))
        cumulative = dict(progress.get("cumulative_accounting", cumulative))
    elif progress_path.exists():
        raise FileExistsError("incomplete Test 2A-3C progress exists; use --resume")
    total_limit = int(training["optimizer"]["accepted_iteration_limit"])
    remaining = total_limit - offset
    if remaining <= 0:
        raise ValueError("latest checkpoint already reached the configured limit")
    fast = FastFixedDiscreteObjective(cache, model_configuration, use_jit=True)
    checkpoint_set = set(training["checkpoint_accepted_iterations"])
    progress_stride = int(training.get("progress_accepted_iteration_stride", 1000))
    if progress_stride < 1:
        raise ValueError("progress_accepted_iteration_stride must be positive")
    checkpoints = {int(value["accepted_iteration"]): value for value in previous_checkpoints}
    run_started = None
    initial_discrete = fast.value(initial)
    monitor_counts = {"objective": 1, "gradient": 0, "physical_metrics": 0}

    def accepted_callback(control, local_index, adapter):
        if local_index == 0:
            return
        global_iteration = offset + local_index
        is_checkpoint = global_iteration in checkpoint_set
        is_progress = global_iteration % progress_stride == 0
        if not is_checkpoint and not is_progress:
            return
        parameters = adapter.pytree_from_vector(control)
        _, gradient = fast.value_and_gradient(parameters)
        monitor_counts["gradient"] += 1
        record = _checkpoint_diagnostics(
            parameters,
            fast,
            model,
            normalization,
            dataset,
            gradient,
            global_iteration,
        )
        monitor_counts["objective"] += 1
        monitor_counts["physical_metrics"] += 1
        if adapter.previous_accepted_control is None:
            relative_step = None
        else:
            parameter_norm = np.linalg.norm(adapter.current_accepted_control)
            relative_step = float(
                np.linalg.norm(
                    adapter.current_accepted_control - adapter.previous_accepted_control
                )
                / max(parameter_norm, np.finfo(np.float64).tiny)
            )
        record["parameter_step_norm_relative_to_parameter_norm"] = relative_step
        record["objective_relative_reduction_from_initial"] = (
            initial_discrete - record["J_disc"]
        ) / initial_discrete
        if not training["initialization"].get("operator_pretraining", False):
            record["objective_relative_reduction_from_seed0"] = record[
                "objective_relative_reduction_from_initial"
            ]
        elapsed = 0.0 if run_started is None else perf_counter() - run_started
        if is_progress:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "accepted_iteration": global_iteration,
                        "J_disc": record["J_disc"],
                        "gradient_norm": record["gradient_norm_J_disc"],
                        "elapsed_wall_seconds_this_process": elapsed,
                        "objective_evaluations_this_process": adapter.value_evaluations,
                        "gradient_evaluations_this_process": adapter.gradient_evaluations,
                        "parameter_pytree_sha256": record["parameter_pytree_sha256"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not is_checkpoint:
            return
        parameter_path = output_root / f"parameters_iter_{global_iteration:05d}.npz"
        save_mlp_parameters_atomic(parameter_path, parameters, model_configuration)
        record.update(
            {
                "parameter_file": str(parameter_path.resolve()),
                "parameter_npz_sha256": _file_sha256(parameter_path),
            }
        )
        checkpoints[global_iteration] = record
        progress = {
            "status": "in_progress",
            "configuration_sha256": _canonical_json_sha256(training),
            "cache_npz_sha256": cache.metadata["cache_npz_sha256"],
            "last_checkpoint_accepted_iteration": global_iteration,
            "last_checkpoint_parameter_file": str(parameter_path.resolve()),
            "last_checkpoint_parameter_npz_sha256": record["parameter_npz_sha256"],
            "last_checkpoint_parameter_pytree_sha256": record[
                "parameter_pytree_sha256"
            ],
            "checkpoint_diagnostics": [checkpoints[key] for key in sorted(checkpoints)],
            "cumulative_accounting": {
                "objective": int(cumulative["objective"]) + adapter.value_evaluations,
                "gradient": int(cumulative["gradient"]) + adapter.gradient_evaluations,
                "HVP": int(cumulative["HVP"]) + adapter.hvp_evaluations,
                "wall_seconds": float(cumulative["wall_seconds"]) + elapsed,
                "monitor_objective": int(cumulative["monitor_objective"])
                + monitor_counts["objective"],
                "monitor_gradient": int(cumulative["monitor_gradient"])
                + monitor_counts["gradient"],
                "monitor_physical_metrics": int(
                    cumulative["monitor_physical_metrics"]
                )
                + monitor_counts["physical_metrics"],
            },
            "resume_contract": training["resume_contract"],
        }
        write_json_record(progress_path, progress)
        write_json_record(
            output_root / f"checkpoint_iter_{global_iteration:05d}.json", record
        )
        print(
            json.dumps(
                {
                    "event": "checkpoint",
                    "accepted_iteration": global_iteration,
                    "J_disc": record["J_disc"],
                    "gradient_norm": record["gradient_norm_J_disc"],
                    "elapsed_wall_seconds_this_process": elapsed,
                    "objective_evaluations_this_process": adapter.value_evaluations,
                    "gradient_evaluations_this_process": adapter.gradient_evaluations,
                    "parameter_pytree_sha256": record[
                        "parameter_pytree_sha256"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    adapter = CompactCheckpointObjective(
        fast.jax_value,
        start_parameters,
        use_jit=True,
        accepted_callback=accepted_callback,
    )
    control = adapter.vector_from_pytree(start_parameters)
    # Untimed compilation before ROL accounting.
    adapter.value(control, 0.0)
    from pyrol.vectors import NumPyVector

    warm_gradient = NumPyVector(np.full(adapter.dimension, np.nan, dtype=np.float64))
    adapter.gradient(warm_gradient, control, 0.0)
    adapter.reset_accounting()
    adapter.accepted_update_count = 0
    adapter.previous_accepted_control = None
    adapter.current_accepted_control = None
    if not resume and 0 in checkpoint_set:
        _, initial_gradient = fast.value_and_gradient(initial)
        monitor_counts["gradient"] += 1
        monitor_counts["physical_metrics"] += 1
        initial_record = _checkpoint_diagnostics(
            initial,
            fast,
            model,
            normalization,
            dataset,
            initial_gradient,
            0,
        )
        monitor_counts["objective"] += 1
        initial_record["parameter_step_norm_relative_to_parameter_norm"] = None
        initial_record["objective_relative_reduction_from_initial"] = 0.0
        parameter_path = output_root / "parameters_iter_00000.npz"
        save_mlp_parameters_atomic(parameter_path, initial, model_configuration)
        initial_record.update(
            {
                "parameter_file": str(parameter_path.resolve()),
                "parameter_npz_sha256": _file_sha256(parameter_path),
            }
        )
        checkpoints[0] = initial_record
        write_json_record(
            output_root / "checkpoint_iter_00000.json", initial_record
        )
        write_json_record(
            progress_path,
            {
                "status": "in_progress",
                "configuration_sha256": _canonical_json_sha256(training),
                "cache_npz_sha256": cache.metadata["cache_npz_sha256"],
                "last_checkpoint_accepted_iteration": 0,
                "last_checkpoint_parameter_file": str(parameter_path.resolve()),
                "last_checkpoint_parameter_npz_sha256": initial_record[
                    "parameter_npz_sha256"
                ],
                "last_checkpoint_parameter_pytree_sha256": initial_record[
                    "parameter_pytree_sha256"
                ],
                "checkpoint_diagnostics": [initial_record],
                "cumulative_accounting": {
                    **cumulative,
                    "monitor_objective": monitor_counts["objective"],
                    "monitor_gradient": monitor_counts["gradient"],
                    "monitor_physical_metrics": monitor_counts[
                        "physical_metrics"
                    ],
                },
                "resume_contract": training["resume_contract"],
            },
        )
    optimizer = training["optimizer"]
    rol_parameters = build_test2a_lbfgs_parameters(
        {
            "gradient_tolerance": optimizer["gradient_tolerance"],
            "step_tolerance": optimizer["step_tolerance"],
            "iteration_limit": remaining,
            "maximum_secant_storage": 20,
        }
    )
    solver = Solver(Problem(adapter, control), rol_parameters)
    print(
        json.dumps(
            {
                "event": "accelerated_fit_start",
                "accepted_iteration_offset": offset,
                "accepted_iteration_limit": total_limit,
                "remaining_iteration_budget": remaining,
                "initial_parameter_pytree_sha256": parameter_pytree_sha256(
                    start_parameters
                ),
                "cache_npz_sha256": cache.metadata["cache_npz_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    run_started = perf_counter()
    solver.solve()
    run_wall = perf_counter() - run_started
    state = solver.getAlgorithmState()
    final_parameters = adapter.pytree_from_vector(control)
    final_iteration = offset + int(state.iter)
    final_parameter_path = output_root / "final_parameters.npz"
    if final_parameter_path.exists() or final_parameter_path.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite final Test 2A-3C parameters")
    save_mlp_parameters_atomic(
        final_parameter_path, final_parameters, model_configuration
    )
    final_value, final_gradient = fast.value_and_gradient(final_parameters)
    monitor_counts["gradient"] += 1
    final_record = _checkpoint_diagnostics(
        final_parameters,
        fast,
        model,
        normalization,
        dataset,
        final_gradient,
        final_iteration,
    )
    if adapter.previous_accepted_control is None:
        final_record["parameter_step_norm_relative_to_parameter_norm"] = None
    else:
        final_parameter_norm = np.linalg.norm(adapter.current_accepted_control)
        final_record["parameter_step_norm_relative_to_parameter_norm"] = float(
            np.linalg.norm(
                adapter.current_accepted_control - adapter.previous_accepted_control
            )
            / max(final_parameter_norm, np.finfo(np.float64).tiny)
        )
    final_record["objective_relative_reduction_from_initial"] = (
        initial_discrete - final_record["J_disc"]
    ) / initial_discrete
    if not training["initialization"].get("operator_pretraining", False):
        final_record["objective_relative_reduction_from_seed0"] = final_record[
            "objective_relative_reduction_from_initial"
        ]
    monitor_counts["objective"] += 1
    monitor_counts["physical_metrics"] += 1
    operator_physics = load_mlp_parameters(
        training["operator_baseline"]["parameter_file"]
    )[0]
    if parameter_pytree_sha256(operator_physics) != training["operator_baseline"][
        "parameter_pytree_sha256"
    ]:
        raise ValueError("frozen operator comparison artifact fingerprint changed")
    operator_discrete, operator_operator = fast.objectives(operator_physics)
    final_flat, _ = ravel_pytree(final_parameters)
    operator_flat, _ = ravel_pytree(operator_physics)
    difference_norm = float(np.linalg.norm(np.asarray(final_flat - operator_flat)))
    operator_parameter_norm = float(np.linalg.norm(np.asarray(operator_flat)))
    predictions_final = physical_predictions(
        final_parameters, model, normalization, dataset.features
    )
    predictions_operator = physical_predictions(
        operator_physics, model, normalization, dataset.features
    )
    operator_gradient = jax.grad(lambda value: fast._objectives(value)[1])(
        final_parameters
    )
    gradient_comparison = objective_gradient_comparison(
        final_record["J_op"],
        operator_gradient,
        final_record["J_disc"],
        final_gradient,
    )
    counts = {
        "objective": int(cumulative["objective"]) + adapter.value_evaluations,
        "gradient": int(cumulative["gradient"]) + adapter.gradient_evaluations,
        "HVP": int(cumulative["HVP"]) + adapter.hvp_evaluations,
    }
    if counts["HVP"] != 0:
        raise RuntimeError("Test 2A-3C unexpectedly used a production HVP")
    result = {
        "status": "complete",
        "benchmark_stage": training["benchmark_stage"],
        "initialization": {
            **training["initialization"],
            "verified": True,
        },
        "optimizer": {
            **optimizer,
            "accepted_iterations": final_iteration,
            "actual_ROL_termination_reason": str(state.statusFlag),
            "objective_evaluations": counts["objective"],
            "gradient_evaluations": counts["gradient"],
            "HVP_evaluations": counts["HVP"],
            "wall_time_seconds": float(cumulative["wall_seconds"]) + run_wall,
            "secant_history_restored_on_resume": False if resume else True,
            "source_optimizer_secant_history_reused": False,
        },
        "final_diagnostics": final_record,
        "checkpoint_diagnostics": [checkpoints[key] for key in sorted(checkpoints)],
        "cross_objective_table": {
            "theta_op": {"J_op": operator_operator, "J_disc": operator_discrete},
            "theta_disc": {
                "J_op": final_record["J_op"],
                "J_disc": final_record["J_disc"],
            },
        },
        "parameter_comparison": {
            "absolute_l2_difference": difference_norm,
            "relative_to_theta_op_l2": difference_norm
            / max(operator_parameter_norm, np.finfo(np.float64).tiny),
            "A_prediction_relative_l2_difference": float(
                np.linalg.norm(predictions_final - predictions_operator)
                / max(np.linalg.norm(predictions_operator), np.finfo(np.float64).tiny)
            ),
            "objective_gradient_comparison_at_theta_disc": gradient_comparison,
        },
        "final_parameter_file": str(final_parameter_path.resolve()),
        "final_parameter_pytree_sha256": parameter_pytree_sha256(final_parameters),
        "fixed_cache": {
            "path": str(Path(cache_path).resolve()),
            "npz_sha256": cache.metadata["cache_npz_sha256"],
            "production_oracle_certified": True,
            "dense_G_or_K_formed": False,
        },
        "truth_state_access": {
            "state_indices": [0, 80],
            "states_after_80_accessed": False,
            "recursive_model_state_propagation": False,
        },
        "checkpoint_monitoring_evaluations": {
            "objective": int(cumulative["monitor_objective"])
            + monitor_counts["objective"],
            "gradient": int(cumulative["monitor_gradient"])
            + monitor_counts["gradient"],
            "physical_metrics": int(cumulative["monitor_physical_metrics"])
            + monitor_counts["physical_metrics"],
        },
    }
    write_json_record(result_path, result)
    write_json_record(
        progress_path,
        {
            "status": "complete",
            "configuration_sha256": _canonical_json_sha256(training),
            "cache_npz_sha256": cache.metadata["cache_npz_sha256"],
            "last_checkpoint_accepted_iteration": final_iteration,
            "last_checkpoint_parameter_file": str(final_parameter_path.resolve()),
            "last_checkpoint_parameter_npz_sha256": _file_sha256(final_parameter_path),
            "last_checkpoint_parameter_pytree_sha256": parameter_pytree_sha256(
                final_parameters
            ),
            "checkpoint_diagnostics": result["checkpoint_diagnostics"],
            "cumulative_accounting": {
                **counts,
                "wall_seconds": result["optimizer"]["wall_time_seconds"],
                "monitor_objective": result[
                    "checkpoint_monitoring_evaluations"
                ]["objective"],
                "monitor_gradient": result[
                    "checkpoint_monitoring_evaluations"
                ]["gradient"],
                "monitor_physical_metrics": result[
                    "checkpoint_monitoring_evaluations"
                ]["physical_metrics"],
            },
        },
    )
    print(
        json.dumps(
            {
                "event": "accelerated_fit_complete",
                "accepted_iterations": final_iteration,
                "J_disc": final_record["J_disc"],
                "gradient_norm": final_record["gradient_norm_J_disc"],
                "termination_reason": str(state.statusFlag),
                "wall_time_seconds": result["optimizer"]["wall_time_seconds"],
                "objective_evaluations": counts["objective"],
                "gradient_evaluations": counts["gradient"],
                "parameter_pytree_sha256": result[
                    "final_parameter_pytree_sha256"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-cache")
    prepare.add_argument("--configuration", required=True)
    prepare.add_argument("--cache", required=True)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--configuration", required=True)
    benchmark.add_argument("--cache", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--production-repeats", type=int, default=1)
    train = subparsers.add_parser("train")
    train.add_argument("--configuration", required=True)
    train.add_argument("--cache", required=True)
    train.add_argument("--output-directory", required=True)
    train.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare-cache":
        metadata = prepare_and_certify_cache(
            arguments.configuration, arguments.cache
        )
        print(
            json.dumps(
                {
                    "event": "fixed_cache_certified",
                    "cache": str(Path(arguments.cache).resolve()),
                    "construction_wall_seconds": metadata[
                        "cache_construction_wall_seconds_before_serialization"
                    ],
                    "oracle_probe_count": len(
                        metadata["oracle_certifications"]
                    ),
                    "production_oracle_certified": metadata[
                        "production_oracle_certified"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elif arguments.command == "benchmark":
        result = benchmark_fixed_cache(
            arguments.configuration,
            arguments.cache,
            arguments.output,
            repeats=arguments.repeats,
            production_repeats=arguments.production_repeats,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    elif arguments.command == "train":
        train_discrete_50k(
            arguments.configuration,
            arguments.cache,
            arguments.output_directory,
            resume=arguments.resume,
        )
    else:
        raise AssertionError("unreachable Test 2A-3C command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CompactCheckpointObjective",
    "FastFixedDiscreteObjective",
    "FixedDiscreteCache",
    "benchmark_fixed_cache",
    "load_discrete_training_configuration",
    "load_fixed_cache",
    "load_training_initial_parameters",
    "prepare_and_certify_cache",
    "save_fixed_cache",
    "train_discrete_50k",
    "validate_resume_record",
)
