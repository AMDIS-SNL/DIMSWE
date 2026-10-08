"""Prepare and certify the Test-2A Problem-B five-objective campaign.

Only explicit ``train`` invocations optimize parameters.  Preparation,
certification, and benchmarking are bounded diagnostics over truth states
0..80 and never launch a detached process.
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

from .learned_physics.parameters import tree_axpy, tree_dot, tree_norm
from .resolved_hidden_c0 import read_json_record, write_json_record
from .test2a_discrete_offline import (
    DeployedDiscreteOfflineObjective,
    DiscretePredictionCache,
    ProductionDiscreteOfflineOperations,
    prepare_production_problem,
)
from .test2a_discrete_training import _matrix_cache_components
from .test2a_embedded_moist import parameter_pytree_sha256
from .test2a_operator import load_operator_dataset, normalization_from_record as a_norm_from_record
from .test2a_problem_b import (
    FourTendencyNormalization,
    NeuralFourTendencyMoistPhysics,
    ProblemBMLPConfiguration,
    ProblemBOperatorDataset,
    ProblemBOperatorObjective,
    SOURCE_ORDER,
    initial_problem_b_parameters,
    load_problem_b_parameters,
    normalization_from_record,
    save_problem_b_parameters,
)


HORIZONS = (1, 2, 5)
BOUNDARY_STEPS = tuple(range(81))
POSTPREFIX_STEPS = tuple(range(80))
PRODUCTION_ARTIFACT_STAGES = {
    "M1": "M1",
    "M2-X-independent": "M2-X-independent",
    "M1-to-M2-X": "M1-to-M2-X",
    "H1": "H1",
    "H2": "H2",
    "H5": "H5",
}
RESOLVED_DIAGNOSTIC_CONFIGURATION_ATTRIBUTES = (
    "sampling_shape",
    "high_wavenumber_fraction",
)


@dataclass(frozen=True)
class ProblemBDiagnosticConfiguration:
    """Minimal resolved-diagnostic interface with resolved-run provenance.

    Problem B uses the same spectral diagnostics as the accepted autonomous
    evaluator.  Values are copied from the ``ResolvedPilotConfiguration``
    reconstructed from the stored truth metadata; they are not independent
    Problem-B scientific settings.
    """

    sampling_shape: tuple[int, int]
    high_wavenumber_fraction: float

    def __post_init__(self):
        shape = tuple(int(value) for value in self.sampling_shape)
        if len(shape) != 2 or any(value < 4 for value in shape):
            raise ValueError("diagnostic sampling_shape must contain two dimensions >= 4")
        high = float(self.high_wavenumber_fraction)
        if not np.isfinite(high) or not 0.0 < high < 1.0:
            raise ValueError("diagnostic high_wavenumber_fraction must be in (0, 1)")
        object.__setattr__(self, "sampling_shape", shape)
        object.__setattr__(self, "high_wavenumber_fraction", high)

    @classmethod
    def from_resolved_pilot(cls, configuration):
        missing = tuple(
            name
            for name in RESOLVED_DIAGNOSTIC_CONFIGURATION_ATTRIBUTES
            if not hasattr(configuration, name)
        )
        if missing:
            raise TypeError(
                "resolved pilot lacks diagnostic configuration attributes: "
                + ", ".join(missing)
            )
        return cls(
            sampling_shape=configuration.sampling_shape,
            high_wavenumber_fraction=configuration.high_wavenumber_fraction,
        )


def _verify_completed_training_artifact(label, path):
    """Validate one immutable production artifact before postprocessing."""

    if label not in PRODUCTION_ARTIFACT_STAGES:
        raise ValueError(f"unknown Problem-B production artifact label {label!r}")
    source = Path(path).resolve()
    if source.name != "final_parameters.npz" or not source.is_file():
        raise FileNotFoundError(f"missing final Problem-B parameters: {source}")
    expected_stage = PRODUCTION_ARTIFACT_STAGES[label]
    records = {}
    for name in ("fit_result.json", "fit_progress.json"):
        record_path = source.parent / name
        if not record_path.is_file():
            raise FileNotFoundError(f"missing completed-fit record: {record_path}")
        record = read_json_record(record_path)
        if record.get("status") != "complete":
            raise ValueError(f"{record_path} does not report status complete")
        if record.get("stage") != expected_stage:
            raise ValueError(
                f"{record_path} stage {record.get('stage')!r} does not match "
                f"{expected_stage!r}"
            )
        if Path(record.get("final_parameter_file", "")).resolve() != source:
            raise ValueError(f"{record_path} points to a different final artifact")
        records[name] = record
    parameters, _, sidecar = load_problem_b_parameters(source)
    pytree_sha = parameter_pytree_sha256(parameters)
    if sidecar.get("parameter_pytree_sha256") != pytree_sha:
        raise ValueError("Problem-B final parameter sidecar SHA mismatch")
    for name, record in records.items():
        if record.get("final_parameter_pytree_sha256") != pytree_sha:
            raise ValueError(f"{name} final parameter SHA mismatch")
    return {
        "label": label,
        "stage": expected_stage,
        "status": "complete",
        "final_parameter_file": str(source),
        "parameter_pytree_sha256": pytree_sha,
        "fit_result_status": records["fit_result.json"]["status"],
        "fit_progress_status": records["fit_progress.json"]["status"],
    }


def production_windows(horizon):
    from .test2a_trajectory import reset_windows

    horizon = int(horizon)
    if horizon not in HORIZONS:
        raise ValueError("Problem B horizon must be 1, 2, or 5")
    return reset_windows(tuple(range(0, 80, horizon)), horizon, "accumulated", (1.0,) * horizon)


def _file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(record):
    return sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_problem_b_configuration(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("benchmark_stage") != "Test 2A Problem B preparation and campaign":
        raise ValueError("not the frozen Test-2A Problem-B configuration")
    if record["truth"] != {
        "allowed_state_indices": [0, 80],
        "states_after_80_forbidden": True,
        "trajectory_configuration": "dimswe/configs/test2a_trajectory_prep.json",
        "c0": 0.14,
        "m2x_oracle_configuration": "dimswe/configs/test2a_deployed_discrete_offline.json",
    }:
        raise ValueError("Problem B truth-support contract changed")
    if record["model"] != {
        "features": ["h", "S", "Qv", "Qc", "B"],
        "layers": [5, 32, 32, 4],
        "activation": "tanh",
        "dtype": "float64",
        "seed": 0,
        "parameter_count": 1380,
        "source_order": ["S", "Qv", "Qc", "Qr"],
        "source_structure_enforced": False,
    }:
        raise ValueError("Problem B model contract changed")
    schedules = record["objectives"]["dense_schedules"]
    expected = {"1": [0, 79, 1], "2": [0, 78, 2], "5": [0, 75, 5]}
    if schedules != expected:
        raise ValueError("Problem B dense window schedules changed")
    optimizer = record["optimizer"]
    if (
        optimizer["library"] != "PyROL/ROL"
        or optimizer["method"] != "line-search L-BFGS"
        or optimizer["maximum_secant_storage"] != 20
        or optimizer["gradient_tolerance"] != 1.0e-8
        or optimizer["step_tolerance"] != 1.0e-12
        or optimizer["production_HVP"] is not False
    ):
        raise ValueError("Problem B optimizer contract changed")
    return record


@dataclass(frozen=True)
class FourSourceFixedCache:
    normalized_features: np.ndarray
    physical_targets: np.ndarray
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
    normalization: FourTendencyNormalization
    metadata: dict

    def __post_init__(self):
        features = np.asarray(self.normalized_features, dtype=np.float64)
        targets = np.asarray(self.physical_targets, dtype=np.float64)
        if features.ndim != 3 or features.shape[-1] != 5:
            raise ValueError("fixed four-source features must be (states,points,5)")
        if targets.shape != (*features.shape[:2], 4):
            raise ValueError("fixed four-source targets must be (states,points,4)")
        if self.w_s_shape[1] != features.shape[1] or self.w_q_shape[1] != features.shape[1]:
            raise ValueError("weak-map sample dimension changed")
        if not np.isfinite(self.normalizer) or self.normalizer <= 0.0:
            raise ValueError("four-source normalizer must be positive")

    @property
    def state_count(self):
        return int(self.normalized_features.shape[0])


class FastFourSourceDiscreteObjective:
    """Exact fixed G4 source-to-state objective with zero solver hot-loop calls."""

    def __init__(self, cache, *, use_jit=True):
        from .test2a_problem_b import build_problem_b_model

        self.cache = cache
        self.model = build_problem_b_model()
        features = jnp.asarray(cache.normalized_features, dtype=jnp.float64)
        targets = jnp.asarray(cache.physical_targets, dtype=jnp.float64)
        w_s = jsparse.BCOO(
            (jnp.asarray(cache.w_s_data), jnp.asarray(cache.w_s_indices)),
            shape=cache.w_s_shape,
        )
        w_q = jsparse.BCOO(
            (jnp.asarray(cache.w_q_data), jnp.asarray(cache.w_q_indices)),
            shape=cache.w_q_shape,
        )
        inverse_q = jsparse.BCOO(
            (
                jnp.asarray(cache.mass_inverse_q_data),
                jnp.asarray(cache.mass_inverse_q_indices),
            ),
            shape=cache.mass_inverse_q_shape,
        )
        inverse_x = jnp.asarray(cache.mass_inverse_s_x)
        inverse_y = jnp.asarray(cache.mass_inverse_s_y)
        grid_order = jnp.asarray(cache.mass_s_grid_order)
        grid_shape = tuple(cache.mass_s_grid_shape)
        scales = jnp.asarray(cache.normalization.output_scales)

        def component_energy(error):
            weak_s = (w_s @ error[:, :, 0].T).T
            weak_s_grid = weak_s[:, grid_order].reshape((error.shape[0], *grid_shape))
            riesz_s_grid = jnp.einsum(
                "ij,bjk,lk->bil", inverse_y, weak_s_grid, inverse_x
            )
            total = jnp.sum(weak_s_grid * riesz_s_grid)
            for component in (1, 2, 3):
                weak_q = (w_q @ error[:, :, component].T).T
                riesz_q = (inverse_q @ weak_q.T).T
                total = total + jnp.sum(weak_q * riesz_q)
            return total

        def objective(parameters):
            coordinates = self.model(parameters, features)
            physical = coordinates * scales
            return component_energy(physical - targets) / float(cache.normalizer)

        self._value = jax.jit(objective) if use_jit else objective
        value_gradient = jax.value_and_grad(objective)
        self._value_gradient = jax.jit(value_gradient) if use_jit else value_gradient

    def value(self, parameters):
        return float(self._value(parameters))

    def jax_value(self, parameters):
        return self._value(parameters)

    def value_and_gradient(self, parameters):
        value, gradient = self._value_gradient(parameters)
        return float(value), gradient

    def gradient(self, parameters):
        return self.value_and_gradient(parameters)[1]


class ProductionFourSourceOperations(ProductionDiscreteOfflineOperations):
    """Immutable-oracle operations without Problem-A's shared-R assertion."""

    def predict(self, parameters, observation):
        payload = observation.payload
        combined = self._combined_kernel(
            parameters,
            payload.packed_state,
            payload.packed_fields,
            payload.moist_parameters,
        )
        source = self.helper._from_device_tree(combined["source"])
        source_dual = self.helper.source_assembly(source)
        tendency = self.helper.state_riesz_representative(
            source_dual, f"test2b_prediction_tendency_{observation.step}"
        )
        return DiscretePredictionCache(
            tendency=tendency,
            auxiliary={"source": source, "rates": {}},
        )


def _carrier_mass_weights(helper):
    from firedrake import TestFunction, TrialFunction, assemble, inner
    from .test2a_discrete_training import _scipy_csr_from_petsc

    carrier = helper.carrier_space
    matrix = assemble(
        inner(TestFunction(carrier), TrialFunction(carrier)) * helper.model.spaces.dx,
        mat_type="aij",
    )
    csr = _scipy_csr_from_petsc(matrix).tocsr()
    diagonal = csr.diagonal()
    residual = csr.copy()
    residual.setdiag(0.0)
    residual.eliminate_zeros()
    maximum = 0.0 if residual.nnz == 0 else float(np.max(np.abs(residual.data)))
    tolerance = 1024.0 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(diagonal))), 1.0
    )
    if maximum > tolerance:
        raise RuntimeError("accepted GLL carrier mass is not diagonal")
    packed = np.asarray(helper.layout.cell_nodes, dtype=np.int64).reshape(-1)
    weights = np.asarray(diagonal[packed], dtype=np.float64)
    if np.any(weights <= 0.0):
        raise RuntimeError("carrier mass weights must be positive")
    return weights, {
        "carrier_dimension": int(carrier.dim()),
        "maximum_offdiagonal_absolute": maximum,
        "diagonal_acceptance_tolerance": tolerance,
        "packed_weight_sum": float(np.sum(weights)),
    }


def _physical_arrays_from_observations(prepared):
    helper = prepared.objective.operations.helper
    features = []
    targets = []
    for observation in prepared.objective.observations:
        payload = observation.payload
        combined = helper.primal_helper._combined_kernel(
            helper._to_device_tree(payload.packed_state),
            helper._to_device_tree(payload.packed_fields),
            helper._to_device_tree(payload.moist_parameters),
        )
        source = helper._from_device_tree(combined["source"])
        features.append(
            np.stack(
                (
                    payload.packed_state["h"],
                    payload.packed_state["S"],
                    payload.packed_state["Qv"],
                    payload.packed_state["Qc"],
                    payload.packed_fields["B"],
                ),
                axis=-1,
            ).reshape(-1, 5)
        )
        targets.append(
            np.stack(tuple(source[name] for name in SOURCE_ORDER), axis=-1).reshape(-1, 4)
        )
    if len(features) != 81:
        raise RuntimeError("Problem B preparation did not obtain states 0..80")
    return np.stack(features), np.stack(targets)


def _postprefix_arrays(configuration, physics):
    from .jax_moist_hvp import JAXMoistEulerHVP
    from .test2a_trajectory_certification import _build_case

    _, case, truth, _, _ = _build_case(
        configuration["truth"]["trajectory_configuration"], maximum_truth_step=80
    )
    analytical = JAXMoistEulerHVP(
        case.helper.moist_child.ufl_oracle, use_jit=True, local_physics=None
    )
    features = []
    targets = []
    with case.physical_c0(float(configuration["truth"]["c0"])):
        for step in POSTPREFIX_STEPS:
            prefix = case.helper.take_fixed_prefix_cached(
                truth[step], case.t0 + step * case.dt, case.dt
            )
            target = analytical.take_forward_step_cached(
                prefix.state_out, case.t0 + step * case.dt, case.dt
            )
            packed = target.packed_state
            features.append(
                np.stack(
                    (
                        packed["h"], packed["S"], packed["Qv"], packed["Qc"],
                        target.packed_fields["B"],
                    ),
                    axis=-1,
                ).reshape(-1, 5)
            )
            targets.append(
                np.stack(
                    tuple(target.source_density[name] for name in SOURCE_ORDER), axis=-1
                ).reshape(-1, 4)
            )
    return case, truth, np.stack(features), np.stack(targets)


def _fixed_energy(targets, matrices):
    from scipy.sparse import coo_matrix

    w_s = coo_matrix(
        (matrices["S"]["data"], (matrices["S"]["indices"][:, 0], matrices["S"]["indices"][:, 1])),
        shape=matrices["S"]["shape"],
    ).tocsr()
    w_q = coo_matrix(
        (matrices["Q"]["data"], (matrices["Q"]["indices"][:, 0], matrices["Q"]["indices"][:, 1])),
        shape=matrices["Q"]["shape"],
    ).tocsr()
    grid_order = np.asarray(matrices["S"]["grid_order"], dtype=np.int64)
    ny, nx = matrices["S"]["grid_shape"]
    inverse_y = matrices["S"]["inverse_y"]
    inverse_x = matrices["S"]["inverse_x"]
    from scipy.sparse import coo_matrix as coo
    inverse_q = coo(
        (
            matrices["Q"]["mass_inverse_data"],
            (
                matrices["Q"]["mass_inverse_indices"][:, 0],
                matrices["Q"]["mass_inverse_indices"][:, 1],
            ),
        ),
        shape=matrices["Q"]["mass_inverse_shape"],
    ).tocsr()
    total = 0.0
    for state in targets:
        weak = np.asarray(w_s @ state[:, 0])
        weak_grid = weak[grid_order].reshape(ny, nx)
        riesz_grid = inverse_y @ weak_grid @ inverse_x.T
        total += float(np.sum(weak_grid * riesz_grid))
        for component in (1, 2, 3):
            weak_q = np.asarray(w_q @ state[:, component])
            total += float(weak_q @ (inverse_q @ weak_q))
    return total


def _cache_from_arrays(features, targets, normalization, matrices, metadata):
    normalizer = _fixed_energy(targets, matrices)
    return FourSourceFixedCache(
        normalized_features=np.asarray(normalization.normalize_features(features)),
        physical_targets=targets,
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
        normalizer=normalizer,
        normalization=normalization,
        metadata=metadata,
    )


def _save_preparation(path, boundary, postprefix, dataset, metadata):
    destination = Path(path)
    sidecar = destination.with_suffix(".json")
    if destination.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite Problem B preparation data")
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "m1_normalized_features": dataset.normalized_features,
        "m1_physical_targets": dataset.physical_targets,
        "m1_spatial_weights": dataset.spatial_weights,
    }
    for prefix, cache in (("m2x", boundary), ("h1", postprefix)):
        for field in cache.__dataclass_fields__:
            if field in ("normalization", "metadata"):
                continue
            value = getattr(cache, field)
            arrays[prefix + "_" + field] = np.asarray(value)
    temporary = destination.with_name(destination.name + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(destination)
    metadata = dict(metadata)
    metadata["preparation_npz_sha256"] = _file_sha256(destination)
    metadata["normalization"] = dataset.normalization.to_record()
    write_json_record(sidecar, metadata)
    return metadata


def _cache_from_archive(archive, prefix, normalization, metadata):
    def array(name):
        return np.array(archive[prefix + "_" + name], copy=True)

    return FourSourceFixedCache(
        normalized_features=array("normalized_features"),
        physical_targets=array("physical_targets"),
        w_s_data=array("w_s_data"),
        w_s_indices=array("w_s_indices"),
        w_s_shape=tuple(int(x) for x in array("w_s_shape")),
        w_q_data=array("w_q_data"),
        w_q_indices=array("w_q_indices"),
        w_q_shape=tuple(int(x) for x in array("w_q_shape")),
        mass_inverse_s_x=array("mass_inverse_s_x"),
        mass_inverse_s_y=array("mass_inverse_s_y"),
        mass_s_grid_order=array("mass_s_grid_order"),
        mass_s_grid_shape=tuple(int(x) for x in array("mass_s_grid_shape")),
        mass_inverse_q_data=array("mass_inverse_q_data"),
        mass_inverse_q_indices=array("mass_inverse_q_indices"),
        mass_inverse_q_shape=tuple(int(x) for x in array("mass_inverse_q_shape")),
        normalizer=float(array("normalizer")),
        normalization=normalization,
        metadata=metadata,
    )


def load_preparation(path):
    source = Path(path)
    metadata = read_json_record(source.with_suffix(".json"))
    if metadata.get("status") != "complete" or metadata.get(
        "states_after_80_accessed", True
    ):
        raise ValueError("Problem B preparation is incomplete or violates truth support")
    if _file_sha256(source) != metadata["preparation_npz_sha256"]:
        raise ValueError("Problem B preparation fingerprint mismatch")
    normalization = normalization_from_record(metadata["normalization"])
    with np.load(source, allow_pickle=False) as archive:
        dataset = ProblemBOperatorDataset(
            normalized_features=np.array(archive["m1_normalized_features"], copy=True),
            physical_targets=np.array(archive["m1_physical_targets"], copy=True),
            spatial_weights=np.array(archive["m1_spatial_weights"], copy=True),
            normalization=normalization,
            metadata={"truth_state_indices": [0, 80], "states_after_80_accessed": False},
        )
        m2x = _cache_from_archive(
            archive, "m2x", normalization,
            {"objective": "M2-X", "truth_state_indices": [0, 80]},
        )
        h1 = _cache_from_archive(
            archive, "h1", normalization,
            {"objective": "H1/M2-Y", "truth_state_indices": [0, 80]},
        )
    return metadata, dataset, m2x, h1


def _build_problem_b_case(configuration, normalization, parameters, maximum_truth_step):
    from dataclasses import replace
    from .hidden_c0 import _flat_values
    from .resolved_hidden_c0 import ResolvedPilotConfiguration
    from .resolved_hidden_c0_driver import build_resolved_hidden_c0_case
    from .resolved_hidden_c0_inference import load_resolved_truth
    from .selected_test1b import load_selected_test1b_plan
    from .test2a_trajectory_certification import load_trajectory_preparation_configuration

    if int(maximum_truth_step) > 80:
        raise ValueError("Problem B may not materialize a truth state after 80")
    selected = load_trajectory_preparation_configuration(
        configuration["truth"]["trajectory_configuration"]
    )
    _, plan = load_selected_test1b_plan(selected["truth"]["selected_plan"])
    inference = plan.inference_configuration(Path(selected["truth"]["run_directory"]).resolve())
    _, loaded = load_resolved_truth(inference, include_heldout=False)
    if tuple(loaded.states) != BOUNDARY_STEPS:
        raise ValueError("Problem B truth loader must expose exactly states 0..80")
    physics = NeuralFourTendencyMoistPhysics(
        parameters, normalization, use_jit=True,
        provenance={"problem": "Test2A-B", "states_after_80_accessed": False},
    )
    pilot = ResolvedPilotConfiguration.from_dict(loaded.metadata["configuration"])
    neural_pilot = replace(
        pilot,
        moist_backend="jax",
        output_directory="/tmp/test2a-problem-b-no-output",
    )
    case = build_resolved_hidden_c0_case(
        neural_pilot, jax_moist_local_physics=physics
    )
    truth = {
        step: case.state_from_values(
            _flat_values(loaded.states[step]), f"test2a_problem_b_truth_{step}"
        )
        for step in range(int(maximum_truth_step) + 1)
    }
    diagnostic_configuration = ProblemBDiagnosticConfiguration.from_resolved_pilot(
        neural_pilot
    )
    return case, truth, physics, diagnostic_configuration


def _direction(parameters):
    flat, unravel = ravel_pytree(parameters)
    values = np.linspace(-0.7, 0.9, int(flat.size), dtype=np.float64)
    values /= np.linalg.norm(values)
    return unravel(jnp.asarray(values, dtype=jnp.float64))


def _gradient_relation(left, right):
    left_flat, _ = ravel_pytree(left)
    right_flat, _ = ravel_pytree(right)
    left_values = np.asarray(left_flat, dtype=np.float64)
    right_values = np.asarray(right_flat, dtype=np.float64)
    difference = left_values - right_values
    dot = float(left_values @ right_values)
    denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    return {
        "absolute_error": float(np.linalg.norm(difference)),
        "relative_error_to_right": float(
            np.linalg.norm(difference)
            / max(np.linalg.norm(right_values), np.finfo(np.float64).tiny)
        ),
        "cosine": None if denominator == 0.0 else dot / denominator,
        "maximum_absolute_component": float(np.max(np.abs(difference))),
    }


def _directional_record(objective, parameters, direction, epsilon=2.0e-6):
    value, gradient = objective.value_and_gradient(parameters)
    adjoint = float(tree_dot(gradient, direction))
    centered = (
        objective.value(tree_axpy(parameters, epsilon, direction))
        - objective.value(tree_axpy(parameters, -epsilon, direction))
    ) / (2.0 * epsilon)
    return {
        "objective": value,
        "gradient_norm": float(tree_norm(gradient)),
        "gradient_parameter_count": sum(
            int(x.size) for x in jax.tree_util.tree_leaves(gradient)
        ),
        "directional_adjoint": adjoint,
        "directional_centered_fd": centered,
        "absolute_discrepancy": abs(adjoint - centered),
        "scale_aware_relative_discrepancy": abs(adjoint - centered)
        / max(abs(adjoint), abs(centered), np.finfo(np.float64).tiny),
    }


def _state_duality(case, objective, parameters, horizon):
    from .hidden_c0 import _copy_function
    from .learned_physics.parameters import tree_zeros

    tape = objective._tape(parameters).windows[0]
    state_direction = _copy_function(
        objective.truth_states[0], f"problem_b_H{horizon}_state_direction"
    )
    with state_direction.dat.vec as vector:
        vector.scale(1.0e-7)
    current = state_direction
    zeros = tree_zeros(parameters)
    for cache in tape.step_caches:
        current = case.helper.take_neural_parameter_tangent_step(
            cache, current, zeros
        ).state_direction_out
    probe_state = _copy_function(
        objective.truth_states[horizon], f"problem_b_H{horizon}_probe"
    )
    with probe_state.dat.vec as vector:
        vector.scale(1.0e-7)
    probe = case.helper.state_mass_map(probe_state, f"problem_b_H{horizon}_probe_dual")
    adjoint = probe
    for cache in reversed(tape.step_caches):
        adjoint = case.helper.take_neural_parameter_adjoint_step(
            cache, adjoint, stop_at_fixed_prefix=False
        ).state_adjoint_in
    left = case.helper.dual_pairing(probe, current)
    right = case.helper.dual_pairing(adjoint, state_direction)
    return {
        "tangent_pairing": left,
        "adjoint_pairing": right,
        "absolute_discrepancy": abs(left - right),
        "relative_discrepancy": abs(left - right)
        / max(abs(left), abs(right), np.finfo(np.float64).tiny),
    }


def certify(configuration_path, preparation_path, output_path):
    from .test2a_trajectory import GlobalMixedMassMetric, NeuralTrajectoryObjective, reset_windows
    configuration = load_problem_b_configuration(configuration_path)
    metadata, dataset, m2x_cache, h1_cache = load_preparation(preparation_path)
    parameters = initial_problem_b_parameters()
    direction = _direction(parameters)
    result = {
        "status": "in_progress",
        "benchmark_stage": "Test 2A Problem B derivative and objective certification",
        "truth_state_indices": [0, 80],
        "states_after_80_accessed": False,
        "parameter_count": 1380,
        "seed0_parameter_pytree_sha256": parameter_pytree_sha256(parameters),
        "preparation_npz_sha256": metadata["preparation_npz_sha256"],
    }
    operator = ProblemBOperatorObjective(dataset)
    result["M1_gradient"] = _directional_record(operator, parameters, direction)

    fast_m2x = FastFourSourceDiscreteObjective(m2x_cache)
    prepared = prepare_production_problem(
        configuration["truth"]["m2x_oracle_configuration"]
    )
    physics = NeuralFourTendencyMoistPhysics(
        parameters, dataset.normalization, use_jit=True
    )
    production_operations = ProductionFourSourceOperations(
        prepared.objective.operations.helper, physics
    )
    production_m2x = DeployedDiscreteOfflineObjective(
        prepared.objective.observations,
        production_operations,
        require_canonical_steps=True,
    )
    fast_value, fast_gradient = fast_m2x.value_and_gradient(parameters)
    oracle_value, oracle_gradient = production_m2x.value_and_gradient(parameters)
    result["M2_X_production_oracle"] = {
        "cached_value": fast_value,
        "production_value": oracle_value,
        "value_absolute_difference": abs(fast_value - oracle_value),
        "value_relative_difference": abs(fast_value - oracle_value)
        / max(abs(oracle_value), np.finfo(np.float64).tiny),
        "gradient": _gradient_relation(fast_gradient, oracle_gradient),
    }

    case, truth, _, _ = _build_problem_b_case(
        configuration, dataset.normalization, parameters, 5
    )
    denominator = float(metadata["common_horizon_denominator"])
    denominator_sha = _canonical_sha256(
        {"value": denominator, "definition": "sum_0^79 ||dt G4 N*(Y_k)||_M^2"}
    )
    metric = GlobalMixedMassMetric(
        case.helper, denominator, denominator_sha256=denominator_sha
    )
    literal_h1 = NeuralTrajectoryObjective(
        case,
        truth,
        reset_windows((0,), 1, "accumulated", (1.0,)),
        metric=metric,
        c0=0.14,
        use_fixed_prefix=True,
    )
    h1_one = FourSourceFixedCache(
        **{
            name: getattr(h1_cache, name)
            for name in h1_cache.__dataclass_fields__
            if name not in ("normalized_features", "physical_targets")
        },
        normalized_features=h1_cache.normalized_features[:1],
        physical_targets=h1_cache.physical_targets[:1],
    )
    fast_h1 = FastFourSourceDiscreteObjective(h1_one)
    fast_h1_value, fast_h1_gradient = fast_h1.value_and_gradient(parameters)
    literal_h1_value, literal_h1_gradient = literal_h1.value_and_gradient(parameters)
    result["H1_cached_vs_literal"] = {
        "cached_value": fast_h1_value,
        "literal_value": literal_h1_value,
        "value_absolute_difference": abs(fast_h1_value - literal_h1_value),
        "value_relative_difference": abs(fast_h1_value - literal_h1_value)
        / max(abs(literal_h1_value), np.finfo(np.float64).tiny),
        "gradient": _gradient_relation(fast_h1_gradient, literal_h1_gradient),
        "literal_directional": _directional_record(literal_h1, parameters, direction),
        "fully_fixed_state_cacheable": True,
    }

    recursive = {}
    for horizon in (2, 5):
        objective = NeuralTrajectoryObjective(
            case,
            truth,
            reset_windows((0,), horizon, "accumulated", (1.0,) * horizon),
            metric=metric,
            c0=0.14,
            use_fixed_prefix=True,
        )
        directional = _directional_record(objective, parameters, direction, epsilon=2.0e-5)
        independent = NeuralTrajectoryObjective(
            case,
            truth,
            reset_windows(tuple(range(horizon)), 1, "accumulated", (1.0,)),
            metric=metric,
            c0=0.14,
            use_fixed_prefix=True,
        )
        full_value, full_gradient = objective.value_and_gradient(parameters)
        reset_value, reset_gradient = independent.value_and_gradient(parameters)
        recursive[f"H{horizon}"] = {
            "directional_gradient": directional,
            "state_tangent_adjoint": _state_duality(case, objective, parameters, horizon),
            "recursive_vs_independent_H1": {
                "recursive_objective": full_value,
                "independent_objective": reset_value,
                "gradient": _gradient_relation(full_gradient, reset_gradient),
            },
        }
    result["recursive_certification"] = recursive
    result["classification"] = {
        "H1": "truth-reset, fixed-state, fully offline/cacheable",
        "H2": "first objective with model-generated-state recursive feedback",
        "H5": "five-step recursive feedback",
    }
    result["status"] = "complete"
    write_json_record(output_path, result)
    return result


def _trajectory_objective(configuration, metadata, normalization, parameters, horizon):
    from .test2a_trajectory import GlobalMixedMassMetric, NeuralTrajectoryObjective
    case, truth, _, _ = _build_problem_b_case(
        configuration, normalization, parameters, 80
    )
    denominator = float(metadata["common_horizon_denominator"])
    metric = GlobalMixedMassMetric(
        case.helper,
        denominator,
        denominator_sha256=_canonical_sha256(
            {"value": denominator, "definition": "sum_0^79 ||dt G4 N*(Y_k)||_M^2"}
        ),
    )
    return NeuralTrajectoryObjective(
        case,
        truth,
        production_windows(horizon),
        metric=metric,
        c0=float(configuration["truth"]["c0"]),
        use_fixed_prefix=True,
    )


def benchmark(
    configuration_path,
    preparation_path,
    output_path,
    repeats=3,
    stages=("M1", "M2-X", "H1", "H2", "H5"),
):
    configuration = load_problem_b_configuration(configuration_path)
    metadata, dataset, m2x_cache, h1_cache = load_preparation(preparation_path)
    parameters = initial_problem_b_parameters()
    objectives = {
        "M1": ProblemBOperatorObjective(dataset),
        "M2-X": FastFourSourceDiscreteObjective(m2x_cache),
        "H1": FastFourSourceDiscreteObjective(h1_cache),
    }
    result = {
        "status": "in_progress",
        "benchmark_stage": "Test 2A Problem B bounded performance benchmark",
        "truth_state_indices": [0, 80],
        "states_after_80_accessed": False,
        "repeats": int(repeats),
        "timings": {},
    }
    selected_stages = tuple(stages)
    allowed = tuple(objectives) + ("H2", "H5")
    if not selected_stages or any(name not in allowed for name in selected_stages):
        raise ValueError("unknown Problem B benchmark stage")
    for name in (value for value in objectives if value in selected_stages):
        objective = objectives[name]
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
        result["timings"][name] = {
            "first_value_seconds": first_value,
            "first_value_gradient_seconds": first_gradient,
            "steady_value_median_seconds": float(np.median(value_times)),
            "steady_value_gradient_median_seconds": float(np.median(gradient_times)),
            "steady_Firedrake_PETSc_solves": 0,
        }
    for horizon in (value for value in (2, 5) if f"H{value}" in selected_stages):
        started = perf_counter()
        objective = _trajectory_objective(
            configuration, metadata, dataset.normalization, parameters, horizon
        )
        setup = perf_counter() - started
        objective.clear_parameter_tape()
        started = perf_counter()
        value = objective.value(parameters)
        value_seconds = perf_counter() - started
        started = perf_counter()
        same_value, gradient = objective.value_and_gradient(parameters)
        gradient_seconds = perf_counter() - started
        result["timings"][f"H{horizon}"] = {
            "fixed_prefix_setup_seconds": setup,
            "full_production_window_count": len(production_windows(horizon)),
            "complete_timesteps": 80,
            "value": value,
            "same_theta_value": same_value,
            "gradient_norm": float(tree_norm(gradient)),
            "value_seconds": value_seconds,
            "gradient_after_same_theta_value_seconds": gradient_seconds,
            "value_plus_gradient_seconds": value_seconds + gradient_seconds,
            "same_theta_tape_reused": objective.work_counts().same_theta_tape_hits > 0,
            "Firedrake_PETSc_actions_present": True,
        }
    result["projected_wall_seconds"] = {}
    for horizon in (value for value in (2, 5) if f"H{value}" in result["timings"]):
        base = result["timings"][f"H{horizon}"]["value_plus_gradient_seconds"]
        result["projected_wall_seconds"][f"H{horizon}"] = {
            str(iterations): iterations * base
            for iterations in (100, 500, 1000)
        }
    result["status"] = "complete"
    write_json_record(output_path, result)
    return result


def smoke(
    configuration_path,
    preparation_path,
    output_path,
    iterations=1,
    stages=("M1", "M2-X", "H1", "H2", "H5"),
):
    from pyrol import Problem, Solver
    from .test2a_discrete_training import CompactCheckpointObjective
    from .test2a_pyrol import build_test2a_lbfgs_parameters

    if int(iterations) < 1 or int(iterations) > 5:
        raise ValueError("Problem B smoke accepts only 1..5 iterations")
    configuration = load_problem_b_configuration(configuration_path)
    metadata, dataset, m2x_cache, h1_cache = load_preparation(preparation_path)
    initial = initial_problem_b_parameters()
    result = {
        "status": "in_progress",
        "interpretation": "NONSCIENTIFIC PROBLEM-B IMPLEMENTATION SMOKE",
        "truth_state_indices": [0, 80],
        "states_after_80_accessed": False,
        "initial_parameter_pytree_sha256": parameter_pytree_sha256(initial),
        "stages": {},
    }
    factories = {
        "M1": lambda: ProblemBOperatorObjective(dataset),
        "M2-X": lambda: FastFourSourceDiscreteObjective(m2x_cache),
        "H1": lambda: FastFourSourceDiscreteObjective(h1_cache),
        "H2": lambda: _trajectory_objective(
            configuration, metadata, dataset.normalization, initial, 2
        ),
        "H5": lambda: _trajectory_objective(
            configuration, metadata, dataset.normalization, initial, 5
        ),
    }
    selected_stages = tuple(stages)
    if not selected_stages or any(name not in factories for name in selected_stages):
        raise ValueError("unknown Problem B smoke stage")
    for name in selected_stages:
        factory = factories[name]
        objective = factory()
        if name in ("H2", "H5"):
            from .test2a_trajectory import TrajectoryPyROLObjective
            adapter = TrajectoryPyROLObjective(objective, initial)
        else:
            adapter = CompactCheckpointObjective(objective.jax_value, initial, use_jit=True)
        control = adapter.vector_from_pytree(initial)
        initial_value = objective.value(initial)
        rol = build_test2a_lbfgs_parameters(
            {
                "gradient_tolerance": 1.0e-8,
                "step_tolerance": 1.0e-12,
                "iteration_limit": int(iterations),
                "maximum_secant_storage": 20,
            }
        )
        started = perf_counter()
        solver = Solver(Problem(adapter, control), rol)
        solver.solve()
        wall = perf_counter() - started
        final = adapter.pytree_from_vector(control)
        final_value = objective.value(final)
        state = solver.getAlgorithmState()
        result["stages"][name] = {
            "initial_objective": initial_value,
            "final_objective": final_value,
            "objective_decreased": bool(final_value < initial_value),
            "accepted_iterations": int(state.iter),
            "termination_reason": str(state.statusFlag),
            "objective_evaluations": int(adapter.value_evaluations),
            "gradient_evaluations": int(adapter.gradient_evaluations),
            "HVP_evaluations": int(adapter.hvp_evaluations),
            "wall_seconds": wall,
            "new_optimizer_process": True,
            "source_secant_history_reused": False,
            "final_parameter_pytree_sha256": parameter_pytree_sha256(final),
        }
        if adapter.hvp_evaluations != 0 or not final_value < initial_value:
            raise RuntimeError(f"Problem B {name} smoke failed")
    result["status"] = "complete"
    write_json_record(output_path, result)
    return result


def _checkpoint_schedule(limit):
    candidates = (0, 10, 25, 50, 75, 100, 500, 1000, 5000, 10000, 25000,
                  50000, 100000, 150000, 200000, int(limit))
    return tuple(sorted({value for value in candidates if 0 <= value <= int(limit)}))


def train_stage(
    configuration_path,
    preparation_path,
    stage,
    output_directory,
    iteration_limit,
    *,
    initial_parameter_file=None,
    expected_initial_sha256=None,
):
    """Run one explicit user-invoked production stage with a new ROL process."""
    from pyrol import Problem, Solver
    from .test2a_discrete_training import CompactCheckpointObjective
    from .test2a_pyrol import build_test2a_lbfgs_parameters

    allowed = ("M1", "M2-X-independent", "M1-to-M2-X", "H1", "H2", "H5")
    if stage not in allowed:
        raise ValueError(f"stage must be one of {allowed}")
    configuration = load_problem_b_configuration(configuration_path)
    metadata, dataset, m2x_cache, h1_cache = load_preparation(preparation_path)
    output = Path(output_directory)
    result_path = output / "fit_result.json"
    progress_path = output / "fit_progress.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite a Problem B stage directory")
    output.mkdir(parents=True)
    if stage in ("M1", "M2-X-independent"):
        if initial_parameter_file is not None:
            raise ValueError(f"{stage} must start from canonical seed 0")
        initial = initial_problem_b_parameters()
        initialization_kind = "canonical_seed0"
        source_file = None
    else:
        if initial_parameter_file is None or expected_initial_sha256 is None:
            raise ValueError(f"{stage} requires an explicit verified warm start")
        initial, _, _ = load_problem_b_parameters(initial_parameter_file)
        actual = parameter_pytree_sha256(initial)
        if actual != expected_initial_sha256:
            raise ValueError(f"{stage} initial parameter fingerprint mismatch")
        initialization_kind = "new_optimizer_from_verified_parameter_artifact"
        source_file = str(Path(initial_parameter_file).resolve())
    initial_sha = parameter_pytree_sha256(initial)
    if stage == "M1":
        objective = ProblemBOperatorObjective(dataset)
    elif stage in ("M2-X-independent", "M1-to-M2-X"):
        objective = FastFourSourceDiscreteObjective(m2x_cache)
    elif stage == "H1":
        objective = FastFourSourceDiscreteObjective(h1_cache)
    else:
        objective = _trajectory_objective(
            configuration, metadata, dataset.normalization, initial, int(stage[1:])
        )
    checkpoints = _checkpoint_schedule(iteration_limit)
    checkpoint_records = []
    checkpoint_zero = output / "checkpoint_000000000_parameters.npz"
    zero_record = save_problem_b_parameters(
        checkpoint_zero,
        initial,
        metadata={"stage": stage, "accepted_iteration": 0, "scientific": True},
    )
    checkpoint_records.append(
        {
            "accepted_iteration": 0,
            "parameter_file": str(checkpoint_zero.resolve()),
            "parameter_pytree_sha256": zero_record["parameter_pytree_sha256"],
            "objective": float(objective.value(initial)),
        }
    )
    started = perf_counter()

    def accepted_callback(control, local_index, adapter):
        if local_index == 0:
            return
        if local_index not in checkpoints and local_index % 100 != 0:
            return
        parameters = adapter.pytree_from_vector(control)
        parameter_sha = parameter_pytree_sha256(parameters)
        value = float(objective.value(parameters))
        if local_index in checkpoints:
            artifact = output / f"checkpoint_{local_index:09d}_parameters.npz"
            record = save_problem_b_parameters(
                artifact,
                parameters,
                metadata={
                    "stage": stage,
                    "accepted_iteration": int(local_index),
                    "scientific": True,
                },
            )
            checkpoint_records.append(
                {
                    "accepted_iteration": int(local_index),
                    "parameter_file": str(artifact.resolve()),
                    "parameter_pytree_sha256": record["parameter_pytree_sha256"],
                    "objective": value,
                }
            )
        write_json_record(
            progress_path,
            {
                "status": "in_progress",
                "stage": stage,
                "accepted_iteration": int(local_index),
                "objective": value,
                "elapsed_wall_seconds": float(perf_counter() - started),
                "objective_evaluations": int(adapter.value_evaluations),
                "gradient_evaluations": int(adapter.gradient_evaluations),
                "parameter_pytree_sha256": parameter_sha,
                "configuration_sha256": _canonical_sha256(configuration),
                "preparation_npz_sha256": metadata["preparation_npz_sha256"],
                "source_optimizer_secant_history_reused": False,
                "parameter_only_restart_restores_secant_history": False,
            },
        )
        print(
            f"Problem-B {stage} accepted={local_index} J={value:.17e} "
            f"elapsed={perf_counter()-started:.1f}s sha={parameter_sha}",
            flush=True,
        )

    if stage in ("H2", "H5"):
        from .test2a_trajectory import TrajectoryPyROLObjective
        adapter = TrajectoryPyROLObjective(
            objective, initial, accepted_callback=accepted_callback
        )
    else:
        adapter = CompactCheckpointObjective(
            objective.jax_value,
            initial,
            use_jit=True,
            accepted_callback=accepted_callback,
        )
    control = adapter.vector_from_pytree(initial)
    rol = build_test2a_lbfgs_parameters(
        {
            "gradient_tolerance": 1.0e-8,
            "step_tolerance": 1.0e-12,
            "iteration_limit": int(iteration_limit),
            "maximum_secant_storage": 20,
        }
    )
    solver = Solver(Problem(adapter, control), rol)
    solver.solve()
    wall = perf_counter() - started
    final = adapter.pytree_from_vector(control)
    final_path = output / "final_parameters.npz"
    final_record = save_problem_b_parameters(
        final_path,
        final,
        metadata={"stage": stage, "artifact_kind": "final", "scientific": True},
    )
    state = solver.getAlgorithmState()
    result = {
        "status": "complete",
        "stage": stage,
        "configuration_sha256": _canonical_sha256(configuration),
        "preparation_npz_sha256": metadata["preparation_npz_sha256"],
        "truth_state_indices": [0, 80],
        "states_after_80_accessed": False,
        "initialization": {
            "kind": initialization_kind,
            "source_parameter_file": source_file,
            "initial_parameter_pytree_sha256": initial_sha,
            "new_optimizer_process": True,
            "source_optimizer_secant_history_reused": False,
        },
        "optimizer": {
            "library": "PyROL/ROL",
            "method": "line-search L-BFGS",
            "maximum_secant_storage": 20,
            "gradient_tolerance": 1.0e-8,
            "step_tolerance": 1.0e-12,
            "iteration_limit": int(iteration_limit),
            "accepted_iterations": int(state.iter),
            "termination_reason": str(state.statusFlag),
            "objective_evaluations": int(adapter.value_evaluations),
            "gradient_evaluations": int(adapter.gradient_evaluations),
            "HVP_evaluations": int(adapter.hvp_evaluations),
        },
        "initial_objective": checkpoint_records[0]["objective"],
        "final_objective": float(objective.value(final)),
        "wall_seconds": float(wall),
        "checkpoint_schedule": list(checkpoints),
        "checkpoints": checkpoint_records,
        "final_parameter_file": str(final_path.resolve()),
        "final_parameter_pytree_sha256": final_record["parameter_pytree_sha256"],
        "parameter_only_restart_restores_secant_history": False,
    }
    if adapter.hvp_evaluations != 0:
        raise RuntimeError("Problem B L-BFGS unexpectedly requested an HVP")
    write_json_record(result_path, result)
    write_json_record(progress_path, {**result, "status": "complete"})
    return result


def _evaluate_artifact(configuration, metadata, dataset, m2x_cache, h1_cache, label, path):
    from .test2a_problem_b import build_problem_b_model, structural_diagnostics

    parameters, _, sidecar = load_problem_b_parameters(path)
    model = build_problem_b_model()
    coordinates = np.asarray(
        model(parameters, jnp.asarray(dataset.normalized_features)), dtype=np.float64
    )
    prediction = coordinates * dataset.normalization.output_scales
    beta2 = float(metadata["normalization"]["sigma_S"] / metadata["normalization"]["sigma_Q"])
    fixed = {
        "J_M1": ProblemBOperatorObjective(dataset).value(parameters),
        "J_M2_X": FastFourSourceDiscreteObjective(m2x_cache).value(parameters),
        "J_H1": FastFourSourceDiscreteObjective(h1_cache).value(parameters),
    }
    fixed["structural_diagnostics_on_boundary_truth_support"] = structural_diagnostics(
        prediction,
        dataset.physical_targets,
        beta2,
        dataset.normalization,
        dataset.spatial_weights,
    )
    fixed["parameter_file"] = str(Path(path).resolve())
    fixed["parameter_pytree_sha256"] = sidecar["parameter_pytree_sha256"]
    fixed["label"] = label

    # Exact dense objectives and an 80-step post-hoc autonomous diagnostic use
    # the same complete split.  They are intentionally deferred to this
    # postprocessing command and never participate in optimizer selection.
    case, truth, _, diagnostic_configuration = _build_problem_b_case(
        configuration, dataset.normalization, parameters, 80
    )
    for horizon in (2, 5):
        objective = _trajectory_objective(
            configuration, metadata, dataset.normalization, parameters, horizon
        )
        fixed[f"J_H{horizon}"] = objective.value(parameters)

    from firedrake import assemble
    from .hidden_c0 import _copy_function
    from .resolved_hidden_c0_driver import ResolvedDiagnosticEvaluator
    from .resolved_hidden_c0_inference import (
        _diagnostic_mismatch,
        _field_trajectory_metric,
        _trajectory_metric,
    )
    from .test2a_apriori_autonomous import source_invariant_diagnostic

    generated = {0: _copy_function(truth[0], f"problem_b_{label}_autonomous_0")}
    with case.physical_c0(float(configuration["truth"]["c0"])):
        for step in range(80):
            generated[step + 1] = _copy_function(
                case.helper.take_forward_step_cached(
                    generated[step], case.t0 + step * case.dt, case.dt,
                    neural_parameters=parameters,
                ).state_out,
                f"problem_b_{label}_autonomous_{step + 1}",
            )
    steps = tuple(range(1, 81))
    truth_proxy = type("TruthProxy", (), {"states": truth})()
    mixed = _trajectory_metric(case, generated, truth_proxy, steps, f"problem_b_{label}_mixed")
    fieldwise = _field_trajectory_metric(
        case, generated, truth_proxy, steps, f"problem_b_{label}_field"
    )
    evaluator = ResolvedDiagnosticEvaluator(case, diagnostic_configuration)
    predicted_diagnostics = []
    truth_diagnostics = []
    source_records = []
    physical_beta2 = float(
        metadata["normalization"]["sigma_S"]
        / metadata["normalization"]["sigma_Q"]
    )
    water_integrals = [float(assemble(
        (generated[0].sub(3) + generated[0].sub(4) + generated[0].sub(5))
        * case.model.spaces.dx
    ))]
    beta_integrals = [float(assemble(
        (generated[0].sub(2) - physical_beta2 * generated[0].sub(3))
        * case.model.spaces.dx
    ))]
    qr_source_max = 0.0
    qr_activity = []
    qr_numerical_tolerance = float(
        64.0 * np.finfo(np.float64).eps * dataset.normalization.sigma_q
    )
    qr_physical_tolerance = float(1.0e-12 * dataset.normalization.sigma_q)
    for step in steps:
        time = case.t0 + step * case.dt
        predicted_diagnostics.append(evaluator.evaluate(generated[step], step, time)[0])
        truth_diagnostics.append(evaluator.evaluate(truth[step], step, time)[0])
        cache = case.helper.moist_helper.take_forward_step_cached(
            generated[step], time, case.dt, neural_parameters=parameters
        )
        local_beta2 = float(cache.parameters["g"] * cache.parameters["L"])
        record = source_invariant_diagnostic(cache.source_density, local_beta2)
        record["step"] = step
        source_records.append(record)
        qr_source_max = max(
            qr_source_max,
            float(np.max(np.abs(np.asarray(cache.source_density["Qr"])))),
        )
        qr_source = np.asarray(cache.source_density["Qr"], dtype=np.float64)
        qr_activity.append(
            {
                "step": step,
                "time": float(time),
                "maximum_absolute_Qr_t": float(np.max(np.abs(qr_source))),
                "rms_Qr_t": float(np.sqrt(np.mean(qr_source * qr_source))),
                "exact_nonzero_fraction": float(np.mean(qr_source != 0.0)),
                "above_float64_scale_fraction": float(
                    np.mean(np.abs(qr_source) > qr_numerical_tolerance)
                ),
                "physically_meaningful_fraction": float(
                    np.mean(np.abs(qr_source) > qr_physical_tolerance)
                ),
            }
        )
        water_integrals.append(float(assemble(
            (generated[step].sub(3) + generated[step].sub(4) + generated[step].sub(5))
            * case.model.spaces.dx
        )))
        beta_integrals.append(float(assemble(
            (generated[step].sub(2) - physical_beta2 * generated[step].sub(3))
            * case.model.spaces.dx
        )))
    times = np.asarray([case.t0 + step * case.dt for step in steps])
    kinetic = _diagnostic_mismatch(
        [entry["kinetic_energy"] for entry in predicted_diagnostics],
        [entry["kinetic_energy"] for entry in truth_diagnostics],
        steps,
        times,
    )
    enstrophy = _diagnostic_mismatch(
        [entry["projected_enstrophy"] for entry in predicted_diagnostics],
        [entry["projected_enstrophy"] for entry in truth_diagnostics],
        steps,
        times,
    )
    exact_qr_steps = [
        record["step"] for record in qr_activity
        if record["exact_nonzero_fraction"] > 0.0
    ]
    physical_qr_steps = [
        record["step"] for record in qr_activity
        if record["physically_meaningful_fraction"] > 0.0
    ]
    fixed["autonomous_training_support_posthoc"] = {
        "mixed_state_error": mixed,
        "fieldwise_errors": fieldwise,
        "kinetic_energy": kinetic,
        "projected_enstrophy": enstrophy,
        "source_defects": source_records,
        "maximum_absolute_predicted_Qr_t": qr_source_max,
        "spurious_Qr_t_activity": qr_activity,
        "spurious_Qr_t_activity_summary": {
            "maximum_absolute_Qr_t": qr_source_max,
            "float64_scale_tolerance": qr_numerical_tolerance,
            "physical_scale_tolerance": qr_physical_tolerance,
            "timesteps_with_exact_nonzero_Qr_t": len(exact_qr_steps),
            "first_exact_nonzero_Qr_t_step": (
                exact_qr_steps[0] if exact_qr_steps else None
            ),
            "timesteps_with_physically_meaningful_Qr_t": len(
                physical_qr_steps
            ),
            "first_physically_meaningful_Qr_t_step": (
                physical_qr_steps[0] if physical_qr_steps else None
            ),
            "definition": (
                "Problem-B direct Qr_t source activity; physical threshold is "
                "1e-12 times frozen sigma_Q"
            ),
        },
        "total_water_integral": water_integrals,
        "S_minus_beta2_Qv_integral": beta_integrals,
        "maximum_accumulated_total_water_drift": float(
            np.max(np.abs(np.asarray(water_integrals) - water_integrals[0]))
        ),
        "maximum_accumulated_S_beta_drift": float(
            np.max(np.abs(np.asarray(beta_integrals) - beta_integrals[0]))
        ),
        "resolved_diagnostic_configuration": {
            "sampling_shape": list(diagnostic_configuration.sampling_shape),
            "high_wavenumber_fraction": float(
                diagnostic_configuration.high_wavenumber_fraction
            ),
            "canonical_source": (
                "ResolvedPilotConfiguration reconstructed from stored truth metadata"
            ),
        },
        "truth_resets_after_initialization": 0,
        "states_after_80_accessed": False,
        "used_for_model_selection": False,
    }
    return fixed


def postprocess(configuration_path, preparation_path, output_path, artifacts):
    configuration = load_problem_b_configuration(configuration_path)
    metadata, dataset, m2x_cache, h1_cache = load_preparation(preparation_path)
    specifications = {}
    for specification in artifacts:
        if "=" not in specification:
            raise ValueError("artifact must be LABEL=PATH")
        label, path = specification.split("=", 1)
        if not label or label in specifications:
            raise ValueError("postprocess labels must be unique and nonempty")
        specifications[label] = path
    if set(specifications) != set(PRODUCTION_ARTIFACT_STAGES):
        missing = sorted(set(PRODUCTION_ARTIFACT_STAGES) - set(specifications))
        extra = sorted(set(specifications) - set(PRODUCTION_ARTIFACT_STAGES))
        raise ValueError(
            f"postprocess requires exactly all six production artifacts; "
            f"missing={missing}, extra={extra}"
        )
    artifact_audit = {
        label: _verify_completed_training_artifact(label, specifications[label])
        for label in PRODUCTION_ARTIFACT_STAGES
    }
    records = {}
    for label, path in specifications.items():
        records[label] = _evaluate_artifact(
            configuration, metadata, dataset, m2x_cache, h1_cache, label, path
        )
    result = {
        "status": "complete",
        "benchmark_stage": "Test 2A Problem B common five-objective/post-hoc evaluator",
        "truth_state_indices": [0, 80],
        "states_after_80_accessed": False,
        "autonomous_metrics_used_for_selection": False,
        "training_artifact_audit": artifact_audit,
        "artifacts": records,
    }
    write_json_record(output_path, result)
    return result


def prepare_data(configuration_path, output_path):
    configuration = load_problem_b_configuration(configuration_path)
    started = perf_counter()
    prepared = prepare_production_problem(configuration["truth"]["m2x_oracle_configuration"])
    features_x, targets_x = _physical_arrays_from_observations(prepared)
    carrier_weights, carrier_audit = _carrier_mass_weights(
        prepared.objective.operations.helper
    )
    _, problem_a_metadata = load_operator_dataset(
        configuration["normalization"]["problem_a_operator_dataset"]
    )
    problem_a_normalization_record = problem_a_metadata["normalization"]
    problem_a_normalization = a_norm_from_record(problem_a_normalization_record)
    weights = np.tile(carrier_weights, features_x.shape[0])
    flat_targets = targets_x.reshape(-1, 4)
    volume_states = float(np.sum(weights))
    rms_s = float(np.sqrt(np.sum(weights * flat_targets[:, 0] ** 2) / volume_states))
    rms_qv = float(np.sqrt(np.sum(weights * flat_targets[:, 1] ** 2) / volume_states))
    rms_qc = float(np.sqrt(np.sum(weights * flat_targets[:, 2] ** 2) / volume_states))
    sigma_q = float(np.sqrt(0.5 * (rms_qv * rms_qv + rms_qc * rms_qc)))
    scale_record = {
        "sigma_S": rms_s,
        "sigma_Q": sigma_q,
        "RMS_Qv": rms_qv,
        "RMS_Qc": rms_qc,
        "truth_state_indices": [0, 80],
        "mass_weighting": carrier_audit,
    }
    normalization = FourTendencyNormalization(
        input_offset=problem_a_normalization.input_offset,
        input_scale=problem_a_normalization.input_scale,
        sigma_s=rms_s,
        sigma_q=sigma_q,
        input_normalization_sha256=_canonical_sha256(problem_a_normalization_record),
        scale_provenance_sha256=_canonical_sha256(scale_record),
    )
    matrices, mass_audit = _matrix_cache_components(
        prepared,
        1.0e-11,
        16,
        1.0e-12,
        periodic_cell_shape=(16, 16),
    )
    physics = NeuralFourTendencyMoistPhysics(
        initial_problem_b_parameters(), normalization, use_jit=True
    )
    case, truth, features_y, targets_y = _postprefix_arrays(configuration, physics)
    boundary_cache = _cache_from_arrays(
        features_x, targets_x, normalization, matrices,
        {"objective": "M2-X", "truth_state_indices": [0, 80]},
    )
    h1_cache = _cache_from_arrays(
        features_y, targets_y, normalization, matrices,
        {"objective": "H1/M2-Y", "truth_state_indices": [0, 80]},
    )
    normalized_features = np.asarray(
        normalization.normalize_features(features_x.reshape(-1, 5)), dtype=np.float64
    )
    dataset = ProblemBOperatorDataset(
        normalized_features=normalized_features,
        physical_targets=flat_targets,
        spatial_weights=weights,
        normalization=normalization,
        metadata={"truth_state_indices": [0, 80], "states_after_80_accessed": False},
    )
    truth_residuals = {
        "water_max_abs": float(np.max(np.abs(
            flat_targets[:, 1] + flat_targets[:, 2] + flat_targets[:, 3]
        ))),
        "beta_max_abs": float(np.max(np.abs(
            flat_targets[:, 0]
            - float(prepared.objective.observations[0].payload.moist_parameters["g"]
                    * prepared.objective.observations[0].payload.moist_parameters["L"])
            * flat_targets[:, 1]
        ))),
        "Qr_max_abs": float(np.max(np.abs(flat_targets[:, 3]))),
    }
    metadata = {
        "status": "complete",
        "benchmark_stage": "Test 2A Problem B preparation",
        "configuration_sha256": _canonical_sha256(configuration),
        "truth_state_indices": [0, 80],
        "states_after_80_accessed": False,
        "sample_count": int(flat_targets.shape[0]),
        "architecture": ProblemBMLPConfiguration().to_record(),
        "seed0_parameter_pytree_sha256": parameter_pytree_sha256(
            initial_problem_b_parameters()
        ),
        "scale_measurements": scale_record,
        "truth_structure_posthoc": truth_residuals,
        "mass_inverse_audit": mass_audit,
        "m1_denominator": ProblemBOperatorObjective(dataset).denominator,
        "m2x_denominator": boundary_cache.normalizer,
        "common_horizon_denominator": float(case.dt**2 * h1_cache.normalizer),
        "m2x_state_count": boundary_cache.state_count,
        "h1_state_count": h1_cache.state_count,
        "preparation_wall_seconds": float(perf_counter() - started),
        "hot_loop_Firedrake_PETSc_solves": {"M1": 0, "M2-X": 0, "H1": 0},
    }
    return _save_preparation(output_path, boundary_cache, h1_cache, dataset, metadata)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--configuration", required=True)
    prepare.add_argument("--output", required=True)
    certification = sub.add_parser("certify")
    certification.add_argument("--configuration", required=True)
    certification.add_argument("--preparation", required=True)
    certification.add_argument("--output", required=True)
    timing = sub.add_parser("benchmark")
    timing.add_argument("--configuration", required=True)
    timing.add_argument("--preparation", required=True)
    timing.add_argument("--output", required=True)
    timing.add_argument("--repeats", type=int, default=3)
    timing.add_argument(
        "--stages", default="M1,M2-X,H1,H2,H5",
        help="comma-separated bounded benchmark stages",
    )
    quick = sub.add_parser("smoke")
    quick.add_argument("--configuration", required=True)
    quick.add_argument("--preparation", required=True)
    quick.add_argument("--output", required=True)
    quick.add_argument("--iterations", type=int, default=1)
    quick.add_argument(
        "--stages", default="M1,M2-X,H1,H2,H5",
        help="comma-separated bounded smoke stages",
    )
    train = sub.add_parser("train")
    train.add_argument("--configuration", required=True)
    train.add_argument("--preparation", required=True)
    train.add_argument("--stage", required=True)
    train.add_argument("--output-directory", required=True)
    train.add_argument("--iteration-limit", type=int, required=True)
    train.add_argument("--initial-parameters")
    train.add_argument("--expected-initial-sha256")
    post = sub.add_parser("postprocess")
    post.add_argument("--configuration", required=True)
    post.add_argument("--preparation", required=True)
    post.add_argument("--output", required=True)
    post.add_argument("--artifact", action="append", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "prepare-data":
        result = prepare_data(args.configuration, args.output)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "certify":
        result = certify(args.configuration, args.preparation, args.output)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "benchmark":
        result = benchmark(
            args.configuration,
            args.preparation,
            args.output,
            args.repeats,
            tuple(value for value in args.stages.split(",") if value),
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "smoke":
        result = smoke(
            args.configuration,
            args.preparation,
            args.output,
            args.iterations,
            tuple(value for value in args.stages.split(",") if value),
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "train":
        result = train_stage(
            args.configuration,
            args.preparation,
            args.stage,
            args.output_directory,
            args.iteration_limit,
            initial_parameter_file=args.initial_parameters,
            expected_initial_sha256=args.expected_initial_sha256,
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "postprocess":
        result = postprocess(
            args.configuration, args.preparation, args.output, args.artifact
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
