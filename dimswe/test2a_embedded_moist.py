"""Frozen Test-2A neural ``A`` with the original deployed analytical ``R``."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from .jax_moist import moist_diagnostics_jax, moist_rates_jax
from .learned_physics.parameters import tree_copy, validate_float64_tree
from .resolved_hidden_c0 import write_json_record
from .test2a_operator import (
    FEATURE_ORDER,
    build_learned_a_model,
    load_mlp_parameters,
    load_operator_dataset,
    load_selected_configuration,
    mlp_configuration_from_record,
    normalization_from_record,
    physical_predictions,
)


PHYSICS_MODE_ANALYTICAL = "analytical_A_original_R"
PHYSICS_MODE_NEURAL_A = "neural_A_original_R"
_STATE_KEYS = ("h", "S", "Qv", "Qc")
_SOURCE_KEYS = ("S", "Qv", "Qc", "Qr")
_FLOAT64 = jnp.dtype(jnp.float64)


def _file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file_sha256(path, expected, description):
    if _file_sha256(path) != expected:
        raise ValueError(f"{description} fingerprint mismatch")


def _verify_parameter_archive(path, configuration):
    dimensions = configuration.layer_dimensions
    expected = {}
    for index, (fan_in, fan_out) in enumerate(
        zip(dimensions[:-1], dimensions[1:])
    ):
        expected[f"layer_{index:03d}_weight"] = (fan_in, fan_out)
        expected[f"layer_{index:03d}_bias"] = (fan_out,)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("frozen neural-A parameter leaf names changed")
        for name, shape in expected.items():
            if archive[name].shape != shape:
                raise ValueError(f"frozen neural-A leaf {name} shape changed")
            if archive[name].dtype != np.dtype(np.float64):
                raise TypeError(f"frozen neural-A leaf {name} must be float64")


def parameter_pytree_sha256(parameters):
    flat, _ = ravel_pytree(validate_float64_tree(parameters, name="parameters"))
    values = np.ascontiguousarray(flat, dtype=np.float64)
    digest = sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _require_tree(name, values, keys):
    result = {}
    for key in keys:
        value = jnp.asarray(values[key])
        if value.dtype != _FLOAT64:
            raise TypeError(f"{name}['{key}'] must be float64")
        result[key] = value
    return result


def load_test2a_embedding_configuration(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if record.get("benchmark_stage") != "Test 2A-2 frozen neural-A embedding":
        raise ValueError("not a selected Test 2A-2 embedding configuration")
    if record["physics"] != {
        "A": "frozen neural A_theta",
        "R": "original deployed analytical rain law",
        "default": False,
        "mode": PHYSICS_MODE_NEURAL_A,
        "source_structure": {
            "Qc": "-h*(A_theta+R_original)",
            "Qr": "h*R_original",
            "Qv": "h*A_theta",
            "S": "h*(g*L)*A_theta",
        },
    }:
        raise ValueError("Test 2A-2 hybrid physics contract changed")
    if record["truth_access"] != {
        "allowed_state_indices": [0, 80],
        "states_after_80_forbidden": True,
    }:
        raise ValueError("Test 2A-2 truth-access contract changed")
    if tuple(record["normalization_contract"]["feature_order"]) != FEATURE_ORDER:
        raise ValueError("Test 2A-2 feature order changed")
    if record["normalization_contract"]["refit_during_embedding"] is not False:
        raise ValueError("embedding normalization must remain frozen")
    return record


class FrozenNeuralAMoistPhysics:
    """Pure-JAX hybrid local physics provider with immutable frozen parameters."""

    physics_mode = PHYSICS_MODE_NEURAL_A

    def __init__(
        self,
        parameters,
        model_configuration,
        normalization,
        *,
        provenance,
        use_jit=True,
    ):
        self.model_configuration = model_configuration
        self.normalization = normalization
        self.provenance = dict(provenance)
        self.use_jit = bool(use_jit)
        frozen_parameters = validate_float64_tree(
            parameters, name="frozen neural-A parameters"
        )
        count = sum(
            int(leaf.size) for leaf in jax.tree_util.tree_leaves(frozen_parameters)
        )
        if count != model_configuration.parameter_count:
            raise ValueError("frozen neural-A parameter count changed")
        self._parameters = frozen_parameters
        self._learned_model = build_learned_a_model(
            model_configuration, normalization
        )

        def combined(state, fields, moist_parameters, neural_parameters):
            state_values = _require_tree("state_q", state, _STATE_KEYS)
            field_values = _require_tree("fields_q", fields, ("B",))
            moist_values = _require_tree(
                "moist_parameters",
                moist_parameters,
                ("g", "q0", "H0", "gamma_r", "qprecip", "L", "configured_dt"),
            )
            network_values = validate_float64_tree(
                neural_parameters, name="neural_parameters"
            )
            original_rates = moist_rates_jax(
                state_values, field_values, moist_values
            )
            learned = self._learned_model(
                network_values,
                state_values,
                {
                    "B": field_values["B"],
                    "beta2": moist_values["g"] * moist_values["L"],
                },
                {"R": original_rates["R"]},
            )
            return {
                "rates": {"A": learned["A"], "R": learned["R"]},
                "source": learned["source"],
            }

        self._combined_explicit = combined
        self.combined_parameterized_kernel = (
            jax.jit(combined) if self.use_jit else combined
        )
        self._combined_frozen = lambda state, fields, moist_parameters: combined(
            state, fields, moist_parameters, self._parameters
        )
        self.combined_kernel = (
            jax.jit(self._combined_frozen) if self.use_jit else self._combined_frozen
        )

        def diagnostics(state, fields, moist_parameters):
            original = moist_diagnostics_jax(state, fields, moist_parameters)
            hybrid = self._combined_frozen(state, fields, moist_parameters)
            return {
                **original,
                "analytical_A_reference": original["A"],
                "neural_A": hybrid["rates"]["A"],
                "A": hybrid["rates"]["A"],
                "R": hybrid["rates"]["R"],
            }

        self.diagnostic_kernel = jax.jit(diagnostics) if self.use_jit else diagnostics

        def parameterized_diagnostics(
            state, fields, moist_parameters, neural_parameters
        ):
            original = moist_diagnostics_jax(state, fields, moist_parameters)
            hybrid = combined(
                state, fields, moist_parameters, neural_parameters
            )
            return {
                **original,
                "analytical_A_reference": original["A"],
                "neural_A": hybrid["rates"]["A"],
                "A": hybrid["rates"]["A"],
                "R": hybrid["rates"]["R"],
            }

        self.diagnostic_parameterized_kernel = (
            jax.jit(parameterized_diagnostics)
            if self.use_jit
            else parameterized_diagnostics
        )

        def state_jvp(state, direction, fields, moist_parameters):
            source = lambda active_state: combined(
                active_state, fields, moist_parameters, self._parameters
            )["source"]
            return jax.jvp(
                source,
                (_require_tree("state_q", state, _STATE_KEYS),),
                (_require_tree("dstate_q", direction, _STATE_KEYS),),
            )

        def state_vjp(state, source_covector, fields, moist_parameters):
            active_state = _require_tree("state_q", state, _STATE_KEYS)
            covector = _require_tree(
                "source_covector_q", source_covector, _SOURCE_KEYS
            )
            source = lambda value: combined(
                value, fields, moist_parameters, self._parameters
            )["source"]
            _, pullback = jax.vjp(source, active_state)
            return pullback(covector)[0]

        def state_differentiated_vjp(
            state,
            source_covector,
            direction,
            source_covector_direction,
            fields,
            moist_parameters,
        ):
            active_state = _require_tree("state_q", state, _STATE_KEYS)
            covector = _require_tree(
                "source_covector_q", source_covector, _SOURCE_KEYS
            )
            state_direction = _require_tree("dstate_q", direction, _STATE_KEYS)
            covector_direction = _require_tree(
                "dsource_covector_q", source_covector_direction, _SOURCE_KEYS
            )

            def vjp_map(local_state, local_covector):
                source = lambda value: combined(
                    value, fields, moist_parameters, self._parameters
                )["source"]
                _, pullback = jax.vjp(source, local_state)
                return pullback(local_covector)[0]

            return jax.jvp(
                vjp_map,
                (active_state, covector),
                (state_direction, covector_direction),
            )

        self.state_jvp_kernel = jax.jit(state_jvp) if self.use_jit else state_jvp
        self.state_vjp_kernel = jax.jit(state_vjp) if self.use_jit else state_vjp
        self.state_differentiated_vjp_kernel = (
            jax.jit(state_differentiated_vjp)
            if self.use_jit
            else state_differentiated_vjp
        )

    @property
    def parameters(self):
        return tree_copy(self._parameters)

    def combined_with_parameters(
        self, state, fields, moist_parameters, neural_parameters
    ):
        return self._combined_explicit(
            state, fields, moist_parameters, neural_parameters
        )

    def parameter_jvp(
        self,
        state,
        parameter_direction,
        fields,
        moist_parameters,
        *,
        base_parameters=None,
    ):
        base = self._parameters if base_parameters is None else validate_float64_tree(
            base_parameters, name="base_parameters"
        )
        source = lambda active_parameters: self._combined_explicit(
            state, fields, moist_parameters, active_parameters
        )["source"]
        return jax.jvp(
            source,
            (base,),
            (
                validate_float64_tree(
                    parameter_direction, name="parameter_direction"
                ),
            ),
        )

    def parameter_vjp(
        self,
        state,
        source_covector,
        fields,
        moist_parameters,
        *,
        base_parameters=None,
    ):
        covector = _require_tree(
            "source_covector_q", source_covector, _SOURCE_KEYS
        )
        source = lambda active_parameters: self._combined_explicit(
            state, fields, moist_parameters, active_parameters
        )["source"]
        base = self._parameters if base_parameters is None else validate_float64_tree(
            base_parameters, name="base_parameters"
        )
        _, pullback = jax.vjp(source, base)
        return pullback(covector)[0]

    def joint_differentiated_vjp(
        self,
        state,
        source_covector,
        state_direction,
        parameter_direction,
        source_covector_direction,
        fields,
        moist_parameters,
        *,
        base_parameters=None,
    ):
        active_state = _require_tree("state_q", state, _STATE_KEYS)
        covector = _require_tree(
            "source_covector_q", source_covector, _SOURCE_KEYS
        )
        dstate = _require_tree("dstate_q", state_direction, _STATE_KEYS)
        dparameters = validate_float64_tree(
            parameter_direction, name="parameter_direction"
        )
        dcovector = _require_tree(
            "dsource_covector_q", source_covector_direction, _SOURCE_KEYS
        )

        def vjp_map(local_state, local_parameters, local_covector):
            source = lambda state_value, parameter_value: self._combined_explicit(
                state_value, fields, moist_parameters, parameter_value
            )["source"]
            _, pullback = jax.vjp(source, local_state, local_parameters)
            return pullback(local_covector)

        base = self._parameters if base_parameters is None else validate_float64_tree(
            base_parameters, name="base_parameters"
        )
        return jax.jvp(
            vjp_map,
            (active_state, base, covector),
            (dstate, dparameters, dcovector),
        )


def load_frozen_neural_a_physics(configuration_path, *, use_jit=True):
    """Load and verify the selected artifact and training-only normalization."""
    record = load_test2a_embedding_configuration(configuration_path)
    frozen = record["frozen_operator"]
    parameter_path = Path(frozen["parameter_file"])
    _require_file_sha256(
        parameter_path, frozen["parameter_npz_sha256"], "frozen neural-A NPZ"
    )
    _require_file_sha256(
        parameter_path.with_suffix(".json"),
        frozen["parameter_sidecar_sha256"],
        "frozen neural-A parameter sidecar",
    )
    configured_model = mlp_configuration_from_record(frozen["architecture"])
    _verify_parameter_archive(parameter_path, configured_model)
    parameters, parameter_configuration = load_mlp_parameters(parameter_path)
    if parameter_pytree_sha256(parameters) != frozen["parameter_pytree_sha256"]:
        raise ValueError("frozen neural-A pytree fingerprint mismatch")
    selected_path = Path(record["selected_operator_configuration"])
    _require_file_sha256(
        selected_path,
        record["selected_operator_configuration_sha256"],
        "selected Test 2A-1 configuration",
    )
    selected = load_selected_configuration(selected_path)
    selected_model = mlp_configuration_from_record(selected["model"])
    if parameter_configuration != selected_model or configured_model != selected_model:
        raise ValueError("frozen neural-A architecture does not match Test 2A-1")
    if configured_model.parameter_count != 1281:
        raise ValueError("frozen neural-A parameter count must be 1281")

    metadata_path = Path(record["dataset_metadata"])
    _require_file_sha256(
        metadata_path,
        record["dataset_metadata_sha256"],
        "Test 2A-1 dataset metadata",
    )
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if (
        metadata.get("truth_state_indices") != [0, 80]
        or metadata.get("states_after_80_accessed", True)
        or metadata["normalization"].get("fitted_truth_state_indices") != [0, 80]
        or metadata["normalization"].get("future_states_used", True)
        or tuple(metadata["normalization"]["feature_order"]) != FEATURE_ORDER
    ):
        raise ValueError("frozen neural-A normalization provenance is invalid")
    normalization = normalization_from_record(metadata["normalization"])

    source_result_path = Path(frozen["source_result"])
    _require_file_sha256(
        source_result_path,
        frozen["source_result_sha256"],
        "frozen neural-A source result",
    )
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if (
        source_result.get("status") != "complete"
        or int(source_result["optimizer"]["additional_accepted_iterations"]) != 45000
        or not np.isclose(
            source_result["final_metrics"]["normalized_mse"],
            0.004285912836972889,
            rtol=0.0,
            atol=32.0 * np.finfo(np.float64).eps,
        )
    ):
        raise ValueError("frozen neural-A source result is not the accepted endpoint")

    provenance = {
        "embedding_configuration": str(Path(configuration_path).resolve()),
        "parameter_file": str(parameter_path.resolve()),
        "parameter_npz_sha256": frozen["parameter_npz_sha256"],
        "parameter_pytree_sha256": frozen["parameter_pytree_sha256"],
        "dataset_metadata": str(metadata_path.resolve()),
        "normalization_fitted_truth_states": [0, 80],
        "states_after_80_accessed": False,
    }
    return FrozenNeuralAMoistPhysics(
        parameters,
        configured_model,
        normalization,
        provenance=provenance,
        use_jit=use_jit,
    )


def certify_frozen_network_prediction_parity(
    embedding_configuration, dataset_path, output
):
    """Compare embedded and standalone Test-2A predictions on states 0..80."""
    physics = load_frozen_neural_a_physics(embedding_configuration)
    dataset, metadata = load_operator_dataset(dataset_path)
    standalone = physical_predictions(
        physics.parameters,
        physics._learned_model.model,
        physics.normalization,
        dataset.features,
    )
    features = dataset.features
    state = {
        name: jnp.asarray(features[:, index], dtype=jnp.float64)
        for index, name in enumerate(_STATE_KEYS)
    }
    fields = {"B": jnp.asarray(features[:, 4], dtype=jnp.float64)}
    embedding_record = load_test2a_embedding_configuration(
        embedding_configuration
    )
    _require_file_sha256(
        embedding_record["moist_activity_audit"],
        embedding_record["moist_activity_audit_sha256"],
        "Test 2 moist-activity audit",
    )
    activity = json.loads(
        Path(embedding_record["moist_activity_audit"]).read_text(
            encoding="utf-8"
        )
    )
    moist_parameters = {
        key: jnp.asarray(value, dtype=jnp.float64)
        for key, value in activity["moist_parameters"].items()
        if key != "beta2"
    }
    embedded = np.asarray(
        physics.combined_kernel(state, fields, moist_parameters)["rates"]["A"],
        dtype=np.float64,
    ).reshape(-1)
    difference = embedded - standalone
    result = {
        "status": "complete",
        "benchmark_stage": "Test 2A-2 pure-network prediction parity",
        "sample_count": dataset.sample_count,
        "truth_state_indices": metadata["truth_state_indices"],
        "states_after_80_accessed": False,
        "bitwise_equal": bool(np.array_equal(embedded, standalone)),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "relative_l2_difference": float(
            np.linalg.norm(difference)
            / max(np.linalg.norm(standalone), np.finfo(np.float64).tiny)
        ),
        "provenance": physics.provenance,
    }
    write_json_record(output, result)
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    parity = subparsers.add_parser("certify-network")
    parity.add_argument("--embedding-configuration", required=True)
    parity.add_argument("--dataset", required=True)
    parity.add_argument("--output", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "certify-network":
        certify_frozen_network_prediction_parity(
            arguments.embedding_configuration,
            arguments.dataset,
            arguments.output,
        )
        return 0
    raise AssertionError("unreachable Test 2A-2 command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "FrozenNeuralAMoistPhysics",
    "PHYSICS_MODE_ANALYTICAL",
    "PHYSICS_MODE_NEURAL_A",
    "certify_frozen_network_prediction_parity",
    "load_frozen_neural_a_physics",
    "load_test2a_embedding_configuration",
    "parameter_pytree_sha256",
)
