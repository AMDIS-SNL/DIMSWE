"""Prepare, certify, benchmark, and explicitly train Test2B representations.

Preparation/certification/benchmark commands are bounded and never optimize.
The ``train`` command exists only for later explicit user invocation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
from jax.flatten_util import ravel_pytree
import numpy as np

from .learned_physics.parameters import tree_axpy, tree_dot, tree_norm
from .resolved_hidden_c0 import ResolvedPilotConfiguration, read_json_record, write_json_record
from .resolved_hidden_c0_driver import build_resolved_hidden_c0_case
from .test2a_discrete_training import _matrix_cache_components
from .test2a_embedded_moist import parameter_pytree_sha256
from .test2a_problem_b_campaign import _carrier_mass_weights
from .test2b_rain_learning import (
    RainActiveNeuralMoistPhysics, RainLearningNormalization, RainMLPConfiguration,
    LINEAR_EXCEEDANCE_VARIANTS, REPRESENTATIONS,
    STRUCTURED_RAIN_VARIANTS, build_model, canonical_sha256,
    initial_parameters, structured_rain_physical_rates,
    load_parameters, save_parameters, source_invariant_diagnostics,
    structural_diagnostics,
)
from .test2b_rain_learning_prepare import file_sha256


HORIZONS = (1, 2, 5)
TRAINING_STEPS = tuple(range(81))


def _sha_record(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_configuration(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("benchmark_stage") != "Test2B rain-active learning preparation/campaign":
        raise ValueError("not a Test2B rain-active learning configuration")
    if record["truth"]["training_states"] != [0, 80] or record["truth"]["heldout_states"] != [81, 160]:
        raise ValueError("Test2B train/held-out split changed")
    if record["model"]["features"] != ["h", "S", "Qv", "Qc", "B"]:
        raise ValueError("Test2B input contract changed")
    if record["objectives"]["dense_schedules"] != {"1": [0, 79, 1], "2": [0, 78, 2], "5": [0, 75, 5]}:
        raise ValueError("Test2B dense schedules changed")
    return record


def _normalization(record):
    return RainLearningNormalization(
        input_offset=np.asarray(record["input_offset"], dtype=np.float64),
        input_scale=np.asarray(record["input_scale"], dtype=np.float64),
        sigma_a=float(record["sigma_A_all_training_support"]),
        sigma_r_active=float(record["sigma_R_active_training_support"]),
        source_scales=np.asarray(record["source_scales"], dtype=np.float64),
        provenance_sha256=str(record["provenance_sha256"]),
    )


class OperatorObjective:
    def __init__(self, representation, features, target_a, target_r, weights, normalization):
        self.representation = representation
        self.model = build_model(representation)
        x = jnp.asarray(features)
        w = jnp.asarray(weights)
        a = jnp.asarray(target_a)
        r = jnp.asarray(target_r)
        h = x[..., 0] * normalization.input_scale[0] + normalization.input_offset[0]
        beta2 = 98.0616
        if representation == "A":
            target = a[..., None]
        elif representation == "B" or representation in STRUCTURED_RAIN_VARIANTS:
            target = jnp.stack((a, r), axis=-1)
        else:
            target = jnp.stack((h * beta2 * a, h * a, -h * (a + r), h * r), axis=-1)
        scales = jnp.asarray(normalization.output_scales(representation))
        normalized_target = target / scales
        denominator = jnp.sum(w[..., None] * normalized_target * normalized_target)
        def objective(parameters):
            raw = self.model(parameters, x)
            if representation in STRUCTURED_RAIN_VARIANTS:
                predicted_a, predicted_r = structured_rain_physical_rates(
                    representation,
                    raw, h, x[..., 3] * normalization.input_scale[3]
                    + normalization.input_offset[3],
                    (
                        normalization.btp_q_precip
                        if representation == "BTP"
                        else normalization.bplus_q_precip
                    ),
                    normalization,
                )
                predicted = jnp.stack((predicted_a, predicted_r), axis=-1)
                error = predicted / scales - normalized_target
            else:
                # Preserve the accepted A/B/C normalized-output objective.
                error = raw - normalized_target
            return jnp.sum(w[..., None] * error * error) / denominator
        self.denominator = float(denominator)
        self._value = jax.jit(objective)
        self._vg = jax.jit(jax.value_and_grad(objective))

    def value(self, parameters): return float(self._value(parameters))
    def jax_value(self, parameters): return self._value(parameters)
    def value_and_gradient(self, parameters):
        value, gradient = self._vg(parameters); return float(value), gradient


class FixedObjective:
    def __init__(self, representation, features, target_a, target_r, matrices, normalizer, normalization):
        self.representation = representation
        self.model = build_model(representation)
        x = jnp.asarray(features)
        a_star, r_star = jnp.asarray(target_a), jnp.asarray(target_r)
        h = x[..., 0] * normalization.input_scale[0] + normalization.input_offset[0]
        beta2 = 98.0616
        w_s = jsparse.BCOO((jnp.asarray(matrices["S"]["data"]), jnp.asarray(matrices["S"]["indices"])), shape=tuple(matrices["S"]["shape"]))
        w_q = jsparse.BCOO((jnp.asarray(matrices["Q"]["data"]), jnp.asarray(matrices["Q"]["indices"])), shape=tuple(matrices["Q"]["shape"]))
        inverse_q = jsparse.BCOO((jnp.asarray(matrices["Q"]["mass_inverse_data"]), jnp.asarray(matrices["Q"]["mass_inverse_indices"])), shape=tuple(matrices["Q"]["mass_inverse_shape"]))
        inverse_x, inverse_y = jnp.asarray(matrices["S"]["inverse_x"]), jnp.asarray(matrices["S"]["inverse_y"])
        grid_order = jnp.asarray(matrices["S"]["grid_order"])
        grid_shape = tuple(int(v) for v in matrices["S"]["grid_shape"])
        scales = jnp.asarray(normalization.output_scales(representation))
        def energy(error):
            weak = (w_s @ error[..., 0].T).T
            grid = weak[:, grid_order].reshape((error.shape[0], *grid_shape))
            riesz = jnp.einsum("ij,bjk,lk->bil", inverse_y, grid, inverse_x)
            total = jnp.sum(grid * riesz)
            for component in (1, 2, 3):
                weak_q = (w_q @ error[..., component].T).T
                total = total + jnp.sum(weak_q * (inverse_q @ weak_q.T).T)
            return total
        truth = jnp.stack((h * beta2 * a_star, h * a_star, -h * (a_star + r_star), h * r_star), axis=-1)
        qc_total = (
            x[..., 3] * normalization.input_scale[3]
            + normalization.input_offset[3]
        )
        def objective(parameters):
            raw = self.model(parameters, x)
            if representation in STRUCTURED_RAIN_VARIANTS:
                aa, rr = structured_rain_physical_rates(
                    representation,
                    raw, h, qc_total,
                    (
                        normalization.btp_q_precip
                        if representation == "BTP"
                        else normalization.bplus_q_precip
                    ),
                    normalization,
                )
                predicted = jnp.stack(
                    (h * beta2 * aa, h * aa, -h * (aa + rr), h * rr),
                    axis=-1,
                )
            else:
                # Preserve the accepted A/B/C physical-output scaling.
                output = raw * scales
            if representation == "A":
                aa, rr = output[..., 0], r_star
                predicted = jnp.stack((h * beta2 * aa, h * aa, -h * (aa + rr), h * rr), axis=-1)
            elif representation == "B":
                aa, rr = output[..., 0], output[..., 1]
                predicted = jnp.stack((h * beta2 * aa, h * aa, -h * (aa + rr), h * rr), axis=-1)
            elif representation == "C":
                predicted = output
            return energy(predicted - truth) / float(normalizer)
        self.normalizer = float(normalizer)
        self._value = jax.jit(objective)
        self._vg = jax.jit(jax.value_and_grad(objective))

    def value(self, parameters): return float(self._value(parameters))
    def jax_value(self, parameters): return self._value(parameters)
    def value_and_gradient(self, parameters):
        value, gradient = self._vg(parameters); return float(value), gradient


def _matrix_energy(target_a, target_r, features, normalization, matrices):
    from dimswe.test2a_problem_b_campaign import _fixed_energy
    h = features[..., 0] * normalization.input_scale[0] + normalization.input_offset[0]
    target = np.stack((h * 98.0616 * target_a, h * target_a, -h * (target_a + target_r), h * target_r), axis=-1)
    return float(_fixed_energy(target, matrices))


def _analytical_arrays(case, adapter, states):
    features, aa, rr = [], [], []
    for state in states:
        result = adapter.evaluate(state, case.dt)
        features.append(np.stack((result.packed_state["h"], result.packed_state["S"], result.packed_state["Qv"], result.packed_state["Qc"], result.packed_fields["B"]), axis=-1).reshape(-1, 5))
        aa.append(np.asarray(result.rates["A"]).reshape(-1))
        rr.append(np.asarray(result.rates["R"]).reshape(-1))
    return np.stack(features), np.stack(aa), np.stack(rr)


def _analytical_case(configuration):
    from .hidden_c0 import _serial_solver_parameters
    from .jax_moist_adapter import JAXMoistEulerPrimal
    truth_root = Path(configuration["truth"]["run_directory"]).resolve()
    metadata = json.loads((truth_root / "metadata.json").read_text(encoding="utf-8"))
    pilot = ResolvedPilotConfiguration(**{**metadata["configuration"], "output_directory": str(truth_root)})
    case = build_resolved_hidden_c0_case(pilot)
    truth = {step: case.state_from_values(np.load(truth_root / "restart" / f"step_{step:08d}.npy", allow_pickle=False), f"test2b_training_truth_{step}") for step in TRAINING_STEPS}
    adapter = JAXMoistEulerPrimal(case.model, _serial_solver_parameters(), use_jit=True)
    return case, truth, adapter


def prepare_data(configuration_path, output_path):
    configuration = load_configuration(configuration_path)
    support = read_json_record(configuration["truth"]["support_audit"])
    normalization = _normalization({
        "input_offset": support["training_normalization"]["input_offset"],
        "input_scale": support["training_normalization"]["input_scale"],
        "sigma_A_all_training_support": support["training_normalization"]["sigma_A"],
        "sigma_R_active_training_support": support["training_normalization"]["sigma_R_active"],
        "source_scales": support["training_normalization"]["source_scales"],
        "provenance_sha256": support["training_normalization"]["provenance_sha256"],
    })
    case, truth, adapter = _analytical_case(configuration)
    carrier_weights, carrier_audit = _carrier_mass_weights(adapter)
    x_features, x_a, x_r = _analytical_arrays(case, adapter, [truth[i] for i in range(81)])
    # An analytical-UFL case has no opt-in neural child, so the public fixed-
    # prefix shortcut is deliberately unavailable.  Replay the accepted full
    # step and retain boundary 5 exactly as the certified truth audit does.
    postprefix = [
        case.helper.take_forward_step_cached(
            truth[i], i * case.dt, case.dt
        ).boundary_states[-2]
        for i in range(80)
    ]
    y_features, y_a, y_r = _analytical_arrays(case, adapter, postprefix)
    x_features = np.asarray(normalization.normalize_features(x_features))
    y_features = np.asarray(normalization.normalize_features(y_features))
    prepared = SimpleNamespace(
        objective=SimpleNamespace(operations=SimpleNamespace(helper=adapter))
    )
    matrices, mass_audit = _matrix_cache_components(prepared, 1.0e-11, 16, 1.0e-12, periodic_cell_shape=(64, 64))
    m2x_denominator = _matrix_energy(x_a, x_r, x_features, normalization, matrices)
    h1_source_denominator = _matrix_energy(y_a, y_r, y_features, normalization, matrices)
    arrays = {"carrier_weights": carrier_weights, "x_features": x_features, "x_A": x_a, "x_R": x_r, "y_features": y_features, "y_A": y_a, "y_R": y_r}
    for component in ("S", "Q"):
        for name, value in matrices[component].items(): arrays[f"matrix_{component}_{name}"] = np.asarray(value)
    destination = Path(output_path)
    if destination.exists() or destination.with_suffix(".json").exists(): raise FileExistsError("refusing to overwrite Test2B preparation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_name(destination.name + ".incomplete").open("wb") as stream: np.savez_compressed(stream, **arrays)
    destination.with_name(destination.name + ".incomplete").replace(destination)
    operator_denominators = {representation: OperatorObjective(representation, x_features, x_a, x_r, np.broadcast_to(carrier_weights, x_a.shape), normalization).denominator for representation in "ABC"}
    metadata = {
        "status": "complete", "truth_states": [0, 80], "heldout_states_accessed": False,
        "truth_manifest_sha256": file_sha256(configuration["truth"]["manifest"]),
        "support_audit_sha256": file_sha256(configuration["truth"]["support_audit"]),
        "normalization": normalization.to_record(), "carrier_mass_audit": carrier_audit,
        "mass_inverse_audit": mass_audit, "operator_denominators": operator_denominators,
        "m2x_denominator": m2x_denominator,
        "common_horizon_denominator": float(case.dt * case.dt * h1_source_denominator),
        "denominator_fingerprints": {
            "M2-X": canonical_sha256({"definition": "sum_0^80 ||G4 N*(X_k)||_M^2", "value": m2x_denominator}),
            "H1-H2-H5": canonical_sha256({"definition": "sum_0^79 ||dt G4 N*(Y_k)||_M^2", "value": float(case.dt * case.dt * h1_source_denominator)}),
        },
        "preparation_npz_sha256": file_sha256(destination),
        "architecture": {r: RainMLPConfiguration(r).to_record() for r in "ABC"},
        "seed0_parameter_sha256": {r: parameter_pytree_sha256(initial_parameters(r)) for r in "ABC"},
    }
    write_json_record(destination.with_suffix(".json"), metadata)
    return metadata


def load_preparation(path):
    source = Path(path); metadata = read_json_record(source.with_suffix(".json"))
    if metadata.get("status") != "complete" or metadata.get("heldout_states_accessed", True): raise ValueError("invalid preparation")
    if file_sha256(source) != metadata["preparation_npz_sha256"]: raise ValueError("preparation SHA mismatch")
    normalization = _normalization(metadata["normalization"])
    with np.load(source, allow_pickle=False) as archive:
        data = {name: np.array(archive[name], copy=True) for name in ("carrier_weights", "x_features", "x_A", "x_R", "y_features", "y_A", "y_R")}
        matrices = {}
        for component in ("S", "Q"):
            prefix = f"matrix_{component}_"; matrices[component] = {name[len(prefix):]: np.array(archive[name], copy=True) for name in archive.files if name.startswith(prefix)}
    return metadata, normalization, data, matrices


def _bplus_scale_record(preparation_path, normalization, data):
    physical = (
        np.asarray(data["x_features"], dtype=np.float64)
        * normalization.input_scale
        + normalization.input_offset
    )
    delta_q = physical[..., 3] / physical[..., 0] - 1.0e-4
    active = delta_q > 0.0
    weights = np.broadcast_to(
        np.asarray(data["carrier_weights"], dtype=np.float64), delta_q.shape
    )
    if not np.any(active):
        raise RuntimeError("BPLUS training support has no positive exceedance")
    delta_q_scale = float(np.sqrt(
        np.sum(weights[active] * delta_q[active] ** 2)
        / np.sum(weights[active])
    ))
    return {
        "definition": (
            "mass-weighted RMS of positive Qc/h-q_precip over frozen "
            "boundary truth states 0..80 at deployed GLL samples"
        ),
        "q_precip": 1.0e-4,
        "training_states": [0, 80],
        "heldout_states_accessed": False,
        "positive_sample_count": int(np.count_nonzero(active)),
        "total_sample_count": int(active.size),
        "carrier_mass_weighting": "fixed_learning_data.npz::carrier_weights",
        "delta_q_scale": delta_q_scale,
        "fixed_learning_data_npz_sha256": file_sha256(preparation_path),
        "fixed_learning_data_sidecar_sha256": file_sha256(
            Path(preparation_path).with_suffix(".json")
        ),
    }


def prepare_bplus_output_map(preparation_path, output_path):
    """Freeze the legacy BPLUS/BTPL linear-exceedance conditioning."""
    _, normalization, data, _ = load_preparation(preparation_path)
    scale = _bplus_scale_record(preparation_path, normalization, data)
    seed_b = parameter_pytree_sha256(initial_parameters("B"))
    seed_bplus = parameter_pytree_sha256(initial_parameters("BPLUS"))
    if seed_bplus != seed_b:
        raise RuntimeError("BPLUS seed-zero parameters differ from B")
    result = {
        "format_version": 1,
        "status": "complete",
        "representation": "BPLUS",
        "scientific_variant": "B_TPL",
        "scientific_change_from_B": (
            "threshold, positivity, and a linear normalized exceedance factor"
        ),
        "base_preparation": str(Path(preparation_path).resolve()),
        "training_data_modified": False,
        "heldout_accessed": False,
        "delta_q_conditioning": scale,
        "sigma_A": float(normalization.sigma_a),
        "sigma_R": float(normalization.sigma_r_active),
        "physical_output_map": (
            "A=sigma_A*a_raw; R_plus=sigma_R*max(0,(Qc/h-q_precip)/"
            "delta_q_scale)*softplus(r_raw)/log(2)"
        ),
        "analytical_gamma_r_or_tau_r_used": False,
        "architecture_B": RainMLPConfiguration("B").to_record(),
        "architecture_BPLUS": RainMLPConfiguration("BPLUS").to_record(),
        "parameter_count": RainMLPConfiguration("BPLUS").parameter_count,
        "seed0_B_parameter_pytree_sha256": seed_b,
        "seed0_BPLUS_parameter_pytree_sha256": seed_bplus,
        "initialization_bitwise_equal_to_B": True,
    }
    result["payload_sha256"] = canonical_sha256(result)
    write_json_record(output_path, result)
    return result


def prepare_btp_output_map(preparation_path, output_path):
    """Freeze the minimal threshold/positivity-only BTP output map."""
    _, normalization, _, _ = load_preparation(preparation_path)
    seed_b = parameter_pytree_sha256(initial_parameters("B"))
    seed_btp = parameter_pytree_sha256(initial_parameters("BTP"))
    if seed_btp != seed_b:
        raise RuntimeError("BTP seed-zero parameters differ from B")
    result = {
        "format_version": 1,
        "status": "complete",
        "representation": "BTP",
        "scientific_variant": "B_TP",
        "scientific_change_from_B": (
            "hard precipitation threshold and nonnegative rain output only"
        ),
        "base_preparation": str(Path(preparation_path).resolve()),
        "fixed_learning_data_npz_sha256": file_sha256(preparation_path),
        "fixed_learning_data_sidecar_sha256": file_sha256(
            Path(preparation_path).with_suffix(".json")
        ),
        "training_states": [0, 80],
        "heldout_accessed": False,
        "training_data_modified": False,
        "q_precip": 1.0e-4,
        "delta_q_scale_used": False,
        "linear_exceedance_factor_used": False,
        "sigma_A": float(normalization.sigma_a),
        "sigma_R": float(normalization.sigma_r_active),
        "physical_output_map": (
            "A=sigma_A*a_raw; R_TP=where(Qc/h>q_precip,"
            "sigma_R*softplus(r_raw)/log(2),0)"
        ),
        "analytical_gamma_r_or_tau_r_used": False,
        "architecture_B": RainMLPConfiguration("B").to_record(),
        "architecture_BTP": RainMLPConfiguration("BTP").to_record(),
        "parameter_count": RainMLPConfiguration("BTP").parameter_count,
        "seed0_B_parameter_pytree_sha256": seed_b,
        "seed0_BTP_parameter_pytree_sha256": seed_btp,
        "initialization_bitwise_equal_to_B": True,
    }
    result["payload_sha256"] = canonical_sha256(result)
    write_json_record(output_path, result)
    return result


def load_bplus_output_map(path, preparation_path, normalization, data):
    record = read_json_record(path)
    if record.get("status") != "complete" or record.get("representation") != "BPLUS":
        raise ValueError("invalid BPLUS output-map preparation")
    payload = dict(record)
    fingerprint = payload.pop("payload_sha256")
    if canonical_sha256(payload) != fingerprint:
        raise ValueError("BPLUS output-map payload fingerprint mismatch")
    expected = _bplus_scale_record(preparation_path, normalization, data)
    if record["delta_q_conditioning"] != expected:
        raise ValueError("BPLUS delta_q conditioning changed")
    if record["sigma_A"] != float(normalization.sigma_a):
        raise ValueError("BPLUS A scale differs from frozen B")
    if record["sigma_R"] != float(normalization.sigma_r_active):
        raise ValueError("BPLUS R scale differs from frozen B")
    if record["parameter_count"] != RainMLPConfiguration("B").parameter_count:
        raise ValueError("BPLUS parameter count differs from B")
    if record["seed0_BPLUS_parameter_pytree_sha256"] != record[
        "seed0_B_parameter_pytree_sha256"
    ]:
        raise ValueError("BPLUS initialization differs from B")
    enriched = replace(
        normalization,
        bplus_delta_q_scale=float(expected["delta_q_scale"]),
        bplus_q_precip=float(expected["q_precip"]),
        bplus_provenance_sha256=str(fingerprint),
    )
    return enriched, record


def load_btp_output_map(path, preparation_path, normalization):
    record = read_json_record(path)
    if record.get("status") != "complete" or record.get("representation") != "BTP":
        raise ValueError("invalid BTP output-map preparation")
    payload = dict(record)
    fingerprint = payload.pop("payload_sha256")
    if canonical_sha256(payload) != fingerprint:
        raise ValueError("BTP output-map payload fingerprint mismatch")
    if record["fixed_learning_data_npz_sha256"] != file_sha256(preparation_path):
        raise ValueError("BTP fixed training data changed")
    if record["fixed_learning_data_sidecar_sha256"] != file_sha256(
        Path(preparation_path).with_suffix(".json")
    ):
        raise ValueError("BTP fixed training-data sidecar changed")
    if record["q_precip"] != 1.0e-4:
        raise ValueError("BTP precipitation threshold changed")
    if record["delta_q_scale_used"] or record["linear_exceedance_factor_used"]:
        raise ValueError("BTP must not use an exceedance scale or factor")
    if record["sigma_A"] != float(normalization.sigma_a):
        raise ValueError("BTP A scale differs from frozen B")
    if record["sigma_R"] != float(normalization.sigma_r_active):
        raise ValueError("BTP R scale differs from frozen B")
    if record["parameter_count"] != RainMLPConfiguration("B").parameter_count:
        raise ValueError("BTP parameter count differs from B")
    if record["seed0_BTP_parameter_pytree_sha256"] != record[
        "seed0_B_parameter_pytree_sha256"
    ]:
        raise ValueError("BTP initialization differs from B")
    enriched = replace(
        normalization,
        btp_q_precip=float(record["q_precip"]),
        btp_provenance_sha256=str(fingerprint),
    )
    return enriched, record


def objectives(preparation_path, representation, rain_output_preparation_path=None):
    metadata, normalization, data, matrices = load_preparation(preparation_path)
    if representation in LINEAR_EXCEEDANCE_VARIANTS:
        if rain_output_preparation_path is None:
            raise ValueError("BTPL requires a rain-output preparation")
        normalization, bplus = load_bplus_output_map(
            rain_output_preparation_path, preparation_path, normalization, data
        )
        metadata = {**metadata, "RAIN_OUTPUT_MAP": bplus}
    elif representation == "BTP":
        if rain_output_preparation_path is None:
            raise ValueError("BTP requires a rain-output preparation")
        normalization, btp = load_btp_output_map(
            rain_output_preparation_path, preparation_path, normalization
        )
        metadata = {**metadata, "RAIN_OUTPUT_MAP": btp}
    elif rain_output_preparation_path is not None:
        raise ValueError(
            "rain-output preparation may only be used with BPLUS/BTPL/BTP"
        )
    weights = np.broadcast_to(data["carrier_weights"], data["x_A"].shape)
    return metadata, normalization, {
        "M1": OperatorObjective(representation, data["x_features"], data["x_A"], data["x_R"], weights, normalization),
        "M2-X": FixedObjective(representation, data["x_features"], data["x_A"], data["x_R"], matrices, metadata["m2x_denominator"], normalization),
        "H1": FixedObjective(representation, data["y_features"], data["y_A"], data["y_R"], matrices, metadata["common_horizon_denominator"] / 10000.0, normalization),
    }


def production_windows(horizon):
    from .test2a_trajectory import reset_windows
    return reset_windows(tuple(range(0, 80, horizon)), horizon, "accumulated", (1.0,) * horizon)


def build_neural_case(configuration, normalization, representation, parameters, maximum_step=80):
    truth_root = Path(configuration["truth"]["run_directory"]).resolve()
    source = json.loads((truth_root / "metadata.json").read_text(encoding="utf-8"))
    pilot = ResolvedPilotConfiguration(**{**source["configuration"], "moist_backend": "jax", "output_directory": "/tmp/test2b-learning-no-output"})
    physics = RainActiveNeuralMoistPhysics(representation, parameters, normalization, use_jit=True, provenance={"truth_states": [0, maximum_step]})
    case = build_resolved_hidden_c0_case(pilot, jax_moist_local_physics=physics)
    truth = {step: case.state_from_values(np.load(truth_root / "restart" / f"step_{step:08d}.npy", allow_pickle=False), f"test2b_neural_truth_{step}") for step in range(maximum_step + 1)}
    return case, truth, physics


def trajectory_objective(configuration, metadata, normalization, representation, parameters, horizon, *, windows=None):
    from .test2a_trajectory import GlobalMixedMassMetric, NeuralTrajectoryObjective
    case, truth, _ = build_neural_case(configuration, normalization, representation, parameters, 80)
    denominator = metadata["common_horizon_denominator"]
    metric = GlobalMixedMassMetric(case.helper, denominator, denominator_sha256=metadata["denominator_fingerprints"]["H1-H2-H5"])
    return case, NeuralTrajectoryObjective(case, truth, production_windows(horizon) if windows is None else windows, metric=metric, c0=0.14, use_fixed_prefix=True)


def _direction(parameters):
    flat, unravel = ravel_pytree(parameters); values = np.linspace(-0.7, 0.9, flat.size); values /= np.linalg.norm(values); return unravel(jnp.asarray(values))


def _directional(objective, parameters, direction, epsilon):
    value, gradient = objective.value_and_gradient(parameters); adjoint = float(tree_dot(gradient, direction)); fd = (objective.value(tree_axpy(parameters, epsilon, direction)) - objective.value(tree_axpy(parameters, -epsilon, direction))) / (2 * epsilon)
    return {"value": value, "gradient_norm": float(tree_norm(gradient)), "adjoint": adjoint, "centered_FD": fd, "absolute_discrepancy": abs(adjoint-fd), "relative_discrepancy": abs(adjoint-fd)/max(abs(adjoint),abs(fd),np.finfo(float).tiny)}


def _duality(case, objective, parameters, horizon, start=0):
    from .hidden_c0 import _copy_function
    from .learned_physics.parameters import tree_zeros
    tape = objective._tape(parameters).windows[0]
    direction = _copy_function(
        objective.truth_states[start], "test2b_state_direction"
    )
    with direction.dat.vec as vector: vector.scale(1e-7)
    current = direction
    for cache in tape.step_caches: current = case.helper.take_neural_parameter_tangent_step(cache, current, tree_zeros(parameters)).state_direction_out
    probe_state = _copy_function(
        objective.truth_states[start + horizon], "test2b_probe_state"
    )
    with probe_state.dat.vec as vector: vector.scale(1e-7)
    probe = case.helper.state_mass_map(probe_state, "test2b_probe_dual"); adjoint = probe
    for cache in reversed(tape.step_caches): adjoint = case.helper.take_neural_parameter_adjoint_step(cache, adjoint, stop_at_fixed_prefix=False).state_adjoint_in
    left, right = case.helper.dual_pairing(probe, current), case.helper.dual_pairing(adjoint, direction)
    return {"tangent_pairing": left, "adjoint_pairing": right, "absolute_discrepancy": abs(left-right), "relative_discrepancy": abs(left-right)/max(abs(left),abs(right),np.finfo(float).tiny)}


class _LiteralMoistChildObjective:
    """Read-only Firedrake oracle for one fixed child-6 source map."""

    def __init__(self, case, state, normalizer):
        from .jax_moist_hvp import JAXMoistEulerHVP
        self.helper = case.helper.moist_helper
        self.state = state
        self.dt = float(case.dt)
        self.normalizer = float(normalizer)
        analytical = JAXMoistEulerHVP(
            case.helper.moist_child.ufl_oracle, use_jit=True, local_physics=None
        )
        self.target = analytical.take_forward_step_cached(
            state, 0.0, self.dt
        ).state_out

    def value_and_gradient(self, parameters):
        from .hidden_c0 import _copy_function
        primal = self.helper.take_forward_step_cached(
            self.state, 0.0, self.dt, neural_parameters=parameters
        )
        residual = _copy_function(primal.state_out, "test2b_literal_child_residual")
        with residual.dat.vec as output, self.target.dat.vec_ro as target:
            output.axpy(-1.0, target)
        dual = self.helper.state_mass_map(residual, "test2b_literal_child_dual")
        value = self.helper.dual_pairing(dual, residual) / self.normalizer
        with dual.dat.vec as vector:
            vector.scale(2.0 / self.normalizer)
        gradient = self.helper.take_parameter_adjoint_step(
            primal, dual
        ).parameter_adjoint
        return float(value), gradient

    def value(self, parameters):
        return self.value_and_gradient(parameters)[0]


def certify(configuration_path, preparation_path, output_path):
    from .test2a_trajectory import reset_windows
    configuration = load_configuration(configuration_path)
    metadata, normalization, data, matrices = load_preparation(preparation_path)
    weights = np.broadcast_to(data["carrier_weights"], data["x_A"].shape)
    result = {"status": "in_progress", "truth_states": [0, 80], "heldout_accessed": False, "representations": {}}
    for representation in "ABC":
        fixed = {
            "M1": OperatorObjective(
                representation, data["x_features"], data["x_A"], data["x_R"],
                weights, normalization,
            ),
            "M2-X": FixedObjective(
                representation, data["x_features"], data["x_A"], data["x_R"],
                matrices, metadata["m2x_denominator"], normalization,
            ),
        }
        parameters = initial_parameters(representation); direction = _direction(parameters)
        record = {"architecture": RainMLPConfiguration(representation).to_record(), "seed0_sha256": parameter_pytree_sha256(parameters), "M1_gradient": _directional(fixed["M1"], parameters, direction, 2e-6), "M2_X_gradient": _directional(fixed["M2-X"], parameters, direction, 2e-6)}
        case, h1 = trajectory_objective(configuration, metadata, normalization, representation, parameters, 1, windows=reset_windows((0,),1,"accumulated",(1.0,)))
        fast_one = FixedObjective(
            representation, data["y_features"][:1], data["y_A"][:1],
            data["y_R"][:1], matrices,
            metadata["common_horizon_denominator"] / 10000.0,
            normalization,
        )
        fv, fg = fast_one.value_and_gradient(parameters); lv, lg = h1.value_and_gradient(parameters)
        ff,_=ravel_pytree(fg); lf,_=ravel_pytree(lg)
        record["H1_cache_literal"] = {"cached_value": fv, "literal_value": lv, "value_absolute_difference": abs(fv-lv), "gradient_absolute_error": float(np.linalg.norm(np.asarray(ff-lf))), "gradient_relative_error": float(np.linalg.norm(np.asarray(ff-lf))/max(np.linalg.norm(np.asarray(lf)),np.finfo(float).tiny)), "directional": _directional(h1,parameters,direction,2e-5)}
        recursive = {}
        for horizon in (2,5):
            case, objective = trajectory_objective(configuration, metadata, normalization, representation, parameters, horizon, windows=reset_windows((0,),horizon,"accumulated",(1.0,)*horizon))
            recursive[f"H{horizon}"] = {"state_tangent_adjoint": _duality(case, objective, parameters, horizon), "parameter_directional": _directional(objective,parameters,direction,2e-5)}
        record["recursive"] = recursive
        state = {"h": jnp.full((2,3),700.),"S":jnp.full((2,3),6800.),"Qv":jnp.full((2,3),1.4),"Qc":jnp.full((2,3),.08)}; fields={"B":jnp.zeros((2,3))}; moist={"g":jnp.asarray(9.80616),"q0":jnp.asarray(.002),"H0":jnp.asarray(750.),"gamma_r":jnp.asarray(.001),"qprecip":jnp.asarray(1e-4),"L":jnp.asarray(10.),"configured_dt":jnp.asarray(100.)}
        source = RainActiveNeuralMoistPhysics(representation,parameters,normalization,use_jit=False).combined_kernel(state,fields,moist)["source"]
        record["arbitrary_output_invariants"] = source_invariant_diagnostics(source,98.0616)
        result["representations"][representation] = record
    result["classification"]={"H1":"fixed post-prefix, truth-reset, exactly cacheable","H2":"first recursive model-generated-state objective","H5":"five-step recursive objective"}; result["status"]="complete"; write_json_record(output_path,result); return result


def certify_oracles(configuration_path, preparation_path, output_path):
    from .test2a_trajectory import reset_windows
    configuration = load_configuration(configuration_path)
    metadata, normalization, data, matrices = load_preparation(preparation_path)
    result = {
        "status": "in_progress", "truth_states": [0, 80],
        "heldout_accessed": False, "representations": {},
    }
    for representation in "ABC":
        parameters = initial_parameters(representation)
        case, truth, _ = build_neural_case(
            configuration, normalization, representation, parameters, 1
        )
        fast = FixedObjective(
            representation, data["x_features"][:1], data["x_A"][:1],
            data["x_R"][:1], matrices, metadata["m2x_denominator"],
            normalization,
        )
        literal = _LiteralMoistChildObjective(
            case, truth[0], 10000.0 * metadata["m2x_denominator"]
        )
        fast_value, fast_gradient = fast.value_and_gradient(parameters)
        literal_value, literal_gradient = literal.value_and_gradient(parameters)
        fast_flat, _ = ravel_pytree(fast_gradient)
        literal_flat, _ = ravel_pytree(literal_gradient)
        difference = np.asarray(fast_flat - literal_flat)
        case_h1, h1 = trajectory_objective(
            configuration, metadata, normalization, representation, parameters,
            1, windows=reset_windows((0,), 1, "accumulated", (1.0,)),
        )
        result["representations"][representation] = {
            "M2_X_cache_vs_literal_child": {
                "cached_value": fast_value, "literal_value": literal_value,
                "value_absolute_difference": abs(fast_value - literal_value),
                "value_relative_difference": abs(fast_value - literal_value)
                / max(abs(literal_value), np.finfo(float).tiny),
                "gradient_absolute_error": float(np.linalg.norm(difference)),
                "gradient_relative_error": float(np.linalg.norm(difference)
                / max(np.linalg.norm(np.asarray(literal_flat)), np.finfo(float).tiny)),
                "maximum_component_difference": float(np.max(np.abs(difference))),
            },
            "H1_state_tangent_adjoint": _duality(case_h1, h1, parameters, 1),
        }
    result["status"] = "complete"
    write_json_record(output_path, result)
    return result


def benchmark(
    configuration_path, preparation_path, output_path, repeats=2,
    representations=("A", "B", "C"), recursive_representation="B",
    rain_output_preparation_path=None,
):
    configuration=load_configuration(configuration_path); result={"status":"in_progress","truth_states":[0,80],"heldout_accessed":False,"representations":{}}
    for representation in tuple(representations):
        metadata,normalization,fixed=objectives(
            preparation_path, representation,
            (
                rain_output_preparation_path
                if representation in STRUCTURED_RAIN_VARIANTS else None
            ),
        ); parameters=initial_parameters(representation); rec={}
        for name,objective in fixed.items():
            t=perf_counter(); objective.value(parameters); first_v=perf_counter()-t; t=perf_counter(); objective.value_and_gradient(parameters); first_g=perf_counter()-t; vs=[];gs=[]
            for _ in range(repeats):
                t=perf_counter();objective.value(parameters);vs.append(perf_counter()-t);t=perf_counter();objective.value_and_gradient(parameters);gs.append(perf_counter()-t)
            rec[name]={"first_value_seconds":first_v,"first_value_gradient_seconds":first_g,"steady_value_median_seconds":float(np.median(vs)),"steady_value_gradient_median_seconds":float(np.median(gs)),"Firedrake_PETSc_solves":0}
        for horizon in ((2, 5) if representation == recursive_representation else ()):
            t=perf_counter();_,objective=trajectory_objective(configuration,metadata,normalization,representation,parameters,horizon);setup=perf_counter()-t; objective.clear_parameter_tape();t=perf_counter();value=objective.value(parameters);vtime=perf_counter()-t;t=perf_counter();same,gradient=objective.value_and_gradient(parameters);gtime=perf_counter()-t
            rec[f"H{horizon}"]={"setup_seconds":setup,"window_count":len(production_windows(horizon)),"complete_timesteps":80,"value":value,"same_theta_value":same,"gradient_norm":float(tree_norm(gradient)),"value_seconds":vtime,"gradient_after_same_theta_value_seconds":gtime,"value_plus_gradient_seconds":vtime+gtime,"same_theta_tape_reused":objective.work_counts().same_theta_tape_hits>0}
        result["representations"][representation]=rec
    result["status"]="complete";write_json_record(output_path,result);return result


def _evaluate_artifact(configuration, preparation_path, representation, label, path):
    from firedrake import assemble
    from .hidden_c0 import _copy_function
    from .resolved_hidden_c0_driver import ResolvedDiagnosticEvaluator
    from .resolved_hidden_c0_inference import (
        _diagnostic_mismatch, _field_trajectory_metric, _trajectory_metric,
    )
    from .test2a_problem_b_campaign import ProblemBDiagnosticConfiguration

    metadata, normalization, data, matrices = load_preparation(preparation_path)
    parameters, sidecar = load_parameters(path, representation)
    weights = np.broadcast_to(data["carrier_weights"], data["x_A"].shape)
    fixed = {
        "J_M1": OperatorObjective(representation, data["x_features"], data["x_A"], data["x_R"], weights, normalization).value(parameters),
        "J_M2_X": FixedObjective(representation, data["x_features"], data["x_A"], data["x_R"], matrices, metadata["m2x_denominator"], normalization).value(parameters),
        "J_H1": FixedObjective(representation, data["y_features"], data["y_A"], data["y_R"], matrices, metadata["common_horizon_denominator"] / 10000.0, normalization).value(parameters),
        "parameter_file": str(Path(path).resolve()),
        "parameter_pytree_sha256": sidecar["parameter_pytree_sha256"],
        "representation": representation, "label": label,
    }
    model = build_model(representation)
    output = np.asarray(model(parameters, jnp.asarray(data["x_features"]))) * normalization.output_scales(representation)
    h = data["x_features"][..., 0] * normalization.input_scale[0] + normalization.input_offset[0]
    truth_source = np.stack((h * 98.0616 * data["x_A"], h * data["x_A"], -h * (data["x_A"] + data["x_R"]), h * data["x_R"]), axis=-1)
    if representation == "A":
        aa, rr = output[..., 0], data["x_R"]
        prediction = np.stack((h * 98.0616 * aa, h * aa, -h * (aa + rr), h * rr), axis=-1)
    elif representation == "B":
        aa, rr = output[..., 0], output[..., 1]
        prediction = np.stack((h * 98.0616 * aa, h * aa, -h * (aa + rr), h * rr), axis=-1)
    else:
        prediction = output
    fixed["structural_diagnostics_on_training_support"] = structural_diagnostics(
        prediction, truth_source, 98.0616, normalization.source_scales,
        weights.reshape(-1),
    )
    for horizon in (2, 5):
        _, objective = trajectory_objective(configuration, metadata, normalization, representation, parameters, horizon)
        fixed[f"J_H{horizon}"] = objective.value(parameters)

    case, truth, _ = build_neural_case(configuration, normalization, representation, parameters, 80)
    generated = {0: _copy_function(truth[0], f"test2b_{label}_autonomous_0")}
    source_records = []
    for step in range(80):
        cache = case.helper.take_forward_step_cached(
            generated[step], step * case.dt, case.dt,
            neural_parameters=parameters,
        )
        generated[step + 1] = _copy_function(cache.state_out, f"test2b_{label}_autonomous_{step+1}")
        moist = cache.children[-1].cache
        source_records.append({
            "step": step, "time": float(step * case.dt),
            **source_invariant_diagnostics(moist.source_density, 98.0616),
            "maximum_absolute_Qr_t": float(np.max(np.abs(np.asarray(moist.source_density["Qr"])))),
            "negative_Qr_t_fraction": float(np.mean(np.asarray(moist.source_density["Qr"]) < 0.0)),
        })
    steps = tuple(range(1, 81)); truth_proxy = type("TruthProxy", (), {"states": truth})()
    mixed = _trajectory_metric(case, generated, truth_proxy, steps, f"test2b_{label}_mixed")
    fieldwise = _field_trajectory_metric(case, generated, truth_proxy, steps, f"test2b_{label}_field")
    diagnostic_configuration = ProblemBDiagnosticConfiguration.from_resolved_pilot(
        ResolvedPilotConfiguration(**{**json.loads((Path(configuration["truth"]["run_directory"]) / "metadata.json").read_text())["configuration"], "output_directory": "/tmp/test2b-postprocess-no-output"})
    )
    evaluator = ResolvedDiagnosticEvaluator(case, diagnostic_configuration)
    predicted = [evaluator.evaluate(generated[step], step, step * case.dt)[0] for step in steps]
    reference = [evaluator.evaluate(truth[step], step, step * case.dt)[0] for step in steps]
    times = np.asarray([step * case.dt for step in steps])
    water = [float(assemble((generated[step].sub(3)+generated[step].sub(4)+generated[step].sub(5))*case.model.spaces.dx)) for step in range(81)]
    thermo = [float(assemble((generated[step].sub(2)-98.0616*generated[step].sub(3))*case.model.spaces.dx)) for step in range(81)]
    fixed["autonomous_training_support_posthoc"] = {
        "mixed_state_error": mixed, "fieldwise_errors": fieldwise,
        "kinetic_energy": _diagnostic_mismatch([r["kinetic_energy"] for r in predicted],[r["kinetic_energy"] for r in reference],steps,times),
        "projected_enstrophy": _diagnostic_mismatch([r["projected_enstrophy"] for r in predicted],[r["projected_enstrophy"] for r in reference],steps,times),
        "source_records": source_records,
        "total_water_integral": water,
        "S_minus_beta2_Qv_integral": thermo,
        "maximum_accumulated_total_water_drift": float(np.max(np.abs(np.asarray(water)-water[0]))),
        "maximum_accumulated_thermodynamic_drift": float(np.max(np.abs(np.asarray(thermo)-thermo[0]))),
        "used_for_model_selection": False,
    }
    return fixed


def postprocess(configuration_path, preparation_path, output_path, artifacts):
    configuration = load_configuration(configuration_path)
    specifications = []
    for specification in artifacts:
        if "=" not in specification or ":" not in specification.split("=", 1)[0]:
            raise ValueError("artifact must be REPRESENTATION:LABEL=PATH")
        identity, path = specification.split("=", 1)
        representation, label = identity.split(":", 1)
        if representation not in "ABC" or not label:
            raise ValueError("invalid postprocess artifact identity")
        specifications.append((representation, label, path))
    result = {
        "status": "in_progress", "training_truth_states": [0, 80],
        "heldout_used_for_selection": False, "artifacts": {},
    }
    for representation, label, path in specifications:
        result["artifacts"][f"{representation}:{label}"] = _evaluate_artifact(
            configuration, preparation_path, representation, label, path
        )
    result["status"] = "complete"
    write_json_record(output_path, result)
    return result


def train(
    configuration_path, preparation_path, representation, stage,
    output_directory, iteration_limit, initial_parameter_file=None,
    rain_output_preparation_path=None,
):
    """Explicit later production entry point; never called by preparation."""
    from pyrol import Problem, Solver
    from .test2a_discrete_training import CompactCheckpointObjective
    from .test2a_pyrol import build_test2a_lbfgs_parameters
    configuration=load_configuration(configuration_path);metadata,normalization,fixed=objectives(preparation_path,representation,rain_output_preparation_path)
    parameters=initial_parameters(representation) if initial_parameter_file is None else load_parameters(initial_parameter_file,representation)[0]
    if stage=="M1": objective=fixed["M1"]
    elif stage=="M2-X": objective=fixed["M2-X"]
    elif stage=="H1": objective=fixed["H1"]
    elif stage in ("H2","H5"): _,objective=trajectory_objective(configuration,metadata,normalization,representation,parameters,int(stage[1:]))
    else: raise ValueError("stage must be M1, M2-X, H1, H2, or H5")
    output=Path(output_directory)
    if output.exists(): raise FileExistsError("refusing to overwrite production stage")
    output.mkdir(parents=True)
    checkpoint_schedule = tuple(sorted(
        value for value in {0, 1, 5, 10, 20, 100, 500, 1000, 5000,
                            10000, int(iteration_limit)}
        if value <= int(iteration_limit)
    ))
    def parameter_metadata(iteration):
        result = {"stage": stage, "accepted_iteration": int(iteration)}
        if representation in STRUCTURED_RAIN_VARIANTS:
            result["rain_output_map_payload_sha256"] = metadata["RAIN_OUTPUT_MAP"][
                "payload_sha256"
            ]
            result["rain_output_variant"] = representation
        return result
    save_parameters(output/"checkpoint_000000000_parameters.npz",representation,parameters,metadata=parameter_metadata(0))
    started=perf_counter()
    def accepted_callback(control, local_index, adapter):
        if local_index == 0 or (
            local_index not in checkpoint_schedule and local_index % 100 != 0
        ):
            return
        current = adapter.pytree_from_vector(control)
        if local_index in checkpoint_schedule:
            record = save_parameters(
                output / f"checkpoint_{local_index:09d}_parameters.npz",
                representation, current,
                metadata=parameter_metadata(local_index),
            )
            parameter_sha = record["parameter_pytree_sha256"]
        else:
            parameter_sha = parameter_pytree_sha256(current)
        progress = {
            "status": "in_progress", "representation": representation,
            "stage": stage, "accepted_iteration": int(local_index),
            "objective": float(objective.value(current)),
            "elapsed_wall_seconds": float(perf_counter() - started),
            "objective_evaluations": int(adapter.value_evaluations),
            "gradient_evaluations": int(adapter.gradient_evaluations),
            "parameter_pytree_sha256": parameter_sha,
            "source_secant_history_reused": False,
            "parameter_only_restart_restores_secant_history": False,
        }
        if representation in STRUCTURED_RAIN_VARIANTS:
            progress["rain_output_map_payload_sha256"] = metadata["RAIN_OUTPUT_MAP"][
                "payload_sha256"
            ]
            progress["rain_output_variant"] = representation
        write_json_record(output / "fit_progress.json", progress)
    if stage in ("H2","H5"):
        from .test2a_trajectory import TrajectoryPyROLObjective
        adapter=TrajectoryPyROLObjective(objective,parameters,accepted_callback=accepted_callback)
    else: adapter=CompactCheckpointObjective(objective.jax_value,parameters,use_jit=True,accepted_callback=accepted_callback)
    control=adapter.vector_from_pytree(parameters)
    solver=Solver(Problem(adapter,control),build_test2a_lbfgs_parameters({"gradient_tolerance":1e-8,"step_tolerance":1e-12,"iteration_limit":int(iteration_limit),"maximum_secant_storage":20}));solver.solve();final=adapter.pytree_from_vector(control);record=save_parameters(output/"final_parameters.npz",representation,final,metadata=parameter_metadata(solver.getAlgorithmState().iter))
    final_result={"status":"complete","representation":representation,"stage":stage,"accepted_iterations":int(solver.getAlgorithmState().iter),"termination_reason":str(solver.getAlgorithmState().statusFlag),"final_objective":float(objective.value(final)),"objective_evaluations":adapter.value_evaluations,"gradient_evaluations":adapter.gradient_evaluations,"wall_seconds":perf_counter()-started,"final_parameter_file":str((output/"final_parameters.npz").resolve()),"final_parameter_pytree_sha256":record["parameter_pytree_sha256"],"source_secant_history_reused":False,"parameter_only_restart_restores_secant_history":False}
    if representation in STRUCTURED_RAIN_VARIANTS:
        final_result["rain_output_map_payload_sha256"] = metadata["RAIN_OUTPUT_MAP"][
            "payload_sha256"
        ]
        final_result["rain_output_variant"] = representation
    write_json_record(output/"fit_result.json",final_result)
    write_json_record(output/"fit_progress.json",final_result)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    for name in ("prepare-data","certify","certify-oracles","benchmark"):
        p=sub.add_parser(name);p.add_argument("--configuration",required=True);p.add_argument("--output",required=True)
        if name!="prepare-data":p.add_argument("--preparation",required=True)
    sub.choices["benchmark"].add_argument("--repeats",type=int,default=2)
    sub.choices["benchmark"].add_argument("--representations",default="A,B,C")
    sub.choices["benchmark"].add_argument("--recursive-representation",choices=REPRESENTATIONS,default="B")
    sub.choices["benchmark"].add_argument("--bplus-preparation")
    sub.choices["benchmark"].add_argument("--rain-output-preparation")
    p=sub.add_parser("prepare-bplus");p.add_argument("--preparation",required=True);p.add_argument("--output",required=True)
    p=sub.add_parser("prepare-btp");p.add_argument("--preparation",required=True);p.add_argument("--output",required=True)
    p=sub.add_parser("train");p.add_argument("--configuration",required=True);p.add_argument("--preparation",required=True);p.add_argument("--representation",choices=REPRESENTATIONS,required=True);p.add_argument("--stage",choices=("M1","M2-X","H1","H2","H5"),required=True);p.add_argument("--output-directory",required=True);p.add_argument("--iteration-limit",type=int,required=True);p.add_argument("--initial-parameters");p.add_argument("--bplus-preparation");p.add_argument("--rain-output-preparation")
    p=sub.add_parser("postprocess");p.add_argument("--configuration",required=True);p.add_argument("--preparation",required=True);p.add_argument("--output",required=True);p.add_argument("--artifact",action="append",required=True)
    args=parser.parse_args(argv)
    if (
        hasattr(args, "bplus_preparation")
        and args.bplus_preparation
        and args.rain_output_preparation
    ):
        parser.error("use only one rain-output preparation argument")
    if args.command=="prepare-data":prepare_data(args.configuration,args.output)
    elif args.command=="prepare-bplus":prepare_bplus_output_map(args.preparation,args.output)
    elif args.command=="prepare-btp":prepare_btp_output_map(args.preparation,args.output)
    elif args.command=="certify":certify(args.configuration,args.preparation,args.output)
    elif args.command=="certify-oracles":certify_oracles(args.configuration,args.preparation,args.output)
    elif args.command=="benchmark":benchmark(
        args.configuration,args.preparation,args.output,args.repeats,
        tuple(value.strip() for value in args.representations.split(",") if value.strip()),
        args.recursive_representation,
        args.rain_output_preparation or args.bplus_preparation,
    )
    elif args.command=="train":train(args.configuration,args.preparation,args.representation,args.stage,args.output_directory,args.iteration_limit,args.initial_parameters,args.rain_output_preparation or args.bplus_preparation)
    else:postprocess(args.configuration,args.preparation,args.output,args.artifact)


if __name__=="__main__":main()
