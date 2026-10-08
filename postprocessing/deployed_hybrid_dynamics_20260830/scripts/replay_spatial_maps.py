#!/usr/bin/env python3
"""Replay one frozen Test-2B hybrid model and cache compact spatial maps.

This is an evaluation-only driver.  It never constructs an optimizer and it
refuses to overwrite a cache.  Boundary quantities are sampled from Xhat_n;
moist rates are sampled from the child-6 input Yhat_n=P(Xhat_n).  At step 160
the prefix and moist child are evaluated diagnostically but are not applied.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from portable_paths import (
    GROUND_TRUTH_PACKAGE as TRUTH_FIGURE_WORKSPACE,
    M1Y_REPOSITORY as M1Y_WORKSPACE,
    PACKAGE_ROOT as OUTPUT_ROOT,
    REFERENCE_REPOSITORY as AUTHORITATIVE,
)

# Production configuration paths are repository-relative.  Run against the
# complete, writable M1-Y workspace while keeping Python from creating bytecode
# beside scientific source files.
sys.dont_write_bytecode = True
sys.path.insert(0, str(M1Y_WORKSPACE))
os.chdir(M1Y_WORKSPACE)
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault(
    "PYOP2_CACHE_DIR", "/private/tmp/dimswe-deployed-hybrid-pyop2"
)
os.environ.setdefault(
    "FIREDRAKE_TSFC_KERNEL_CACHE_DIR",
    "/private/tmp/dimswe-deployed-hybrid-tsfc",
)
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/dimswe-deployed-hybrid-xdg")

from firedrake import SpatialCoordinate, assemble  # noqa: E402

from dimswe.hidden_c0 import STATE_FIELDS, _copy_function  # noqa: E402
from dimswe.resolved_hidden_c0 import ResolvedPilotConfiguration  # noqa: E402
from dimswe.resolved_hidden_c0_driver import (  # noqa: E402
    ResolvedDiagnosticEvaluator,
    _kinetic_energy,
)
from dimswe.resolved_hidden_c0_inference import (  # noqa: E402
    _state_squared_difference,
)
from dimswe.test2a_problem_b_campaign import (  # noqa: E402
    ProblemBDiagnosticConfiguration,
)
from dimswe.test2b_rain_learning import (  # noqa: E402
    SOURCE_ORDER,
    load_parameters,
)
from dimswe.test2b_rain_learning_campaign import (  # noqa: E402
    build_neural_case,
    load_configuration,
    load_preparation,
)
from dimswe.test2b_representation_a_postprocess import (  # noqa: E402
    _field_integral,
)
from dimswe.ufl_helpers import curl2D  # noqa: E402


INTERIOR_GLL = np.asarray((5, 6, 9, 10), dtype=np.int64)
BETA2 = 98.0616
METHODS = ("M1Y", "H1", "H2", "H5")
REPRESENTATIONS = ("A", "B", "C")
HISTORICAL_DIRECTORIES = {
    "H1": "h1-from-m1-m20-5k",
    "H2": "h2-from-h1-m20-20",
    "H5": "h5-from-h2-m20-20",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def checkpoint_path(representation: str, method: str) -> Path:
    if method == "M1Y":
        return (
            M1Y_WORKSPACE
            / "external-results/m1y-test2b-20260828/production"
            / f"representation-{representation}/m1y-seed0-m20-10k"
            / "final_parameters.npz"
        )
    return (
        AUTHORITATIVE
        / "external-results/test2b-rain-active-learning/production"
        / f"representation-{representation}"
        / HISTORICAL_DIRECTORIES[method]
        / "final_parameters.npz"
    )


def accepted_record_path(representation: str, method: str) -> Path:
    if method == "M1Y":
        return (
            M1Y_WORKSPACE
            / "external-results/m1y-test2b-20260828/evaluation"
            / f"representation_{representation}_matched.json"
        )
    return (
        AUTHORITATIVE
        / "external-results/test2b-rain-active-learning/production"
        / f"representation-{representation}"
        / f"representation_{representation.lower()}_final_comparison.json"
    )


def accepted_autonomous(representation: str, method: str):
    source = accepted_record_path(representation, method)
    record = json.loads(source.read_text(encoding="utf-8"))
    if method == "M1Y":
        result = record["standard_M1_Y"]["autonomous"]
    else:
        result = record["autonomous"][method]
    return source, result


def plotting_grid(case, truth_cache):
    primal = case.helper.moist_helper.primal_helper
    coordinate = SpatialCoordinate(case.model.mesh)
    _, packed_x = primal.interpolate_and_pack(
        coordinate[0], "deployed_map_coordinate_x"
    )
    _, packed_y = primal.interpolate_and_pack(
        coordinate[1], "deployed_map_coordinate_y"
    )
    packed_x = np.asarray(packed_x, dtype=np.float64)
    packed_y = np.asarray(packed_y, dtype=np.float64)
    x_flat = packed_x[:, INTERIOR_GLL].reshape(-1)
    y_flat = packed_y[:, INTERIOR_GLL].reshape(-1)
    x_key = np.round(x_flat, decimals=6)
    y_key = np.round(y_flat, decimals=6)
    order = np.lexsort((x_key, y_key))
    nx_plot = int(np.asarray(truth_cache["x_km"]).size)
    ny_plot = int(np.asarray(truth_cache["y_km"]).size)
    if (nx_plot, ny_plot) != (128, 128):
        raise RuntimeError("accepted truth plotting-grid shape changed")
    if order.size != nx_plot * ny_plot:
        raise RuntimeError("interior-GLL plotting-grid size changed")
    x_grid = x_key[order].reshape(ny_plot, nx_plot)
    y_grid = y_key[order].reshape(ny_plot, nx_plot)
    x_km = x_grid[0] / 1000.0
    y_km = y_grid[:, 0] / 1000.0
    if not (
        np.array_equal(x_km, truth_cache["x_km"])
        and np.array_equal(y_km, truth_cache["y_km"])
    ):
        raise RuntimeError("deployed plotting grid does not equal truth grid")

    def grid(packed):
        values = np.asarray(packed, dtype=np.float64)
        return values[:, INTERIOR_GLL].reshape(-1)[order].reshape(
            ny_plot, nx_plot
        )

    return x_km, y_km, grid


def representation_c_rates(moist, normalization):
    """Exact scale-weighted two-rate projection used by accepted C audit."""
    predicted = np.stack(
        [
            np.asarray(moist.source_density[name], dtype=np.float64).reshape(-1)
            for name in SOURCE_ORDER
        ],
        axis=-1,
    )
    packed_h = np.asarray(moist.packed_state["h"], dtype=np.float64)
    h = packed_h.reshape(-1)
    scales = np.asarray(normalization.source_scales, dtype=np.float64)
    basis_physical = np.asarray(
        ((BETA2, 1.0, -1.0, 0.0), (0.0, 0.0, -1.0, 1.0)),
        dtype=np.float64,
    ).T
    basis = basis_physical / scales[:, None]
    normalized = predicted / scales
    coefficients = normalized @ basis @ np.linalg.inv(basis.T @ basis)
    residual = normalized - coefficients @ basis.T
    a_effective = (coefficients[:, 0] / h).reshape(packed_h.shape)
    r_effective = (coefficients[:, 1] / h).reshape(packed_h.shape)
    residual_norm = np.sqrt(np.sum(residual * residual, axis=-1)).reshape(
        packed_h.shape
    )
    return a_effective, r_effective, residual_norm, {
        "component_order": list(SOURCE_ORDER),
        "source_scales": scales.tolist(),
        "basis": basis.tolist(),
        "definition": (
            "scale-weighted least-squares projection of the four predicted "
            "source densities onto columns hA(beta2,1,-1,0) and "
            "hR(0,0,-1,1), identical to "
            "test2b_representation_c_postprocess._projection_diagnostics"
        ),
    }


def difference_record(candidate, reference):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        return {
            "shape_match": False,
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
            "passed": False,
        }
    absolute = np.abs(candidate - reference)
    relative = absolute / np.maximum(
        np.abs(reference), np.finfo(np.float64).tiny
    )
    return {
        "shape_match": True,
        "candidate_shape": list(candidate.shape),
        "reference_shape": list(reference.shape),
        "exact_array_equal": bool(np.array_equal(candidate, reference)),
        "maximum_absolute_difference": float(np.max(absolute)),
        "maximum_relative_difference": float(np.max(relative)),
        "allclose_rtol_5e-13_atol_1e-12": bool(
            np.allclose(candidate, reference, rtol=5.0e-13, atol=1.0e-12)
        ),
        "passed": bool(
            np.allclose(candidate, reference, rtol=5.0e-13, atol=1.0e-12)
        ),
    }


def replay(representation: str, method: str, destination: Path):
    started = perf_counter()
    if representation not in REPRESENTATIONS or method not in METHODS:
        raise ValueError("unsupported representation/method")
    sidecar_path = destination.with_suffix(".json")
    if destination.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    configuration_path = M1Y_WORKSPACE / "dimswe/configs/test2b_rain_active_learning.json"
    authoritative_configuration = AUTHORITATIVE / "dimswe/configs/test2b_rain_active_learning.json"
    preparation_path = (
        M1Y_WORKSPACE
        / "external-results/test2b-rain-active-learning/preparation"
        / "fixed_learning_data.npz"
    )
    configuration = load_configuration(configuration_path)
    _, normalization, data, _ = load_preparation(preparation_path)
    checkpoint = checkpoint_path(representation, method)
    parameters, parameter_sidecar = load_parameters(checkpoint, representation)
    accepted_path, accepted = accepted_autonomous(representation, method)

    immutable_hashes = {
        "configuration_copy": file_sha256(configuration_path),
        "configuration_authoritative": file_sha256(authoritative_configuration),
        "preparation": file_sha256(preparation_path),
        "preparation_sidecar": file_sha256(preparation_path.with_suffix(".json")),
        "checkpoint": file_sha256(checkpoint),
        "checkpoint_sidecar": file_sha256(checkpoint.with_suffix(".json")),
        "accepted_scalar_record": file_sha256(accepted_path),
    }
    if immutable_hashes["configuration_copy"] != immutable_hashes[
        "configuration_authoritative"
    ]:
        raise RuntimeError("historical configuration copy differs from authority")

    print(
        f"building frozen case for Representation {representation} {method}",
        flush=True,
    )
    case, truth, _ = build_neural_case(
        configuration, normalization, representation, parameters, 160
    )
    truth_root = Path(configuration["truth"]["run_directory"]).resolve()
    truth_metadata_path = truth_root / "metadata.json"
    truth_audit_path = truth_root / "rain_activity_audit.json"
    immutable_hashes.update(
        {
            "truth_metadata": file_sha256(truth_metadata_path),
            "truth_rain_audit": file_sha256(truth_audit_path),
        }
    )
    truth_metadata = json.loads(truth_metadata_path.read_text(encoding="utf-8"))
    pilot = ResolvedPilotConfiguration(
        **{
            **truth_metadata["configuration"],
            "output_directory": "/tmp/deployed-hybrid-dynamics-no-output",
        }
    )
    diagnostic_configuration = (
        ProblemBDiagnosticConfiguration.from_resolved_pilot(pilot)
    )
    evaluator = ResolvedDiagnosticEvaluator(case, diagnostic_configuration)

    truth_cache_path = (
        TRUTH_FIGURE_WORKSPACE
        / "outputs/ground_truth_figures_20260829/data/test2b_truth_maps.npz"
    )
    immutable_hashes["truth_map_cache"] = file_sha256(truth_cache_path)
    with np.load(truth_cache_path, allow_pickle=False) as archive:
        truth_cache = {name: np.array(archive[name], copy=True) for name in archive.files}
    x_km, y_km, grid = plotting_grid(case, truth_cache)

    shape = (161, 128, 128)
    maps = {
        name: np.empty(shape, dtype=np.float32)
        for name in (
            "relative_vorticity_1e5_s-1",
            "supersaturation_percent",
            "specific_cloud_g_kg-1",
            "specific_rain_ug_kg-1",
            "A_g_kg-1_h-1",
            "R_ug_kg-1_h-1",
        )
    }
    if representation == "C":
        maps["source_manifold_residual_normalized"] = np.empty(
            shape, dtype=np.float32
        )

    scalar = {
        name: np.empty(161, dtype=np.float64)
        for name in (
            "Qv_mass",
            "Qc_mass",
            "Qr_mass",
            "total_water_mass",
            "kinetic_energy",
            "projected_relative_vorticity_squared_half_integral",
            "projected_relative_vorticity_squared_integral",
            "mixed_state_error_numerator",
            "mixed_state_error_denominator",
            "mixed_state_relative_error",
        )
    }
    zero = case.new_state(f"spatial_{representation}_{method}_zero")
    zero.assign(0)
    current = _copy_function(
        truth[0], f"spatial_{representation}_{method}_autonomous_0"
    )
    field_index = {name: index for index, name in enumerate(STATE_FIELDS)}
    primal = case.helper.moist_helper.primal_helper
    _, packed_b = primal.interpolate_and_pack(
        primal.term.B, f"spatial_{representation}_{method}_topography"
    )
    packed_b = np.asarray(packed_b, dtype=np.float64)
    c_projection = None

    for step in range(161):
        target = truth[step]
        qv_mass = _field_integral(case, current.sub(field_index["Qv"]))
        qc_mass = _field_integral(case, current.sub(field_index["Qc"]))
        qr_mass = _field_integral(case, current.sub(field_index["Qr"]))
        scalar["Qv_mass"][step] = qv_mass
        scalar["Qc_mass"][step] = qc_mass
        scalar["Qr_mass"][step] = qr_mass
        scalar["total_water_mass"][step] = qv_mass + qc_mass + qr_mass
        scalar["kinetic_energy"][step] = _kinetic_energy(case, current)

        evaluator.workspace.assign(current)
        evaluator.vorticity_solver.solve()
        vorticity_squared = float(
            assemble(
                evaluator.vorticity
                * evaluator.vorticity
                * case.model.spaces.dx
            )
        )
        scalar[
            "projected_relative_vorticity_squared_half_integral"
        ][step] = 0.5 * vorticity_squared
        scalar["projected_relative_vorticity_squared_integral"][step] = (
            vorticity_squared
        )
        numerator = _state_squared_difference(
            case,
            current,
            target,
            f"spatial_{representation}_{method}_state_residual_{step}",
        )
        denominator = _state_squared_difference(
            case,
            target,
            zero,
            f"spatial_{representation}_{method}_state_target_{step}",
        )
        scalar["mixed_state_error_numerator"][step] = numerator
        scalar["mixed_state_error_denominator"][step] = denominator
        scalar["mixed_state_relative_error"][step] = np.sqrt(
            numerator / denominator
        )

        packed = {}
        for name in ("h", "S", "Qv", "Qc", "Qr"):
            _, packed[name] = primal.interpolate_and_pack(
                current.sub(field_index[name]),
                f"spatial_{representation}_{method}_{name}_{step}",
            )
            packed[name] = np.asarray(packed[name], dtype=np.float64)
        _, packed_curl = primal.interpolate_and_pack(
            curl2D(current.sub(field_index["v"])),
            f"spatial_{representation}_{method}_curl_{step}",
        )
        h_grid = grid(packed["h"])
        qv_grid = grid(packed["Qv"]) / h_grid
        s_grid = grid(packed["S"]) / h_grid
        qsat = (
            0.002
            * 750.0
            / (h_grid + grid(packed_b))
            * np.exp(20.0 * (1.0 - s_grid / 9.80616))
        )
        maps["relative_vorticity_1e5_s-1"][step] = (
            grid(packed_curl) * 1.0e5
        )
        maps["supersaturation_percent"][step] = 100.0 * (
            qv_grid / qsat - 1.0
        )
        maps["specific_cloud_g_kg-1"][step] = (
            grid(packed["Qc"]) / h_grid * 1.0e3
        )
        maps["specific_rain_ug_kg-1"][step] = (
            grid(packed["Qr"]) / h_grid * 1.0e9
        )

        if step < 160:
            cache = case.helper.take_forward_step_cached(
                current,
                step * case.dt,
                case.dt,
                neural_parameters=parameters,
            )
            moist = cache.children[-1].cache
            next_state = _copy_function(
                cache.state_out,
                f"spatial_{representation}_{method}_autonomous_{step + 1}",
            )
        else:
            prefix = case.helper.take_fixed_prefix_cached(
                current, step * case.dt, case.dt
            )
            moist = case.helper.moist_helper.take_forward_step_cached(
                prefix.state_out,
                step * case.dt,
                case.dt,
                neural_parameters=parameters,
            )
            next_state = None

        if representation in ("A", "B"):
            a_rate = np.asarray(moist.rates["A"], dtype=np.float64)
            r_rate = np.asarray(moist.rates["R"], dtype=np.float64)
        else:
            a_rate, r_rate, residual, c_projection = representation_c_rates(
                moist, normalization
            )
            maps["source_manifold_residual_normalized"][step] = grid(residual)
        maps["A_g_kg-1_h-1"][step] = grid(a_rate) * 3.6e6
        maps["R_ug_kg-1_h-1"][step] = grid(r_rate) * 3.6e12
        if next_state is not None:
            current = next_state

        if step % 10 == 0 or step == 160:
            print(
                f"Representation {representation} {method}: "
                f"cached step {step}/160; elapsed={perf_counter()-started:.1f}s",
                flush=True,
            )

    arrays = {
        "step": np.arange(161, dtype=np.int64),
        "time_s": np.arange(161, dtype=np.float64) * float(case.dt),
        "x_km": x_km,
        "y_km": y_km,
        **maps,
        **scalar,
    }

    mixed_summary = {
        "final": float(scalar["mixed_state_relative_error"][-1]),
        "maximum": float(np.max(scalar["mixed_state_relative_error"])),
        "maximum_step": int(np.argmax(scalar["mixed_state_relative_error"])),
        "accumulated": float(
            np.sqrt(
                np.sum(scalar["mixed_state_error_numerator"])
                / np.sum(scalar["mixed_state_error_denominator"])
            )
        ),
    }
    parity = {
        "Qc_mass": difference_record(
            scalar["Qc_mass"], accepted["boundary_timeseries"]["Qc_mass"]
        ),
        "Qr_mass": difference_record(
            scalar["Qr_mass"], accepted["boundary_timeseries"]["Qr_mass"]
        ),
        "total_water_mass": difference_record(
            scalar["total_water_mass"],
            accepted["boundary_timeseries"]["total_water_mass"],
        ),
        "kinetic_energy": difference_record(
            scalar["kinetic_energy"],
            accepted["flow"]["kinetic_energy"]["predicted"],
        ),
        "projected_relative_vorticity_squared_half_integral": difference_record(
            scalar["projected_relative_vorticity_squared_half_integral"],
            accepted["flow"]["projected_enstrophy"]["predicted"],
        ),
        "mixed_state_final": difference_record(
            [mixed_summary["final"]],
            [accepted["mixed_state_error"]["ALL"]["final"]],
        ),
        "mixed_state_maximum": difference_record(
            [mixed_summary["maximum"]],
            [accepted["mixed_state_error"]["ALL"]["maximum"]],
        ),
        "mixed_state_accumulated": difference_record(
            [mixed_summary["accumulated"]],
            [accepted["mixed_state_error"]["ALL"]["accumulated"]],
        ),
    }
    parity_passed = all(row["passed"] for row in parity.values())

    atomic_npz(destination, arrays)
    cache_hash = file_sha256(destination)
    metadata = {
        "status": "complete" if parity_passed else "failed_parity",
        "evaluation_only": True,
        "optimizer_instantiated": False,
        "truth_generated": False,
        "representation": representation,
        "training_method": method,
        "boundary_state": "model-generated Xhat_n, n=0..160",
        "moist_call_state": "model-generated Yhat_n=P(Xhat_n)",
        "final_rate_convention": (
            "at n=160 the complete fixed prefix and moist child are evaluated "
            "diagnostically; the moist update is not applied to the trajectory"
        ),
        "frame_count": 161,
        "steps": [0, 160],
        "times_s": [0.0, 16000.0],
        "plot_grid": {
            "shape_yx": [128, 128],
            "definition": (
                "two interior GLL nodes per coordinate per quadrilateral "
                "cell; exact ordering matched to accepted truth cache"
            ),
            "truth_grid_exact_array_equal": True,
        },
        "variables": {
            "relative_vorticity_1e5_s-1": "1e5 curl2D(v) at boundary Xhat",
            "supersaturation_percent": "100(qv/qsat-1) at boundary Xhat",
            "specific_cloud_g_kg-1": "1e3 Qc/h at boundary Xhat",
            "specific_rain_ug_kg-1": "1e9 Qr/h at boundary Xhat",
            "A_g_kg-1_h-1": (
                "3.6e6 learned A at Yhat" if representation != "C" else
                "3.6e6 effective A from accepted physical two-rate projection"
            ),
            "R_ug_kg-1_h-1": (
                "3.6e12 analytical R at Yhat" if representation == "A" else
                "3.6e12 learned R at Yhat" if representation == "B" else
                "3.6e12 effective R from accepted physical two-rate projection"
            ),
        },
        "representation_c_projection": c_projection,
        "normalization": normalization.to_record(),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": immutable_hashes["checkpoint"],
            "sidecar": str(checkpoint.with_suffix(".json")),
            "sidecar_sha256": immutable_hashes["checkpoint_sidecar"],
            "parameter_pytree_sha256": parameter_sidecar[
                "parameter_pytree_sha256"
            ],
        },
        "configuration": str(configuration_path),
        "preparation": str(preparation_path),
        "truth_root": str(truth_root),
        "accepted_scalar_record": str(accepted_path),
        "immutable_hashes": immutable_hashes,
        "cache": str(destination),
        "cache_sha256": cache_hash,
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        "mixed_state_summary": mixed_summary,
        "parity": parity,
        "parity_passed": parity_passed,
        "wall_seconds": float(perf_counter() - started),
    }
    write_json(sidecar_path, metadata)
    print(
        f"Representation {representation} {method}: parity "
        f"{'PASS' if parity_passed else 'FAIL'}; cache={destination}",
        flush=True,
    )
    if not parity_passed:
        raise RuntimeError("replay does not reproduce accepted scalar record")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    replay(args.representation, args.method, args.output.resolve())


if __name__ == "__main__":
    main()
