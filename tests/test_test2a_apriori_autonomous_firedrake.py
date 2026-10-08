"""Tiny full-split construction smoke for the Test 2A-3B evaluator."""

from types import SimpleNamespace

import numpy as np
from firedrake import assemble

from dimswe.resolved_hidden_c0 import ResolvedPilotConfiguration
from dimswe.resolved_hidden_c0_driver import (
    ResolvedDiagnosticEvaluator,
    _kinetic_energy,
    build_resolved_hidden_c0_case,
)
from dimswe.resolved_hidden_c0_inference import (
    _diagnostic_mismatch,
    _field_trajectory_metric,
    _trajectory_metric,
)
from dimswe.test2a_apriori_autonomous import load_compatible_neural_physics
from dimswe.test2a_discrete_training import _matrix_cache_components


EMBEDDING = "dimswe/configs/test2a_embedded_neural_a.json"
PARAMETERS = (
    "external-results/test2a/optimizer-study/continuation-m20-plus45000/"
    "continuation_final_parameters.npz"
)
FINGERPRINT = "f7d2fd9577ba2a824d3df6c1f8c90a425e6964e1fb5962508315509b247bed56"
CHILD_ORDER = (
    "dry_rk4_0",
    "dry_rk4_1",
    "hyperviscosity_euler",
    "dg_ssprk43_0",
    "dg_ssprk43_1",
    "moist_euler",
)


def test_tiny_neural_resolved_case_uses_complete_opt_in_split(tmp_path):
    physics = load_compatible_neural_physics(
        EMBEDDING,
        PARAMETERS,
        expected_pytree_sha256=FINGERPRINT,
        use_jit=True,
    )
    configuration = ResolvedPilotConfiguration(
        case="doublevortex",
        nx=4,
        ny=4,
        dt=100.0,
        nsteps=1,
        output_stride=1,
        c0=0.14,
        s=3.2,
        moist_backend="jax",
        seed=0,
        output_directory=str(tmp_path),
    )
    case = build_resolved_hidden_c0_case(
        configuration, jax_moist_local_physics=physics
    )
    with case.physical_c0(0.14):
        cache = case.helper.take_forward_step_cached(
            case.initial_state, case.t0, case.dt
        )
    assert cache.forward_child_order == CHILD_ORDER
    moist = [child for child in cache.children if child.name == "moist_euler"]
    assert len(moist) == 1
    assert moist[0].cache.physics_mode == "neural_A_original_R"
    assert all(
        np.all(np.isfinite(np.asarray(field.dat.data_ro)))
        for field in cache.state_out.subfunctions
    )
    prediction = {1: cache.state_out}
    truth = type("TinyTruth", (), {"states": {1: cache.state_out}})()
    mixed = _trajectory_metric(case, prediction, truth, (1,), "test2a3b_mixed")
    fields = _field_trajectory_metric(
        case, prediction, truth, (1,), "test2a3b_fields"
    )
    assert mixed["maximum"] == 0.0
    assert all(record["maximum"] == 0.0 for record in fields.values())
    evaluator = ResolvedDiagnosticEvaluator(case, configuration)
    evaluator.workspace.assign(cache.state_out)
    evaluator.vorticity_solver.solve()
    diagnostics = {
        "kinetic_energy": _kinetic_energy(case, cache.state_out),
        "projected_enstrophy": 0.5
        * float(
            assemble(
                evaluator.vorticity
                * evaluator.vorticity
                * case.model.spaces.dx
            )
        ),
    }
    for key in diagnostics:
        mismatch = _diagnostic_mismatch(
            [diagnostics[key]], [diagnostics[key]], (1,), (case.dt,)
        )
        assert mismatch["maximum_absolute_mismatch"] == 0.0
    prepared = SimpleNamespace(
        objective=SimpleNamespace(
            operations=SimpleNamespace(helper=case.helper.moist_helper)
        )
    )
    _, mass_audit = _matrix_cache_components(
        prepared,
        2.0e-13,
        16,
        2.0e-13,
        periodic_cell_shape=(4, 4),
    )
    assert mass_audit["S"]["indexing"] == (
        "cell-topology-aware periodic tensor ordering"
    )
    assert mass_audit["S"]["periodic_grid_shape"] == [12, 12]
    assert mass_audit["S"]["interpolated_coordinates_form_complete_grid"] is False
    assert mass_audit["S"]["tensor_factorization_relative_residual"] < 2.0e-13
    assert mass_audit["S"]["action_certification"][
        "maximum_relative_l2_error"
    ] < 2.0e-13
