"""Focused checks for the exact deployed hyperviscosity Euler-child audit."""

from __future__ import annotations

import numpy as np

from dimswe.hyperviscosity_stability import (
    FieldEigenvalueEstimate,
    stability_rows,
    verify_tiny_dense_oracle,
)


def _synthetic_estimate():
    return FieldEigenvalueEstimate(
        field="h",
        space="synthetic CG(3)",
        value_size=1,
        dofs=4,
        mass_nonzeros=4,
        stiffness_nonzeros=12,
        mass_diagonal_min=1.0,
        mass_diagonal_max=1.0,
        relative_mass_offdiagonal_max=0.0,
        relative_stiffness_symmetry_defect=0.0,
        r=0.5,
        s=3.2,
        laplacian_mu_max=2.0,
        laplacian_mu_residual_relative=1.0e-14,
        laplacian_mu_certified_upper_bound=2.2,
        lambda_max=0.04,
        lambda_max_certified_upper_bound=0.05,
        eigensolve_wall_time_seconds=0.01,
    )


def test_euler_stability_rows_use_exact_interval_and_conservative_bound():
    rows = stability_rows(
        "synthetic",
        16,
        16,
        400.0,
        (0.07, 0.14),
        (_synthetic_estimate(),),
        safety_factor=0.8,
    )
    stable, unstable = rows
    np.testing.assert_allclose(stable.sigma, 1.12)
    np.testing.assert_allclose(stable.sigma_certified_upper_bound, 1.4)
    assert stable.euler_amplification_bound == 1.0
    assert stable.upper_bound_certifies_stability
    np.testing.assert_allclose(stable.dt_max, 2.0 / (0.07 * 0.04))
    np.testing.assert_allclose(
        stable.recommended_dt, 0.8 * 2.0 / (0.07 * 0.05)
    )
    np.testing.assert_allclose(unstable.sigma, 2.24)
    np.testing.assert_allclose(unstable.euler_amplification_bound, 1.8)
    assert unstable.ritz_detects_instability
    assert not unstable.upper_bound_certifies_stability


def test_tiny_production_operator_matches_dense_oracle_and_deployed_child():
    result = verify_tiny_dense_oracle(
        case="doublevortex",
        nx=2,
        ny=2,
        dt=100.0,
        c0=0.14,
        s=3.2,
        dense_oracle_max_dofs=512,
        relative_tolerance=2.0e-10,
    )
    assert result.passed
    assert {item["field"] for item in result.field_comparisons} == {"v", "h", "S"}
    assert max(
        item["relative_error"] for item in result.field_comparisons
    ) < 2.0e-10
    assert max(result.deployed_child_relative_errors.values()) < 2.0e-10
    assert max(result.inactive_field_relative_errors.values()) == 0.0
