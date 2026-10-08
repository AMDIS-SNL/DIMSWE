import jax
import jax.numpy as jnp
import numpy as np

from dimswe.test2a_embedded_moist import parameter_pytree_sha256
from dimswe.test2b_rain_learning import (
    RainActiveNeuralMoistPhysics,
    RainLearningNormalization,
    RainMLPConfiguration,
    bplus_physical_rates,
    btp_physical_rates,
    build_model,
    initial_parameters,
    source_invariant_diagnostics,
)


jax.config.update("jax_enable_x64", True)


DELTA_Q_SCALE = 1.9902871261559997e-6
Q_PRECIP = 1.0e-4


def normalization():
    return RainLearningNormalization(
        input_offset=np.zeros(5),
        input_scale=np.ones(5),
        sigma_a=9.052258655848717e-8,
        sigma_r_active=1.9902871261559996e-11,
        source_scales=np.asarray(
            (6.671477765500949e-3, 6.803353979030477e-5,
             6.80335397581467e-5, 1.5076498196845062e-8)
        ),
        provenance_sha256="0" * 64,
        bplus_delta_q_scale=DELTA_Q_SCALE,
        bplus_q_precip=Q_PRECIP,
        bplus_provenance_sha256="1" * 64,
    )


def btp_normalization():
    return RainLearningNormalization(
        input_offset=np.zeros(5),
        input_scale=np.ones(5),
        sigma_a=9.052258655848717e-8,
        sigma_r_active=1.9902871261559996e-11,
        source_scales=np.asarray(
            (6.671477765500949e-3, 6.803353979030477e-5,
             6.80335397581467e-5, 1.5076498196845062e-8)
        ),
        provenance_sha256="0" * 64,
        btp_q_precip=Q_PRECIP,
        btp_provenance_sha256="2" * 64,
    )


def local_inputs(delta_q):
    shape = (2, 3)
    h = jnp.full(shape, 700.0, dtype=jnp.float64)
    state = {
        "h": h,
        "S": jnp.full(shape, 6800.0, dtype=jnp.float64),
        "Qv": jnp.full(shape, 1.4, dtype=jnp.float64),
        "Qc": h * (Q_PRECIP + delta_q),
    }
    fields = {"B": jnp.zeros(shape, dtype=jnp.float64)}
    moist = {
        "g": jnp.asarray(9.80616),
        "q0": jnp.asarray(0.002),
        "H0": jnp.asarray(750.0),
        "gamma_r": jnp.asarray(0.001),
        "qprecip": jnp.asarray(Q_PRECIP),
        "L": jnp.asarray(10.0),
        "configured_dt": jnp.asarray(100.0),
    }
    return state, fields, moist


def test_parameter_count_and_seed_initialization_are_identical_to_b():
    assert RainMLPConfiguration("B").parameter_count == 1314
    assert RainMLPConfiguration("BPLUS").parameter_count == 1314
    assert RainMLPConfiguration("BTPL").parameter_count == 1314
    assert RainMLPConfiguration("BTP").parameter_count == 1314
    b = initial_parameters("B")
    bplus = initial_parameters("BPLUS")
    btpl = initial_parameters("BTPL")
    btp = initial_parameters("BTP")
    assert parameter_pytree_sha256(b) == parameter_pytree_sha256(bplus)
    assert parameter_pytree_sha256(b) == parameter_pytree_sha256(btpl)
    assert parameter_pytree_sha256(b) == parameter_pytree_sha256(btp)
    for candidate in (bplus, btpl, btp):
        for left, right in zip(
            jax.tree_util.tree_leaves(b), jax.tree_util.tree_leaves(candidate)
        ):
            assert np.array_equal(np.asarray(left), np.asarray(right))


def test_btp_is_exactly_gated_nonnegative_and_has_no_exceedance_factor():
    normalizer = btp_normalization()
    raw = jnp.asarray(
        ((0.2, -1000.0), (0.2, -3.0), (0.2, 0.0), (0.2, 1000.0)),
        dtype=jnp.float64,
    )
    h = jnp.full(4, 700.0, dtype=jnp.float64)
    qc = h * jnp.asarray(
        (Q_PRECIP - 1.0e-8, Q_PRECIP, Q_PRECIP + 1.0e-12,
         Q_PRECIP + 5.0e-5),
        dtype=jnp.float64,
    )
    _, rain = btp_physical_rates(raw, h, qc, Q_PRECIP, normalizer)
    assert float(rain[0]) == 0.0
    assert float(rain[1]) == 0.0
    assert np.all(np.asarray(rain[2:]) >= 0.0)
    raw_equal = jnp.asarray(((0.0, 0.4), (0.0, 0.4)))
    _, equal_rain = btp_physical_rates(
        raw_equal,
        h[:2],
        h[:2] * jnp.asarray((Q_PRECIP + 1.0e-12, Q_PRECIP + 5.0e-5)),
        Q_PRECIP,
        normalizer,
    )
    assert np.array_equal(np.asarray(equal_rain[:1]), np.asarray(equal_rain[1:]))
    assert float(equal_rain[0]) > 0.0


def test_btp_gate_state_derivative_is_zero_on_each_smooth_side():
    normalizer = btp_normalization()
    raw = jnp.asarray(((0.25, -0.4),), dtype=jnp.float64)
    h = jnp.asarray((700.0,), dtype=jnp.float64)
    rain = lambda qc: btp_physical_rates(
        raw, h, qc, Q_PRECIP, normalizer
    )[1]
    direction = jnp.ones(1, dtype=jnp.float64)
    for qc in (
        h * (Q_PRECIP - 1.0e-6),
        h * (Q_PRECIP + 1.0e-6),
    ):
        _, tangent = jax.jvp(rain, (qc,), (direction,))
        assert float(tangent[0]) == 0.0


def test_threshold_exact_zero_continuity_and_positivity():
    normalizer = normalization()
    raw = jnp.asarray(((-1000.0, -1000.0), (2.0, -3.0), (4.0, 1000.0)))
    h = jnp.full(3, 700.0)
    qc = h * jnp.asarray(
        (Q_PRECIP - DELTA_Q_SCALE, Q_PRECIP, Q_PRECIP + DELTA_Q_SCALE)
    )
    _, rain = bplus_physical_rates(raw, h, qc, Q_PRECIP, normalizer)
    assert float(rain[0]) == 0.0
    assert float(rain[1]) == 0.0
    assert float(rain[2]) >= 0.0
    small = []
    for factor in (1.0e-2, 1.0e-4, 1.0e-6):
        _, value = bplus_physical_rates(
            raw[1:2], h[:1],
            h[:1] * (Q_PRECIP + factor * DELTA_Q_SCALE),
            Q_PRECIP, normalizer,
        )
        small.append(float(value[0]))
    assert small[2] < small[1] < small[0]
    assert small[2] > 0.0
    raw_rain_sweep = jnp.asarray(
        ((0.0, -1000.0), (0.0, -3.0), (0.0, 0.0), (0.0, 1000.0)),
        dtype=jnp.float64,
    )
    _, rain_sweep = bplus_physical_rates(
        raw_rain_sweep,
        jnp.full(4, 700.0, dtype=jnp.float64),
        jnp.full(
            4, 700.0 * (Q_PRECIP + DELTA_Q_SCALE), dtype=jnp.float64
        ),
        Q_PRECIP,
        normalizer,
    )
    assert np.all(np.asarray(rain_sweep) >= 0.0)


def test_rain_state_derivative_is_zero_below_and_correct_above_threshold():
    normalizer = normalization()
    raw = jnp.asarray(((0.25, -0.4),), dtype=jnp.float64)
    h = jnp.asarray((700.0,), dtype=jnp.float64)
    direction = jnp.asarray((1.0,), dtype=jnp.float64)
    rain = lambda qc: bplus_physical_rates(
        raw, h, qc, Q_PRECIP, normalizer
    )[1]
    below = h * (Q_PRECIP - DELTA_Q_SCALE)
    above = h * (Q_PRECIP + DELTA_Q_SCALE)
    _, tangent_below = jax.jvp(rain, (below,), (direction,))
    _, tangent_above = jax.jvp(rain, (above,), (direction,))
    expected = (
        normalizer.sigma_r_active
        * jax.nn.softplus(raw[0, 1])
        / jnp.log(2.0)
        / DELTA_Q_SCALE
        / h[0]
    )
    assert float(tangent_below[0]) == 0.0
    assert np.isclose(float(tangent_above[0]), float(expected), rtol=2e-14)


def test_bplus_preserves_b_a_head_and_exact_source_structure():
    normalizer = normalization()
    parameters = initial_parameters("B")
    state, fields, moist = local_inputs(DELTA_Q_SCALE)
    b = RainActiveNeuralMoistPhysics(
        "B", parameters, normalizer, use_jit=False
    ).combined_kernel(state, fields, moist)
    bplus = RainActiveNeuralMoistPhysics(
        "BPLUS", parameters, normalizer, use_jit=False
    ).combined_kernel(state, fields, moist)
    assert np.array_equal(np.asarray(b["rates"]["A"]), np.asarray(bplus["rates"]["A"]))
    assert np.all(np.asarray(bplus["rates"]["R"]) >= 0.0)
    audit = source_invariant_diagnostics(bplus["source"], 98.0616)
    assert audit["water_maximum_absolute"] <= 1.0e-18
    assert audit["S_minus_beta2_Qv_maximum_absolute"] <= 1.0e-16


def test_btp_preserves_b_a_head_and_exact_source_structure():
    normalizer = btp_normalization()
    parameters = initial_parameters("B")
    state, fields, moist = local_inputs(DELTA_Q_SCALE)
    b = RainActiveNeuralMoistPhysics(
        "B", parameters, normalizer, use_jit=False
    ).combined_kernel(state, fields, moist)
    btp = RainActiveNeuralMoistPhysics(
        "BTP", parameters, normalizer, use_jit=False
    ).combined_kernel(state, fields, moist)
    assert np.array_equal(np.asarray(b["rates"]["A"]), np.asarray(btp["rates"]["A"]))
    assert np.all(np.asarray(btp["rates"]["R"]) >= 0.0)
    audit = source_invariant_diagnostics(btp["source"], 98.0616)
    assert audit["water_maximum_absolute"] <= 1.0e-18
    assert audit["S_minus_beta2_Qv_maximum_absolute"] <= 1.0e-16


def test_btpl_canonical_identifier_matches_legacy_bplus_exactly():
    normalizer = normalization()
    parameters = initial_parameters("BPLUS")
    state, fields, moist = local_inputs(DELTA_Q_SCALE)
    legacy = RainActiveNeuralMoistPhysics(
        "BPLUS", parameters, normalizer, use_jit=False
    ).combined_kernel(state, fields, moist)
    canonical = RainActiveNeuralMoistPhysics(
        "BTPL", parameters, normalizer, use_jit=False
    ).combined_kernel(state, fields, moist)
    for group in ("rates", "source"):
        for name in legacy[group]:
            assert np.array_equal(
                np.asarray(legacy[group][name]), np.asarray(canonical[group][name])
            )


def test_frozen_b_output_map_remains_numerically_unchanged():
    normalizer = normalization()
    parameters = initial_parameters("B")
    state, fields, moist = local_inputs(DELTA_Q_SCALE)
    provider = RainActiveNeuralMoistPhysics(
        "B", parameters, normalizer, use_jit=False
    )
    actual = provider.combined_kernel(state, fields, moist)
    features = jnp.stack(
        (state["h"], state["S"], state["Qv"], state["Qc"], fields["B"]),
        axis=-1,
    )
    physical = build_model("B")(
        parameters, normalizer.normalize_features(features)
    ) * jnp.asarray((normalizer.sigma_a, normalizer.sigma_r_active))
    a, rain = physical[..., 0], physical[..., 1]
    h = state["h"]
    expected = {
        "S": h * 98.0616 * a,
        "Qv": h * a,
        "Qc": -h * (a + rain),
        "Qr": h * rain,
    }
    for name in expected:
        assert np.array_equal(
            np.asarray(actual["source"][name]), np.asarray(expected[name])
        )


def test_parameter_tangent_adjoint_and_directional_derivative_above_kink():
    normalizer = normalization()
    parameters = initial_parameters("BPLUS")
    provider = RainActiveNeuralMoistPhysics(
        "BPLUS", parameters, normalizer, use_jit=False
    )
    state, fields, moist = local_inputs(2.0 * DELTA_Q_SCALE)
    direction = jax.tree_util.tree_map(
        lambda value: jnp.full_like(value, 1.0e-4), parameters
    )
    source, tangent = provider.parameter_jvp(
        state, direction, fields, moist
    )
    covector = {
        name: jnp.full_like(value, 0.2 + index)
        for index, (name, value) in enumerate(source.items())
    }
    adjoint = provider.parameter_vjp(state, covector, fields, moist)
    left = sum(jnp.vdot(tangent[name], covector[name]) for name in tangent)
    right = sum(
        jnp.vdot(direction["layers"][index][leaf], adjoint["layers"][index][leaf])
        for index in range(3) for leaf in ("weight", "bias")
    )
    assert float(abs(left - right)) <= 5.0e-13 * max(
        abs(float(left)), abs(float(right)), 1.0
    )
    epsilon = 1.0e-3
    plus = jax.tree_util.tree_map(
        lambda base, delta: base + epsilon * delta, parameters, direction
    )
    minus = jax.tree_util.tree_map(
        lambda base, delta: base - epsilon * delta, parameters, direction
    )
    scalar = lambda active: sum(
        jnp.vdot(
            provider.combined_with_parameters(
                state, fields, moist, active
            )["source"][name],
            covector[name],
        )
        for name in covector
    )
    finite_difference = (scalar(plus) - scalar(minus)) / (2.0 * epsilon)
    assert np.isclose(
        float(left), float(finite_difference), rtol=2.0e-8, atol=1.0e-13
    )
