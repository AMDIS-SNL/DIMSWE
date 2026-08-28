"""Cheap contracts for matched Test-2A Method-1/Method-2 long fits."""

import json

import jax
import numpy as np

from dimswe.test2a_discrete_training import load_discrete_training_configuration
from dimswe.test2a_embedded_moist import parameter_pytree_sha256
from dimswe.test2a_fair_longfit import (
    CANONICAL_SEED_SHA256,
    load_fair_operator_configuration,
)
from dimswe.test2a_operator import (
    initialize_mlp,
    load_selected_configuration,
    mlp_configuration_from_record,
)


OPERATOR = "dimswe/configs/test2a_fair_operator_200k.json"
DISCRETE = "dimswe/configs/test2a_fair_discrete_200k.json"


def test_matched_longfit_optimizer_and_checkpoint_contracts():
    operator = load_fair_operator_configuration(OPERATOR)
    discrete = load_discrete_training_configuration(DISCRETE)
    assert operator["optimizer"] == discrete["optimizer"]
    assert operator["optimizer"] == {
        "library": "PyROL/ROL",
        "method": "line-search L-BFGS",
        "maximum_secant_storage": 20,
        "accepted_iteration_limit": 200000,
        "gradient_tolerance": 1.0e-8,
        "step_tolerance": 1.0e-12,
        "exact_gradients": True,
        "production_HVP": False,
    }
    assert operator["checkpoint_accepted_iterations"] == discrete[
        "checkpoint_accepted_iterations"
    ] == [25000, 50000, 75000, 100000, 150000, 200000]


def test_fair_methods_share_exact_seed0_initial_parameters():
    selected = load_selected_configuration(
        "dimswe/configs/test2a_selected_operator.json"
    )
    parameters = initialize_mlp(mlp_configuration_from_record(selected["model"]))
    assert parameter_pytree_sha256(parameters) == CANONICAL_SEED_SHA256
    operator = json.load(open(OPERATOR, encoding="utf-8"))
    discrete = json.load(open(DISCRETE, encoding="utf-8"))
    assert operator["initialization"]["parameter_pytree_sha256"] == CANONICAL_SEED_SHA256
    assert discrete["initialization"]["parameter_pytree_sha256"] == CANONICAL_SEED_SHA256


def test_parameter_only_restart_discloses_missing_secant_history():
    for path in (OPERATOR, DISCRETE):
        record = json.load(open(path, encoding="utf-8"))
        assert "secant history is process-local" in record["resume_contract"]
        assert "not restored" in record["resume_contract"]

