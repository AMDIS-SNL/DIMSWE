"""Opt-in, model-agnostic learned-physics experimentation interfaces.

The package initializer deliberately imports no Firedrake or PyROL objects.
Production-solver adapters live in separate opt-in modules.
"""

from .closure import (
    FeatureMap,
    LearnedPhysicsModel,
    OutputMap,
    ParameterizedModel,
)
from .experiment import (
    BENCHMARK_CONTRACTS,
    BenchmarkContract,
    ExperimentDefinition,
    ExperimentResult,
    LearnedPhysicsRole,
    TruthDataset,
    TruthMetadata,
    load_truth_dataset,
    save_experiment_result,
    save_truth_dataset,
    summarize_experiment_result,
)
from .objectives import (
    DiscreteOfflineExample,
    LocalOfflineExample,
    LossAccumulation,
    RolloutExample,
    TrainingMode,
    TruthResetWindow,
    apriori_offline,
    discrete_offline,
    objective_for_mode,
    rollout,
    squared_l2_loss,
    truth_reset,
)
from .parameters import (
    Float64TreeError,
    tree_all_finite,
    tree_axpy,
    tree_copy,
    tree_dot,
    tree_norm,
    tree_zeros,
    validate_float64_tree,
)

__all__ = (
    "BENCHMARK_CONTRACTS",
    "BenchmarkContract",
    "DiscreteOfflineExample",
    "ExperimentDefinition",
    "ExperimentResult",
    "FeatureMap",
    "Float64TreeError",
    "LearnedPhysicsModel",
    "LearnedPhysicsRole",
    "LocalOfflineExample",
    "LossAccumulation",
    "OutputMap",
    "ParameterizedModel",
    "RolloutExample",
    "TrainingMode",
    "TruthDataset",
    "TruthMetadata",
    "TruthResetWindow",
    "apriori_offline",
    "discrete_offline",
    "load_truth_dataset",
    "objective_for_mode",
    "rollout",
    "save_experiment_result",
    "save_truth_dataset",
    "summarize_experiment_result",
    "squared_l2_loss",
    "tree_all_finite",
    "tree_axpy",
    "tree_copy",
    "tree_dot",
    "tree_norm",
    "tree_zeros",
    "truth_reset",
    "validate_float64_tree",
)
