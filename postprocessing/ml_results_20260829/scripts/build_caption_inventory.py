#!/usr/bin/env python3
"""Consolidate table/figure captions and generate a figure inventory."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAPTIONS = ROOT / "captions" / "ML_RESULTS_CAPTION_DRAFTS.md"
FIGURES = ROOT / "figures"

TABLE_CAPTIONS = [
    (
        "Table 1 — Data and evaluation protocol",
        "Data and evaluation protocol for Test 2A and Test 2B. Test 2A states 0--80 "
        "are training states; no evaluation or test set was defined, and states 81--160 "
        "were unused by the recorded learning studies. For Test 2B, states 0--80 are "
        "training states and states 81--160 are temporally adjacent evaluation states. "
        "The evaluation states did not influence stopping or model selection.",
    ),
    (
        "Table 2 — Main Test 2B training-run contracts",
        "Frozen contracts for M1-Y, H1/M2-Y, H2, and H5 in Representations A, B, "
        "and C. All networks use the production feature order (h,S,Qv,Qc,B), float64, "
        "and seed 0. H1, H2, and H5 are sequential continuations with unequal budgets, "
        "so objective and optimization history change together.",
    ),
    (
        "Table 3 — Final local moist-physics accuracy at Y*",
        "Final frozen-network errors at truth-derived pre-moist states Y*=P(X*) for "
        "Test 2B training states 0--80 and evaluation states 81--160. The evaluation "
        "states did not influence optimization. A, R, and source-component quantities "
        "remain separate because the representations learn different targets.",
    ),
    (
        "Table 5 — Main rain-event and water-partition diagnostics",
        "Test 2B rain-event and water-partition diagnostics for truth and the deployed "
        "M1-Y, H1, H2, and H5 models. "
        "Representation A uses analytical R on the model-generated state, Representation "
        "B learns R directly, and Representation C is reported through its effective "
        "Qr-source rain diagnostic. The A/B source identities are imposed by construction; "
        "their residuals are not evidence that conservation was learned.",
    ),
    (
        "Supplementary table — Local moist-physics accuracy at deployed Yhat",
        "Stored local-law errors on model-generated pre-moist states Yhat=P(Xhat) for "
        "M1-Y, H1, H2, and H5. Coverage is uniform across Representations A, B, and C, "
        "but no single cross-representation scalar is formed because the learned targets differ.",
    ),
    (
        "Supplementary tables — Complete campaign results",
        "The complete training contracts, X-based direct errors, cross-objective matrix, "
        "rain diagnostics, and Test 2A training-state results are retained unchanged in "
        "the supplementary table directory.",
    ),
]

FIGURE_TITLES = {
    "ML1_main_optimization": "ML-1 — Optimization of M1-Y, H1/M2-Y, H2, and H5",
    "ML2_main_training_evaluation": "ML-2 — Training and evaluation of the fitted objectives",
    "ML3_main_callsite_physical_accuracy": "ML-3 — Local moist-physics accuracy at Y*",
    "ML4_main_deployed_physical_diagnostics": "ML-4 — Deployed physical diagnostics",
    "ML5A_main_global_trajectories": "ML-5A — Global trajectories, Representation A",
    "ML5B_main_global_trajectories": "ML-5B — Global trajectories, Representation B",
    "ML5C_main_global_trajectories": "ML-5C — Global trajectories, Representation C",
    "ML1_optimization_progress_test2b": "Supplementary — Complete Test 2B optimization histories",
    "ML2_objective_training_evaluation_history_test2b": "Supplementary — Complete nonrecursive objective histories",
    "ML3_m1x_m1y_cross_state_final": "Supplementary — Final M1-X/M1-Y cross-state comparison",
    "ML4_frozen_model_objective_matrix_test2b": "Supplementary — Test 2B objective matrix",
    "ML5_deployed_physical_diagnostics_test2b": "Supplementary — Complete deployed physical diagnostics",
    "ML6_global_trajectories_representation_A": "Supplementary — Prior Representation A trajectory subset",
    "ML6_global_trajectories_representation_B": "Supplementary — Prior Representation B trajectory subset",
    "ML6_global_trajectories_representation_C": "Supplementary — Prior Representation C trajectory subset",
    "ML1_test2a_recorded_optimization_progress": "Supplementary ML-1 — Test 2A training objectives",
    "ML2_posthoc_direct_history_test2b_all_models": "Supplementary ML-2 — Direct error for all Test 2B models",
    "ML2_common_x_direct_history_test2b": "Supplementary ML-2 — Common-X direct physical-law histories",
    "ML3_m1x_m1y_cross_state_checkpoint_history": "Supplementary ML-3 — M1-X/M1-Y error histories",
    "ML3_supplement_callsite_source_components": "Supplementary ML-3 — Representation C source-component accuracy at Y*",
    "ML4_frozen_model_objective_matrix_test2a": "Supplementary ML-4 — Test 2A objective matrix",
    "ML5_representation_c_source_structure": "Supplementary ML-5 — Representation C source identities",
    "ML6_global_trajectories_representation_A_all_models": "Supplementary ML-6A — All Representation A trajectories",
    "ML6_global_trajectories_representation_B_all_models": "Supplementary ML-6B — All Representation B trajectories",
    "ML6_global_trajectories_representation_C_all_models": "Supplementary ML-6C — All Representation C trajectories",
    "ML_test2a_training_interval_deployment": "Supplementary Test 2A deployment diagnostics",
}


def main() -> None:
    CAPTIONS.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Machine-Learning Results caption drafts",
        "",
        "These captions use training/evaluation terminology while retaining the methodological qualifications needed to interpret each quantity.",
        "",
        "## Tables",
        "",
    ]
    for title, caption in TABLE_CAPTIONS:
        lines.extend([f"### {title}", "", caption, ""])

    inventory: list[dict[str, object]] = []
    for class_name in ["main", "supplement"]:
        lines.extend([f"## {class_name.title()} figures", ""])
        for sidecar in sorted((FIGURES / class_name).glob("*.json")):
            payload = json.loads(sidecar.read_text())
            figure_id = payload["figure_id"]
            lines.extend(
                [
                    f"### {FIGURE_TITLES.get(figure_id, figure_id)}",
                    "",
                    payload["draft_caption"],
                    "",
                ]
            )
            inventory.append(
                {
                    "figure_id": figure_id,
                    "class": class_name,
                    "scientific_question": payload["scientific_question"],
                    "quantity_kind": payload["quantity_kind"],
                    "support_classification": payload["support_classification"],
                    "model_labels": "; ".join(payload.get("model_labels", [])),
                    "representations": "; ".join(payload.get("representations", [])),
                    "pdf_path": payload["files"]["pdf"]["path"],
                    "pdf_sha256": payload["files"]["pdf"]["sha256"],
                    "png_path": payload["files"]["png_300dpi"]["path"],
                    "png_sha256": payload["files"]["png_300dpi"]["sha256"],
                    "csv_path": payload["files"]["csv"]["path"],
                    "csv_sha256": payload["files"]["csv"]["sha256"],
                    "json_sidecar_path": str(sidecar),
                    "draft_caption": payload["draft_caption"],
                }
            )
    CAPTIONS.write_text("\n".join(lines) + "\n")
    inventory_path = FIGURES / "FIGURE_INVENTORY.csv"
    with inventory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    print({"status": "complete", "figure_count": len(inventory), "caption_file": str(CAPTIONS)})


if __name__ == "__main__":
    main()
