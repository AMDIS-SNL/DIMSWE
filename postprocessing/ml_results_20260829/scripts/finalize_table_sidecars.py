#!/usr/bin/env python3
"""Create exact machine-readable table sidecars and the Test-2A supplement.

CSV/JSON/Markdown parsing and formatting only.  No model, truth, prefix, or
optimizer code is imported or executed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "tables"
AUDIT = ROOT.parent / "ml_results_audit_20260829"
MODEL_ORDER = ["M1-X", "M1-Y", "M2-X-independent", "warm M2-X", "H1", "H2", "H5"]
T2A_ORDER = ["M1-X", "M2-X-independent", "warm M2-X", "H1", "H2", "H5"]


def tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def tex_table(headers: list[str], rows: list[list[object]]) -> str:
    columns = "l" * len(headers)
    lines = [r"\begin{tabular}{" + columns + "}", r"\hline"]
    lines.append(" & ".join(tex_escape(x) for x in headers) + r" \\")
    lines.append(r"\hline")
    for row in rows:
        lines.append(" & ".join(tex_escape(x) for x in row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(v: object) -> str:
        return str(v).replace("|", r"\|").replace("\n", " ")
    lines = ["| " + " | ".join(cell(x) for x in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(cell(x) for x in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def fmt(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == 0:
        return "0"
    if abs(number) < 1e-2 or abs(number) >= 1e4:
        return f"{number:.3e}"
    return f"{number:.3g}"


def table1() -> None:
    rows = [
        {
            "physical_case": "Test 2A",
            "objective_family": "M1-X / M2-X",
            "mesh_dt": "16x16 elements / 100 s",
            "stored_states": 161,
            "training_indices_and_count": "0--80; 81 states; 331,776 spatial samples",
            "evaluation_and_target_state": "X; local law or fixed deployed moist child",
            "feature_order": "(h,S,Qv,Qc,B)",
            "normalization_output_scaling_weighting": "input statistics on X 0--80; objective-specific frozen output scale; packed/deployed mass metric",
            "evaluation_terminology_and_limit": "TRAINING SUPPORT only",
        },
        {
            "physical_case": "Test 2A",
            "objective_family": "H1",
            "mesh_dt": "16x16 elements / 100 s",
            "stored_states": 161,
            "training_indices_and_count": "starts 0--79; 80 one-step windows; 327,680 spatial starts",
            "evaluation_and_target_state": "Y=P(X); target X_(n+1)*",
            "feature_order": "(h,S,Qv,Qc,B)",
            "normalization_output_scaling_weighting": "same frozen X scaling; deployed mixed-state mass metric",
            "evaluation_terminology_and_limit": "TRAINING-SUPPORT WINDOWS",
        },
        {
            "physical_case": "Test 2A",
            "objective_family": "H2 / H5",
            "mesh_dt": "16x16 elements / 100 s",
            "stored_states": 161,
            "training_indices_and_count": "40 H2 / 16 H5 recursive windows within 0--80",
            "evaluation_and_target_state": "first Y=P(X), then model-recursive states; truth boundary targets",
            "feature_order": "(h,S,Qv,Qc,B)",
            "normalization_output_scaling_weighting": "same frozen scaling; accumulated mixed-state mass metric",
            "evaluation_terminology_and_limit": "no held-out/test set; 81--160 UNUSED BY RECORDED LEARNING",
        },
        {
            "physical_case": "Test 2B",
            "objective_family": "M1-X / M2-X",
            "mesh_dt": "64x64 elements / 100 s",
            "stored_states": 161,
            "training_indices_and_count": "0--80; 81 states; 5,308,416 spatial samples",
            "evaluation_and_target_state": "X; local law or fixed deployed moist child",
            "feature_order": "(h,S,Qv,Qc,B)",
            "normalization_output_scaling_weighting": "input statistics/output scales on X 0--80; carrier-mass weighting",
            "evaluation_terminology_and_limit": "TRAINING TRUTH SUPPORT",
        },
        {
            "physical_case": "Test 2B",
            "objective_family": "M1-Y / H1",
            "mesh_dt": "64x64 elements / 100 s",
            "stored_states": 161,
            "training_indices_and_count": "M1-Y 0--80: 5,308,416 samples; H1 starts 0--79: 80 windows",
            "evaluation_and_target_state": "Y=P(X); local law at Y or target X_(n+1)*",
            "feature_order": "(h,S,Qv,Qc,B)",
            "normalization_output_scaling_weighting": "historical X-fitted scaling retained; carrier/mixed-state weighting",
            "evaluation_terminology_and_limit": "TRAINING TRUTH SUPPORT",
        },
        {
            "physical_case": "Test 2B",
            "objective_family": "H2 / H5",
            "mesh_dt": "64x64 elements / 100 s",
            "stored_states": 161,
            "training_indices_and_count": "40 H2 / 16 H5 recursive windows within 0--80",
            "evaluation_and_target_state": "first Y=P(X), then model-recursive states; truth boundary targets",
            "feature_order": "(h,S,Qv,Qc,B)",
            "normalization_output_scaling_weighting": "same frozen X scaling; accumulated mixed-state metric",
            "evaluation_terminology_and_limit": "81--160 HELD-OUT TRUTH SUPPORT; post-hoc only; temporally adjacent",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "table1_data_supports.csv", index=False)
    headers = [
        "Case", "family", "mesh / dt", "stored", "training indices / fitted count",
        "evaluation / target", "features", "scaling / weighting", "terminology / limitation",
    ]
    display = [[
        r["physical_case"], r["objective_family"], r["mesh_dt"], r["stored_states"],
        r["training_indices_and_count"], r["evaluation_and_target_state"], r["feature_order"],
        r["normalization_output_scaling_weighting"], r["evaluation_terminology_and_limit"],
    ] for r in rows]
    note = (
        "Test 2A has no historically defined held-out/test set. Test 2B states 81--160 "
        "are temporally adjacent held-out truth support, and all held-out quantities in "
        "this package are post-hoc rather than stopping/model-selection signals."
    )
    (TABLES / "table1_data_supports.md").write_text(
        "# Table 1. Data supports and evaluation protocol\n\n" + md_table(headers, display) + "\n" + note + "\n"
    )
    (TABLES / "table1_data_supports.tex").write_text(tex_table(headers, display) + "% " + note + "\n")


def table2() -> None:
    matrix = pd.read_csv(DATA / "completed_objective_matrix.csv")
    inventory = pd.read_csv(AUDIT / "ML_RUN_INVENTORY.csv")
    selected = inventory[inventory["run_id"].isin(matrix["run_id"])].copy()
    if len(selected) != 33:
        raise RuntimeError(f"expected 33 main inventory rows, got {len(selected)}")
    selected.to_csv(TABLES / "table2_training_contracts_provenance.csv", index=False)
    selected["model_label"] = selected["run_id"].map(matrix.set_index("run_id")["model_label"])
    selected["model_order"] = selected["model_label"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    selected = selected.sort_values(["physical_case", "representation", "model_order"])
    display = pd.DataFrame(
        {
            "physical_case": selected["physical_case"],
            "representation": selected["representation"],
            "model_label": selected["model_label"],
            "evaluation_state": selected["training_evaluation_state"],
            "target_state": selected["target_state"],
            "architecture_parameter_count": selected["architecture"] + " (" + selected["parameter_count"].astype(str) + ")",
            "initialization": selected["initialization_kind"] + selected["initialization_source"].fillna("").map(lambda x: f" from {x}" if x else ""),
            "iteration_budget": selected["iteration_budget"],
            "accepted_iterations": selected["accepted_iterations_this_run"],
            "optimizer": selected["optimizer"].fillna("PyROL/ROL line-search L-BFGS") + "; m=" + selected["lbfgs_memory"].fillna(20).astype(str),
            "stop_reason": selected["stopping_reason"],
        }
    )
    display.to_csv(TABLES / "table2_training_contracts.csv", index=False)


def table3() -> None:
    data = pd.read_csv(DATA / "final_direct_metrics.csv")
    t2b = data[
        data["physical_case"].eq("Test 2B")
        & data["evaluation_state"].eq("X")
        & data["support"].isin(["TRAINING TRUTH SUPPORT", "HELD-OUT TRUTH SUPPORT"])
    ].copy()
    keep = (
        ((t2b["representation"].eq("A")) & t2b["quantity"].eq("A") & t2b["metric"].isin(["physical_RMS_error", "relative_RMS_error", "signed_mass_weighted_bias", "correlation"]))
        | ((t2b["representation"].eq("B")) & (
            (t2b["quantity"].eq("A") & t2b["metric"].eq("relative_RMS_error"))
            | (t2b["quantity"].isin(["R_all", "R_truth_active"]) & t2b["metric"].eq("relative_RMS_error"))
            | (t2b["quantity"].eq("R_activation") & t2b["metric"].isin(["false_positive_rate_given_truth_inactive", "false_negative_rate_given_truth_active"]))
        ))
        | ((t2b["representation"].eq("C")) & (
            (t2b["quantity"].str.startswith("source_components.") & t2b["metric"].eq("physical_RMS_error"))
            | (t2b["quantity"].isin(["effective_A", "effective_R"]) & t2b["metric"].eq("relative_RMS_error"))
            | (t2b["quantity"].eq("source_structure") & t2b["metric"].eq("normalized_off_manifold_RMS"))
        ))
    )
    t2b[keep].to_csv(TABLES / "table3_final_direct_accuracy.csv", index=False)

    t2a = data[data["physical_case"].eq("Test 2A")].copy()
    t2a.to_csv(TABLES / "tableS3_test2a_training_support_accuracy.csv", index=False)
    lines_md = ["# Table S3. Test 2A training-support direct accuracy", ""]
    lines_tex: list[str] = []
    a = t2a[t2a["representation"].eq("A")]
    alook = a.set_index(["model_label", "quantity", "metric"])["value"]
    ahead = ["Model", "A RMS", "A rel. RMS", "A max abs.", "A corr.", "sign accuracy"]
    arows = []
    for model in T2A_ORDER:
        get = lambda metric: alook.get((model, "A", metric), None)
        arows.append([model, fmt(get("physical_RMS_error")), fmt(get("relative_RMS_error")), fmt(get("maximum_absolute_error")), fmt(get("correlation")), fmt(get("sign_accuracy"))])
    lines_md += ["## Representation A", "", md_table(ahead, arows)]
    lines_tex += ["% Representation A", tex_table(ahead, arows)]

    c = t2a[t2a["representation"].eq("C")]
    clook = c.set_index(["model_label", "quantity", "metric"])["value"]
    chead = ["Model", "S RMS", "Qv RMS", "Qc RMS", "Qr RMS", "off-manifold", "water defect", "thermo. defect"]
    crows = []
    for model in T2A_ORDER:
        get = lambda quantity, metric: clook.get((model, quantity, metric), None)
        crows.append([
            model,
            *[fmt(get(f"source_components.{q}", "physical_RMS_error")) for q in ["S", "Qv", "Qc", "Qr"]],
            fmt(get("source_structure", "normalized_off_manifold_RMS")),
            fmt(get("source_structure", "water_source_defect_RMS")),
            fmt(get("source_structure", "S_minus_beta2_Qv_defect_RMS")),
        ])
    lines_md += ["## Representation C", "", md_table(chead, crows)]
    lines_md += [
        "All quantities are on Test 2A TRAINING SUPPORT 0--80. No historical held-out/test protocol exists, and states 81--160 were not evaluated for this table."
    ]
    lines_tex += ["% Representation C", tex_table(chead, crows)]
    (TABLES / "tableS3_test2a_training_support_accuracy.md").write_text("\n".join(lines_md) + "\n")
    (TABLES / "tableS3_test2a_training_support_accuracy.tex").write_text("\n".join(lines_tex))


def table4_and_5() -> None:
    pd.read_csv(DATA / "completed_objective_matrix.csv").to_csv(TABLES / "table4_objective_matrix.csv", index=False)
    pd.read_csv(DATA / "rain_event_diagnostics.csv").to_csv(TABLES / "table5_rain_events.csv", index=False)


def history_coverage() -> None:
    direct = pd.read_csv(DATA / "checkpoint_direct_histories.csv")
    objectives = pd.read_csv(DATA / "checkpoint_training_objectives.csv")
    rows = []
    for rep in "ABC":
        for model in MODEL_ORDER:
            d = direct[direct["representation"].eq(rep) & direct["model_label"].eq(model)]
            o = objectives[
                objectives["physical_case"].eq("Test 2B")
                & objectives["representation"].eq(rep)
                & objectives["model_label"].eq(model)
            ]
            x_train_points = d[d["evaluation_state"].eq("X") & d["support"].str.startswith("TRAINING")]["checkpoint_iteration"].nunique()
            x_heldout_points = d[d["evaluation_state"].eq("X") & d["support"].str.startswith("HELD-OUT")]["checkpoint_iteration"].nunique()
            y_train_points = d[d["evaluation_state"].eq("Y") & d["support"].str.startswith("TRAINING")]["checkpoint_iteration"].nunique()
            y_heldout_points = d[d["evaluation_state"].eq("Y") & d["support"].str.startswith("HELD-OUT")]["checkpoint_iteration"].nunique()
            rows.append(
                {
                    "physical_case": "Test 2B",
                    "representation": rep,
                    "model_label": model,
                    "saved_checkpoint_count": d["checkpoint_iteration"].nunique(),
                    "direct_X_training_point_count": x_train_points,
                    "direct_X_training_availability": "CHECKPOINT_HISTORY" if x_train_points > 1 else "FINAL_ONLY",
                    "direct_X_heldout_point_count": x_heldout_points,
                    "direct_X_heldout_availability": "CHECKPOINT_HISTORY" if x_heldout_points > 1 else "FINAL_ONLY",
                    "direct_Y_training_point_count": y_train_points,
                    "direct_Y_training_availability": (
                        "CHECKPOINT_HISTORY" if y_train_points > 1 else
                        "FINAL_ONLY_DIAGNOSTIC" if y_train_points == 1 else "NOT_AVAILABLE"
                    ),
                    "direct_Y_heldout_point_count": y_heldout_points,
                    "direct_Y_heldout_availability": (
                        "CHECKPOINT_HISTORY" if y_heldout_points > 1 else
                        "FINAL_ONLY_DIAGNOSTIC" if y_heldout_points == 1 else "NOT_AVAILABLE"
                    ),
                    "fitted_objective_point_count": o["checkpoint_iteration"].nunique(),
                    "fitted_objective_history_status": (
                        "ENDPOINTS_ONLY_RECURSIVE_HISTORY_DEFERRED"
                        if model in ["H2", "H5"]
                        else "FIXED_ARRAY_RECONSTRUCTED"
                        if model in ["M1-X", "M1-Y"]
                        else "FIXED_MAP_RECONSTRUCTED"
                    ),
                    "history_kinds": "; ".join(sorted(o["history_kind"].dropna().unique())),
                }
            )
    pd.DataFrame(rows).to_csv(DATA / "checkpoint_history_coverage.csv", index=False)
    objectives[
        objectives["physical_case"].eq("Test 2B") & objectives["model_label"].isin(["H2", "H5"])
    ].to_csv(DATA / "recursive_objective_endpoints.csv", index=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite-accepted-assets",
        action="store_true",
        help="allow this script to recreate accepted table/data sidecars",
    )
    return parser


def main(argv=None) -> None:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if not arguments.overwrite_accepted_assets:
        parser.error(
            "refusing to write accepted table/data sidecars without "
            "--overwrite-accepted-assets"
        )
    table1()
    table2()
    table3()
    table4_and_5()
    history_coverage()
    print(
        {
            "status": "complete",
            "table_csv_count": len(list(TABLES.glob("*.csv"))),
            "history_coverage_rows": len(pd.read_csv(DATA / "checkpoint_history_coverage.csv")),
            "recursive_endpoint_rows": len(pd.read_csv(DATA / "recursive_objective_endpoints.csv")),
        }
    )


if __name__ == "__main__":
    main()
