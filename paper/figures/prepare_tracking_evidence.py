#!/usr/bin/env python3
"""Validate archived tracking summaries and regenerate the paper's tracking table.

Uses only Python's standard library and the committed data/tracking snapshots;
no network, simulation, model inference, or scratch results are required.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
from statistics import mean


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/tracking"
COMMIT = "b5549db7a34750949300f5de6c7321d18e889451"
CONDITIONS = ("CR", "DA", "LS")
MODELS = {
    "baseline": "DOM",
    "velocity": "DOM + velocity",
    "coordinate": "KR-Coordinate",
    "coordinate_velocity": "KR-Coordinate + velocity",
}
SOURCES = {
    "tracking": "results/tracking_table1_partial_20261009/summary.json",
    "tracking_csv": "results/tracking_table1_partial_20261009/tracking_CR_DA_LS.csv",
    "speed": "results/tracking_rescored_3cm_20261008/summary.json",
    "speed_csv": "results/tracking_rescored_3cm_20261008/per_episode.csv",
    "speed_rmse": "results/coordinate_velocity_plot_20261007/summary.json",
}


def close(actual, expected, context):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (
        context, actual, expected
    )


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_evidence():
    provenance = json.loads((DATA / "source_provenance.json").read_text())
    assert provenance["commit"] == COMMIT
    source_hashes = {row["path"]: row["sha256"] for row in provenance["files"]}
    checked = []
    for path in SOURCES.values():
        digest = hashlib.sha256((DATA / path).read_bytes()).hexdigest()
        assert digest == source_hashes[path], f"Source hash mismatch: {path}"
        checked.append({"source_path": path, "file": f"data/tracking/{path}",
                        "sha256": digest})

    tracking = json.loads((DATA / SOURCES["tracking"]).read_text())
    tracking_csv = {row["Model"]: row for row in read_csv(DATA / SOURCES["tracking_csv"])}
    speed = json.loads((DATA / SOURCES["speed"]).read_text())
    speed_episodes = read_csv(DATA / SOURCES["speed_csv"])
    speed_rmse = json.loads((DATA / SOURCES["speed_rmse"]).read_text())
    assert tracking["scoring_version"] == speed["scoring_version"] == 4
    assert set(tracking["models"]) == set(MODELS)
    assert len(tracking_csv) == len(MODELS)
    assert len(speed_episodes) == speed["recorded_rollouts"] == 120
    assert set(row["model"] for row in speed_episodes) == set(MODELS)

    rows = []
    total_episodes = 0
    for model_id, label in MODELS.items():
        source = tracking["models"][model_id]
        csv_row = tracking_csv[source["label"]]
        row = {"id": model_id, "label": label, "conditions": {}}
        for condition in CONDITIONS:
            cell = source["dimensions"][condition]
            assert cell["episodes"] == 200
            assert cell["completion_percent"] == 100
            coverage = cell["success_percent"]
            rmse = cell["mean_hover_rmse_mm"]
            close(coverage, float(csv_row[f"{condition}_success_percent"]),
                  (model_id, condition, "coverage"))
            close(rmse, float(csv_row[f"{condition}_RMSE_mm"]),
                  (model_id, condition, "RMSE"))
            row["conditions"][condition] = {
                "coverage_percent": coverage, "rmse_mm": rmse,
                "episodes": cell["episodes"], "completed_count": cell["episodes"],
            }
            total_episodes += cell["episodes"]
        row["mean"] = {
            "coverage_percent": mean(c["coverage_percent"] for c in row["conditions"].values()),
            "rmse_mm": mean(c["rmse_mm"] for c in row["conditions"].values()),
        }
        close(row["mean"]["coverage_percent"], source["three_dimension_mean_success_percent"],
              (model_id, "summary mean"))
        close(row["mean"]["coverage_percent"], float(csv_row["Three_dimension_mean_percent"]),
              (model_id, "CSV mean"))
        rows.append(row)
    assert total_episodes == tracking["rollouts"] == 2400

    sweep = []
    reference_episodes = None
    for model_id, label in MODELS.items():
        row = {"id": model_id, "label": label, "speed_multipliers": [1.0, 1.5, 2.0],
               "coverage_percent": [], "completed_count": [], "requested_count": [],
               "rmse_completed_mm": []}
        all_episodes = []
        for factor in row["speed_multipliers"]:
            group_key = f"{model_id}_{factor:g}x"
            group = speed["groups"][group_key]
            observations = [e for e in speed_episodes if e["model"] == model_id
                            and float(e["speed_multiplier"]) == factor]
            episode_ids = {int(e["episode"]) for e in observations}
            assert len(observations) == len(episode_ids) == group["episodes"] == 10
            if reference_episodes is None:
                reference_episodes = episode_ids
            assert episode_ids == reference_episodes, (model_id, factor, "unmatched episodes")
            completed = sum(e["completed"] == "True" for e in observations)
            assert completed == group["completed"]
            close(mean(float(e["success_percent"]) for e in observations),
                  group["success_percent"], (group_key, "coverage"))
            old_group = speed_rmse["groups"][group_key]
            assert old_group["expected"] == old_group["recorded"] == 10
            assert old_group["completed"] == completed
            row["coverage_percent"].append(group["success_percent"])
            row["completed_count"].append(completed)
            row["requested_count"].append(10)
            row["rmse_completed_mm"].append(1000 * old_group["completed_mean_hover_rmse_m"])
            all_episodes.extend(observations)
        assert len(all_episodes) == speed["models"][model_id]["episodes"] == 30
        row["mean_coverage_percent"] = mean(float(e["success_percent"]) for e in all_episodes)
        close(row["mean_coverage_percent"], speed["models"][model_id]["success_percent"],
              (model_id, "speed mean"))
        sweep.append(row)

    return {
        "schema_version": 1,
        "metric_definitions": {
            "coverage_percent": tracking["success_definition"],
            "rmse_mm": "Arithmetic mean of per-episode 3D end-effector-to-hover-target position RMSE, in mm. CR/DA/LS mean weights conditions equally; each has 200 episodes.",
            "speed_rmse_completed_mm": "Arithmetic mean of position RMSE over completed rollouts only. The KR-Coordinate 2x group has 9 completed rollouts; all other groups have 10. This differs from coverage's all-requested-rollout denominator.",
            "completion": "Reaching the requested simulation horizon, independently of tracking coverage; not task success.",
        },
        "protocol": {
            "conditions": {"CR": "Lateral direction reversals", "DA": "Changing three-axis motion",
                           "LS": "30-second continuous following, not manipulation sequencing"},
            "condition_horizons_s": {"CR": 10, "DA": 10, "LS": 30},
            "tracking_rollouts": total_episodes,
            "speed_sweep_rollouts": len(speed_episodes),
            "hover_target": "10 cm above the apple's visual top",
            "expert_warmup_s": 3,
            "velocity": "Recorded or simulator-provided current object velocity; no camera-derived estimator",
            "inference": "Simulation pauses during policy inference",
            "scope": "Three tracking conditions in one room; separate from the manipulation benchmark",
        },
        "rows": rows,
        "speed_sweep": sweep,
        "provenance": {
            "repository": provenance["repository"], "branch": provenance["branch"],
            "commit": COMMIT, "retrieved_date": provenance["retrieved_date"],
            "validation_scope": "CR/DA/LS summaries cross-checked against their committed aggregate CSV; speed coverage recomputed from the per-episode CSV. Original simulator telemetry was not independently revalidated.",
            "sources": checked,
        },
    }


def build_table(evidence):
    groups = (*CONDITIONS, "Mean")
    def cell(row, group):
        return row["mean"] if group == "Mean" else row["conditions"][group]

    best = {(group, metric): select(cell(row, group)[metric] for row in evidence["rows"])
            for group in groups
            for metric, select in (("coverage_percent", max), ("rmse_mm", min))}
    lines = [
        "% Generated by paper/figures/prepare_tracking_evidence.py; edit the generator.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Controlled hover tracking. Coverage is time within 3\,cm (\%); RMSE is mean per-episode position error (mm). Each condition has 200 matched trials per model.}",
        r"\label{tab:tracking}",
        r"\begingroup",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\footnotesize",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l*{8}{r}@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{CR} & \multicolumn{2}{c}{DA} & \multicolumn{2}{c}{LS} & \multicolumn{2}{c}{Mean} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
        r"Model & Coverage $\uparrow$ & RMSE $\downarrow$ & Coverage $\uparrow$ & RMSE $\downarrow$ & Coverage $\uparrow$ & RMSE $\downarrow$ & Coverage $\uparrow$ & RMSE $\downarrow$ \\",
        r"\midrule",
    ]
    for row in evidence["rows"]:
        values = []
        for group in groups:
            for metric in ("coverage_percent", "rmse_mm"):
                value = cell(row, group)[metric]
                formatted = f"{value:.2f}"
                if value == best[group, metric]:
                    formatted = r"\textbf{" + formatted + "}"
                values.append(formatted)
        lines.append(row["label"] + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\endgroup", r"\end{table*}", ""])
    return "\n".join(lines)


def main():
    evidence = build_evidence()
    (ROOT / "tracking_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    table_path = ROOT.parent / "tables/tracking_results.tex"
    table_path.parent.mkdir(exist_ok=True)
    table_path.write_text(build_table(evidence))
    print("Validated 2,400 tracking rollouts and 120 speed-sweep episode scores.")
    print("Wrote paper/figures/tracking_evidence.json and paper/tables/tracking_results.tex")


if __name__ == "__main__":
    main()
