"""Select on development data, report frozen test scores and paired outcomes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chest_agent.config import ROOT, DATA


def checked_result(path, manifest):
    result = json.loads(path.read_text())
    if not result["complete"]:
        raise ValueError("Incomplete experiment: " + str(path))
    if (
        result["frozen"]["manifest_sha256"]
        != hashlib.sha256(manifest.read_bytes()).hexdigest()
    ):
        raise ValueError("Manifest differs")
    examples = {e["case_id"]: e for e in json.loads(manifest.read_text())["examples"]}
    variants = result["frozen"]["variants"]
    seen = set()
    for row in result["records"]:
        key = (row["case_id"], row["variant"])
        if key in seen:
            raise ValueError("Duplicated task")
        seen.add(key)
        if row["reference_answer"] != examples[row["case_id"]]["reference_answer"]:
            raise ValueError("Changed answer")
        if row["correct"] != (
            row["status"] == "completed"
            and row["prediction"] == row["reference_answer"]
        ):
            raise ValueError("Incorrect grading")
    if len(seen) != len(examples) * len(variants):
        raise ValueError("Missing tasks")
    return result


def paired(first, second, first_variant="direct", second_variant="direct"):
    a = {r["case_id"]: r for r in first["records"] if r["variant"] == first_variant}
    b = {r["case_id"]: r for r in second["records"] if r["variant"] == second_variant}
    if set(a) != set(b):
        raise ValueError("Different patients cannot be paired")
    return {
        "patients": len(a),
        "both_correct": sum(a[k]["correct"] and b[k]["correct"] for k in a),
        "first_only_correct": sum(a[k]["correct"] and not b[k]["correct"] for k in a),
        "second_only_correct": sum(not a[k]["correct"] and b[k]["correct"] for k in a),
        "both_wrong": sum(not a[k]["correct"] and not b[k]["correct"] for k in a),
    }


def build(directory, output, models):
    regression = DATA / "independent/manifest.json"
    test = DATA / "medical_test/manifest.json"
    develop = {
        m: checked_result(directory / f"{m}_regression.json", regression)
        for m in models
    }
    winner = sorted(
        models,
        key=lambda m: (
            -develop[m]["summary"]["direct"]["accuracy"],
            develop[m]["summary"]["direct"]["median_seconds"],
            m,
        ),
    )[0]
    tests = {m: checked_result(directory / f"{m}_test.json", test) for m in models}
    pipelines = {
        r["frozen"]["pipeline_sha256"] for r in [*develop.values(), *tests.values()]
    }
    if len(pipelines) != 1:
        raise ValueError("Models used different pipeline code")
    result = {
        "task": "minimal_CXR_MCQ",
        "test_patients": 50,
        "selection_patients": 12,
        "selection_policy": "Highest development direct correctness, then lowest development median latency.",
        "selected_model": winner,
        "clinical_accuracy_validated": False,
        "pipeline_sha256": pipelines.pop(),
        "models": {
            m: {
                "development": develop[m]["summary"],
                "test": tests[m]["summary"],
                "frozen_configuration": tests[m]["frozen"],
                "gpu_peak_reserved_gib": max(
                    r["gpu_peak_reserved_gib"] or 0 for r in tests[m]["records"]
                ),
            }
            for m in models
        },
        "model_pairs": {m: paired(tests[models[0]], tests[m]) for m in models[1:]},
        "tool_test": None,
        "tool_tests": {},
        "limitations": [
            "Fifty public single-radiograph diagnostic MCQs, not the full 2500-question benchmark.",
            "Question-only protocol differs from prior full-report evaluation with additional history.",
            "Public model pretraining contamination is unknown; no clinical expert grading.",
            "Model-estimated tool consistency is not clinical validation.",
        ],
    }
    for model in models:
        tools_path = directory / f"{model}_test_tools.json"
        if not tools_path.exists():
            continue
        tools = checked_result(tools_path, test)
        if tools["frozen"]["pipeline_sha256"] != result["pipeline_sha256"]:
            raise ValueError("Tool pipeline code changed")
        result["tool_tests"][model] = {
            "summary": tools["summary"],
            "raw_vs_validated": paired(tools, tools, "tools", "validated"),
            "direct_vs_validated": paired(tests[model], tools, "direct", "validated"),
        }
    result["tool_test"] = result["tool_tests"].get(winner)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "selected_model": winner,
                "test_scores": {m: tests[m]["summary"] for m in models},
                "tool_test": result["tool_test"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=DATA / "medical_comparison")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "medical_model_validation.json"
    )
    parser.add_argument(
        "--models", nargs="+", default=["qwen32", "lingshu7", "lingshu32"]
    )
    args = parser.parse_args()
    build(args.directory, args.output, args.models)
