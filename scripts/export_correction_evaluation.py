"""Audit completed correction trials and publish aggregate development measurements."""

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image

from chest_agent import store
from chest_agent.correction_metrics import summarize
from scripts.prepare_medxpert import pixel_hash

ROOT = Path(__file__).resolve().parent.parent


def export(manifest_path, results_path, output):
    manifest = json.loads(manifest_path.read_text())
    results = json.loads(results_path.read_text())
    if (
        results["frozen"]["manifest_sha256"]
        != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Results belong to another manifest")
    if not results["complete"] or len(results["records"]) != len(manifest["examples"]):
        raise ValueError("Cannot publish a final score for incomplete trials")
    examples = {e["case_id"]: e for e in manifest["examples"]}
    seen = set()
    decisions, latencies, peaks, transitions = Counter(), [], [], Counter()
    triggered, new_chunks, failed_turns = 0, 0, 0
    for row in results["records"]:
        if row["case_id"] in seen or row["case_id"] not in examples:
            raise ValueError("Duplicate or unknown trial")
        seen.add(row["case_id"])
        example = examples[row["case_id"]]
        run = row["run"]
        if (
            run["context"] != example["context"]
            or run["question"] != example["question"]
            or row["reference_answer"] != example["reference_answer"]
        ):
            raise ValueError("Trial input or reference changed")
        case = store.get_case(run["case_id"])
        with Image.open(case["image_path"]) as image:
            if pixel_hash(image) != example["pixel_sha256"]:
                raise ValueError("Uploaded image pixels differ from frozen input")
        result = run.get("result") or {}
        messages = result.get("collaboration", {}).get("messages", [])
        draft = next(
            (
                m["output"].get("answer_choice")
                for m in messages
                if m["agent"] == "clinical" and m["status"] == "completed"
            ),
            None,
        )
        final = result.get("report", {}).get("answer_choice")
        if (
            row["draft_choice"] != draft
            or row["final_choice"] != final
            or row["status"] != run["status"]
        ):
            raise ValueError("Stored grading fields do not match raw agent outputs")
        transitions[
            ("correct" if draft == row["reference_answer"] else "wrong_or_unavailable")
            + "_to_"
            + (
                "correct"
                if final == row["reference_answer"] and run["status"] == "completed"
                else "wrong_or_unavailable"
            )
        ] += 1
        collaboration = result.get("collaboration", {})
        retrieval = collaboration.get("dispute_retrieval", {})
        triggered += int(retrieval.get("triggered", False))
        new_chunks += len(retrieval.get("new_evidence_ids", []))
        decision = collaboration.get("adjudication")
        if decision:
            decisions[decision["decision"]] += 1
        failed_turns += sum(m["status"] == "failed" for m in messages)
        if result:
            latencies.append(result["metrics"]["elapsed_seconds"])
            peaks.append(result["metrics"]["gpu_peak_reserved_gib"])
    summary = {
        "type": "development diagnostic-correction MCQ evaluation",
        "source": manifest["source"],
        "source_revision": manifest["source_revision"],
        "pipeline": results["frozen"],
        "metrics": summarize(results["records"]),
        "answer_transitions": dict(transitions),
        "disputes_triggered": triggered,
        "new_reference_chunks": new_chunks,
        "adjudication_counts": dict(decisions),
        "failed_agent_turns": failed_turns,
        "median_seconds_completed": round(statistics.median(latencies), 2)
        if latencies
        else None,
        "peak_reserved_gib": max(peaks) if peaks else None,
        "holdout_evaluated": manifest["split"] == "holdout",
        "limitations": manifest["limitations"]
        + [
            "Development measurements, not a new independent test score.",
            "Draft versus final isolates later pipeline steps, not standalone-model versus full-team ablation.",
            "MCQ label agreement does not establish open-ended diagnostic or clinical correctness.",
            "Frozen ChestAgentBench 60-case score remains unchanged.",
        ],
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary["metrics"], ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/medxpert/development/manifest.json",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "data/correction/development12/results.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evals/correction-development.json"
    )
    args = parser.parse_args()
    export(args.manifest, args.results, args.output)
