"""Audit a frozen multi-model report evaluation and export public aggregate metrics."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.audit_full_report_results import checked, quality


def summarize(manifest, results):
    data = checked(results, manifest, ["verified"])
    records = data["records"]
    role_counts = {
        role: {"completed": 0, "failed": 0}
        for role in ("radiology", "clinical", "review", "coordinator")
    }
    expected = {item["agent"]: item for item in data["frozen"]["health"]["agent_team"]}
    models = {role: item["model"] for role, item in expected.items()}
    if len(set(models.values())) < 3:
        raise ValueError("Evaluation did not configure three distinct models")
    complete_teams = 0
    for row in records:
        run = row["run"]
        team = (run.get("result") or {}).get("collaboration", {})
        if run.get("result") and team.get("protocol") != "heterogeneous-review-v1":
            raise ValueError("Report used an unexpected collaboration protocol")
        roles = set()
        failed = set()
        for message in team.get("messages", []):
            role = message["agent"]
            if role not in role_counts:
                raise ValueError("Unknown agent role")
            identity = expected[role]
            if any(
                message.get(key) != identity.get(key)
                for key in ("model", "backend", "quantization", "source_revision")
            ):
                raise ValueError(
                    "Agent model identity changed during the frozen evaluation"
                )
            if message["status"] == "completed":
                roles.add(role)
            elif message["status"] == "failed":
                failed.add(role)
        for role in role_counts:
            role_counts[role]["completed"] += int(role in roles)
            role_counts[role]["failed"] += int(role in failed)
        complete_teams += int(set(role_counts) <= roles and not failed)
    return {
        "protocol": "heterogeneous-review-v1",
        "current_team_evaluation": "complete",
        "model": " / ".join(dict.fromkeys(models.values())),
        "models_by_agent": models,
        "agent_models": list(expected.values()),
        "mode": "verified",
        "test_patients": len(records),
        "test": data["summary"]["modes"]["verified"],
        "report_quality": quality(records),
        "collaboration": {"complete_teams": complete_teams, "agents": role_counts},
        "development": None,
        "test_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "pipeline_sha256": data["frozen"]["pipeline_sha256"],
        "test_outcomes_used_for_selection": False,
        "clinical_accuracy_validated": False,
        "limitations": [
            "Twenty public diagnostic MCQs within full reports; not clinical expert grading.",
            "Public pretraining contamination unknown.",
            "Independent model review and citation validity do not prove medical correctness.",
            "Serial auxiliary model loading is included in report latency.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "evals/summary.json")
    args = parser.parse_args()
    result = summarize(args.manifest, args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Audited summary saved:", args.output)
