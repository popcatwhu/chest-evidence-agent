"""Evaluate curated diagnostic cases with labels kept outside model inputs.

Input JSON: [{title, context, image_path, reference_diagnosis, accepted_names?}].
Results use explicit string matches: an engineering check, not expert clinical grading.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chest_agent import store
from chest_agent.workflow import execute

parser = argparse.ArgumentParser()
parser.add_argument("cases", type=Path)
parser.add_argument(
    "--modes",
    nargs="+",
    choices=["direct", "tools", "verified"],
    default=["direct", "tools", "verified"],
)
parser.add_argument("--output", type=Path, default=Path("data/evaluation.json"))
parser.add_argument(
    "--task", choices=["diagnosis", "multiple_choice"], default="diagnosis"
)
args = parser.parse_args()
store.init_db()
records = []
for example in json.loads(args.cases.read_text()):
    # Never send reference diagnosis, accepted names, or benchmark explanations to the agent.
    image = example.get("image_path")
    if image:
        image = str((args.cases.parent / image).resolve())
    for mode in args.modes:
        # Separate case histories prevent one baseline's report influencing another.
        case = store.create_case(
            example["title"] + " · " + mode, example["context"], image
        )
        question = example.get("question", "最可能诊断是什么？请给出证据和鉴别诊断。")
        run_id = store.create_run(case["id"], question, mode)
        execute(run_id)
        run = store.get_run(run_id)
        result = run["result"]
        prediction = (
            (
                result["report"]["answer_choice"]
                if args.task == "multiple_choice"
                else result["report"]["most_likely"]["name"]
            )
            if result
            else None
        )
        aliases = [example["reference_diagnosis"], *example.get("accepted_names", [])]
        records.append(
            {
                "case_id": case["id"],
                "mode": mode,
                "status": run["status"],
                "run_id": run_id,
                "prediction": prediction,
                "reference": example["reference_diagnosis"],
                "name_match": bool(
                    prediction
                    and any(a.casefold() == prediction.casefold() for a in aliases)
                ),
                "metrics": result["metrics"] if result else None,
                "verification": result["verification"] if result else None,
                "error": run["error"],
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(mode, run["status"], prediction, flush=True)
summary = {
    mode: {
        "tasks": sum(r["mode"] == mode for r in records),
        "completed": sum(
            r["mode"] == mode and r["status"] == "completed" for r in records
        ),
        "exact_name_matches": sum(
            r["mode"] == mode and r["name_match"] for r in records
        ),
    }
    for mode in args.modes
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
