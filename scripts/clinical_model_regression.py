"""Run known clinical development inputs through the real report workflow.

This checks application behavior, not independent diagnostic accuracy.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chest_agent import config, store
from chest_agent.workflow import execute


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    store.init_db()
    examples = json.loads(
        (config.DATA / "quality/cases.json").read_text()
    ) + json.loads((config.DATA / "quality/heldout.json").read_text())
    records = []
    for example in examples:
        case = store.create_case(
            config.MODEL.name + " 临床开发回归 · " + example["title"],
            example["context"],
            str((config.DATA / example["image_path"]).resolve()),
        )
        run_id = store.create_run(
            case["id"],
            "请综合病史、已提供检查和原始胸片，给出最可能的具体诊断、鉴别诊断及支持和反对证据；明确还需哪些检查才能确认。",
            "verified",
        )
        execute(run_id)
        run = store.get_run(run_id)
        records.append(
            {
                "title": example["title"],
                "reference": example["reference_diagnosis"],
                "run": run,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "purpose": "known-development clinical application regression",
                    "model": config.MODEL.name,
                    "cxr_expert_enabled": config.CXR_EXPERT_ENABLED,
                    "clinical_accuracy_validated": False,
                    "records": records,
                    "complete": len(records) == len(examples),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(
            example["title"],
            run["status"],
            run["result"]["report"]["most_likely"]["name"]
            if run["result"]
            else run["error"],
            flush=True,
        )
