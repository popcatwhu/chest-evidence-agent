"""Evaluate initial clinical drafts versus final reports without exposing labels to agents."""

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

import requests

from chest_agent.correction_metrics import summarize
from chest_agent.fingerprint import source_fingerprint

ROOT = Path(__file__).resolve().parent.parent


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def evaluate(manifest_path, output, base, allow_holdout=False):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("split") == "holdout" and not allow_holdout:
        raise ValueError(
            "Holdout is reserved. Use --evaluate-holdout only after freezing the pipeline."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    health = requests.get(base + "/api/health", timeout=30).json()
    health.pop("model_loaded", None)
    pipeline_hash = source_fingerprint()
    if health.get("pipeline_source_sha256") != pipeline_hash:
        raise ValueError(
            "Running server differs from local pipeline; restart before evaluation"
        )

    def corpus_hash():
        digest = hashlib.sha256()
        for path in sorted((ROOT / "data/knowledge").glob("*.json")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    frozen = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
        ).hexdigest(),
        "split": manifest.get("split"),
        "mode": "verified",
        "health": health,
        "pipeline_source_sha256": pipeline_hash,
        "corpus_sha256": corpus_hash(),
    }
    result = {"frozen": frozen, "records": [], "complete": False}
    if output.exists():
        result = json.loads(output.read_text())
        if result["frozen"] != frozen:
            raise ValueError("Cannot resume results with different code or data")
    else:
        write(output, result)
    seen = {r["case_id"] for r in result["records"]}
    for example in manifest["examples"]:
        if example["case_id"] in seen:
            continue
        if (
            source_fingerprint() != pipeline_hash
            or corpus_hash() != frozen["corpus_sha256"]
        ):
            raise ValueError("Pipeline or reference corpus changed during evaluation")
        image_path = ROOT / "data" / example["image_path"]
        if (
            hashlib.sha256(image_path.read_bytes()).hexdigest()
            != example["image_sha256"]
        ):
            raise ValueError("Input image differs from frozen manifest")
        # No answers, captions, diagnoses, explanations, or evaluator-side options
        # metadata enter this request; task options are already in question.
        pending_path = output.parent / "pending.json"
        pending = (
            json.loads(pending_path.read_text()) if pending_path.exists() else None
        )
        if pending and pending["case_id"] in seen:
            pending_path.unlink()
            pending = None
        if pending and (
            pending["case_id"] != example["case_id"]
            or pending.get("manifest_sha256") != frozen["manifest_sha256"]
        ):
            raise ValueError(
                "Pending trial belongs to another sample; refusing to replace it"
            )
        if pending:
            run_id = pending["run_id"]
        else:
            with image_path.open("rb") as image:
                response = requests.post(
                    base + "/api/cases",
                    data={"title": example["title"], "context": example["context"]},
                    files={"image": ("chest.png", image, "image/png")},
                    timeout=30,
                )
            response.raise_for_status()
            case_id = response.json()["id"]
            response = requests.post(
                base + f"/api/cases/{case_id}/runs",
                json={"question": example["question"], "mode": "verified"},
                timeout=30,
            )
            response.raise_for_status()
            run_id = response.json()["run_id"]
            pending = {
                "case_id": example["case_id"],
                "run_id": run_id,
                "manifest_sha256": frozen["manifest_sha256"],
            }
            write(pending_path, pending)
        deadline = time.monotonic() + 1200
        while True:
            response = requests.get(base + "/api/runs/" + run_id, timeout=30)
            response.raise_for_status()
            run = response.json()
            if run["status"] in {"completed", "failed", "interrupted"}:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Run still pending: {run_id}; do not replace this trial"
                )
            time.sleep(2)
        completed = run.get("result") or {}
        messages = completed.get("collaboration", {}).get("messages", [])
        draft = next(
            (
                m["output"].get("answer_choice")
                for m in messages
                if m["agent"] == "clinical" and m["status"] == "completed"
            ),
            None,
        )
        row = {
            "case_id": example["case_id"],
            "status": run["status"],
            "reference_answer": example["reference_answer"],
            "draft_choice": draft,
            "final_choice": completed.get("report", {}).get("answer_choice"),
            "run": run,
        }
        result["records"].append(row)
        result["summary"] = summarize(result["records"])
        result["complete"] = len(result["records"]) == len(manifest["examples"])
        write(output, result)
        (output.parent / "pending.json").unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "case_id": row["case_id"],
                    "status": row["status"],
                    "summary": result["summary"],
                }
            ),
            flush=True,
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/medxpert/development/manifest.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/correction/results.json"
    )
    parser.add_argument("--base", default="http://127.0.0.1:7860")
    parser.add_argument("--evaluate-holdout", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate(args.manifest, args.output, args.base, args.evaluate_holdout)
