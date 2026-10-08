"""Run a resumable report batch, publish progress, then audit and publish metrics."""

import argparse
import fcntl
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.summarize_team_evaluation import summarize
from scripts.export_frontend_evaluation import export
from scripts.evaluate_http import fingerprint
from chest_agent import config


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:7860")
    parser.add_argument("--push-results", action="store_true")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.results = args.results.resolve()
    args.results.parent.mkdir(parents=True, exist_ok=True)
    lock = args.results.with_suffix(".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    )
    total = len(json.loads(args.manifest.read_text())["examples"])
    progress_file = args.results.with_name(args.results.stem + "_progress.json")
    public_progress = config.DATA / "evaluation_progress.json"
    status = {"state": "running", "total": total, "completed": 0, "error": None}

    def publish():
        if progress_file.exists():
            try:
                progress = json.loads(progress_file.read_text())
                status["completed"] = progress["completed_tasks"]
            except (json.JSONDecodeError, OSError):
                pass  # The evaluator may be replacing its progress file.
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(public_progress, status)
        write_json(args.results.parent / "status.json", status)

    publish()
    try:
        # Terminal failures are retained by the evaluator; resume only interrupted runs.
        for attempt in range(3):
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts/evaluate_full_reports.py"),
                    "--manifest",
                    str(args.manifest),
                    "--output",
                    str(args.results),
                    "--base",
                    args.base,
                    "--modes",
                    "verified",
                ],
                cwd=ROOT,
            )
            while child.poll() is None:
                publish()
                time.sleep(3)
            if child.returncode == 0:
                break
            if attempt == 2:
                raise RuntimeError(
                    "Evaluation interrupted; rerun the same command to resume"
                )
            time.sleep(5)
        raw = json.loads(args.results.read_text())
        if raw["frozen"]["pipeline_sha256"] != fingerprint(
            (ROOT / "chest_agent").glob("*.py")
        ):
            raise ValueError(
                "Inference source changed during evaluation; automatic publication stopped"
            )
        if raw["frozen"]["knowledge_sha256"] != fingerprint(
            (ROOT / "data/knowledge").glob("*.json")
        ):
            raise ValueError(
                "Knowledge corpus changed during evaluation; automatic publication stopped"
            )
        summary = summarize(args.manifest, args.results)
        summary["evaluated_commit"] = source_commit
        write_json(args.results.parent / "audited_summary.json", summary)
        write_json(ROOT / "evals/summary.json", summary)
        export()
        if args.push_results:
            head = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
                .decode()
                .strip()
            )
            if head == source_commit:
                files = ["evals/summary.json", "static/evaluation.json"]
                commit = subprocess.run(
                    [
                        "git",
                        "commit",
                        "--only",
                        "-m",
                        f"Publish audited {total}-patient evaluation",
                        "--",
                        *files,
                    ],
                    cwd=ROOT,
                )
                if commit.returncode == 0:
                    pushed = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
                    status["github_update"] = (
                        "complete" if pushed.returncode == 0 else "push_failed"
                    )
                else:
                    status["github_update"] = "commit_failed"
            else:
                status["github_update"] = "skipped_repository_changed"
        status["state"] = "complete"
        publish()
        print("Audited evaluation published:", total, "patients", flush=True)
    except Exception as error:
        status["state"] = "interrupted"
        status["error"] = str(error)
        publish()
        raise


if __name__ == "__main__":
    main()
