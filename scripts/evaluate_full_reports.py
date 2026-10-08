"""Evaluate Agent reports with progress logging separate from grading."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.evaluate_http import evaluate


class BlindProgress:
    def __init__(self, path):
        self.path = path

    def write(self, text):
        # Deliberately discard prediction/reference logging in progress logs.
        if self.path.exists():
            result = json.loads(self.path.read_text())
            state = {
                "completed_tasks": len(result["records"]),
                "complete": result["complete"],
                "pending": bool(result.get("pending")),
            }
            self.path.with_name(self.path.stem + "_progress.json").write_text(
                json.dumps(state, indent=2)
            )
        return len(text)

    def flush(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:7860")
    parser.add_argument(
        "--modes", nargs="+", choices=["direct", "verified"], default=["verified"]
    )
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    with redirect_stdout(BlindProgress(args.output)):
        evaluate(args)
    print("Full-report evaluation finished; inspect the saved result file.", flush=True)
