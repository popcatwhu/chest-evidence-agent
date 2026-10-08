"""Publish the Agent evaluation summary to the workbench."""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def export():
    record = json.loads((ROOT / "evals/summary.json").read_text())
    record["snapshot_exported_at"] = datetime.now(timezone.utc).isoformat()
    (ROOT / "static/evaluation.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    export()
