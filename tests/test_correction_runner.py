import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import evaluate_corrections as evaluator


def test_holdout_requires_explicit_evaluation_flag(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"split": "holdout"}))
    with pytest.raises(ValueError, match="reserved"):
        evaluator.evaluate(manifest, tmp_path / "results.json", "http://unused")


def test_pending_trial_is_resumed_without_submitting_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    monkeypatch.setattr(
        evaluator.subprocess,
        "check_output",
        lambda *a, **kw: "test-head" if kw.get("text") else b"",
    )
    image = tmp_path / "data/image.png"
    image.parent.mkdir()
    image.write_bytes(b"fake-test-image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "split": "development",
                "examples": [
                    {
                        "case_id": "sample",
                        "image_path": "image.png",
                        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "reference_answer": "B",
                    }
                ],
            }
        )
    )
    (tmp_path / "pending.json").write_text(
        json.dumps(
            {
                "case_id": "sample",
                "run_id": "existing-run",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        )
    )

    def get(url, **kwargs):
        data = (
            {"pipeline_source_sha256": evaluator.source_fingerprint()}
            if url.endswith("health")
            else {
                "status": "completed",
                "result": {
                    "report": {"answer_choice": "B"},
                    "collaboration": {
                        "messages": [
                            {
                                "agent": "clinical",
                                "status": "completed",
                                "output": {"answer_choice": "A"},
                            }
                        ]
                    },
                },
            }
        )
        return SimpleNamespace(json=lambda: data, raise_for_status=lambda: None)

    monkeypatch.setattr(evaluator.requests, "get", get)
    monkeypatch.setattr(
        evaluator.requests,
        "post",
        lambda *a, **kw: pytest.fail("Pending run must not be replaced"),
    )
    result = evaluator.evaluate(manifest, tmp_path / "results.json", "http://test")
    assert result["complete"] and result["summary"]["wrong_drafts_corrected"] == 1
    assert not (tmp_path / "pending.json").exists()
