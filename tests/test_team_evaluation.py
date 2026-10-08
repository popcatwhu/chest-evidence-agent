import pytest

from scripts import summarize_team_evaluation as export


def fixture_result(models):
    roster = [
        {
            "agent": role,
            "model": model,
            "backend": "local",
            "quantization": "NF4",
            "source_revision": {"revision": role},
        }
        for role, model in models.items()
    ]
    return {
        "records": [],
        "frozen": {"health": {"agent_team": roster}, "pipeline_sha256": "frozen"},
        "summary": {"modes": {"verified": {"tasks": 1, "completed": 1, "correct": 0}}},
    }


def test_summary_rejects_shared_model_disguised_as_three_agents(tmp_path, monkeypatch):
    data = fixture_result(
        {
            role: "same-model"
            for role in ["radiology", "clinical", "review", "coordinator"]
        }
    )
    monkeypatch.setattr(export, "checked", lambda *args: data)
    with pytest.raises(ValueError, match="three distinct models"):
        export.summarize(tmp_path / "manifest.json", tmp_path / "results.json")


def test_summary_rejects_model_identity_change(tmp_path, monkeypatch):
    data = fixture_result(
        {
            "radiology": "nv",
            "clinical": "lingshu",
            "review": "qwen",
            "coordinator": "lingshu",
        }
    )
    message = {
        **data["frozen"]["health"]["agent_team"][0],
        "model": "changed",
        "status": "completed",
    }
    data["records"] = [
        {
            "run": {
                "result": {
                    "collaboration": {
                        "protocol": "heterogeneous-review-v1",
                        "messages": [message],
                    }
                }
            }
        }
    ]
    monkeypatch.setattr(export, "checked", lambda *args: data)
    with pytest.raises(ValueError, match="identity changed"):
        export.summarize(tmp_path / "manifest.json", tmp_path / "results.json")


def test_failed_reviewer_is_not_counted_as_complete_collaboration(
    tmp_path, monkeypatch
):
    data = fixture_result(
        {
            "radiology": "nv",
            "clinical": "lingshu",
            "review": "qwen",
            "coordinator": "lingshu",
        }
    )
    messages = [
        {**item, "status": "failed" if item["agent"] == "review" else "completed"}
        for item in data["frozen"]["health"]["agent_team"]
    ]
    data["records"] = [
        {
            "run": {
                "result": {
                    "collaboration": {
                        "protocol": "heterogeneous-review-v1",
                        "messages": messages,
                    }
                }
            }
        }
    ]
    monkeypatch.setattr(export, "checked", lambda *args: data)
    monkeypatch.setattr(export, "quality", lambda rows: {})
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    summary = export.summarize(manifest, tmp_path / "results.json")
    assert summary["collaboration"]["complete_teams"] == 0
    assert summary["collaboration"]["agents"]["review"]["failed"] == 1
    assert summary["test"]["completed"] == 1
