from pathlib import Path
from types import SimpleNamespace

import pytest

from chest_agent import agents, workflow
from chest_agent.llm import ModelBackend
from chest_agent.schemas import Observation, Report, Review, CaseProfile, Evidence


def test_models_have_independent_configuration(monkeypatch):
    first = ModelBackend(Path("/models/one"), "local", True)
    second = ModelBackend(Path("/models/two"), "local", False)
    monkeypatch.setattr(agents.config, "MODEL", Path("/models/changed-default"))
    assert first.model_path != second.model_path
    assert first.load_nf4 and not second.load_nf4
    assert first.metrics is not second.metrics


def test_auxiliary_pool_releases_previous_model_before_new_call():
    order = []
    pool = agents.AuxiliaryModels()
    first = SimpleNamespace(release=lambda: order.append("release-first"))
    second = SimpleNamespace(release=lambda: order.append("release-second"))
    with pool.use(first):
        order.append("first-call")
    with pool.use(second):
        order.append("second-call")
    assert order == ["first-call", "release-first", "second-call"]


def test_radiology_task_does_not_receive_history_or_clinical_conclusion(monkeypatch):
    monkeypatch.setattr(workflow, "log", lambda *args: None)

    def infer(prompt, schema, image, **kwargs):
        assert "secret history" not in prompt
        assert "lung cancer" not in prompt
        assert image == "chest.png"
        return Observation(observations=["opacity"], limitations=["uncertain location"])

    monkeypatch.setattr(agents.radiologist, "run", infer)
    result = workflow.observe(
        {
            "mode": "verified",
            "case": {"context": "secret history", "image_path": "chest.png"},
            "evidence": [],
            "report": {"diagnosis": "lung cancer"},
        }
    )
    assert result["agent_messages"][0]["agent"] == "radiology"
    assert result["evidence"][0].source_type == "model-generated observation"


def test_failed_independent_review_is_not_reported_as_passed(monkeypatch):
    monkeypatch.setattr(workflow, "log", lambda *args: None)
    report = Report.model_validate(
        {
            "most_likely": {
                "name": "候选",
                "support": [{"text": "fever", "evidence_ids": ["C-1"]}],
            }
        }
    )

    def fail(*args, **kwargs):
        raise RuntimeError("review model unavailable")

    monkeypatch.setattr(agents.reviewer, "run", fail)
    result = workflow.verify(
        {
            "mode": "verified",
            "case": {},
            "report": report,
            "profile": CaseProfile(facts=[]),
            "evidence": [
                Evidence(id="C-1", kind="case", title="history", content="fever")
            ],
        }
    )
    assert result["agent_messages"][0]["status"] == "failed"
    assert result["review_notes"]


def test_reviewer_uses_its_own_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        agents.reviewer.model,
        "json",
        lambda *args, **kwargs: calls.append(args) or Review(issues=[]),
    )
    monkeypatch.setattr(
        agents.backend,
        "json",
        lambda *args, **kwargs: pytest.fail(
            "Primary model must not act as independent reviewer"
        ),
    )
    monkeypatch.setattr(agents.auxiliary_models, "active", None)
    result = agents.reviewer.run("actual task", Review)
    assert result.issues == []
    assert "actual task" in calls[0][0]
