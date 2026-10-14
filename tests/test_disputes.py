import pytest

from chest_agent import agents, workflow
from chest_agent.disputes import (
    Adjudication,
    DisputePlan,
    adjudication_errors,
    disagrees,
)
from chest_agent.independent_review import BlindOpinion, IndexedReview
from chest_agent.schemas import Report, CaseProfile, Evidence


def state(blind_choice="B"):
    report = Report.model_validate(
        {
            "answer_choice": "A",
            "most_likely": {
                "name": "肺炎",
                "support": [{"text": "发热", "evidence_ids": ["C-1"]}],
            },
        }
    )
    return {
        "mode": "verified",
        "case": {"context": "fever"},
        "repairs": 0,
        "question": "最可能诊断？\nA. 肺炎\nB. 肺水肿",
        "report": report,
        "profile": CaseProfile(facts=[]),
        "blind_opinion": BlindOpinion(
            answer_choice=blind_choice, candidates=[report.most_likely]
        ),
        "evidence": [
            Evidence(id="C-1", kind="case", title="history", content="fever"),
            Evidence(
                id="K-1",
                kind="knowledge",
                title="guide",
                content="old text",
                source="https://example.org/old",
            ),
        ],
    }


def test_agreement_does_not_call_search_or_coordinator(monkeypatch):
    monkeypatch.setattr(
        agents.coordinator, "run", lambda *a, **kw: pytest.fail("unnecessary call")
    )
    assert not disagrees(state("A"))
    result = workflow.investigate_dispute(state("A"))
    assert not result["dispute_retrieval"]["triggered"]


def test_dispute_search_preserves_ids_deduplicates_and_bounds_evidence(monkeypatch):
    monkeypatch.setattr(workflow, "log", lambda *a: None)
    monkeypatch.setattr(
        agents.coordinator,
        "run",
        lambda *a, **kw: DisputePlan(
            disputed_point="infection or edema",
            queries=["pneumonia versus edema", "second query"],
            decision_criteria=["patient-specific discriminating morphology"],
        ),
    )

    def retrieve(query, **kwargs):
        assert kwargs == {"limit": 2, "max_per_source": 1}
        return [
            state()["evidence"][1],
            Evidence(
                id="K-1",
                kind="knowledge",
                title="new guide",
                content="new text" * 200,
                source="https://example.org/new",
            ),
        ]

    monkeypatch.setattr(workflow, "retrieve", retrieve)
    result = workflow.investigate_dispute(state())
    assert [e.id for e in result["evidence"]] == ["C-1", "K-1", "K-2"]
    assert result["evidence"][1].content == "old text"
    assert len(result["evidence"][-1].content) <= 700
    assert result["dispute_retrieval"]["new_evidence_ids"] == ["K-2"]


@pytest.mark.parametrize(
    "refs,choice", [(["K-1"], "B"), (["missing"], "B"), (["C-1"], "E"), (["C-1"], "A")]
)
def test_unsupported_revision_is_rejected(refs, choice):
    decision = Adjudication(
        decision="revise", preferred_choice=choice, reason="change", evidence_ids=refs
    )
    assert adjudication_errors(decision, state())


def test_valid_revision_triggers_one_repair_and_cannot_loop(monkeypatch):
    monkeypatch.setattr(workflow, "log", lambda *a: None)
    s = state()
    s["adjudication"] = Adjudication(
        decision="revise",
        preferred_choice="B",
        reason="原始患者依据支持另一候选",
        evidence_ids=["C-1"],
    ).model_dump()
    monkeypatch.setattr(
        agents.reviewer, "run", lambda *a, **kw: IndexedReview(issues=[])
    )
    result = workflow.verify(s)
    assert workflow.route({**s, **result}) == "repair"
    s["repairs"] = 1
    result = workflow.verify(s)
    assert workflow.route({**s, **result}) == "update"


def test_unresolved_dispute_remains_visible_without_forcing_revision(monkeypatch):
    monkeypatch.setattr(workflow, "log", lambda *a: None)
    s = state()
    s["dispute_plan"] = {"queries": ["differential"]}
    monkeypatch.setattr(
        agents.coordinator,
        "run",
        lambda *a, **kw: Adjudication(
            decision="unresolved",
            preferred_choice=None,
            reason="没有决定性证据",
            evidence_ids=["C-1"],
        ),
    )
    result = workflow.adjudicate_dispute(s)
    assert result["collaboration_notes"]
    assert result["adjudication"]["decision"] == "unresolved"
