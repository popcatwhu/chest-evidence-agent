import pytest
from pydantic import ValidationError

from chest_agent import agents, workflow
from chest_agent.independent_review import (
    BlindOpinion,
    BlindChoiceOpinion,
    IndexedReview,
    blind_prompt,
    source_index,
    report_index,
    resolve_review,
    SelectedCaseProfile,
    resolve_facts,
)
from chest_agent.schemas import Candidate, Claim, Evidence, Report, CaseProfile


def report(text="发热", choice=None):
    return Report(
        answer_choice=choice,
        most_likely=Candidate(
            name="肺炎", support=[Claim(text=text, evidence_ids=["C-1"])]
        ),
    )


def state(draft=None):
    return {
        "mode": "verified",
        "case": {"context": "reported fever"},
        "question": "诊断是什么？\nA. 肺炎\nB. 心衰",
        "profile": CaseProfile(facts=[]),
        "evidence": [
            Evidence(id="C-1", kind="case", title="history", content="reported fever")
        ],
        "report": draft or report(),
        "repairs": 0,
    }


def test_blind_context_excludes_every_other_models_opinion(monkeypatch):
    s = state()
    s.update(plan="SECRET_PLAN", clinical_contrast="SECRET_CONTRAST")
    s["report"] = report("SECRET_DRAFT")
    s["evidence"].append(
        Evidence(
            id="I-observe", kind="image", title="observer", content="SECRET_OBSERVER"
        )
    )
    s["evidence"].append(
        Evidence(
            id="K-1", kind="knowledge", title="retrieval", content="SECRET_RETRIEVAL"
        )
    )
    prompt = blind_prompt(s)
    assert "SECRET" not in prompt
    assert "reported fever" in prompt and "A. 肺炎" in prompt
    monkeypatch.setattr(workflow, "log", lambda *args: None)
    monkeypatch.setattr(
        agents.reviewer,
        "run",
        lambda *args, **kwargs: BlindOpinion(
            candidates=[report("reported fever").most_likely]
        ),
    )
    result = workflow.consult(s)
    assert result["agent_messages"][0]["output"]["phase"] == "blind"


def test_clinical_draft_does_not_read_blind_opinion():
    s = state()
    s["blind_opinion"] = BlindOpinion(
        candidates=[
            Candidate(
                name="SECRET_BLIND",
                support=[Claim(text="reported fever", evidence_ids=["C-1"])],
            )
        ]
    )
    assert "SECRET_BLIND" not in workflow.report_prompt(s)


def test_template_copy_and_missing_choice_cannot_pass_as_real_opinions():
    with pytest.raises(ValidationError):
        BlindOpinion(
            candidates=[
                Candidate(
                    name="独立候选疾病",
                    support=[Claim(text="患者自身的支持依据", evidence_ids=["C-1"])],
                )
            ]
        )
    with pytest.raises(ValidationError):
        BlindChoiceOpinion(candidates=[report().most_likely])
    with pytest.raises(ValidationError):
        IndexedReview.model_validate({})


def test_exact_span_selection_preserves_decimals_unicode_and_failed_tools():
    evidence = [
        Evidence(
            id="C-1",
            kind="case",
            title="history",
            content="LDH 3.5 mmol/L.\n发热。无胸痛。",
        ),
        Evidence(id="I-original", kind="image", title="image", content="placeholder"),
        Evidence(
            id="I-fail", kind="image", title="tool", content="error", status="failed"
        ),
    ]
    rows = source_index(evidence)
    assert rows and all(r["evidence_id"] == "C-1" for r in rows)
    assert any("3.5" in r["text"] for r in rows)
    for row in rows:
        assert evidence[0].content[row["start"] : row["end"]] == row["text"]
    review = IndexedReview.model_validate(
        {
            "issues": [
                {"report_id": "R-1", "source_id": rows[0]["id"], "reason": "原文反驳"}
            ]
        }
    )
    resolved, spans, notes = resolve_review(
        review, report_index(report(choice="A")), rows
    )
    assert resolved.issues[0].report_quote == "A"
    assert resolved.issues[0].source_quote == rows[0]["text"]
    assert spans[0]["end"] == rows[0]["end"] and not notes


def test_invalid_ids_cannot_create_accepted_challenges():
    result, spans, notes = resolve_review(
        IndexedReview.model_validate(
            {
                "issues": [
                    {"report_id": "R-999", "source_id": "C-1-S1-1", "reason": "invalid"}
                ]
            }
        ),
        report_index(report()),
        source_index(state()["evidence"]),
    )
    assert not result.issues and not spans and notes


def test_fact_selection_copies_source_instead_of_regenerating_values():
    rows = source_index(
        [
            Evidence(
                id="C-1",
                kind="case",
                title="history",
                content="CD4 12 cells/mm3. LDH 650 U/L.",
            ),
            Evidence(
                id="K-1",
                kind="knowledge",
                title="guide",
                content="CD4 below 200 is a risk factor.",
            ),
        ]
    )
    selection = SelectedCaseProfile.model_validate(
        {
            "facts": [
                {"name": "CD4", "source_id": "C-1-S1-1"},
                {"name": "LDH", "source_id": "C-1-S2-1"},
                {"name": "invented", "source_id": "K-1-S1-1"},
                {"name": "missing", "source_id": "C-99-S1-1"},
            ]
        }
    )
    profile, notes = resolve_facts(selection, rows)
    assert len(profile.facts) == 2 and len(notes) == 2
    assert profile.facts[0].value == "CD4 12 cells/mm3."
    assert profile.facts[1].source_quote == "LDH 650 U/L."
    assert all(f.source_id == "C-1" for f in profile.facts)
    assert profile.facts[0].source_span.id == "C-1-S1-1"
    assert rows[0]["text"].strip() == profile.facts[0].source_quote


def test_known_snippet_references_resolve_to_evidence_without_accepting_unknown_ids():
    from chest_agent.clinical_contrast import IndexedClinicalContrast, resolve_contrast

    raw = IndexedClinicalContrast.model_validate(
        {
            "key_case_clues": [{"name": "fever", "source_id": "C-1-S1-1"}],
            "hypotheses": [
                {
                    "name": "肺炎",
                    "support": [{"text": "fever", "evidence_ids": ["C-1-S1-1"]}],
                },
                {
                    "name": "心衰",
                    "support": [{"text": "unknown", "evidence_ids": ["C-99-S1-1"]}],
                },
            ],
        }
    )
    resolved, notes = resolve_contrast(raw, source_index(state()["evidence"]))
    assert resolved.hypotheses[0].support[0].evidence_ids == ["C-1"]
    assert resolved.hypotheses[1].support[0].evidence_ids == ["C-99-S1-1"]
    assert resolved.key_case_clues[0].source_span.id == "C-1-S1-1"
    assert not notes


@pytest.mark.parametrize("wrong", [True, False])
def test_injected_false_claim_and_correct_control(monkeypatch, wrong):
    monkeypatch.setattr(workflow, "log", lambda *args: None)
    s = state(report("无发热" if wrong else "reported fever"))

    def review(prompt, schema, *args, **kwargs):
        assert schema is IndexedReview
        assert s["question"] in prompt
        return IndexedReview.model_validate(
            {
                "issues": [
                    {
                        "report_id": "R-2",
                        "source_id": "C-1-S1-1",
                        "reason": "报告写无发热，病史明确有发热",
                    }
                ]
                if wrong
                else []
            }
        )

    monkeypatch.setattr(agents.reviewer, "run", review)
    result = workflow.verify(s)
    accepted = result["agent_messages"][-1]["output"]["accepted_challenges"]
    assert bool(accepted) == wrong
    assert workflow.route({**s, **result}) == ("repair" if wrong else "update")


def test_disagreement_is_visible_but_cannot_force_repair(monkeypatch):
    monkeypatch.setattr(workflow, "log", lambda *args: None)
    s = state(report("reported fever", choice="A"))
    s["blind_opinion"] = BlindOpinion(
        answer_choice="B", candidates=[report().most_likely]
    )
    monkeypatch.setattr(
        agents.reviewer, "run", lambda *args, **kwargs: IndexedReview(issues=[])
    )
    result = workflow.verify(s)
    assert result["review_notes"]
    assert not result["issues"]
    assert workflow.route({**s, **result}) == "update"
