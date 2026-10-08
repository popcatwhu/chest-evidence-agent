from chest_agent.schemas import Evidence, CaseProfile, Report, Review
from chest_agent.quality import ground_facts, anchor_review, known_test_conflicts


def report():
    return Report.model_validate(
        {
            "assessment": "有限",
            "most_likely": {
                "name": "疑似感染",
                "support": [{"text": "患者干咳", "evidence_ids": ["C-1"]}],
            },
            "missing_information": ["CD4细胞计数和LDH具体数值"],
        }
    )


def test_facts_reject_invented_numbers_and_translated_quotes():
    evidence = [
        Evidence(
            id="C-1",
            kind="case",
            title="病例",
            content="CD4 count: 60 cells/mm3; dry cough.",
        )
    ]
    profile = CaseProfile(
        facts=[
            {
                "name": "CD4",
                "value": "6",
                "source_id": "C-1",
                "source_quote": "CD4 count: 60 cells/mm3",
            },
            {
                "name": "症状",
                "value": "干咳",
                "source_id": "C-1",
                "source_quote": "干咳",
            },
            {
                "name": "CD4",
                "value": "60",
                "source_id": "C-1",
                "source_quote": "CD4 count: 60 cells/mm3",
            },
        ]
    )
    accepted, rejected = ground_facts(profile, evidence)
    assert len(rejected) == 2
    assert accepted.facts[0].value == "60"


def test_generic_review_reminder_cannot_trigger_report_repair():
    evidence = [Evidence(id="C-1", kind="case", title="病例", content="干咳")]
    review = Review(
        issues=[
            {
                "report_quote": "模型分数不是患病概率",
                "reason": "注意风险",
                "evidence_ids": ["C-1"],
                "source_quote": "干咳",
            }
        ]
    )
    accepted, notes = anchor_review(review, report(), evidence)
    assert accepted == [] and notes


def test_review_requires_actual_report_quote_and_valid_evidence():
    evidence = [Evidence(id="C-1", kind="case", title="病例", content="患者无咳嗽")]
    review = Review(
        issues=[
            {
                "report_quote": "患者干咳",
                "reason": "病例明确否认咳嗽",
                "evidence_ids": ["C-1"],
                "source_quote": "患者无咳嗽",
            }
        ]
    )
    accepted, notes = anchor_review(review, report(), evidence)
    assert len(accepted) == 1 and not notes


def test_known_cd4_is_not_missing_but_ldh_numeric_result_can_be():
    profile = CaseProfile(
        facts=[
            {
                "name": "CD4细胞计数",
                "value": "6 cells/mm³",
                "source_id": "C-1",
                "source_quote": "CD4细胞计数6 cells/mm³",
            },
            {
                "name": "LDH",
                "value": "升高",
                "source_id": "C-1",
                "source_quote": "LDH升高",
            },
        ]
    )
    issues = known_test_conflicts(report(), profile)
    assert len(issues) == 1 and "CD4" in issues[0]


def test_generic_fact_name_still_preserves_known_cd4_result():
    profile = CaseProfile(
        facts=[
            {
                "name": "检查",
                "value": "CD4细胞计数6 cells/mm³",
                "source_id": "C-1",
                "source_quote": "CD4细胞计数6 cells/mm³",
            }
        ]
    )
    assert any("CD4" in issue for issue in known_test_conflicts(report(), profile))


def test_cd4_label_number_is_not_a_provided_count():
    profile = CaseProfile(
        facts=[
            {
                "name": "CD4",
                "value": "CD4降低",
                "source_id": "C-1",
                "source_quote": "CD4降低",
            }
        ]
    )
    assert known_test_conflicts(report(), profile) == []


def test_visual_observation_is_not_a_classifier_score():
    evidence = [
        Evidence(id="I-observe", kind="image", title="观察", content="肺部有阴影")
    ]
    output = report()
    output.most_likely.support[0].text = "肺部有阴影"
    review = Review(
        issues=[
            {
                "report_quote": "肺部有阴影",
                "source_quote": "肺部有阴影",
                "reason": "把分类分数当成概率",
                "evidence_ids": ["I-observe"],
            }
        ]
    )
    accepted, notes = anchor_review(review, output, evidence)
    assert not accepted and notes


def test_classifier_disclaimer_cannot_discredit_original_image_claim():
    evidence = [
        Evidence(id="I-original", kind="image", title="原图", content="原始胸片"),
        Evidence(
            id="I-classify", kind="image", title="分类", content="分数不是疾病概率"
        ),
    ]
    output = report()
    output.most_likely.support[0].text = "胸片存在阴影"
    output.most_likely.support[0].evidence_ids = ["I-original"]
    review = Review(
        issues=[
            {
                "report_quote": "胸片存在阴影",
                "source_quote": "分数不是疾病概率",
                "reason": "把分类分数当成疾病表现",
                "evidence_ids": ["I-classify"],
            }
        ]
    )
    accepted, notes = anchor_review(review, output, evidence)
    assert not accepted and notes
