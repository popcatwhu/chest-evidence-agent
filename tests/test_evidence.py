from chest_agent.schemas import Evidence, Report, validate_evidence


def report(refs):
    return Report.model_validate(
        {
            "assessment": "有限",
            "most_likely": {
                "name": "待评估的候选",
                "support": [{"text": "仅供验证引用", "evidence_ids": refs}],
            },
        }
    )


def test_failed_tool_cannot_support_diagnosis():
    evidence = [
        Evidence(
            id="I-classify",
            kind="image",
            title="分类",
            content="failed",
            status="failed",
        )
    ]
    assert any(
        "失败" in issue for issue in validate_evidence(report(["I-classify"]), evidence)
    )


def test_knowledge_is_not_patient_evidence():
    evidence = [
        Evidence(id="K-1", kind="knowledge", title="资料", content="general knowledge")
    ]
    assert any(
        "患者自身证据" in issue
        for issue in validate_evidence(report(["K-1"]), evidence)
    )


def test_valid_patient_reference():
    evidence = [
        Evidence(id="C-1", kind="case", title="病例", content="reported symptom")
    ]
    assert validate_evidence(report(["C-1"]), evidence) == []


def test_classifier_score_cannot_be_used_to_rule_out_disease():
    evidence = [
        Evidence(id="I-classify", kind="image", title="分类", content="model score")
    ]
    output = report(["I-classify"])
    output.most_likely.support[0].text = "胸片未见心脏增大"
    assert any("排除疾病" in issue for issue in validate_evidence(output, evidence))


def test_tool_only_reasoning_cannot_claim_direct_image_verification():
    evidence = [
        Evidence(id="I-original", kind="image", title="原图", content="原始胸片")
    ]
    assert any(
        "未直接读取" in issue
        for issue in validate_evidence(
            report(["I-original"]), evidence, allow_original=False
        )
    )
