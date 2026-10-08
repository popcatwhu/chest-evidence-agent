import json
from chest_agent.schemas import Report, Evidence
from chest_agent.clinical_consistency import clinical_inconsistencies


def test_drug_use_does_not_prove_immune_suppression():
    report = Report.model_validate(
        {
            "most_likely": {
                "name": "肺孢子菌肺炎",
                "support": [
                    {
                        "text": "患者静脉药物使用，符合PCP高危免疫抑制人群",
                        "evidence_ids": ["C-1"],
                    }
                ],
            }
        }
    )
    evidence = [
        Evidence(
            id="C-1",
            kind="case",
            title="病例",
            content="History of IV drug abuse, fever.",
        ),
        Evidence(
            id="K-1",
            kind="knowledge",
            title="NIH PCP",
            content="immune risk",
            diagnostic_rules=[{"id": "nih-pcp-risk"}],
        ),
    ]
    assert clinical_inconsistencies(report, evidence)
    evidence[0].content += " AIDS with CD4 count 6 cells/mm3."
    assert not clinical_inconsistencies(report, evidence)


def test_collapse_interpretation_does_not_explain_contralateral_shift():
    report = Report.model_validate(
        {
            "most_likely": {
                "name": "左侧肺不张",
                "support": [{"text": "左胸腔不透光", "evidence_ids": ["I-radiology"]}],
            }
        }
    )
    evidence = [
        Evidence(
            id="I-radiology",
            kind="image",
            title="专用模型",
            content=json.dumps(
                {
                    "visual_findings": [
                        "near complete opacification of the left hemi thorax with mediastinal shift to the right"
                    ]
                }
            ),
        ),
        Evidence(
            id="K-1",
            kind="knowledge",
            title="Opaque hemithorax",
            content="contralateral shift",
        ),
    ]
    assert clinical_inconsistencies(report, evidence)
    report.most_likely.name = "胸腔占位伴压迫性肺不张"
    assert not clinical_inconsistencies(report, evidence)
