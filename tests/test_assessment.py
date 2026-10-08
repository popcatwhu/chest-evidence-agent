from chest_agent.schemas import Report


def test_missing_confidence_metadata_does_not_claim_sufficient_evidence():
    report = Report.model_validate(
        {
            "most_likely": {
                "name": "待确认候选",
                "support": [{"text": "患者资料", "evidence_ids": ["C-1"]}],
            }
        }
    )
    assert report.assessment == "有限"
