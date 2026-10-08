from chest_agent.schemas import Report, Evidence
from chest_agent.recommendations import check_recommendations


def source():
    return Evidence(
        id="K-1",
        kind="knowledge",
        title="NIH diagnostic guideline",
        content="Routine culture does not establish PCP.",
        diagnostic_rules=[
            {
                "id": "pcp-culture",
                "condition_terms": ["肺孢子菌", "Pneumocystis"],
                "unsupported_patterns": ["痰培养", "sputum culture"],
                "reason": "常规培养不能确认PCP",
            }
        ],
    )


def report(check, purpose):
    return Report.model_validate(
        {
            "assessment": "有限",
            "most_likely": {
                "name": "疑似肺孢子菌肺炎",
                "support": [{"text": "患者资料", "evidence_ids": ["C-1"]}],
            },
            "recommended_checks": [
                {"name": check, "purpose": purpose, "evidence_ids": ["K-1"]}
            ],
        }
    )


def test_sputum_culture_cannot_confirm_pcp():
    assert any(
        "冲突" in issue
        for issue in check_recommendations(report("痰培养", "确认病原体"), [source()])
    )


def test_full_sputum_name_is_not_missed_by_culture_constraint():
    s = source()
    s.diagnostic_rules[0]["unsupported_patterns"] = ["痰(?:液)?培养"]
    assert any(
        "冲突" in issue
        for issue in check_recommendations(report("痰液培养", "确认肺孢子菌感染"), [s])
    )


def test_culture_for_other_bacteria_is_not_rejected():
    assert check_recommendations(report("痰培养", "鉴别其他细菌感染"), [source()]) == []


def test_induced_sputum_pcr_is_not_routine_culture():
    assert check_recommendations(report("诱导痰PCR", "确认肺孢子菌"), [source()]) == []


def test_invalid_recommendation_source_is_flagged():
    r = report("呼吸道PCR", "确认病原体")
    r.recommended_checks[0].evidence_ids = ["K-missing"]
    assert any("来源" in issue for issue in check_recommendations(r, [source()]))


def test_unreferenced_legacy_advice_is_not_silently_verified():
    r = report("呼吸道PCR", "确认病原体")
    r.recommended_checks = []
    r.next_checks = ["痰培养"]
    assert any("缺少用途" in issue for issue in check_recommendations(r, [source()]))


def test_only_investigating_other_bacteria_does_not_confirm_primary():
    s = source()
    s.diagnostic_rules.append(
        {
            "id": "confirm-pcp",
            "condition_terms": ["肺孢子菌"],
            "required_confirmation_patterns": ["PCR", "灌洗", "染色"],
            "reason": "需要针对肺孢子菌的确认检查",
        }
    )
    issues = check_recommendations(report("痰培养", "鉴别其他细菌感染"), [s])
    assert any("未说明" in issue for issue in issues)
