from chest_agent.schemas import Report
from chest_agent.diagnosis_checks import diagnosis_issues


def candidate(name):
    return {"name": name, "support": [{"text": "病例依据", "evidence_ids": ["C-1"]}]}


def test_aliases_do_not_count_as_independent_differential_diagnoses():
    report = Report.model_validate(
        {
            "most_likely": candidate("疑似肺孢子菌肺炎"),
            "differentials": [
                candidate("Pneumocystis jirovecii pneumonia"),
                candidate("卡氏肺孢子虫肺炎"),
            ],
        }
    )
    assert len(diagnosis_issues(report)) == 2


def test_shared_disease_family_does_not_erase_distinct_etiology():
    report = Report.model_validate(
        {
            "most_likely": candidate("自发性气胸"),
            "differentials": [
                candidate("月经相关气胸"),
                candidate("肺孢子菌肺炎并发气胸"),
            ],
        }
    )
    assert diagnosis_issues(report) == []


def test_exact_duplicates_are_detected_without_known_aliases():
    report = Report.model_validate(
        {"most_likely": candidate("肺癌"), "differentials": [candidate("肺癌")]}
    )
    assert diagnosis_issues(report)
