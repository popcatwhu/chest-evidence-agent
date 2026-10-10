from chest_agent.llm import ModelBackend
from chest_agent.schemas import Review
from chest_agent import config


def test_format_retry_restarts_from_task_without_echoing_invalid_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "DATA", tmp_path)
    model = ModelBackend()
    prompts = []
    outputs = iter(
        [
            '{"properties":{"issues":[]}}',
            '{"issues":[{"report_quote":"实际报告句子","reason":"具体证据问题","evidence_ids":["C-1"],"source_quote":"实际证据句子"}]}',
        ]
    )

    def generate(prompt, *a, **kw):
        prompts.append(prompt)
        return next(outputs)

    monkeypatch.setattr(model, "generate", generate)
    review = model.json("复核报告", Review)
    assert review.issues[0].reason == "具体证据问题"
    assert '"properties"' not in prompts[1]
    assert "复核报告" in prompts[1]
    assert "从原始证据重新生成" in prompts[1]
    assert len(list((tmp_path / "traces").glob("*.json"))) == 2


def test_unsupported_alternative_is_omitted_not_fabricated(tmp_path, monkeypatch):
    from chest_agent.schemas import Report
    import json

    monkeypatch.setattr(config, "DATA", tmp_path)
    model = ModelBackend()
    payload = {
        "assessment": "有限",
        "most_likely": {
            "name": "候选",
            "support": [{"text": "患者资料", "evidence_ids": ["C-1"]}],
        },
        "differentials": [{"name": "没有证据的候选", "support": []}],
    }
    monkeypatch.setattr(model, "generate", lambda *a, **kw: json.dumps(payload))
    result = model.json("分析", Report)
    assert not result.differentials
    assert result.most_likely.support[0].evidence_ids == ["C-1"]


def test_structured_checks_in_legacy_field_preserve_real_citations(
    tmp_path, monkeypatch
):
    from chest_agent.schemas import Report
    import json

    monkeypatch.setattr(config, "DATA", tmp_path)
    model = ModelBackend()
    payload = {
        "assessment": "有限",
        "most_likely": {
            "name": "候选",
            "support": [{"text": "资料", "evidence_ids": ["C-1"]}],
        },
        "next_checks": [
            {"name": "诱导痰PCR", "purpose": "确认病原体", "evidence_ids": ["K-2"]}
        ],
    }
    monkeypatch.setattr(model, "generate", lambda *a, **kw: json.dumps(payload))
    result = model.json("分析", Report)
    assert result.next_checks == ["诱导痰PCR"]
    assert result.recommended_checks[0].evidence_ids == ["K-2"]
