from chest_agent.workflow import update_report
from chest_agent.schemas import Report, Evidence
from chest_agent import workflow


def test_reanalysis_without_new_information_does_not_fabricate_change(monkeypatch):
    output = Report.model_validate(
        {
            "assessment": "有限",
            "most_likely": {
                "name": "当前候选",
                "support": [{"text": "症状", "evidence_ids": ["C-1"]}],
            },
        }
    )
    state = {
        "report": output,
        "previous": {"most_likely": {"name": "上次候选"}},
        "previous_context": "原始病史",
        "evidence": [Evidence(id="C-1", kind="case", title="病例", content="原始病史")],
    }
    monkeypatch.setattr(workflow, "log", lambda *a: None)

    def should_not_call(*a, **kw):
        raise AssertionError("没有新资料时不应生成不存在的新证据")

    monkeypatch.setattr(workflow.backend, "json", should_not_call)
    result = update_report(state)
    assert "没有新增病例资料" in result["report"].change_summary
