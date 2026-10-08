import json
import pytest
from lmformatenforcer import JsonSchemaParser
from pydantic import ValidationError
from chest_agent import config, workflow
from chest_agent.llm import ModelBackend
from chest_agent.schemas import ChoiceReport, Report, report_schema


def payload():
    return {
        "answer_choice": "B",
        "most_likely": {
            "name": "candidate",
            "support": [{"text": "finding", "evidence_ids": ["I-original"]}],
        },
    }


def accepts(value):
    parser = JsonSchemaParser(ChoiceReport.model_json_schema())
    for char in json.dumps(value):
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


def test_mcq_requires_a_letter_but_open_diagnosis_does_not():
    assert report_schema("Which diagnosis?\nA) First\nB) Second") is ChoiceReport
    assert report_schema("AIDS患者，请分析胸片") is Report
    for missing in [None, "B)", "G"]:
        data = payload()
        data["answer_choice"] = missing
        with pytest.raises(ValidationError):
            ChoiceReport.model_validate(data)
    data = payload()
    del data["answer_choice"]
    with pytest.raises(ValidationError):
        ChoiceReport.model_validate(data)
    assert Report.model_validate(data).answer_choice is None


def test_decode_constraints_reject_reached_benchmark_format_failures():
    assert accepts(payload())
    data = payload()
    data["answer_choice"] = None
    assert not accepts(data)
    data = payload()
    data["most_likely"]["support"][0] = {
        "claim": "finding",
        "evidence_ids": ["I-original"],
    }
    assert not accepts(data)
    data = payload()
    data["differentials"] = [data["most_likely"]] * 3
    assert not accepts(data)


def test_backend_sends_schema_and_requires_mcq_letter(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    backend = ModelBackend()
    seen = []

    def generate(prompt, *args, **kwargs):
        seen.append((prompt, kwargs["json_schema"]))
        return json.dumps(payload())

    monkeypatch.setattr(backend, "generate", generate)
    assert backend.json("Choose an option", ChoiceReport).answer_choice == "B"
    assert "不能为null" in seen[0][0]
    assert "answer_choice" in seen[0][1]["required"]
    assert seen[0][1]["properties"]["answer_choice"]["enum"] == list("ABCDEF")


def test_choice_schema_is_kept_during_report_repair(monkeypatch):
    monkeypatch.setattr(config, "RECHECK_IMAGE", True)
    monkeypatch.setattr(workflow, "log", lambda *args: None)
    calls = []

    def generate(prompt, schema, image=None, **kwargs):
        calls.append((schema, image))
        return schema.model_validate(payload())

    monkeypatch.setattr(workflow.backend, "json", generate)
    state = {
        "mode": "verified",
        "case": {"image_path": "x.png"},
        "question": "A) first\nB) second",
        "evidence": [],
        "issues": [],
        "repairs": 0,
    }
    state.update(workflow.diagnose(state))
    workflow.repair(state)
    assert calls == [(ChoiceReport, "x.png"), (ChoiceReport, "x.png")]


def test_real_token_filter_allows_chinese_medical_strings():
    import torch
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    from chest_agent.constrained import schema_prefix_function

    text = '{"text":"肺孢子菌肺炎"}'
    chars = sorted(set(text))
    eos = len(chars)
    data = TokenEnforcerTokenizerData(
        [(i, c, False) for i, c in enumerate(chars)],
        lambda ids: "".join(chars[i] for i in ids if i < eos),
        eos,
        False,
        eos + 1,
    )
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    prefix = schema_prefix_function(data, schema)
    ids = [eos]
    for char in text:
        token = chars.index(char)
        assert token in prefix(0, torch.tensor(ids)), char
        ids.append(token)
    assert eos in prefix(0, torch.tensor(ids))
