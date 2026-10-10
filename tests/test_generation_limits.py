import json

import pytest
import torch
from lmformatenforcer import JsonSchemaParser
from pydantic import ValidationError

from chest_agent import config
from chest_agent.generation import RepetitionStop, repeated_suffix
from chest_agent.llm import ModelBackend
from chest_agent.schemas import Candidate, ChoiceReport


def report():
    return {
        "answer_choice": "C",
        "most_likely": {
            "name": "候选诊断",
            "support": [{"text": "已提供症状", "evidence_ids": ["C-1"]}],
        },
    }


def test_decode_closes_diagnosis_name_at_length_bound():
    parser = JsonSchemaParser(ChoiceReport.model_json_schema())
    prefix = '{"answer_choice":"C","most_likely":{"name":"candidate '
    for char in prefix:
        assert char in parser.get_allowed_characters()
        parser = parser.add_character(char)
    for _ in range(120 - len("candidate ")):
        parser = parser.add_character("a")
    assert "a" not in parser.get_allowed_characters()
    assert '"' in parser.get_allowed_characters()


@pytest.mark.parametrize("name", ["x" * 121, "disease [K-2]", ""])
def test_invalid_name_is_rejected_without_silent_trimming(name):
    with pytest.raises(ValidationError):
        Candidate.model_validate(
            {"name": name, "support": [{"text": "finding", "evidence_ids": ["C-1"]}]}
        )


def test_loop_detector_preserves_short_repeats_and_ignores_prompt():
    pattern = list(range(20))
    assert repeated_suffix(pattern * 6)
    assert not repeated_suffix(pattern * 3)
    assert not repeated_suffix(list(range(192)))
    prompt = pattern * 8
    stop = RepetitionStop(len(prompt))
    assert not stop(torch.tensor([prompt + [1, 2, 3]]), None).item()
    assert stop(torch.tensor([prompt + list(range(16)) * 6]), None).item()


def test_bounded_name_constraint_preserves_chinese_generation():
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    from chest_agent.constrained import schema_prefix_function

    text = '{"name":"肺孢子菌肺炎"}'
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
        "properties": {"name": Candidate.model_json_schema()["properties"]["name"]},
        "required": ["name"],
        "additionalProperties": False,
    }
    prefix = schema_prefix_function(data, schema)
    ids = [eos]
    for char in text:
        token = chars.index(char)
        assert token in prefix(0, torch.tensor(ids))
        ids.append(token)
    assert eos in prefix(0, torch.tensor(ids))


def test_retry_discards_looping_partial_output_and_keeps_original_task(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "DATA", tmp_path)
    model = ModelBackend()
    prompts = []
    bad = '{"most_likely":{"name":"' + "[K-2] " * 100
    outputs = iter([bad, json.dumps(report(), ensure_ascii=False)])

    def generate(prompt, *args, **kwargs):
        prompts.append(prompt)
        model.metrics.append({"repetition_stopped": len(prompts) == 1})
        return next(outputs)

    monkeypatch.setattr(model, "generate", generate)
    result = model.json("病例原文必须保留", ChoiceReport)
    assert result.answer_choice == "C"
    assert len(prompts) == 2
    assert bad not in prompts[1]
    assert "病例原文必须保留" in prompts[1]
    assert "不能续写上次输出" in prompts[1]
