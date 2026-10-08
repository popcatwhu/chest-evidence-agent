import pytest
from chest_agent.llm import parse_json


def test_missing_list_bracket_preserves_patient_text_and_values():
    raw = '{"most_likely":{"name":"候选"},"differentials":[{"name":"另一候选"},"findings":[],"CD4":6}'
    result = parse_json(
        raw, root_keys=["most_likely", "differentials", "findings", "CD4"]
    )
    assert result["most_likely"]["name"] == "候选"
    assert result["differentials"][0]["name"] == "另一候选"
    assert result["CD4"] == 6


def test_truncated_report_cannot_be_replaced_with_valid_nested_candidate():
    with pytest.raises(ValueError, match="截断"):
        parse_json('{"most_likely":{"name":"候选","support":[]}')


def test_missing_value_cannot_be_invented_by_syntax_repair():
    with pytest.raises(ValueError):
        parse_json('{"CD4":,"name":"患者"}')
