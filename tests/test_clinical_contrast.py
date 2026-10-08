import importlib.util
from pathlib import Path
from chest_agent.schemas import Evidence

from chest_agent import clinical_contrast as m


def hypothesis(name='感染',ref='C-1'):
    return {'name':name,'support':[{'text':'Fever reported','evidence_ids':[ref]}]}


def test_knowledge_cannot_be_promoted_to_patient_clue():
    evidence=[Evidence(id='K-1',kind='knowledge',title='资料',content='fever is typical')]
    raw=m.ClinicalContrast.model_validate({'key_case_clues':[{'name':'fever','value':'present',
        'source_id':'K-1','source_quote':'fever is typical'}]})
    grounded,notes=m.ground_contrast(raw,evidence)
    assert not grounded.key_case_clues
    assert any('原文' in n for n in notes)


def test_fabricated_numeric_clue_is_rejected():
    evidence=[Evidence(id='C-1',kind='case',title='资料',content='CD4 count10 cells/mm3')]
    raw=m.ClinicalContrast.model_validate({'key_case_clues':[{'name':'CD4','value':'100',
        'source_id':'C-1','source_quote':'CD4 count10 cells/mm3'}]})
    grounded,notes=m.ground_contrast(raw,evidence)
    assert not grounded.key_case_clues


def test_failed_tool_and_repeated_candidate_are_not_valid_contrasts():
    evidence=[Evidence(id='C-1',kind='case',title='资料',content='fever'),
        Evidence(id='I-failed',kind='image',title='工具',content='none',status='failed')]
    raw=m.ClinicalContrast.model_validate({'hypotheses':[hypothesis('肺炎'),hypothesis('肺炎'),hypothesis('肿瘤','I-failed')]})
    grounded,notes=m.ground_contrast(raw,evidence)
    assert len(grounded.hypotheses)==1
    assert any('同义' in n for n in notes)
    assert any('失败' in n for n in notes)


def test_conflict_requires_actual_quotes_and_distinct_sources():
    evidence=[Evidence(id='I-a',kind='image',title='观察',content='{"observations":["no shift"]}'),
        Evidence(id='I-b',kind='image',title='观察',content='rightward shift')]
    raw=m.ClinicalContrast.model_validate({'observation_conflicts':[{'first_source':'I-a',
        'first_quote':'no shift','second_source':'I-b','second_quote':'rightward shift','description':'shift disagreement'}]})
    grounded,_=m.ground_contrast(raw,evidence)
    assert len(grounded.observation_conflicts)==1
    raw.observation_conflicts[0].second_quote='invented shift'
    grounded,notes=m.ground_contrast(raw,evidence)
    assert not grounded.observation_conflicts
    assert any('无法匹配' in n for n in notes)


def test_distinct_subtypes_remain_hypotheses_without_overwriting_diagnosis():
    evidence=[Evidence(id='C-1',kind='case',title='资料',content='recurrent episodes')]
    raw=m.ClinicalContrast.model_validate({'hypotheses':[hypothesis('原发性气胸'),hypothesis('月经相关气胸')]})
    grounded,notes=m.ground_contrast(raw,evidence)
    assert len(grounded.hypotheses)==2
    assert not notes


def test_valid_literal_clue_is_retained_and_paraphrase_not_labeled_fact():
    evidence=[Evidence(id='C-1',kind='case',title='资料',content='Episodes every30-40days, followed by menstruation.')]
    raw=m.ClinicalContrast.model_validate({'key_case_clues':[{'name':'recurrence','value':'every30-40days',
        'source_id':'C-1','source_quote':'Episodes every30-40days'}]})
    grounded,notes=m.ground_contrast(raw,evidence)
    assert len(grounded.key_case_clues)==1
    raw.key_case_clues[0].value='onset after menstruation'
    grounded,notes=m.ground_contrast(raw,evidence)
    assert not grounded.key_case_clues
    assert any('直接片段' in n for n in notes)
