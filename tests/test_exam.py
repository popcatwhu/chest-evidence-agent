import pytest
from pydantic import ValidationError
from chest_agent import exam


def test_exam_protocol_does_not_inherit_report_instructions_or_labels(monkeypatch):
    captured=[]
    def respond(prompt,schema,image,**kwargs):
        captured.append((prompt,schema,image))
        return exam.ExamAnswer(reason='Visible opacity matches option B.',answer_choice='B')
    monkeypatch.setattr(exam.backend,'json',respond)
    answer=exam.answer('A) first\nB) second','image.png')
    assert answer.answer_choice=='B'
    assert captured[0][1] is exam.ExamAnswer
    assert 'clinical report' in captured[0][0]
    assert 'recommended_checks' not in captured[0][0]
    with pytest.raises(ValidationError):exam.ExamAnswer(reason='x',answer_choice=None)


def test_validator_is_question_blind_and_does_not_accept_unknown_or_duplicate_claims(monkeypatch):
    captured=[]
    def respond(prompt,schema,image,**kwargs):
        captured.append(prompt)
        return exam.ToolChecks(checks=[
            exam.ToolCheck(claim_id='F-1',status='supported',visual_basis='Visible'),
            exam.ToolCheck(claim_id='F-1',status='supported',visual_basis='Visible'),
            exam.ToolCheck(claim_id='F-2',status='contradicted',visual_basis='Pleural line visible'),
            exam.ToolCheck(claim_id='invented',status='supported',visual_basis='Visible')])
    monkeypatch.setattr(exam.backend,'json',respond)
    result=exam.validate_claims([{'claim_id':'F-1','text':'clear lungs'},
        {'claim_id':'F-2','text':'no pneumothorax'},{'claim_id':'F-3','text':'heart size normal'}],'image.png')
    assert result['accepted']==[]
    assert {r['claim_id'] for r in result['checks']}=={'F-2'}
    assert any(r['claim_id']=='F-3' for r in result['invalid'])
    assert 'Question:' not in captured[0]


def test_validated_prompt_filters_tool_claims_but_keeps_classifier_control_constant(monkeypatch):
    prompts=[]
    monkeypatch.setattr(exam.backend,'json',lambda prompt,*a,**k:prompts.append(prompt) or exam.ExamAnswer(reason='x',answer_choice='A'))
    tools={'scores':{'Mass':0.6},'claims':[{'claim_id':'F-1','text':'wrong negative'},
                                         {'claim_id':'F-2','text':'supported opacity'}]}
    exam.answer('A) first\nB) second','image.png','tools',tools)
    exam.answer('A) first\nB) second','image.png','validated',tools,{'accepted':[tools['claims'][1]]})
    assert 'wrong negative' in prompts[0] and 'wrong negative' not in prompts[1]
    assert 'Mass=0.600' in prompts[0] and 'Mass=0.600' in prompts[1]
    assert 'supported opacity' in prompts[1]
