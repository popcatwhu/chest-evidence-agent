import pytest
from chest_agent import store, workflow,config
from chest_agent.schemas import Plan, Report, Review, CaseProfile, Observation


@pytest.fixture(autouse=True)
def no_external_models_in_workflow_unit_tests(monkeypatch):
    monkeypatch.setattr(config,'CXR_EXPERT_ENABLED',False)
    monkeypatch.setattr(config,'CLINICAL_CONTRAST',False)


def make_report(ref):
    return Report.model_validate({'assessment':'有限', 'most_likely':{'name':'候选诊断',
        'support':[{'text':'需进一步评估','evidence_ids':[ref]}]}})


def test_failed_tool_triggers_repair_and_keeps_failure_visible(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB',tmp_path/'workflow.sqlite3')
    store.init_db()
    case=store.create_case('test','已报告症状','example.png')
    run=store.create_run(case['id'],'分析','verified')
    replies=iter([CaseProfile(facts=[]),Observation(observations=[],limitations=[]),Plan(tools=['classify'],query='pneumonia',reason='检查影像'),
                  make_report('I-classify'),Review(issues=[]),make_report('C-1'),Review(issues=[])])
    monkeypatch.setattr(workflow.backend,'json',lambda *a,**kw:next(replies))
    def fail(*args):
        raise RuntimeError('工具失败测试')
    monkeypatch.setattr(workflow.image_tools,'classify',fail)
    monkeypatch.setattr(workflow,'retrieve',lambda query:[])
    workflow.execute(run)
    result=store.get_run(run)
    assert result['status']=='completed'
    assert result['result']['metrics']['repairs']==1
    assert result['result']['verification']['issues']==[]
    assert any(e['status']=='failed' for e in result['result']['evidence'])


def test_unfixed_evidence_is_flagged_after_bounded_repair(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB',tmp_path/'workflow.sqlite3')
    store.init_db()
    case=store.create_case('test','已报告症状')
    run=store.create_run(case['id'],'分析','verified')
    replies=iter([CaseProfile(facts=[]),Plan(tools=[],query='pneumonia',reason='分析'),make_report('missing'),Review(issues=[]),
                  make_report('missing'),Review(issues=[])])
    monkeypatch.setattr(workflow.backend,'json',lambda *a,**kw:next(replies))
    monkeypatch.setattr(workflow,'retrieve',lambda query:[])
    workflow.execute(run)
    result=store.get_run(run)
    assert result['result']['verification']['status']=='needs_review'
    assert result['result']['metrics']['repairs']==1


def test_model_failure_is_persisted(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB',tmp_path/'workflow.sqlite3')
    store.init_db()
    case=store.create_case('test','资料')
    run=store.create_run(case['id'],'分析','direct')
    def fail(*a,**kw):
        raise ValueError('无效模型输出')
    monkeypatch.setattr(workflow.backend,'json',fail)
    workflow.execute(run)
    assert store.get_run(run)['status']=='failed'
    assert store.get_run(run)['result'] is None


def test_invalid_review_keeps_valid_draft_without_claiming_passed(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB',tmp_path/'workflow.sqlite3')
    store.init_db()
    case=store.create_case('test','已报告症状')
    run=store.create_run(case['id'],'分析','verified')
    def response(prompt,schema,*a,**kw):
        if schema is CaseProfile:return CaseProfile(facts=[])
        if schema is Plan:return Plan(tools=[],query='pneumonia',reason='分析')
        if schema is Report:return make_report('C-1')
        if schema is Review:raise ValueError('invalid reviewer output')
    monkeypatch.setattr(workflow.backend,'json',response)
    monkeypatch.setattr(workflow,'retrieve',lambda query:[])
    workflow.execute(run)
    result=store.get_run(run)
    assert result['status']=='completed'
    assert result['result']['verification']['status']=='needs_review'
    assert result['result']['verification']['notes']


def test_original_image_reaches_draft_review_and_repair(monkeypatch):
    monkeypatch.setattr(config,'CXR_EXPERT_ENABLED',True)
    monkeypatch.setattr(config,'RECHECK_IMAGE',True)
    calls=[]
    def response(prompt,schema,image=None,**kwargs):
        calls.append((schema,image))
        return Review(issues=[]) if schema is Review else make_report('I-original')
    monkeypatch.setattr(workflow.backend,'json',response)
    state={'mode':'verified','case':{'image_path':'chest.png'},'question':'分析',
           'evidence':[], 'profile':CaseProfile(facts=[]),'repairs':0}
    monkeypatch.setattr(workflow,'log',lambda *args:None)
    state.update(workflow.diagnose(state))
    state.update(workflow.verify(state))
    state.update(workflow.repair(state))
    assert calls==[(Report,'chest.png'),(Review,'chest.png'),(Report,'chest.png')]
    monkeypatch.setattr(config,'RECHECK_IMAGE',False)
    assert workflow.reasoning_image(state) is None
    assert '不能引用I-original' in workflow.report_prompt(state)
    state['mode']='direct'
    assert workflow.reasoning_image(state)=='chest.png'
