import json
from chest_agent import workflow,config
from chest_agent.schemas import Evidence,Plan,Report,CaseProfile


def test_search_retains_history_and_adds_plan_and_both_observers(monkeypatch):
    captured=[]
    monkeypatch.setattr(workflow,'log',lambda *args:None)
    monkeypatch.setattr(workflow,'retrieve',lambda query:captured.append(query) or [])
    state={'mode':'verified','evidence':[
        Evidence(id='C-1',kind='case',title='病史',content='recurrent episodes after menstruation'),
        Evidence(id='I-observe',kind='image',title='观察',content=json.dumps({'observations':['peripheral lucency']})),
        Evidence(id='I-radiology',kind='image',title='观察',content=json.dumps({'visual_findings':['pleural line']}))],
        'plan':Plan(query='pneumothorax differential',reason='比较',hypotheses=['catamenial pneumothorax'])}
    workflow.search(state)
    assert all(term in captured[0] for term in ['after menstruation','pneumothorax differential',
        'catamenial pneumothorax','peripheral lucency','pleural line'])


def test_report_never_requests_unsupported_checks_in_direct_mode():
    state={'mode':'direct','case':{'image_path':None},'question':'分析','evidence':[],
           'profile':CaseProfile(facts=[])}
    assert '没有相关K资料' in workflow.report_prompt(state)


def test_reader_identity_changes_tool_cache_key(tmp_path,monkeypatch):
    from chest_agent import exam
    folder=tmp_path/'reader';folder.mkdir()
    (folder/'config.json').write_text('{}');(folder/'source_revision.json').write_text('{}')
    monkeypatch.setattr(config,'NVREASON_MODEL',folder)
    monkeypatch.setattr(config,'CXR_FINDINGS_MODEL',folder)
    monkeypatch.setattr(config,'CXR_READER','iamjb');original=exam.tool_signature()
    monkeypatch.setattr(config,'CXR_READER','nvreason');assert exam.tool_signature()!=original
