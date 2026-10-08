import importlib.util
from pathlib import Path
import sys
import pytest
from chest_agent import config,store
from chest_agent.schemas import CaseProfile, Observation, Plan, Report, Review

from chest_agent import clinical_contrast as contrast, workflow as wf


def report():
    return Report.model_validate({'most_likely':{'name':'候选A','support':[{'text':'症状已提供','evidence_ids':['C-1']}]}})


def test_enabled_contrast_reaches_report_and_is_saved(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'CXR_EXPERT_ENABLED',False)
    monkeypatch.setattr(store,'DB',tmp_path/'test.sqlite3');store.init_db()
    case=store.create_case('test','reported fever');rid=store.create_run(case['id'],'分析','verified')
    prompts=[]
    def reply(prompt,schema,*args,**kwargs):
        if schema is CaseProfile:return CaseProfile(facts=[])
        if schema is Plan:return Plan(query='fever',reason='分析')
        if schema is contrast.ClinicalContrast:
            return contrast.ClinicalContrast.model_validate({'key_case_clues':[{'name':'fever','value':'fever',
                'source_id':'C-1','source_quote':'reported fever'}],'hypotheses':[
                {'name':name,'support':[{'text':'症状已提供','evidence_ids':['C-1']}]} for name in ['候选A','候选B']]})
        if schema is Report:prompts.append(prompt);return report()
        if schema is Review:return Review(issues=[])
        raise AssertionError(schema)
    monkeypatch.setattr(wf.backend,'json',reply);monkeypatch.setattr(wf,'retrieve',lambda q:[])
    wf.execute(rid);run=store.get_run(rid)
    assert run['status']=='completed'
    assert run['result']['clinical_contrast_enabled']
    assert len(run['result']['clinical_contrast']['hypotheses'])==2
    assert '不是新增患者事实' in prompts[0]
    assert run['result']['verification']['status']=='passed_checks'


def test_failed_contrast_is_visible_without_losing_valid_report(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'DB',tmp_path/'test.sqlite3');store.init_db()
    case=store.create_case('test','reported fever');rid=store.create_run(case['id'],'分析','verified')
    def reply(prompt,schema,*args,**kwargs):
        if schema is CaseProfile:return CaseProfile(facts=[])
        if schema is Plan:return Plan(query='fever',reason='分析')
        if schema is contrast.ClinicalContrast:raise ValueError('invalid matrix')
        if schema is Report:return report()
        if schema is Review:return Review(issues=[])
    monkeypatch.setattr(wf.backend,'json',reply);monkeypatch.setattr(wf,'retrieve',lambda q:[])
    wf.execute(rid);run=store.get_run(rid)
    assert run['status']=='completed'
    assert run['result']['clinical_contrast'] is None
    assert run['result']['verification']['status']=='needs_review'
    assert any('临床证据对照未完成' in n for n in run['result']['verification']['notes'])


def test_comparison_modes_do_not_call_contrast_model(monkeypatch):
    monkeypatch.setattr(wf.backend,'json',lambda *a,**k:pytest.fail('Unexpected extra call'))
    for mode in ['direct','tools']:
        assert wf.contrast({'mode':mode})['clinical_contrast'] is None
