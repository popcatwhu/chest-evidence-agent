"""Select a clinical contrast on development data, then unseal one frozen report test."""
import argparse
from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import statistics
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent.parent
os.chdir(ROOT);sys.path.insert(0,str(ROOT))


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(path)


def deterministic_issues(run):
    from chest_agent.schemas import Report,Evidence,CaseProfile,validate_evidence
    from chest_agent.quality import known_test_conflicts
    from chest_agent.recommendations import check_recommendations
    from chest_agent.clinical_consistency import clinical_inconsistencies
    from chest_agent.diagnosis_checks import diagnosis_issues
    if not run.get('result'):return ['Report did not complete']
    result=run['result'];report=Report.model_validate(result['report'])
    evidence=[Evidence.model_validate(e) for e in result['evidence']]
    profile=CaseProfile.model_validate(result.get('profile',{'facts':[]}))
    return list(dict.fromkeys(validate_evidence(report,evidence,allow_original=result['reasoning_reads_original_image'])
        +known_test_conflicts(report,profile)+check_recommendations(report,evidence,require_sources=run['mode']!='direct')
        +clinical_inconsistencies(report,evidence)+diagnosis_issues(report)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--paused-service-pid',type=int,required=True)
    args=parser.parse_args()
    from chest_agent import config,store
    from chest_agent.workflow import execute
    from scripts.run_quality_v3 import summarize as clinical_summary
    from scripts.evaluate_http import summarize,fingerprint
    r=config.DATA/'quality_v4';store.init_db();config.CLINICAL_CONTRAST=True
    with zipfile.ZipFile(r/'candidate_pipeline.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in [ROOT/'chest_agent',config.DATA/'knowledge']:
            for file in folder.glob('*.py' if folder.name=='chest_agent' else '*.json'):
                z.write(file,str(file.relative_to(ROOT)))
        z.write(r/'policy.json','policy.json')
    sys.argv=['clinical_model_regression.py','--output',str(r/'candidate_clinical.json')]
    runpy.run_path(str(ROOT/'scripts/clinical_model_regression.py'),run_name='__main__')
    manifest=json.loads((config.DATA/'independent/manifest.json').read_text());records=[]
    for example in manifest['examples']:
        image=config.DATA/example['image_path']
        assert hashlib.sha256(image.read_bytes()).hexdigest()==example['image_sha256']
        case=store.create_case(example['title']+' · 临床对照开发',example['context'],str(image))
        rid=store.create_run(case['id'],example['question'],'verified');execute(rid);run=store.get_run(rid)
        prediction=run['result']['report']['answer_choice'] if run.get('result') else None
        records.append({'case_id':example['case_id'],'question_id':example['question_id'],'mode':'verified',
            'status':run['status'],'prediction':prediction,'reference_answer':example['reference_answer'],
            'correct':run['status']=='completed' and prediction==example['reference_answer'],'run':run})
        write(r/'candidate_dev12.json',{'records':records,'summary':summarize(records,['verified']),
            'complete':len(records)==len(manifest['examples']),'clinical_accuracy_validated':False})
        print('Development',example['case_id'],run['status'],bool(records[-1]['correct']),flush=True)
    baseline_clinical=clinical_summary(config.DATA/'quality_v3/clinical_legacy.json')
    candidate_clinical=clinical_summary(r/'candidate_clinical.json')
    baseline_dev=json.loads((r/'baseline_dev12.json').read_text())
    baseline_correct=baseline_dev['summary']['modes']['verified']['correct']
    candidate_correct=sum(x['correct'] for x in records)
    old_runs=json.loads((config.DATA/'quality_v3/clinical_legacy.json').read_text())['records']
    new_runs=json.loads((r/'candidate_clinical.json').read_text())['records']
    old_issues=sum(len(deterministic_issues(x['run'])) for x in old_runs)+sum(len(deterministic_issues(x['run'])) for x in baseline_dev['records'])
    new_issues=sum(len(deterministic_issues(x['run'])) for x in new_runs)+sum(len(deterministic_issues(x['run'])) for x in records)
    old_score=(baseline_clinical['specific_matches'],-old_issues,baseline_correct)
    new_score=(candidate_clinical['specific_matches'],-new_issues,candidate_correct)
    promoted=(candidate_clinical['specific_matches']>=baseline_clinical['specific_matches'] and
              candidate_correct>=baseline_correct and new_score>old_score)
    decision={'clinical_contrast_enabled':promoted,'baseline_specific_matches':baseline_clinical['specific_matches'],
        'candidate_specific_matches':candidate_clinical['specific_matches'],
        'baseline_development_choices_correct':baseline_correct,'candidate_development_choices_correct':candidate_correct,
        'baseline_deterministic_report_issues':old_issues,'candidate_deterministic_report_issues':new_issues,
        'development_baseline':baseline_clinical,'development_candidate':candidate_clinical,
        'policy':json.loads((r/'policy.json').read_text()),'test_outcomes_used_for_selection':False}
    # This decision is committed before reading any held-out baseline grading.
    write(r/'deployment_decision.json',decision)
    print('Development selection locked:',promoted,flush=True)
    config.CLINICAL_CONTRAST=promoted
    if promoted:
        held=json.loads((r/'test20/manifest.json').read_text());tests=[]
        frozen={'manifest_sha256':hashlib.sha256((r/'test20/manifest.json').read_bytes()).hexdigest(),
            'pipeline_sha256':fingerprint((ROOT/'chest_agent').glob('*.py')),
            'knowledge_sha256':fingerprint((config.DATA/'knowledge').glob('*.json')),
            'clinical_contrast':True,'modes':['verified']}
        for example in held['examples']:
            image=config.DATA/example['image_path'];assert hashlib.sha256(image.read_bytes()).hexdigest()==example['image_sha256']
            case=store.create_case(example['title']+' · 完整报告对照',example['context'],str(image))
            rid=store.create_run(case['id'],example['question'],'verified');execute(rid);run=store.get_run(rid)
            prediction=run['result']['report']['answer_choice'] if run.get('result') else None
            tests.append({'case_id':example['case_id'],'question_id':example['question_id'],'mode':'verified',
                'status':run['status'],'prediction':prediction,'reference_answer':example['reference_answer'],
                'correct':run['status']=='completed' and prediction==example['reference_answer'],'run':run})
            write(r/'candidate_test20.json',{'records':tests,'summary':summarize(tests,['verified']),
                'frozen':frozen,'complete':len(tests)==len(held['examples']),'clinical_accuracy_validated':False})
            print('Frozen test progress:',len(tests),'/20',flush=True)
    baseline_test=json.loads((r/'baseline_test20.json').read_text())
    result={'task':'full_report_quality_v4','clinical_accuracy_validated':False,'development_selection':decision,
        'frozen_test_patients':20,'test_baseline':baseline_test['summary'],
        'test_candidate':json.loads((r/'candidate_test20.json').read_text())['summary'] if promoted else None,
        'test_manifest_sha256':hashlib.sha256((r/'test20/manifest.json').read_bytes()).hexdigest(),
        'limitations':['Exact choice scoring evaluates diagnostic MCQ conclusions within full reports, not clinical expert correctness.',
            'Public pretraining contamination unknown; known6 open-case name matches are development evidence.']}
    write(ROOT/'full_report_quality_validation.json',result)
    path=Path(f'/proc/{args.paused_service_pid}/cmdline')
    if path.exists():
        assert b'uvicorn' in path.read_bytes() and b'chest_agent.app:app' in path.read_bytes()
        os.kill(args.paused_service_pid,signal.SIGTERM)
        for _ in range(100):
            if not path.exists():break
            time.sleep(.1)
    config.INFERENCE_PAUSED=False
    print('Restoring resident app; clinical contrast:',promoted,flush=True)
    import uvicorn
    uvicorn.run('chest_agent.app:app',host='127.0.0.1',port=7860)


if __name__=='__main__':main()
