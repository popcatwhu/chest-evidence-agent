"""Audit fixed inputs, exact grading, paired outcomes and report limitations."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.evaluate_http import summarize


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


def checked(path,manifest,modes):
    from PIL import Image,ImageChops
    data=json.loads(path.read_text())
    if not data['complete']:raise ValueError('Experiment is incomplete')
    if data['frozen']['manifest_sha256']!=hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError('Input manifest changed')
    examples={e['case_id']:e for e in json.loads(manifest.read_text())['examples']}
    seen=set()
    for row in data['records']:
        key=(row['case_id'],row['mode'])
        if key in seen or row['mode'] not in modes:raise ValueError('Duplicate or unknown strategy')
        seen.add(key);example=examples[row['case_id']];run=row['run']
        if run['context']!=example['context'] or run['question']!=example['question']:
            raise ValueError('Strategies received different history or questions')
        actual=run['result']['report']['answer_choice'] if run.get('result') else None
        if row['prediction']!=actual or row['reference_answer']!=example['reference_answer']:
            raise ValueError('Prediction or reference was changed')
        if row['correct']!=(row['status']=='completed' and actual==example['reference_answer']):
            raise ValueError('Wrong grading; failures must remain wrong')
        if row['status']!=run['status']:raise ValueError('Terminal status changed')
        from chest_agent import store
        case=store.get_case(run['case_id'])
        if not case:raise ValueError('Original case snapshot is missing')
        with Image.open(ROOT/'data'/example['image_path']) as expected,Image.open(case['image_path']) as observed:
            first,second=expected.convert('RGB'),observed.convert('RGB')
            if first.size!=second.size or ImageChops.difference(first,second).getbbox():
                raise ValueError('Image pixels differ between strategies')
    if len(seen)!=len(examples)*len(modes):raise ValueError('Missing fixed tasks')
    if data['summary']!=summarize(data['records'],modes):
        raise ValueError('Reported summary differs from actual fixed tasks')
    return data


def quality(rows):
    completed=[r['run'] for r in rows if r['run'].get('result')]
    result={'tasks':len(rows),'completed':len(completed),
        'failed_or_interrupted':len(rows)-len(completed),
        'terminal_errors':[{'case_id':r['case_id'],'error':r['run']['error']} for r in rows if not r['run'].get('result')],
        'deterministic_report_issues':sum(len(deterministic_issues(r)) for r in completed),
        'needs_review_reports':sum(r['result']['verification']['status']=='needs_review' for r in completed),
        'median_report_seconds':statistics.median(r['result']['metrics']['elapsed_seconds'] for r in completed) if completed else None,
        'gpu_peak_reserved_gib':max((r['result']['metrics']['gpu_peak_reserved_gib'] or 0 for r in completed),default=0),
        'usable_contrasts':sum(bool(r['result'].get('clinical_contrast')) for r in completed),
        'literal_clues_retained':sum(len((r['result'].get('clinical_contrast') or {}).get('key_case_clues',[])) for r in completed),
        'quoted_observation_conflicts':sum(len((r['result'].get('clinical_contrast') or {}).get('observation_conflicts',[])) for r in completed)}
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=checked(args.results,args.manifest,['verified'])
    audit={'complete_tasks_and_grading_checked':True,
           'clinical_expert_scoring_performed':False,
           'test':data['summary']['modes']['verified'],
           'report_quality':quality(data['records'])}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print('Audit saved:',args.output)


if __name__=='__main__':main()
