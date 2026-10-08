"""Audit fixed inputs, exact grading, paired outcomes and report limitations."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.run_full_report_iteration import deterministic_issues
from scripts.build_medical_comparison import paired
from scripts.evaluate_http import summarize


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


def pair(first,second):
    # The original paired helper uses 'variant'; retain evaluator modes via temporary copies.
    a={'records':[{**r,'variant':r['mode']} for r in first]}
    b={'records':[{**r,'variant':r['mode']} for r in second]}
    out=paired(a,b,first[0]['mode'],second[0]['mode'])
    n=out['first_only_correct']+out['second_only_correct'];k=min(out['first_only_correct'],out['second_only_correct'])
    out['mcnemar_exact_two_sided_p']=min(1,2*sum(math.comb(n,i) for i in range(k+1))/2**n)
    out['changed_patients']=[{'case_id':x['case_id'],'first_correct':x['correct'],
        'second_correct':next(y['correct'] for y in second if y['case_id']==x['case_id'])}
        for x in first if x['correct']!=next(y['correct'] for y in second if y['case_id']==x['case_id'])]
    return out


def main():
    root=ROOT/'data/quality_v4';manifest=root/'test20/manifest.json'
    result=json.loads((ROOT/'full_report_quality_validation.json').read_text())
    baseline=checked(root/'baseline_test20.json',manifest,['direct','verified'])
    groups={mode:[r for r in baseline['records'] if r['mode']==mode] for mode in ['direct','verified']}
    if result['development_selection']['clinical_contrast_enabled']:
        candidate=checked(root/'candidate_test20.json',manifest,['verified'])
        groups['candidate_verified']=candidate['records']
    audit={'same_history_questions_and_image_pixels':True,
        'complete_tasks_and_grading_checked':True,'clinical_expert_scoring_performed':False,
        'report_quality':{k:quality(v) for k,v in groups.items()},
        'paired':{'direct_vs_baseline_verified':pair(groups['direct'],groups['verified'])}}
    if 'candidate_verified' in groups:
        audit['paired']['baseline_vs_candidate_verified']=pair(groups['verified'],groups['candidate_verified'])
    result['audit']=audit
    result['archives']={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ['baseline_pipeline.zip','candidate_pipeline.zip']}
    result['audit_limits']=['Source and citation checks prove provenance, not truth of the diagnosis or image interpretation.',
        'needs_review counts include unfinished or ungrounded model review; distinguish them from deterministic report errors.',
        'Full reports answer published MCQs with additional original patient history; do not pool with earlier minimal MCQ scores.']
    (ROOT/'full_report_quality_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
