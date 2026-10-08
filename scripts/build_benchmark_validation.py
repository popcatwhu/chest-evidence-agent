"""Build an auditable scorecard without changing predictions or reference labels."""
import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_http import summarize

ROOT=Path(__file__).resolve().parent.parent


def build(manifest_path,results_path,output_path):
    manifest=json.loads(manifest_path.read_text())
    results=json.loads(results_path.read_text())
    assert results['frozen']['manifest_sha256']==hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    examples={e['case_id']:e for e in manifest['examples']}
    assert len(examples)==len(manifest['examples'])
    assert not set(examples).intersection(manifest['excluded_development_case_ids'])
    records=results['records'];seen=set()
    for row in records:
        key=(row['case_id'],row['mode'])
        assert key not in seen;seen.add(key)
        example=examples[row['case_id']]
        assert row['question_id']==example['question_id']
        assert row['reference_answer']==example['reference_answer']
        assert row['correct']==(row['status']=='completed' and row['prediction']==row['reference_answer'])
    complete=len(records)==len(examples)*len(results['frozen']['modes'])
    if not complete:raise ValueError('Refusing to publish a final scorecard for an incomplete run')
    score=summarize(records,results['frozen']['modes'])
    for mode,summary in score['modes'].items():
        rows=[r for r in records if r['mode']==mode]
        valid=[r['run']['result'] for r in rows if r['run'].get('result')]
        summary['status_counts']=dict(collections.Counter(r['status'] for r in rows))
        summary['completed_but_missing_choice']=sum(r['status']=='completed' and r['prediction'] is None for r in rows)
        summary['completed_but_invalid_choice']=sum(r['status']=='completed' and r['prediction'] is not None and r['prediction'] not in list('ABCDEF') for r in rows)
        summary['median_seconds_completed_only']=round(statistics.median(r['metrics']['elapsed_seconds'] for r in valid),2) if valid else None
        summary['verification_counts']=dict(collections.Counter(r['verification']['status'] for r in valid))
    validation={'task':'public_patient_disjoint_multiple_choice_sample','patients':len(examples),'tasks':len(records),
        'complete':True,'scoring':'Exact answer_choice letter; failed tasks and missing letters count as incorrect.',
        'frozen':results['frozen'],'summary':score,
        'model_names':sorted({r['run']['result']['model'] for r in records if r['run'].get('result')}),
        'sampled_gpu_peak_mib':max(r['sampled_gpu_peak_mib'] for r in records),
        'sampled_gpu_peak_gib':round(max(r['sampled_gpu_peak_mib'] for r in records)/1024,2),
        'cases':[{'case_id':cid,'reference_answer':examples[cid]['reference_answer'],
                  'predictions':{r['mode']:{'choice':r['prediction'],'correct':r['correct'],'status':r['status'],
                       'run_id':r['run']['id'],'error':r['run'].get('error')} for r in records if r['case_id']==cid}}
                 for cid in examples],
        'raw_results':str(results_path.relative_to(ROOT)),
        'clinical_accuracy_validated':False,'clinical_expert_reviewed':False,
        'limitations':manifest['limitations']+['Results evaluate the frozen pipeline, not later code changes.',
            'Errors include output-contract failures; scores cannot isolate medical reasoning ability.',
            'Model observations and review pass/fail do not establish diagnostic correctness.']}
    output_path.write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    print(json.dumps(validation['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    # Support both module and direct-script execution.
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=ROOT/'data/independent/manifest.json')
    parser.add_argument('--results',type=Path,default=ROOT/'data/independent/results.json')
    parser.add_argument('--output',type=Path,default=ROOT/'independent_validation.json')
    args=parser.parse_args()
    build(args.manifest,args.results,args.output)
