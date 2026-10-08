"""Export evaluator-only comparison material; no automatic clinical error grading."""
import argparse
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent


def options(question):
    return dict(re.findall(r'(?ms)^\s*([A-F])\)\s*(.*?)(?=^\s*[A-F]\)|\Z)',question))


def export(manifest_path,results_path,output_path):
    manifest=json.loads(manifest_path.read_text())
    results=json.loads(results_path.read_text())
    examples={e['case_id']:e for e in manifest['examples']}
    reviews=[]
    for record in results['records']:
        example=examples[record['case_id']]
        choices=options(example['question'])
        result=record['run'].get('result') or {}
        report=result.get('report') or {}
        reviews.append({'case_id':record['case_id'],'mode':record['mode'],'run_id':record['run']['id'],
            'status':record['status'],'correct_option':record['correct'],
            'selected_option':record['prediction'],'selected_option_text':choices.get(record['prediction']),
            'reference_option':record['reference_answer'],'reference_option_text':choices.get(record['reference_answer']),
            'evaluator_only_image_description':example['evaluator_only_caption'],
            'evaluator_only_explanation':example['evaluator_only_explanation'],
            'primary':report.get('most_likely'),'differentials':report.get('differentials'),
            'visual_model_observations':[e for e in result.get('evidence',[]) if e['id'] in {'I-observe','I-radiology'}],
            'classification_output':[e for e in result.get('evidence',[]) if e['id']=='I-classify'],
            'verification':result.get('verification'),
            'error':record['run'].get('error'),
            'review':{'error_stage':None,'description':None,'evidence':None,'reviewer':None}})
    output={'task':'multiple_choice','complete':results['complete'],'clinical_expert_reviewed':False,
        'instructions':'Evaluator-side only. Review stages: image_observation / reasoning_or_option_selection / evidence_use / structured_output / benchmark_ambiguity / undetermined. Do not infer clinical accuracy from option matches.',
        'records':reviews}
    # Never overwrite human annotations during re-export.
    if output_path.exists():
        old=json.loads(output_path.read_text())
        annotations={(r['case_id'],r['mode']):r['review'] for r in old['records']}
        for row in reviews:row['review']=annotations.get((row['case_id'],row['mode']),row['review'])
    output_path.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print('Exported',len(reviews),'evaluator-only rows to',output_path)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=ROOT/'data/independent/manifest.json')
    parser.add_argument('--results',type=Path,default=ROOT/'data/independent/results.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/independent/error_review.json')
    args=parser.parse_args()
    export(args.manifest,args.results,args.output)
