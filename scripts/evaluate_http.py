"""Resumable, paired MCQ evaluation against the resident GPU service.

Reference fields are never sent to the service. Failures remain in the accuracy
denominator. Exact option grading is separate from report/verification success.
"""
import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent.parent


def fingerprint(paths):
    digest=hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def interval(correct,total):
    if not total:return None
    z=1.96;p=correct/total;denom=1+z*z/total
    center=(p+z*z/(2*total))/denom
    width=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return [round(center-width,4),round(center+width,4)]


def summarize(records,modes):
    summary={}
    for mode in modes:
        rows=[r for r in records if r['mode']==mode]
        n=len(rows);correct=sum(r['correct'] for r in rows)
        summary[mode]={'tasks':n,'completed':sum(r['status']=='completed' for r in rows),
            'correct':correct,'accuracy':correct/n if n else None,'wilson_95_ci':interval(correct,n),
            'invalid_or_missing_choices':sum(r['prediction'] not in 'ABCDEF' if isinstance(r['prediction'],str) and len(r['prediction'])==1 else True for r in rows)}
    paired={}
    by_id={}
    for row in records:by_id.setdefault(row['case_id'],{})[row['mode']]=row
    if 'direct' in modes and 'verified' in modes:
        pairs=[r for r in by_id.values() if 'direct' in r and 'verified' in r]
        paired={'patients':len(pairs),'both_correct':sum(r['direct']['correct'] and r['verified']['correct'] for r in pairs),
                'direct_only_correct':sum(r['direct']['correct'] and not r['verified']['correct'] for r in pairs),
                'verified_only_correct':sum(r['verified']['correct'] and not r['direct']['correct'] for r in pairs),
                'both_wrong':sum(not r['direct']['correct'] and not r['verified']['correct'] for r in pairs)}
    return {'modes':summary,'paired':paired}


def evaluate(args):
    manifest=json.loads(args.manifest.read_text())
    health=requests.get(args.base+'/api/health',timeout=15).json()
    health.pop('model_loaded',None)  # Runtime warm-up state is not configuration.
    frozen={'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            'pipeline_sha256':fingerprint((ROOT/'chest_agent').glob('*.py')),
            'knowledge_sha256':fingerprint((ROOT/'data/knowledge').glob('*.json')),
            'modes':args.modes,'health':health}
    records=[]
    pending=None
    if args.output.exists():
        previous=json.loads(args.output.read_text())
        if previous['frozen']!=frozen:raise ValueError('Cannot resume with changed inputs, pipeline, knowledge, modes or service configuration')
        records=previous['records']
        pending=previous.get('pending')
    completed={(r['case_id'],r['mode']) for r in records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def checkpoint():
        result={'task':'multiple_choice','frozen':frozen,'records':records,'pending':pending,
                'summary':summarize(records,args.modes),
                'complete':len(records)==len(manifest['examples'])*len(args.modes),
                'clinical_accuracy_validated':False}
        temporary=args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2));temporary.replace(args.output)
    for example in manifest['examples']:
        image=ROOT/'data'/example['image_path']
        if hashlib.sha256(image.read_bytes()).hexdigest()!=example['image_sha256']:
            raise ValueError('Image changed after freezing')
        for mode in args.modes:
            if (example['case_id'],mode) in completed:continue
            # Explicit allowlist: answers, diagnoses, captions and explanations stay local.
            if pending:
                if (pending['case_id'],pending['mode'])!=(example['case_id'],mode):
                    raise ValueError('Pending run does not match the next frozen task')
                rid=pending['run_id']
            else:
                with image.open('rb') as uploaded:
                    response=requests.post(args.base+'/api/cases',data={'title':example['title']+' · '+mode,
                        'context':example['context']},files={'image':('chest.png',uploaded,'image/png')},timeout=30)
                response.raise_for_status();cid=response.json()['id']
                response=requests.post(args.base+f'/api/cases/{cid}/runs',json={'question':example['question'],'mode':mode},timeout=30)
                response.raise_for_status();rid=response.json()['run_id']
                pending={'case_id':example['case_id'],'mode':mode,'run_id':rid,'sampled_gpu_peak_mib':0}
                checkpoint()
            start=time.monotonic();peak=pending['sampled_gpu_peak_mib']
            while True:
                response=requests.get(args.base+'/api/runs/'+rid,timeout=30)
                response.raise_for_status();run=response.json()
                sample=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
                if sample.returncode==0:peak=max(peak,int(sample.stdout.splitlines()[0]))
                pending['sampled_gpu_peak_mib']=peak
                if run['status'] in {'completed','failed','interrupted'}:break
                if time.monotonic()-start>args.timeout:
                    checkpoint()
                    raise TimeoutError(f'Run {rid} is still pending; retained on server, do not silently count or drop it')
                time.sleep(2)
            report=run['result']['report'] if run.get('result') else None
            choice=report.get('answer_choice') if report else None
            record={'case_id':example['case_id'],'question_id':example['question_id'],'mode':mode,
                'status':run['status'],'prediction':choice,'reference_answer':example['reference_answer'],
                'correct':run['status']=='completed' and choice==example['reference_answer'],
                'sampled_gpu_peak_mib':peak,'run':run}
            records.append(record)
            pending=None
            checkpoint()
            print(example['case_id'],mode,run['status'],'prediction',choice,'reference',example['reference_answer'],
                  'correct',record['correct'],flush=True)
    print(json.dumps(summarize(records,args.modes),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=ROOT/'data/independent/manifest.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/independent/results.json')
    parser.add_argument('--modes',nargs='+',choices=['direct','verified'],default=['verified'])
    parser.add_argument('--base',default='http://127.0.0.1:7860')
    parser.add_argument('--timeout',type=int,default=900)
    evaluate(parser.parse_args())
