"""Serial development comparison, frozen prospective MCQs, then restore the resident app."""
import argparse
import hashlib
import gc
import json
import os
from pathlib import Path
import re
import runpy
import signal
import statistics
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)


def label_match(reference,name):
    """Grader-only semantic name match; neither reference nor grading patterns enter inference."""
    if re.search(r'排除|不支持|不考虑',name):return False,False
    if re.search(r'肺孢子',reference):
        match=bool(re.search(r'肺孢子菌|肺孢子虫|Pneumocystis|\bPJP\b|\bPCP\b',name,re.I));return match,match
    if '脓毒性' in reference:
        match=bool(re.search(r'脓毒性肺栓塞|感染性肺栓塞|septic.*(?:pulmonary|embol)',name,re.I));return match,match
    if '纤维' in reference:
        match=bool(re.search(r'孤立性.*纤维|solitary.*fibrous|\bSFTP?\b',name,re.I))
        return match,match
    if re.search(r'endobronchial',reference,re.I):
        return bool(re.search(r'支气管(?:内膜)?结核|endobronchial.*tuberculosis|\bEBTB\b',name,re.I)),bool(re.search(r'结核|tuberculosis|\bTB\b',name,re.I))
    if re.search(r'tension',reference,re.I):
        return bool(re.search(r'张力性.*气胸|tension.*pneumothorax',name,re.I)),bool(re.search(r'气胸|pneumothorax',name,re.I))
    if re.search(r'catamenial',reference,re.I):
        return bool(re.search(r'月经.*气胸|气胸.*月经|catamenial.*pneumothorax',name,re.I)),bool(re.search(r'气胸|pneumothorax',name,re.I))
    raise ValueError('Unsupported reference label in development grader')


def summarize(path,reader=None):
    record=json.loads(path.read_text())
    patients={};rows=[];missing_reader=0
    for example in record['records']:
        run=example['run'];result=run.get('result')
        name=result['report']['most_likely']['name'] if result else None
        strict,broad=label_match(example['reference'],name) if name else (False,False)
        patient=re.search(r'\d{3,5}',example['title']).group()
        row={'patient_id':patient,'title':example['title'],'candidate':name,'specific_match':strict,
             'broad_match':broad,'status':run['status'],'run_id':run['id'],
             'review_issues':len(result['verification']['issues'])+len(result['verification'].get('notes',[])) if result else 1}
        rows.append(row);patients.setdefault(patient,row)
        if reader=='nvreason' and not any(e['id']=='I-radiology' and e['status']=='completed'
            and 'NV-Reason' in e['content'] for e in (result or {}).get('evidence',[])):
            missing_reader+=1
    warm=[x['run']['result']['metrics']['elapsed_seconds'] for x in record['records'][1:] if x['run'].get('result')]
    return {'complete':record['complete'],'inputs':len(rows),'patients':len(patients),
        'specific_matches':sum(x['specific_match'] for x in patients.values()),
        'broad_matches':sum(x['broad_match'] for x in patients.values()),
        'unresolved_issues':sum(x['review_issues'] for x in rows),
        'median_warm_seconds':statistics.median(warm) if warm else None,
        'missing_required_reader':missing_reader,'records':rows}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--paused-service-pid',type=int,required=True)
    args=parser.parse_args()
    from chest_agent import config
    from chest_agent.llm import backend
    from scripts.benchmark_mcq import evaluate
    from argparse import Namespace
    directory=config.DATA/'quality_v3';directory.mkdir(exist_ok=True)
    # Record the actual pipeline, corpus and test inputs before any outcomes are observed.
    with zipfile.ZipFile(directory/'evaluated_pipeline.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for base in [ROOT/'chest_agent',config.DATA/'knowledge']:
            for file in base.glob('*.py' if base.name=='chest_agent' else '*.json'):
                archive.write(file,str(file.relative_to(ROOT)))
        for file in [Path(__file__),ROOT/'scripts/benchmark_mcq.py',directory/'policy.json',directory/'test20/manifest.json']:
            archive.write(file,str(file.relative_to(ROOT)))
    clinical={}
    for reader in ['legacy','nvreason']:
        config.CXR_READER=reader
        if reader=='nvreason':
            while not (config.NVREASON_MODEL/'source_revision.json').exists():
                print('Waiting for official CXR weight checksums',flush=True);time.sleep(5)
            import torch
            torch.cuda.empty_cache()
        output=directory/f'clinical_{reader}.json'
        if not output.exists() or not json.loads(output.read_text()).get('complete'):
            sys.argv=['clinical_model_regression.py','--output',str(output)]
            runpy.run_path(str(ROOT/'scripts/clinical_model_regression.py'),run_name='__main__')
        clinical[reader]=summarize(output,reader)
        print('Clinical summary',reader,json.dumps(clinical[reader],ensure_ascii=False),flush=True)
    eligible=[r for r,s in clinical.items() if s['complete'] and not s['missing_required_reader']]
    if not eligible:raise RuntimeError('No completed reader candidate')
    selected=sorted(eligible,key=lambda r:(-clinical[r]['specific_matches'],-clinical[r]['broad_matches'],
        clinical[r]['unresolved_issues'],clinical[r]['median_warm_seconds'],r))[0]
    selection={'reader':selected,'policy':json.loads((directory/'policy.json').read_text()),'development':clinical,
               'clinical_accuracy_validated':False}
    (directory/'reader_selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    print('Reader selected BEFORE prospective test:',selected,flush=True)
    # Always finish both prospective readers; do not change the development selection using test scores.
    tests={}
    for reader,variants in [('legacy',['direct','tools']),('nvreason',['tools'])]:
        config.CXR_READER=reader
        options=Namespace(manifest=directory/'test20/manifest.json',output=directory/f'test20_{reader}.json',variants=variants)
        evaluate(options);tests[reader]=json.loads(options.output.read_text())['summary']
    config.CXR_READER=selected
    from chest_agent.cxr_reader import reader as nv_reader
    from chest_agent.tools import image_tools
    # Release the unselected expert from GPU; keep its versioned records for review.
    if selected=='legacy' and nv_reader.model is not None:
        nv_reader.model=None;nv_reader.processor=None
    if selected=='nvreason' and image_tools.findings_model is not None:
        image_tools.findings_model=None;image_tools.findings_processor=None;image_tools.findings_tokenizer=None
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    result={'task':'diagnostic_quality_v3','clinical_accuracy_validated':False,
        'baseline_development':summarize(directory/'baseline_clinical.json'),
        'development':clinical,'selected_reader':selected,'selection_used_test_scores':False,
        'prospective_test_patients':20,'prospective_test':tests,
        'test_manifest_sha256':hashlib.sha256((directory/'test20/manifest.json').read_bytes()).hexdigest(),
        'limitations':['Known-case label matches are development evidence, not clinical accuracy.',
            '20 new public MCQs; pretraining contamination unknown; no clinical expert grading.',
            'Source and pipeline changes are combined; reader comparison holds them fixed.']}
    (ROOT/'diagnostic_quality_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    command=Path(f'/proc/{args.paused_service_pid}/cmdline')
    if command.exists():
        content=command.read_bytes()
        if b'uvicorn' not in content or b'chest_agent.app:app' not in content:
            raise RuntimeError('Paused service identity changed')
        os.kill(args.paused_service_pid,signal.SIGTERM)
        for _ in range(100):
            if not command.exists():break
            time.sleep(.1)
    config.INFERENCE_PAUSED=False
    print('Restoring app with resident models; selected reader:',selected,flush=True)
    import uvicorn
    uvicorn.run('chest_agent.app:app',host='127.0.0.1',port=7860)


if __name__=='__main__':
    main()
