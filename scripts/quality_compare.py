"""Fixed development-case comparison through the resident local GPU service."""
import argparse
import json
import time
from pathlib import Path
import requests
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('label')
parser.add_argument('--cases',type=Path,default=Path('data/quality/cases.json'))
args=parser.parse_args()
base='http://127.0.0.1:7860'
records=[]
for example in json.loads(args.cases.read_text()):
    image=(Path('data')/example['image_path']).resolve()
    with image.open('rb') as f:
        response=requests.post(base+'/api/cases',data={'title':f"质量对照 {args.label} · {example['title']}",
            'context':example['context']},files={'image':('chest.png',f,'image/png')},timeout=15)
    response.raise_for_status()
    case_id=response.json()['id']
    response=requests.post(base+f'/api/cases/{case_id}/runs',json={'question':
        '请综合病史、已提供检查和胸片，给出最可能的具体诊断、鉴别诊断及支持和反对证据；明确还需哪些检查才能确认。',
        'mode':'verified'},timeout=15)
    response.raise_for_status();run_id=response.json()['run_id']
    peak_gpu_mib=0
    for attempt in range(240):
        sampled=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True)
        peak_gpu_mib=max(peak_gpu_mib,int(sampled.stdout.strip().splitlines()[0]))
        run=requests.get(base+'/api/runs/'+run_id,timeout=15).json()
        if run['status'] in ['completed','failed','interrupted']:break
        time.sleep(2)
    else:raise RuntimeError('Run timeout')
    report=run['result']['report'] if run['result'] else None
    records.append({'title':example['title'],'reference':example['reference_diagnosis'],
        'accepted_names':example.get('accepted_names',[]),'sampled_gpu_peak_mib':peak_gpu_mib,'run':run})
    destination=Path(f'data/quality/{args.label}.json')
    destination.write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print(example['title'],run['status'],report['most_likely']['name'] if report else run['error'],flush=True)
