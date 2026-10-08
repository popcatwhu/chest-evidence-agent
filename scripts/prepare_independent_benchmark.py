"""Freeze a seeded, patient-disjoint single-chest-radiograph MCQ sample.

Only history and the published question are inference inputs. Captions, answers,
explanations and final diagnoses remain in evaluator-side records.
"""
import argparse
import hashlib
import io
import json
import random
import re
import zipfile
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
EXCLUDED={'6399','10698','11583','10304','1373','14432'}


def prepare(limit=12,seed=20261007,destination=None,excluded=None):
    source=ROOT/'data/benchmark/source'
    metadata=json.loads((ROOT.parent/'MedRAX/data/eurorad_metadata.json').read_text())
    candidates={}
    excluded=EXCLUDED | set(excluded or [])
    for line in (source/'metadata.jsonl').read_text().splitlines():
        row=json.loads(line)
        cid=str(row['case_id'])
        if cid in excluded or cid not in metadata or 'diagnosis' not in row['categories'].split(',') or len(row['images'])!=1:
            continue
        figure=Path(row['images'][0]).stem.replace('figure_','').lower()
        captions=[sub['caption'] for group in metadata[cid]['figures'] for sub in group.get('subfigures',[])
                  if sub['number'].lower().replace('figure ','').replace(' ','')==figure]
        caption=' '.join(captions)
        if not re.search(r'(chest|thorax|thoracic).{0,40}(radiograph|x.ray)|(radiograph|x.ray).{0,40}(chest|thorax|thoracic)',caption,re.I):
            continue
        if re.search(r'follow.up|post.operat|after|CT scan|computed tomography',caption,re.I):
            continue
        # Use the first eligible question per patient in the published file order.
        candidates.setdefault(cid,(row,caption))
    ids=sorted(candidates)
    random.Random(seed).shuffle(ids)
    if limit>len(ids):raise ValueError('Not enough eligible patients')
    destination=Path(destination).resolve() if destination else ROOT/'data/independent'
    destination.mkdir(parents=True,exist_ok=True)
    examples=[]
    with zipfile.ZipFile(source/'figures.zip') as archive:
        for cid in ids[:limit]:
            row,caption=candidates[cid]
            patient=metadata[cid]
            image=archive.read(row['images'][0])
            path=destination/f'{cid}.png'
            Image.open(io.BytesIO(image)).convert('RGB').save(path)
            examples.append({'case_id':cid,'title':f'独立选择题 {cid}',
                'context':f"Age: {patient['age']}; sex: {patient['gender']}.\nClinical history: {patient['history']}\nOnly the history and the matching chest radiograph are provided.",
                'image_path':str(path.relative_to(ROOT/'data')),'question':row['question'],
                'reference_answer':row['answer'],'question_id':row['full_question_id'],
                'categories':row['categories'],'reference_diagnosis':patient['diagnosis'],
                'evaluator_only_caption':caption,'evaluator_only_explanation':row['explanation'],
                'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest={'task':'multiple_choice','seed':seed,'selection':'First eligible question per patient; caption-confirmed initial chest radiograph; seeded random sample',
        'excluded_development_case_ids':sorted(excluded),'eligible_patients':len(candidates),
        'source':'https://huggingface.co/datasets/wanglab/chestagentbench',
        'metadata_sha256':hashlib.sha256((source/'metadata.jsonl').read_bytes()).hexdigest(),
        'limitations':['Small public sample; possible model pretraining contamination.',
                      'MCQ accuracy does not measure open-ended diagnostic or clinical accuracy.',
                      'Published questions and labels are used unchanged; some require information beyond one image.'],
        'examples':examples}
    target=destination/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=manifest:
        raise ValueError('Refusing to overwrite a different frozen manifest')
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'patients':len(examples),'case_ids':[e['case_id'] for e in examples],
                      'manifest_sha256':hashlib.sha256(target.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=12)
    parser.add_argument('--seed',type=int,default=20261007)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--exclude-manifest',type=Path)
    args=parser.parse_args()
    excluded=[]
    if args.exclude_manifest:excluded=[e['case_id'] for e in json.loads(args.exclude_manifest.read_text())['examples']]
    prepare(args.limit,args.seed,args.output_dir,excluded)
