"""Prepare a small ChestAgentBench development subset, keeping answers in evaluation files."""
import argparse
import json
from pathlib import Path
import requests
import zipfile
from huggingface_hub import hf_hub_download

parser=argparse.ArgumentParser()
parser.add_argument('--limit',type=int,default=5)
parser.add_argument('--output',type=Path,default=Path('data/benchmark/development.json'))
args=parser.parse_args()
args.output.parent.mkdir(parents=True,exist_ok=True)
response=requests.get('https://huggingface.co/datasets/wanglab/chestagentbench/resolve/main/metadata.jsonl',timeout=60)
response.raise_for_status()
# Original Eurorad image URLs return 403 on this host. Use the public benchmark archive.
archive=hf_hub_download('wanglab/chestagentbench','figures.zip',repo_type='dataset',
    local_dir=str(args.output.parent/'source'))
figures=zipfile.ZipFile(archive)
examples=[]
seen=set()
for line in response.text.splitlines():
    row=json.loads(line)
    if 'diagnosis' not in row['categories'] or len(row['images'])!=1 or row['case_id'] in seen:
        continue
    try:
        from PIL import Image
        import io
        path=args.output.parent/f"case-{row['case_id']}.png"
        Image.open(io.BytesIO(figures.read(row['images'][0]))).convert('RGB').save(path)
    except Exception as e:
        print('Skip',row['case_id'],str(e),flush=True)
        continue
    seen.add(row['case_id'])
    examples.append({'title':f"ChestAgentBench {row['case_id']}",
        'context':'仅提供胸片与问题中的病例信息，没有额外病史。',
        'image_path':path.name,'question':row['question'],
        'reference_diagnosis':row['answer'],'question_id':row['full_question_id'],
        'source':'https://huggingface.co/datasets/wanglab/chestagentbench'})
    print('Prepared',row['case_id'],flush=True)
    if len(examples)>=args.limit:
        break
args.output.write_text(json.dumps(examples,ensure_ascii=False,indent=2))
print('Development examples:',len(examples),'Not an independent test set.')
