"""Download pinned official, non-gated medical model weights without executing code."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download

ROOT=Path(__file__).resolve().parent.parent
MODELS={
    '7b':('lingshu-medical-mllm/Lingshu-7B','b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9'),
    '32b':('lingshu-medical-mllm/Lingshu-32B','36b98277cacb60db86f34b75ce0540b1ea35183c'),
    'cxr3b':('nvidia/NV-Reason-CXR-3B','056bd0383b35226554da9dc5866e095df174ae19'),
}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('size',choices=MODELS)
    args=parser.parse_args()
    repo,revision=MODELS[args.size]
    info=HfApi().model_info(repo,revision=revision)
    if info.gated:raise RuntimeError('Model requires an access agreement')
    destination=ROOT.parent/'models'/repo.split('/')[-1]
    print('Downloading',repo,revision,'to',destination,flush=True)
    snapshot_download(repo,revision=revision,local_dir=str(destination),max_workers=4,
        allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','README.md','LICENSE'])
    (destination/'source_revision.json').write_text(json.dumps({'repository':repo,'revision':revision},indent=2))
    print('Completed',destination,flush=True)
