"""Wait for checksum-verified weights and a free GPU, then evaluate the medical 32B."""
import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent.parent


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--save-quantized',action='store_true')
    args=parser.parse_args();model=ROOT.parent/'models/Lingshu-32B'
    print('Waiting for verified Lingshu-32B weights and an idle GPU',flush=True)
    while True:
        ready=(model/'source_revision.json').exists()
        memory=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
        idle=memory.returncode==0 and int(memory.stdout.splitlines()[0])<1500
        if ready and idle:break
        time.sleep(5)
    os.environ['CHEST_MODEL_PATH']=str(model)
    os.environ['CHEST_LOAD_NF4']='1'
    sys.path.insert(0,str(ROOT))
    from scripts.benchmark_mcq import evaluate
    from chest_agent.llm import backend
    from argparse import Namespace
    options=Namespace(manifest=ROOT/'data/independent/manifest.json',
        output=ROOT/'data/medical_comparison/lingshu32_regression.json',variants=['direct'])
    evaluate(options)
    options.manifest=ROOT/'data/medical_test/manifest.json'
    options.output=ROOT/'data/medical_comparison/lingshu32_test.json'
    evaluate(options)
    if args.save_quantized:
        target=ROOT.parent/'models/Lingshu-32B-NF4'
        target.mkdir(parents=True,exist_ok=True)
        backend.model.save_pretrained(target,safe_serialization=True,max_shard_size='4GB')
        backend.processor.save_pretrained(target)
        # The slow processor saves vocab/merges but omits the original fast tokenizer.
        # Preserve the official tokenizer file for the application's readiness check.
        shutil.copyfile(model/'tokenizer.json',target/'tokenizer.json')
        (target/'source_revision.json').write_text(json.dumps({'repository':'lingshu-medical-mllm/Lingshu-32B',
            'revision':'36b98277cacb60db86f34b75ce0540b1ea35183c',
            'conversion':'bitsandbytes NF4 double quantization; visual and lm_head preserved'},indent=2))
        print('Saved reusable NF4 checkpoint',target,flush=True)
