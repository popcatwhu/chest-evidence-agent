"""Save a standalone local NF4 checkpoint, without requiring benchmark data."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();target=args.output.resolve()
    if source==target:raise ValueError('Output must differ from the original model directory')
    if target.exists() and any(target.iterdir()):raise ValueError('Output directory must be empty')
    cfg=json.loads((source/'config.json').read_text())
    if cfg.get('model_type')!='qwen2_5_vl':raise ValueError('This converter supports Qwen2.5-VL architecture only')
    if cfg.get('quantization_config'):raise ValueError('Source is already quantized')
    import torch
    from transformers import AutoProcessor,BitsAndBytesConfig,Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():raise RuntimeError('Conversion requires a supported CUDA GPU')
    processor=AutoProcessor.from_pretrained(source,local_files_only=True,use_fast=False)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(source,local_files_only=True,
        dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=['visual','lm_head']))
    target.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(target,safe_serialization=True,max_shard_size='4GB')
    processor.save_pretrained(target)
    shutil.copyfile(source/'tokenizer.json',target/'tokenizer.json')
    revision=source/'source_revision.json'
    provenance=json.loads(revision.read_text()) if revision.exists() else {'local_source':source.name}
    provenance['conversion']='NF4 double quantization; BF16 compute; visual and lm_head preserved'
    (target/'source_revision.json').write_text(json.dumps(provenance,indent=2))
    print('Saved NF4 checkpoint:',target)


if __name__=='__main__':main()
