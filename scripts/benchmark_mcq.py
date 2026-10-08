"""GPU-serial minimal MCQ benchmark; answers never enter the inference function."""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from chest_agent import config
from chest_agent.llm import backend
from chest_agent import exam
from scripts.evaluate_http import fingerprint,interval


def summarize(records,variants):
    output={}
    for variant in variants:
        rows=[r for r in records if r['variant']==variant]
        n=len(rows);correct=sum(r['correct'] for r in rows)
        output[variant]={'tasks':n,'completed':sum(r['status']=='completed' for r in rows),
            'correct':correct,'accuracy':correct/n if n else None,'wilson_95_ci':interval(correct,n),
            'median_seconds':round(statistics.median(r['seconds'] for r in rows),2) if rows else None}
    return output


def evaluate(args):
    import torch
    if not backend.ready():raise RuntimeError('Model files are not complete')
    manifest=json.loads(args.manifest.read_text())
    source=config.MODEL/'source_revision.json'
    frozen={'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'pipeline_sha256':fingerprint((config.ROOT/'chest_agent').glob('*.py')),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model':str(config.MODEL.resolve()),'model_config_sha256':hashlib.sha256((config.MODEL/'config.json').read_bytes()).hexdigest(),
        'source_revision':json.loads(source.read_text()) if source.exists() else None,
        'load_nf4':config.LOAD_NF4,'constrained_json':config.CONSTRAINED_JSON,
        'cxr_reader':config.CXR_READER,
        'max_input_tokens':config.MAX_INPUT_TOKENS,'variants':args.variants,
        'question_only':True,'protocol':'original image and published question; short English reasoning and exact letter; no clinical report or extra history'}
    records=[]
    if args.output.exists():
        previous=json.loads(args.output.read_text())
        if previous['frozen']!=frozen:raise ValueError('Cannot resume with different frozen settings')
        records=previous['records']
    seen={(r['case_id'],r['variant']) for r in records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        result={'task':'minimal_multiple_choice','frozen':frozen,'records':records,
                'summary':summarize(records,args.variants),
                'complete':len(records)==len(manifest['examples'])*len(args.variants),
                'clinical_accuracy_validated':False}
        temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,indent=2));temp.replace(args.output)
    for example in manifest['examples']:
        image=config.DATA/example['image_path']
        if hashlib.sha256(image.read_bytes()).hexdigest()!=example['image_sha256']:raise ValueError('Image changed')
        tools=validation=None
        for variant in args.variants:
            if (example['case_id'],variant) in seen:continue
            start=time.monotonic();metric_start=len(backend.metrics)
            if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
            answer=None;error=None
            try:
                if variant!='direct' and tools is None:tools=exam.raw_tools(str(image))
                if variant=='validated' and validation is None:validation=exam.validate_claims(tools['claims'],str(image))
                answer=exam.answer(example['question'],str(image),variant,tools,validation)
                status='completed'
            except Exception as exception:
                status='failed';error=str(exception)
                if torch.cuda.is_available():torch.cuda.empty_cache()
            prediction=answer.answer_choice if answer else None
            rows={'case_id':example['case_id'],'question_id':example['question_id'],'variant':variant,
                'status':status,'prediction':prediction,'reason':answer.reason if answer else None,
                'reference_answer':example['reference_answer'],'correct':status=='completed' and prediction==example['reference_answer'],
                'error':error,'seconds':round(time.monotonic()-start,2),
                'gpu_peak_allocated_gib':round(torch.cuda.max_memory_allocated()/2**30,2) if torch.cuda.is_available() else None,
                'gpu_peak_reserved_gib':round(torch.cuda.max_memory_reserved()/2**30,2) if torch.cuda.is_available() else None,
                'model_calls':backend.metrics[metric_start:],
                'tool_inputs':tools if variant!='direct' else None,
                'tool_validation':validation if variant=='validated' else None}
            if backend.model is not None:
                visual=getattr(getattr(backend.model,'model',backend.model),'visual',None)
                if visual is not None:
                    rows['vision_parameter_dtypes']=sorted({str(p.dtype) for p in visual.parameters()})
                    rows['quantized_visual_modules']=sum(type(m).__name__=='Linear4bit' for m in visual.modules())
            records.append(rows);save()
            print(config.MODEL.name,example['case_id'],variant,status,prediction,'correct',rows['correct'],rows['seconds'],flush=True)
    print(json.dumps(summarize(records,args.variants),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=config.DATA/'independent/manifest.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variants',nargs='+',choices=['direct','tools','validated'],default=['direct'])
    parser.add_argument('--additional-manifest',type=Path)
    parser.add_argument('--additional-output',type=Path)
    args=parser.parse_args()
    if bool(args.additional_manifest)!=bool(args.additional_output):parser.error('Specify both additional manifest and output')
    evaluate(args)
    if args.additional_manifest:
        args.manifest=args.additional_manifest;args.output=args.additional_output
        evaluate(args)
