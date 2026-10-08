"""Independent CXR reader with image-only input and immutable, versioned output caching."""
import hashlib
import json
from pathlib import Path

from . import config
from .schemas import Evidence, Observation

PROMPT = ('Find abnormalities and support devices. Describe directly visible morphology and '
          'locations on this chest radiograph. No clinical history is provided. Do not infer '
          'an underlying etiology or assume unprovided prior images. State uncertainty clearly.')


class CXRReader:
    def __init__(self):
        self.model = self.processor = None

    def findings(self, image_path):
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from .llm import backend

        revision = config.NVREASON_MODEL / 'source_revision.json'
        if not revision.exists():
            raise RuntimeError('胸片专用模型尚未完成权重校验')
        signature = hashlib.sha256(revision.read_bytes() + Path(__file__).read_bytes()).hexdigest()
        image_hash = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        cache = config.DATA / 'cxr_reader_cache'
        cache.mkdir(exist_ok=True)
        path = cache / f'{image_hash}-{signature[:16]}.json'
        if path.exists():
            return Evidence.model_validate(json.loads(path.read_text()))
        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(config.NVREASON_MODEL,
                local_files_only=True, use_fast=False, min_pixels=256*28*28, max_pixels=768*28*28)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.NVREASON_MODEL, local_files_only=True, dtype=torch.bfloat16,
                device_map='cuda:0', attn_implementation='sdpa').eval()
        image = Image.open(image_path).convert('RGB')
        text = self.processor.apply_chat_template([{'role':'user','content':[
            {'type':'image','image':image},{'type':'text','text':PROMPT}]}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors='pt').to('cuda:0')
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=1536, do_sample=False)
        raw = self.processor.batch_decode(tokens[:,inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        # Do not treat narrative reasoning, guessed devices, or etiology as patient observations.
        extracted = backend.json('Extract only directly described visual morphology and location '
            'from this fallible model report. Keep conflicts and uncertainty in limitations. '
            'Do not add findings, disease diagnoses, patient history, unseen comparisons or support devices. '
            'Return separate concise observations, not a paragraph containing multiple diagnoses.\n'
            + raw, Observation, max_tokens=600)
        evidence = Evidence(id='I-radiology', kind='image', title='NV-Reason胸片专用模型的观察',
            content=json.dumps({'model':'nvidia/NV-Reason-CXR-3B','source_revision':json.loads(revision.read_text()),
                'visual_findings':extracted.observations,'limitations':extracted.limitations,
                'raw_report':raw,'extraction':'Fallible model extraction; original output retained',
                'image_sha256':image_hash,'pipeline_sha256':signature,
                'limits':'模型观察未经临床核实；与原图及病史冲突时不能当作已确认事实。'},ensure_ascii=False),
            source_type='CXR-trained model-generated findings')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(evidence.model_dump_json(indent=2)); temporary.replace(path)
        return evidence


reader = CXRReader()
