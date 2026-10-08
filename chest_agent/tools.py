import json
from pathlib import Path
from PIL import Image
import numpy as np
from .config import DATA,CXR_FINDINGS_MODEL
from .schemas import Evidence
from .radiology import split_findings


class ImageTools:
    """TorchXRayVision tools used by MedRAX, isolated from its optional heavy imports."""
    def __init__(self):
        self.classifier = self.segmenter = None
        self.findings_model = self.findings_processor = self.findings_tokenizer = None

    def findings(self,image_path,run_id):
        from . import config
        if config.CXR_READER=='nvreason':
            from .cxr_reader import reader
            return reader.findings(image_path)
        if config.CXR_READER!='legacy':
            raise ValueError('未知的胸片观察模型：'+config.CXR_READER)
        import torch
        from transformers import VisionEncoderDecoderModel,ViTImageProcessor,BertTokenizer
        if self.findings_model is None:
            self.findings_model=VisionEncoderDecoderModel.from_pretrained(CXR_FINDINGS_MODEL,local_files_only=True).eval().cuda()
            self.findings_processor=ViTImageProcessor.from_pretrained(CXR_FINDINGS_MODEL,local_files_only=True)
            self.findings_tokenizer=BertTokenizer.from_pretrained(CXR_FINDINGS_MODEL,local_files_only=True)
        image=Image.open(image_path).convert('RGB')
        pixels=self.findings_processor(images=image,return_tensors='pt').pixel_values.cuda()
        expected=self.findings_model.config.encoder.image_size
        size=tuple(expected) if isinstance(expected,(list,tuple)) else (expected,expected)
        if tuple(pixels.shape[-2:])!=size:
            pixels=torch.nn.functional.interpolate(pixels,size=size,mode='bilinear',align_corners=False)
        with torch.inference_mode():
            tokens=self.findings_model.generate(pixels,max_new_tokens=128,num_beams=2,
                decoder_start_token_id=self.findings_tokenizer.cls_token_id,
                eos_token_id=self.findings_tokenizer.sep_token_id,pad_token_id=self.findings_tokenizer.pad_token_id)
        text=self.findings_tokenizer.batch_decode(tokens,skip_special_tokens=True)[0]
        return Evidence(id='I-radiology',kind='image',title='专用胸片报告模型的观察',
            content=json.dumps({'model':'IAMJB/chexpert-mimic-cxr-findings-baseline',**split_findings(text),
                'limits':'研究模型生成，不是放射科医生报告；可能漏诊或虚构，不支持单幅图像的纵向比较。'},ensure_ascii=False),
            source_type='CXR-trained model-generated findings')

    def classify(self, image_path, run_id):
        import torch
        import torchxrayvision as xrv
        if self.classifier is None:
            self.classifier = xrv.models.DenseNet(weights="densenet121-res224-all").eval().cuda()
        img = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        img = xrv.datasets.normalize(img, 255)[None]
        img = xrv.datasets.XRayCenterCrop()(img)
        img = xrv.datasets.XRayResizer(224)(img)
        with torch.inference_mode():
            scores = self.classifier(torch.from_numpy(img).unsqueeze(0).cuda()).cpu()[0].tolist()
        result = dict(zip(self.classifier.pathologies, scores))
        return Evidence(id="I-classify", kind="image", title="胸片分类模型输出",
            content=json.dumps({"model": "DenseNet121-res224-all", "scores": result,
                "interpretation": "模型分数用于辅助观察，不是患者患病概率；低分不排除疾病。"}, ensure_ascii=False))

    def segment(self, image_path, run_id):
        import torch
        import torchxrayvision as xrv
        if self.segmenter is None:
            self.segmenter = xrv.baseline_models.chestx_det.PSPNet().eval().cuda()
        original = Image.open(image_path).convert("L")
        arr = np.asarray(original, dtype=np.float32)
        crop = xrv.datasets.XRayCenterCrop()(xrv.datasets.normalize(arr, 255)[None])
        crop = xrv.datasets.XRayResizer(512)(crop)
        with torch.inference_mode():
            logits = self.segmenter(torch.from_numpy(crop).unsqueeze(0).cuda())
            masks = (torch.sigmoid(logits).cpu().numpy()[0] > 0.5)
        # Display masks in transformed square coordinates; no fabricated physical units.
        crop_size = min(original.size)
        left = (original.width - crop_size) // 2
        top = (original.height - crop_size) // 2
        base = np.asarray(original.crop((left, top, left+crop_size, top+crop_size)).resize((512,512)).convert("RGB")).copy()
        labels = {4: "Left Lung", 5: "Right Lung", 8: "Heart"}
        colors = {4: (56,189,248), 5: (45,212,191), 8: (251,113,133)}
        metrics = {}
        for index, name in labels.items():
            mask = masks[index]
            base[mask] = (0.65 * base[mask] + 0.35 * np.array(colors[index])).astype(np.uint8)
            metrics[name] = {"area_pixels_at_512": int(mask.sum())}
        out = DATA / "artifacts" / f"{run_id}-segmentation.png"
        Image.fromarray(base).save(out)
        return Evidence(id="I-segment", kind="image", title="肺与心脏区域分割",
            content=json.dumps({"model": "ChestX-Det PSPNet", "metrics": metrics,
                "note": "分割图为中心裁剪后的512像素图，仅展示模型区域；不含真实物理尺寸，也不能单独判定疾病。"}, ensure_ascii=False),
            artifact=f"/artifacts/{out.name}")


image_tools = ImageTools()


def retrieve(query, limit=6):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    synonyms={'AIDS':'HIV immunosuppression','IV drug abuse':'intravenous drug use',
        'IV drug use':'intravenous drug use','nonproductive cough':'dry cough',
        'dyspnoea':'dyspnea shortness of breath','结节':'nodular opacities nodules','肿块':'pleural mass'}
    for term,expansion in synonyms.items():
        if term.casefold() in query.casefold():query+=' '+expansion
    documents = []
    for path in sorted((DATA / "knowledge").glob("*.json")):
        doc = json.loads(path.read_text())
        text = doc["text"]
        paragraphs=text.split('\n\n')
        chunks=[]
        for paragraph in paragraphs:
            for offset in range(0,len(paragraph),800):
                chunks.append(paragraph[offset:offset+1000])
        documents.extend({**doc,'text':chunk} for chunk in chunks if chunk.strip())
    if not documents:
        return []
    # Character ngrams support Chinese query terms and English source terminology.
    vectorizer=TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5))
    matrix=vectorizer.fit_transform([d['title']+' '+d['text'] for d in documents])
    scores=cosine_similarity(vectorizer.transform([query]),matrix)[0]
    for i,doc in enumerate(documents):
        if any(alias.casefold() in query.casefold() for alias in doc.get('aliases',[]) if len(alias)>=3):
            scores[i]+=0.15
    indices = []
    counts = {}
    for i in np.argsort(scores)[::-1]:
        url = documents[i]["url"]
        if scores[i] <= max(0.025,float(scores.max())*.3) or counts.get(url,0) >= 2:
            continue
        indices.append(i)
        counts[url] = counts.get(url,0) + 1
        if len(indices) == limit:
            break
    return [Evidence(id=f"K-{rank+1}", kind="knowledge", title=documents[i]["title"],
        content=documents[i]["text"], source=documents[i]["url"],source_type=documents[i].get('type'),
        diagnostic_rules=documents[i].get('diagnostic_rules',[])) for rank,i in enumerate(indices) if scores[i] > 0]
