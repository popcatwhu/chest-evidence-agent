import base64
import io
import json
import re
import time
from uuid import uuid4
from threading import Lock
from collections import Counter
from PIL import Image
from . import config


def parse_json(text, root_keys=()):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = text.find("{")
    if start < 0:
        raise ValueError("模型未返回JSON对象")
    raw = text[start:].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        from json_repair import repair_json

        string_pattern = r'"(?:\\.|[^"\\])*"'
        unquoted = re.sub(string_pattern, "", raw)
        # Do not turn a truncated answer into a completed medical report.
        if not raw.endswith("}") or unquoted.count("{") != unquoted.count("}"):
            raise ValueError("模型输出被截断；不能把内层候选对象当成完整报告")
        # The reached failure is a missing array terminator before a root field.
        # Use known root-field names, never clinical values, to restore hierarchy.
        stack = []
        inserts = []
        previous = None
        tokens = list(re.finditer(string_pattern + r"|[{}\[\],:]", raw))
        for i, token in enumerate(tokens):
            value = token.group()
            if (
                value.startswith('"')
                and stack == ["{", "["]
                and previous
                and previous.group() == ","
            ):
                key = json.loads(value)
                if (
                    key in root_keys
                    and i + 1 < len(tokens)
                    and tokens[i + 1].group() == ":"
                ):
                    inserts.append(previous.start())
                    stack.pop()
            elif value in "{[":
                stack.append(value)
            elif value in "}]" and stack:
                stack.pop()
            previous = token
        corrected = raw
        for index in reversed(inserts):
            corrected = corrected[:index] + "]" + corrected[index:]
        repaired = repair_json(corrected, return_objects=True)
        if not isinstance(repaired, dict):
            raise ValueError("修复后不是JSON对象")
        fixed = json.dumps(repaired, ensure_ascii=False)

        def literal_values(value):
            strings = Counter(
                json.loads(token) for token in re.findall(string_pattern, value)
            )
            rest = re.sub(string_pattern, "", value)
            primitives = Counter(
                re.findall(
                    r"(?<!\w)(?:-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)(?!\w)",
                    rest,
                )
            )
            return strings, primitives

        if literal_values(raw) != literal_values(fixed):
            raise ValueError("JSON语法修复改变了字段、文字或数值，拒绝使用")
        return repaired


class ModelBackend:
    def __init__(self, model_path=None, backend_type=None, load_nf4=None):
        self._model_path = model_path
        self._backend_type = backend_type
        self._load_nf4 = load_nf4
        self.model = self.processor = None
        self.lock = Lock()
        self.metrics = []
        self.enforcer_tokenizer_data = None

    @property
    def model_path(self):
        return self._model_path if self._model_path is not None else config.MODEL

    @property
    def backend_type(self):
        return self._backend_type if self._backend_type is not None else config.BACKEND

    @property
    def load_nf4(self):
        return self._load_nf4 if self._load_nf4 is not None else config.LOAD_NF4

    def release(self):
        import gc
        import torch

        with self.lock:
            self.model = self.processor = self.enforcer_tokenizer_data = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def ready(self):
        if self.backend_type == "api":
            import os

            return bool(os.getenv("OPENAI_API_KEY") and config.API_MODEL)
        try:
            if not all(
                (self.model_path / name).exists()
                for name in [
                    "config.json",
                    "preprocessor_config.json",
                    "tokenizer_config.json",
                    "tokenizer.json",
                ]
            ):
                return False
            index = json.loads(
                (self.model_path / "model.safetensors.index.json").read_text()
            )
            return all(
                (self.model_path / name).exists()
                for name in set(index["weight_map"].values())
            )
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return False

    def generate(self, prompt, image_path=None, max_tokens=1800, json_schema=None):
        start = time.monotonic()
        with self.lock:
            if self.backend_type == "api":
                from openai import OpenAI

                content = [{"type": "text", "text": prompt}]
                if image_path:
                    img = Image.open(image_path).convert("RGB")
                    img.thumbnail((1024, 1024))
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG")
                    data = base64.b64encode(buffer.getvalue()).decode()
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{data}"},
                        }
                    )
                response = OpenAI(timeout=120, max_retries=1).chat.completions.create(
                    model=config.API_MODEL,
                    messages=[{"role": "user", "content": content}],
                    temperature=0,
                    max_tokens=max_tokens,
                )
                output = response.choices[0].message.content
                tokens = response.usage.completion_tokens if response.usage else None
            else:
                import torch
                from transformers import (
                    AutoProcessor,
                    Qwen2_5_VLForConditionalGeneration,
                    Qwen3VLForConditionalGeneration,
                )

                if self.model is None:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_path,
                        local_files_only=True,
                        use_fast=False,
                        min_pixels=256 * 28 * 28,
                        max_pixels=768 * 28 * 28,
                    )
                    model_type = json.loads(
                        (self.model_path / "config.json").read_text()
                    )["model_type"]
                    model_classes = {
                        "qwen3_vl": Qwen3VLForConditionalGeneration,
                        "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
                    }
                    if model_type not in model_classes:
                        raise ValueError(f"尚未支持模型架构：{model_type}")
                    model_class = model_classes[model_type]
                    load_options = {}
                    model_config = json.loads(
                        (self.model_path / "config.json").read_text()
                    )
                    if self.load_nf4 and not model_config.get("quantization_config"):
                        from transformers import BitsAndBytesConfig

                        load_options["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            llm_int8_skip_modules=["visual", "lm_head"],
                        )
                    self.model = model_class.from_pretrained(
                        self.model_path,
                        local_files_only=True,
                        dtype=torch.bfloat16,
                        device_map="cuda:0",
                        attn_implementation="sdpa",
                        **load_options,
                    ).eval()
                content = []
                image = None
                if image_path:
                    image = Image.open(image_path).convert("RGB")
                    content.append({"type": "image", "image": image})
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text], images=[image] if image else None, return_tensors="pt"
                ).to("cuda:0")
                input_tokens = inputs.input_ids.shape[1]
                if input_tokens > config.MAX_INPUT_TOKENS:
                    raise ValueError(
                        f"输入为{input_tokens} tokens，超过当前显存配置的{config.MAX_INPUT_TOKENS}上限"
                    )
                generation_options = {}
                if json_schema is not None and config.CONSTRAINED_JSON:
                    from lmformatenforcer.integrations.transformers import (
                        build_token_enforcer_tokenizer_data,
                    )
                    from .constrained import schema_prefix_function

                    if self.enforcer_tokenizer_data is None:
                        self.enforcer_tokenizer_data = (
                            build_token_enforcer_tokenizer_data(
                                self.processor.tokenizer
                            )
                        )
                    generation_options["prefix_allowed_tokens_fn"] = (
                        schema_prefix_function(
                            self.enforcer_tokenizer_data, json_schema
                        )
                    )
                from transformers import StoppingCriteriaList
                from .generation import RepetitionStop

                repetition_stop = RepetitionStop(inputs.input_ids.shape[1])
                with torch.inference_mode():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        stopping_criteria=StoppingCriteriaList([repetition_stop]),
                        **generation_options,
                    )
                trimmed = generated[:, inputs.input_ids.shape[1] :]
                tokens = trimmed.shape[1]
                output = self.processor.batch_decode(trimmed, skip_special_tokens=True)[
                    0
                ]
            metric = {
                "model": self.model_path.name,
                "seconds": round(time.monotonic() - start, 2),
                "output_tokens": tokens,
            }
            metric["constrained_json"] = bool(
                json_schema is not None
                and config.CONSTRAINED_JSON
                and self.backend_type == "local"
            )
            if self.backend_type != "api":
                metric["input_tokens"] = input_tokens
            if self.backend_type != "api":
                metric["repetition_stopped"] = repetition_stop.triggered
                metric["output_budget_reached"] = tokens >= max_tokens
            self.metrics.append(metric)
            return output

    def json(self, prompt, schema, image_path=None, max_tokens=1800):
        is_report = schema.__name__ in {"Report", "ChoiceReport"}
        templates = {
            "ExamAnswer": {
                "reason": "Brief image-grounded reasoning, at most two sentences.",
                "answer_choice": "One letter A to F",
            },
            "ToolChecks": {
                "checks": [
                    {
                        "claim_id": "Actual F-number",
                        "status": "supported/contradicted/uncertain",
                        "visual_basis": "Brief visible morphology, location or limitation.",
                    }
                ]
            },
            "Plan": {
                "tools": ["classify"],
                "query": "英文检索词",
                "reason": "选择工具的理由",
                "hypotheses": ["待验证的具体疾病"],
            },
            "CaseProfile": {
                "facts": [
                    {
                        "name": "检查或病史项目",
                        "value": "原文中的值",
                        "source_id": "真实C编号",
                        "source_quote": "逐字复制的原文片段",
                    }
                ]
            },
            "ClinicalContrast": {
                "key_case_clues": [
                    {
                        "name": "Actual distinctive patient clue",
                        "value": "Verbatim source substring",
                        "source_id": "Actual C-number",
                        "source_quote": "Verbatim patient-source quote",
                    }
                ],
                "hypotheses": [
                    {
                        "name": "真实候选疾病",
                        "support": [
                            {
                                "text": "Short patient-specific supporting reason",
                                "evidence_ids": ["Actual evidence ID"],
                            }
                        ],
                        "against": [],
                        "still_unknown": ["Unprovided confirmation evidence"],
                    },
                    {
                        "name": "另一个真实不同的候选疾病",
                        "support": [
                            {
                                "text": "Alternative patient-specific reason",
                                "evidence_ids": ["Actual evidence ID"],
                            }
                        ],
                        "against": [],
                        "still_unknown": [],
                    },
                ],
                "observation_conflicts": [],
                "remaining_questions": [],
            },
            "Observation": {
                "observations": ["图中能观察到的形态与位置"],
                "limitations": ["图像无法确认的事项"],
            },
            "Report": {
                "answer_choice": None,
                "assessment": "有限",
                "most_likely": {
                    "name": "候选诊断名称",
                    "support": [{"text": "具体依据", "evidence_ids": ["实际证据ID"]}],
                    "against": [],
                },
                "differentials": [
                    {
                        "name": "有证据支持的鉴别诊断",
                        "support": [
                            {"text": "支持理由", "evidence_ids": ["实际患者证据ID"]}
                        ],
                        "against": [],
                    }
                ],
                "findings": [
                    {"text": "具体影像观察", "evidence_ids": ["实际图像证据ID"]}
                ],
                "missing_information": [],
                "next_checks": [],
                "recommended_checks": [
                    {
                        "name": "具体检查",
                        "purpose": "用于确认或鉴别什么",
                        "evidence_ids": ["真实资料K编号"],
                    }
                ],
                "change_summary": "本次判断变化",
            },
            "Review": {
                "issues": [
                    {
                        "report_quote": "逐字复制报告中有问题的句子",
                        "reason": "具体问题与对应原文证据",
                        "evidence_ids": ["真实证据ID"],
                        "source_quote": "逐字复制的矛盾证据原文",
                    }
                ]
            },
            "Change": {
                "summary": "新证据如何影响判断",
                "evidence_ids": ["实际新增证据ID"],
            },
        }
        template = templates["Report" if is_report else schema.__name__].copy()
        if schema.__name__ == "ChoiceReport":
            template["answer_choice"] = "选项字母"
            prompt += "\n本题有明确选项，answer_choice必须是A至F中的一个大写字母，不能为null。"
        specification = json.dumps(template, ensure_ascii=False)
        instructions = (
            prompt + "\n只输出结果JSON，不要输出JSON Schema或properties、$defs。"
            "以下仅为字段结构示例，所有占位内容必须替换为本次结果。\n" + specification
        )
        if is_report:
            instructions += (
                "\n诊断name只写疾病名称，最多120字符，不能包含证据编号或方括号。"
                "引用只放在evidence_ids中，每条最多4个编号，不在名称中重复引用。"
                "每条依据不超过400字符，用短句说明。"
            )
        initial_instructions = instructions
        trace_dir = config.DATA / "traces"
        trace_dir.mkdir(exist_ok=True)
        for attempt in range(2):
            raw = self.generate(
                instructions,
                image_path,
                max_tokens,
                json_schema=schema.model_json_schema(),
            )
            (trace_dir / f"{uuid4().hex}.json").write_text(
                json.dumps(
                    {"schema": schema.__name__, "raw_output": raw},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            try:
                if self.metrics and self.metrics[-1].get("repetition_stopped"):
                    raise ValueError("检测到重复输出循环，本次结果未完成")
                payload = parse_json(raw, schema.model_fields.keys())
                if is_report and isinstance(payload.get("next_checks"), list):
                    structured = [
                        c for c in payload["next_checks"] if isinstance(c, dict)
                    ]
                    if structured and not payload.get("recommended_checks"):
                        # Some local outputs put the requested structured advice under the
                        # legacy field. Preserve its actual purpose and citations, then validate.
                        payload["recommended_checks"] = structured
                        payload["next_checks"] = [c.get("name", "") for c in structured]
                if is_report and isinstance(payload.get("differentials"), list):
                    # A proposed alternative with zero supporting evidence must not become
                    # a diagnosis; omit it rather than inventing support or failing a valid primary.
                    omitted = [
                        c.get("name", "")
                        for c in payload["differentials"]
                        if isinstance(c, dict) and c.get("support") == []
                    ]
                    payload["differentials"] = [
                        c
                        for c in payload["differentials"]
                        if not (isinstance(c, dict) and c.get("support") == [])
                    ]
                    if omitted and self.metrics:
                        self.metrics[-1]["omitted_unsupported_candidates"] = omitted
                if is_report and payload.get("recommended_checks"):
                    payload["next_checks"] = [
                        c["name"]
                        for c in payload["recommended_checks"]
                        if isinstance(c, dict) and "name" in c
                    ]
                return schema.model_validate(payload)
            except ValueError as error:
                if attempt:
                    raise
                # Start again from original evidence; never feed a looping partial
                # answer back as working memory or complete it by inventing values.
                retry = (
                    "\n上次输出未完成或结构无效。请从原始证据重新生成简短完整JSON，"
                    "不能续写上次输出，只使用当前字段结构，不补造事实或引用。"
                )
                if is_report:
                    retry += (
                        "诊断名称只写疾病，引用只写在evidence_ids。"
                        "最多保留一个有依据的鉴别项，每项只写最关键依据，"
                        "无实际依据的列表留空。"
                    )
                instructions = (
                    initial_instructions + retry + "\n错误类型：" + type(error).__name__
                )


backend = ModelBackend()
