"""Separate model agents with bounded tasks and one auxiliary GPU residency slot."""

from contextlib import contextmanager
from dataclasses import dataclass
import json
from threading import Lock

from . import config
from .llm import ModelBackend, backend


class AuxiliaryModels:
    def __init__(self):
        self.lock = Lock()
        self.active = None

    @contextmanager
    def use(self, model):
        with self.lock:
            if self.active is not None and self.active is not model:
                self.active.release()
            self.active = model
            # Generation caches from the primary model need not occupy free VRAM.
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield model


auxiliary_models = AuxiliaryModels()


@dataclass
class ModelAgent:
    name: str
    model: ModelBackend
    instruction: str
    auxiliary: bool = False

    def run(self, prompt, schema, image=None, max_tokens=800):
        task = self.instruction + "\n\n" + prompt
        start = len(self.model.metrics)
        try:
            if self.auxiliary:
                with auxiliary_models.use(self.model):
                    return self.model.json(task, schema, image, max_tokens=max_tokens)
            return self.model.json(task, schema, image, max_tokens=max_tokens)
        finally:
            for metric in self.model.metrics[start:]:
                metric["agent"] = self.name

    def identity(self):
        path = self.model.model_path
        revision = path / "source_revision.json"
        return {
            "agent": self.name,
            "model": config.API_MODEL
            if self.model.backend_type == "api"
            else path.name,
            "backend": self.model.backend_type,
            "quantization": "NF4"
            if self.model.load_nf4 or "NF4" in path.name
            else "BF16",
            "source_revision": json.loads(revision.read_text())
            if revision.exists()
            else None,
        }


radiologist = ModelAgent(
    "radiology",
    ModelBackend(
        model_path=config.RADIOLOGY_MODEL, backend_type="local", load_nf4=False
    ),
    "You are the radiology agent. Independently inspect the supplied chest image. "
    "You do not receive the clinical agent's diagnosis or patient history. "
    "Report visible morphology and locations, separating uncertainty from observation. "
    "Do not infer etiologies, unseen comparisons, or unprovided patient facts.",
    auxiliary=True,
)
clinician = ModelAgent(
    "clinical",
    backend,
    "你是临床鉴别诊断 Agent。基于病史、原图和可追溯证据提出候选疾病。"
    "影像 Agent 的输出是待核实的模型观察，不能当成已确认病史。",
)
reviewer = ModelAgent(
    "review",
    ModelBackend(model_path=config.REVIEW_MODEL, backend_type="local", load_nf4=True),
    "You are an independent evidence-review agent using a different model. "
    "Inspect the supplied report against original patient evidence and image. "
    "Challenge specific unsupported statements with exact quotations and source IDs. "
    "Do not defer to another agent's authority or majority agreement.",
    auxiliary=True,
)
coordinator = ModelAgent(
    "coordinator",
    backend,
    "你是协调 Agent。依据原始证据处理独立审查 Agent 的质疑，"
    "修订临床 Agent 的报告。不能通过投票确定诊断，不得把 Agent 意见当成新增证据。",
)


def roster():
    return [
        agent.identity() for agent in (radiologist, clinician, reviewer, coordinator)
    ]


def message(agent, status, output=None, error=None):
    return {**agent.identity(), "status": status, "output": output, "error": error}
