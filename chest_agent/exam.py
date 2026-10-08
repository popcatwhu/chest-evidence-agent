"""Minimal MCQ inference and question-blind checks of generated CXR assertions.

Reference answers, captions and final diagnoses have no place in these functions.
Tool consistency is model-estimated and does not constitute clinical validation.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal
from pydantic import Field
from . import config
from .llm import backend
from .schemas import StrictModel, ChoiceLetter
from .tools import image_tools


class ExamAnswer(StrictModel):
    reason: str = Field(min_length=1, max_length=900)
    answer_choice: ChoiceLetter


class ToolCheck(StrictModel):
    claim_id: str
    status: Literal["supported", "contradicted", "uncertain"]
    visual_basis: str = Field(min_length=1, max_length=350)


class ToolChecks(StrictModel):
    checks: list[ToolCheck] = Field(max_length=16)


def tool_signature():
    digest = hashlib.sha256()
    digest.update(config.CXR_READER.encode())
    paths = [
        Path(__file__).with_name("tools.py"),
        Path(__file__).with_name("radiology.py"),
    ]
    if config.CXR_READER == "nvreason":
        paths.extend(
            [
                Path(__file__).with_name("cxr_reader.py"),
                config.NVREASON_MODEL / "config.json",
                config.NVREASON_MODEL / "source_revision.json",
            ]
        )
    else:
        paths.append(config.CXR_FINDINGS_MODEL / "config.json")
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def raw_tools(image_path):
    image_sha = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    signature = tool_signature()
    cache = config.DATA / "exam_tools"
    cache.mkdir(exist_ok=True)
    destination = cache / f"{image_sha}-{signature[:12]}.json"
    if destination.exists():
        return json.loads(destination.read_text())
    result = {
        "image_sha256": image_sha,
        "tool_signature": signature,
        "claims": [],
        "scores": {},
        "errors": [],
    }
    try:
        evidence = image_tools.findings(image_path, "exam")
        content = json.loads(evidence.content)
        result["radiology_raw"] = content
        for i, text in enumerate(content["visual_findings"]):
            result["claims"].append({"claim_id": f"F-{i + 1}", "text": text})
    except Exception as error:
        result["errors"].append({"tool": "radiology", "error": str(error)})
    try:
        result["scores"] = json.loads(image_tools.classify(image_path, "exam").content)[
            "scores"
        ]
    except Exception as error:
        result["errors"].append({"tool": "classify", "error": str(error)})
    # Failed tools may be retried; they never become normal image findings.
    if not result["errors"]:
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def validate_claims(claims, image_path):
    if not claims:
        return {"checks": [], "accepted": [], "invalid": []}
    response = backend.json(
        "Independently check each generated radiograph statement against the attached original image. "
        "The statements are fallible model outputs, not ground truth. You do not receive a diagnosis or question. "
        "Mark supported only when visible morphology and location support the complete statement; "
        "contradicted when visible evidence opposes it; uncertain when the image does not establish it. "
        "For negative statements explicitly inspect the relevant area. If only part of a compound statement "
        "is supported, use uncertain. Describe visible evidence in at most 20 words per item. "
        "Return exactly one item per supplied claim_id. Do not make disease or etiologic diagnoses.\nStatements:\n"
        + json.dumps(claims, ensure_ascii=False),
        ToolChecks,
        image_path,
        max_tokens=1100,
    )
    expected = {c["claim_id"]: c for c in claims}
    counts = {key: sum(c.claim_id == key for c in response.checks) for key in expected}
    valid = [
        c for c in response.checks if c.claim_id in expected and counts[c.claim_id] == 1
    ]
    invalid = [
        c.model_dump()
        for c in response.checks
        if c.claim_id not in expected or counts.get(c.claim_id) != 1
    ]
    invalid += [
        {"claim_id": key, "reason": "missing or duplicated"}
        for key in expected
        if counts[key] != 1
    ]
    accepted = [expected[c.claim_id] for c in valid if c.status == "supported"]
    return {
        "checks": [c.model_dump() for c in valid],
        "accepted": accepted,
        "invalid": invalid,
    }


def auxiliary_text(tools, claims):
    scores = sorted(tools["scores"].items(), key=lambda p: p[1], reverse=True)[:5]
    return (
        "Auxiliary classifier scores, not patient probabilities; low scores cannot exclude disease: "
        + ", ".join(f"{k}={v:.3f}" for k, v in scores)
        + "\nFallible model-generated observations:\n"
        + json.dumps(claims, ensure_ascii=False)
        + "\nReconcile these with the original image; they do not override direct image evidence."
    )


def answer(question, image_path, variant="direct", tools=None, validation=None):
    if variant not in {"direct", "tools", "validated"}:
        raise ValueError("Unknown exam variant")
    auxiliary = ""
    if variant != "direct":
        if tools is None:
            tools = raw_tools(image_path)
        claims = tools["claims"]
        if variant == "validated":
            if validation is None:
                validation = validate_claims(claims, image_path)
            claims = validation["accepted"]
        auxiliary = "\n" + auxiliary_text(tools, claims)
    prompt = (
        "Answer this six-option chest-radiograph question using the original image and information "
        "in the question. Compare the options, explain the decisive evidence in at most two short English "
        "sentences, then choose exactly one uppercase letter A to F. Do not generate a clinical report, "
        "additional examinations or prescriptions.\nQuestion:\n" + question + auxiliary
    )
    return backend.json(prompt, ExamAnswer, image_path, max_tokens=320)
