"""Bounded evidence searches and adjudication for independent diagnostic disagreement."""

import json
import re
from typing import Literal, Annotated

from pydantic import Field, field_validator

from .diagnosis_checks import canonical_name
from .schemas import EvidenceID, StrictModel
from .independent_review import source_index


class DisputePlan(StrictModel):
    disputed_point: str = Field(min_length=1, max_length=240)
    queries: list[Annotated[str, Field(min_length=1, max_length=240)]] = Field(
        min_length=1, max_length=2
    )
    decision_criteria: list[Annotated[str, Field(min_length=1, max_length=240)]] = (
        Field(min_length=1, max_length=3)
    )

    @field_validator("queries")
    @classmethod
    def queries_are_not_template(cls, value):
        if "English differential diagnosis query" in value:
            raise ValueError("检索查询仍是结构示例占位内容")
        return value


class Adjudication(StrictModel):
    decision: Literal["keep", "revise", "unresolved"]
    preferred_choice: str | None = Field(default=None, max_length=1)
    reason: str = Field(min_length=1, max_length=600)
    evidence_ids: list[EvidenceID] = Field(min_length=1, max_length=5)
    criterion_source_id: EvidenceID | None = None

    @field_validator("reason")
    @classmethod
    def reason_is_not_template(cls, value):
        if value == "患者证据如何支持或反驳两个候选，缺少哪些决定性依据":
            raise ValueError("裁决理由仍是结构示例占位内容")
        return value


def disagrees(state):
    opinion = state.get("blind_opinion")
    if not opinion or not opinion.candidates:
        return False
    report = state["report"]
    if opinion.answer_choice and report.answer_choice:
        return opinion.answer_choice != report.answer_choice
    return canonical_name(opinion.candidates[0].name) != canonical_name(
        report.most_likely.name
    )


def dispute_context(state):
    return (
        "任务与选项（不是患者事实）："
        + state.get("question", "")
        + "\n临床草稿："
        + state["report"].model_dump_json()
        + "\n独立意见（不是新证据）："
        + state["blind_opinion"].model_dump_json()
    )


def professional_source(description):
    text = (description or "").casefold()
    if re.search(r"public health information|patient information|consumer", text):
        return False
    professional = bool(re.search(r"professional|peer.reviewed", text))
    guideline = "guideline" in text and not re.search(
        r"not (?:a |an )?(?:clinical )?guideline", text
    )
    return professional or bool(guideline)


def adjudication_errors(decision, state, allow_original=True):
    available = {e.id: e for e in state["evidence"] if e.status == "completed"}
    errors = []
    if any(ref not in available for ref in decision.evidence_ids):
        errors.append("分歧裁决引用不存在或失败的证据")
    if "I-original" in decision.evidence_ids and not allow_original:
        errors.append("裁决未直接读取原图，不能引用I-original声称视觉核验")
    if decision.decision == "revise":
        criterion = next(
            (
                r
                for r in source_index(state["evidence"])
                if r["id"] == decision.criterion_source_id
            ),
            None,
        )
        if (
            not criterion
            or criterion["kind"] != "knowledge"
            or not professional_source(criterion.get("source_type"))
        ):
            errors.append(
                "改诊断缺少可定位的专业医学判据，不能仅凭另一个模型的意见修改"
            )
        elif criterion["evidence_id"] not in decision.evidence_ids:
            errors.append("裁决没有引用判据所在的资料证据")
    if not any(
        available.get(ref)
        and available[ref].kind in {"case", "image"}
        and ref not in {"I-classify", "I-segment"}
        for ref in decision.evidence_ids
    ):
        errors.append("分歧裁决缺少患者病史或实际影像依据")
    if state["report"].answer_choice:
        letters = set(
            re.findall(r"(?m)^\s*([A-F])[).、]\s*", state.get("question", ""))
        )
        if decision.preferred_choice and decision.preferred_choice not in letters:
            errors.append("分歧裁决选择不存在的选项")
        if decision.decision == "revise" and (
            not decision.preferred_choice
            or decision.preferred_choice == state["report"].answer_choice
        ):
            errors.append("修订裁决没有提出与草稿不同的有效选项")
        if decision.decision == "keep" and decision.preferred_choice not in {
            None,
            state["report"].answer_choice,
        }:
            errors.append("保留裁决与其选项不一致")
    elif decision.preferred_choice is not None:
        errors.append("非选择题裁决不能指定选项字母")
    return errors


def repair_dispute_context(state):
    decision = state.get("adjudication")
    if not decision:
        return ""
    return "\n分歧裁决（模型建议，必须再次对照患者证据）：" + json.dumps(
        decision, ensure_ascii=False
    )
