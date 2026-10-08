from typing import Literal
import re
from pydantic import BaseModel, Field, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(BaseModel):
    id: str
    kind: Literal["case", "image", "knowledge"]
    title: str
    content: str
    status: Literal["completed", "failed"] = "completed"
    source: str | None = None
    artifact: str | None = None
    source_type: str | None = None
    diagnostic_rules: list[dict] = Field(default_factory=list)


class Plan(StrictModel):
    tools: list[Literal["classify", "segment"]] = Field(default_factory=list)
    query: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    hypotheses: list[str] = Field(default_factory=list,max_length=4)


class Fact(StrictModel):
    name: str
    value: str
    source_id: str
    source_quote: str = Field(min_length=1)


class CaseProfile(StrictModel):
    facts: list[Fact]


class Observation(StrictModel):
    observations: list[str]
    limitations: list[str]


class Claim(StrictModel):
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class Candidate(StrictModel):
    name: str = Field(min_length=1)
    support: list[Claim] = Field(min_length=1)
    against: list[Claim] = Field(default_factory=list)


class RecommendedCheck(StrictModel):
    name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


ChoiceLetter = Literal['A','B','C','D','E','F']


class Report(StrictModel):
    answer_choice: ChoiceLetter | None = None
    # Missing confidence metadata never promotes a draft to sufficient evidence.
    assessment: Literal["有限", "较充分", "存在冲突"] = "有限"
    most_likely: Candidate
    differentials: list[Candidate] = Field(default_factory=list, max_length=2)
    findings: list[Claim] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)
    recommended_checks: list[RecommendedCheck] = Field(default_factory=list,max_length=8)
    change_summary: str = "首次分析"


class ChoiceReport(Report):
    answer_choice: ChoiceLetter


def report_schema(question: str):
    letters=set(re.findall(r'(?m)^\s*([A-F])[).、]\s*',question))
    return ChoiceReport if {'A','B'} <= letters else Report


class ReviewIssue(StrictModel):
    source_quote: str = Field(min_length=1)
    source_type: str | None = None
    report_quote: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class Review(StrictModel):
    issues: list[ReviewIssue]


class Change(StrictModel):
    summary: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


def validate_evidence(report: Report, evidence: list[Evidence],allow_original=True) -> list[str]:
    """Verify citation integrity and patient evidence; this is not clinical validation."""
    available = {e.id: e for e in evidence if e.status == "completed"}
    issues = []
    if re.search(r'肺野密度|纹理增|浸润影|弥漫性阴影',report.most_likely.name):
        issues.append('主结论仍是影像表现，没有形成明确的疾病候选，需要进一步评估')
    candidates = [report.most_likely, *report.differentials]
    claims = [*report.findings]
    for candidate in candidates:
        claims.extend(candidate.support + candidate.against)
        if not any(
            available.get(ref) and available[ref].kind in {"case", "image"}
            for claim in candidate.support for ref in claim.evidence_ids
        ):
            issues.append(f"候选诊断 {candidate.name} 缺少患者自身证据")
    for claim in claims:
        if (claim.evidence_ids == ["I-classify"] and
                re.search(r"未见|未发现|排除|不存在|没有|正常",claim.text)):
            issues.append("分类分数不能直接证明影像正常或排除疾病：" + claim.text)
        for ref in claim.evidence_ids:
            if ref=='I-original' and not allow_original:
                issues.append('本轮诊断推理未直接读取原图，应引用实际接收的工具影像观察，不能引用I-original假装直接核验')
            if ref not in available:
                issues.append(f"结论引用不存在或失败的证据：{ref}")
    return list(dict.fromkeys(issues))
