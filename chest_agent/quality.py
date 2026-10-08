"""Grounding checks for source facts and actionable report review."""

import re
import unicodedata
from .schemas import CaseProfile, Report, Evidence, Review


def normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def ground_facts(profile: CaseProfile, evidence: list[Evidence]):
    sources = {
        e.id: e.content
        for e in evidence
        if e.kind == "case" and e.status == "completed"
    }
    accepted = []
    rejected = []
    for fact in profile.facts:
        source = sources.get(fact.source_id, "")
        if not source or normalized(fact.source_quote) not in normalized(source):
            rejected.append(f"{fact.name}：引用原文不匹配")
            continue
        numbers = re.findall(
            r"\d+(?:\.\d+)?", unicodedata.normalize("NFKC", fact.value)
        )
        quoted_numbers = set(
            re.findall(
                r"\d+(?:\.\d+)?", unicodedata.normalize("NFKC", fact.source_quote)
            )
        )
        if any(n not in quoted_numbers for n in numbers):
            rejected.append(f"{fact.name}：数值没有原文依据")
            continue
        accepted.append(fact)
    return CaseProfile(facts=accepted), rejected


def report_texts(report: Report):
    candidates = [report.most_likely, *report.differentials]
    return (
        [c.name for c in candidates]
        + [claim.text for c in candidates for claim in c.support + c.against]
        + [claim.text for claim in report.findings]
        + report.missing_information
        + report.next_checks
    )


def anchor_review(review: Review, report: Report, evidence: list[Evidence]):
    texts = report_texts(report)
    candidates = [report.most_likely, *report.differentials]
    claims = [
        *report.findings,
        *[claim for c in candidates for claim in c.support + c.against],
    ]
    cited_by_text = {claim.text: set(claim.evidence_ids) for claim in claims}
    cited_by_text.update(
        {
            c.name: {ref for claim in c.support for ref in claim.evidence_ids}
            for c in candidates
        }
    )
    available = {e.id: e for e in evidence if e.status == "completed"}
    accepted = []
    rejected = []
    for issue in review.issues:
        if issue.report_quote not in texts:
            rejected.append("复核意见无法对应报告原句：" + issue.report_quote)
        elif not all(ref in available for ref in issue.evidence_ids):
            rejected.append("复核意见引用无效证据")
        elif not any(
            normalized(issue.source_quote) in normalized(available[ref].content)
            for ref in issue.evidence_ids
        ):
            rejected.append("复核意见缺乏可匹配的证据原文")
        elif re.search(r"分类|分数|概率", issue.reason) and (
            "I-classify" not in issue.evidence_ids
            or "I-classify" not in cited_by_text.get(issue.report_quote, set())
        ):
            rejected.append("复核将其他证据误认为分类分数")
        else:
            accepted.append(issue)
    return accepted, rejected


def known_test_conflicts(report: Report, profile: CaseProfile):
    """Reject requests for already provided test results; leave repeat testing recommendations alone."""
    issues = []
    for fact in profile.facts:
        # Match a full test name/abbreviation rather than generic symptoms or demographic facts.
        name = fact.name
        identifier = re.search(
            r"CD4|LDH|CRP|WBC|ESR|血氧|血压|体温|血糖|血红蛋白",
            name + " " + fact.value,
            re.I,
        )
        if not identifier:
            continue
        for missing in report.missing_information:
            if normalized(identifier.group()) not in normalized(missing):
                continue
            if re.search(r"具体|数值|水平|计数", missing) and not re.search(
                r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])", fact.value
            ):
                continue  # "elevated" does not supply a numerical result.
            issues.append(
                f"已提供{name}={fact.value}（{fact.source_id}），不能再次列为缺失资料：{missing}"
            )
    return list(dict.fromkeys(issues))
