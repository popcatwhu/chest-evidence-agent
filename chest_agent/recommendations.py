"""Source-provenanced confirmation-test checks; rules describe diseases, never patient IDs."""

import re
from .schemas import Report, Evidence
from .quality import normalized


def check_recommendations(
    report: Report, evidence: list[Evidence], require_sources=True
):
    available = {
        e.id: e for e in evidence if e.kind == "knowledge" and e.status == "completed"
    }
    issues = []
    if require_sources and report.next_checks and not report.recommended_checks:
        issues.append("进一步检查建议缺少用途与诊断资料引用，请提供recommended_checks")
    for check in report.recommended_checks:
        if not any(ref in available for ref in check.evidence_ids):
            issues.append(f"检查建议“{check.name}”缺少有效的诊断资料来源")
    primary = normalized(report.most_likely.name)
    proposals = [
        (check.name + "：" + check.purpose) for check in report.recommended_checks
    ]
    proposals += report.missing_information
    if not report.recommended_checks:
        proposals += report.next_checks
    seen = set()
    for e in available.values():
        for rule in e.diagnostic_rules:
            if rule["id"] in seen or not any(
                normalized(term) in primary for term in rule["condition_terms"]
            ):
                continue
            seen.add(rule["id"])
            if rule.get("required_confirmation_patterns") and not any(
                re.search(pattern, proposal, re.I)
                for pattern in rule["required_confirmation_patterns"]
                for proposal in proposals
            ):
                issues.append(
                    f"未说明针对当前候选的确认方法（{e.id}）：{rule['reason']}"
                )
            for proposal in proposals:
                if re.search(
                    r"不推荐|不建议|不要|不能靠|avoid|not recommended", proposal, re.I
                ):
                    continue
                if not any(
                    re.search(pattern, proposal, re.I)
                    for pattern in rule.get("unsupported_patterns", [])
                ):
                    continue
                # Cultures to investigate a competing bacterial infection can be appropriate;
                # they cannot establish Pneumocystis itself.
                if re.search(r"细菌|bacterial|鉴别其他|排查其他", proposal, re.I):
                    continue
                if rule.get("confirmation_only") and not re.search(
                    r"确诊|确认病原体|confirm|definitive", proposal, re.I
                ):
                    continue
                issues.append(f"“{proposal}”与{e.id}诊断资料冲突：{rule['reason']}")
    return list(dict.fromkeys(issues))
