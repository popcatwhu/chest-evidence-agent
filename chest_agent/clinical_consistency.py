"""Check narrow, source-backed inconsistencies reached by the development workflow."""

import json
import re
from .schemas import Report, Evidence


def clinical_inconsistencies(report: Report, evidence: list[Evidence]):
    issues = []
    sources = [e for e in evidence if e.kind == "knowledge" and e.status == "completed"]
    case_text = " ".join(e.content for e in evidence if e.kind == "case")
    primary = report.most_likely.name
    if any(
        s.diagnostic_rules
        and any(r.get("id", "").startswith("nih-pcp") for r in s.diagnostic_rules)
        for s in sources
    ):
        if re.search(r"肺孢子菌|肺孢子虫|PCP|PJP|Pneumocystis", primary, re.I):
            immune_basis = bool(
                re.search(
                    r"AIDS|艾滋病|immunosuppression|免疫抑制|移植|transplant|corticosteroid",
                    case_text,
                    re.I,
                )
            )
            cd4 = re.search(r"CD4[^\n\d]{0,35}(\d+(?:\.\d+)?)", case_text, re.I)
            immune_basis = immune_basis or bool(cd4 and float(cd4.group(1)) < 200)
            if not immune_basis:
                for claim in report.most_likely.support:
                    if re.search(
                        r"高危|免疫抑制|HIV|AIDS", claim.text, re.I
                    ) and not re.search(
                        r"未知|未提供|待查|待确认|可能感染|需.*检查", claim.text
                    ):
                        issues.append(
                            "报告把未提供的HIV/免疫抑制状态当作已知高危事实；静脉药物使用史不能代替免疫检查结果："
                            + claim.text
                        )
    expert = next(
        (e for e in evidence if e.id == "I-radiology" and e.status == "completed"), None
    )
    volume_source = next((e for e in sources if "opaque" in e.title.casefold()), None)
    if expert and volume_source:
        observations = " ".join(json.loads(expert.content)["visual_findings"])
        left = re.search(
            r"opacification of (?:the )?left hemi\s*thorax", observations, re.I
        )
        right = re.search(
            r"opacification of (?:the )?right hemi\s*thorax", observations, re.I
        )
        shift_right = re.search(
            r"(?:mediastinal )?shift to (?:the )?right", observations, re.I
        )
        shift_left = re.search(
            r"(?:mediastinal )?shift to (?:the )?left", observations, re.I
        )
        if (left and shift_right) or (right and shift_left):
            if re.search(
                r"不张|塌陷|atelectasis|collapse", primary, re.I
            ) and not re.search(r"占位|肿块|积液|mass|effusion", primary, re.I):
                issues.append(
                    f"单纯肺不张不能充分解释专用模型观察到的对侧纵隔移位（{volume_source.id}）；应鉴别占位或大量积液，并通过CT/超声复核，而不是直接接受模型的肺塌陷解释。"
                )
    return issues
