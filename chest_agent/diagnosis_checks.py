"""Check equivalent disease names without selecting or replacing the diagnosis."""
import re
from .quality import normalized

# Human PCP/PJP nomenclature: https://www.cdc.gov/pneumocystis-pneumonia/about/
PJP_NAMES = {
    '肺孢子菌肺炎', '肺孢子虫肺炎', '卡氏肺孢子虫肺炎', '卡氏肺孢子菌肺炎',
    '肺孢子菌感染', '肺孢子虫感染', 'PCP', 'PJP', 'Pneumocystis pneumonia',
    'Pneumocystis jirovecii pneumonia', 'Pneumocystis carinii pneumonia',
}


def canonical_name(name):
    name = re.sub(r'^(?:疑似|考虑|可能|suspected\s+|probable\s+|possible\s+)', '', name, flags=re.I)
    name = normalized(name).strip(' ()（）?？。.')
    return 'pjp' if name in {normalized(x) for x in PJP_NAMES} else name


def diagnosis_issues(report):
    seen = {canonical_name(report.most_likely.name): report.most_likely.name}
    issues = []
    for candidate in report.differentials:
        key = canonical_name(candidate.name)
        if key in seen:
            issues.append(f'鉴别项“{candidate.name}”与“{seen[key]}”是相同或同义诊断，不能作为不同病因重复列出。')
        else:
            seen[key] = candidate.name
    return issues
