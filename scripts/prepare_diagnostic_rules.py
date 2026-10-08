"""General disease-level constraints from NIH PCP diagnostic guidance."""
import json
from pathlib import Path

path=Path('data/knowledge/nih_pcp.json')
doc=json.loads(path.read_text())
sentence=('Pneumocystis cannot be routinely cultivated. Routine sputum culture does not establish PCP; '
    'respiratory sample staining or PCR is relevant. LDH and beta-D-glucan are not independently confirmatory.')
if sentence not in doc['text']:doc['text']+='\n\n'+sentence
doc['diagnostic_rules']=[
    {'id':'nih-pcp-confirmation-method','condition_terms':['Pneumocystis','肺孢子菌','肺孢子虫','PCP','PJP'],
     'required_confirmation_patterns':['PCR','聚合酶','染色','免疫荧光','stain','bronchoalveolar','支气管肺泡灌洗','BAL','活检','biopsy'],
     'reason':'确认PCP应说明从合适呼吸道或组织标本检测肺孢子菌的方法；仅排查其他细菌感染不能确认PCP。'},
    {'id':'nih-pcp-culture','condition_terms':['Pneumocystis','肺孢子菌','肺孢子虫','PCP','PJP'],
     'unsupported_patterns':['痰(?:液)?培养','sputum culture','常规培养','routine culture'],
     'reason':'肺孢子菌无法常规培养，普通痰培养不能确诊PCP；应依据合适呼吸道标本的染色或PCR，培养其他细菌需明确鉴别目的。'},
    {'id':'nih-pcp-nonspecific-marker','condition_terms':['Pneumocystis','肺孢子菌','肺孢子虫','PCP','PJP'],
     'unsupported_patterns':['LDH','乳酸脱氢酶','β.?D.?葡聚糖','beta.?D.?glucan'],
     'confirmation_only':True,'reason':'LDH和β-D-葡聚糖只能辅助评估，不能独立确诊PCP。'},
]
path.write_text(json.dumps(doc,ensure_ascii=False,indent=2))
print('NIH confirmation constraints prepared; source:',doc['url'])
