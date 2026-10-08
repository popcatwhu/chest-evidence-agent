"""Pair public case histories with matching benchmark chest images, withholding outcomes."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chest_agent import store
from chest_agent.config import ROOT, DATA

metadata = json.loads((ROOT.parent / "MedRAX/data/eurorad_metadata.json").read_text())
store.init_db()
examples = []
references = {
    "6399": (
        "肺孢子菌肺炎",
        ["肺孢子虫肺炎", "肺孢子菌肺炎（PJP）", "Pneumocystis jirovecii pneumonia"],
    ),
    "10698": ("胸膜孤立性纤维性肿瘤", ["胸膜纤维瘤", "胸膜纤维性肿瘤"]),
    "11583": ("脓毒性肺栓塞", ["感染性肺栓塞", "脓毒性肺栓塞（感染性心内膜炎相关）"]),
}
existing = {c["title"] for c in store.list_cases()}
for case_id, (reference, aliases) in references.items():
    source = metadata[case_id]
    image = DATA / "benchmark" / f"case-{case_id}.png"
    if not image.exists():
        continue
    context = (
        f"公开病例 {case_id}。年龄：{source['age']}；性别：{source['gender']}。\n"
        f"病史（原文）：{source['history']}\n"
        "输入仅含病史与匹配的胸片。其他检查结果未提供。\n"
        f"来源：https://www.eurorad.org/case/{case_id}；病历元数据来自MedRAX，图像来自ChestAgentBench。"
    )
    title = f"公开病例 {case_id} · 病史与胸片"
    if title not in existing:
        case = store.create_case(title, context, str(image))
        print("Case created:", case["id"], case_id)
    examples.append(
        {
            "title": title,
            "context": context,
            "image_path": f"benchmark/case-{case_id}.png",
            "reference_diagnosis": reference,
            "accepted_names": aliases,
            "reference_source": source["diagnosis"],
        }
    )
(DATA / "curated_development.json").write_text(
    json.dumps(examples, ensure_ascii=False, indent=2)
)
print("Curated development cases:", len(examples), "Not an independent test set.")
