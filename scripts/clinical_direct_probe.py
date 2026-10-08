"""Known-case diagnostic probe without retrieved knowledge or generated tool observations."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chest_agent import config
from chest_agent.llm import backend
from chest_agent.schemas import Report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    examples = json.loads(
        (config.DATA / "quality/cases.json").read_text()
    ) + json.loads((config.DATA / "quality/heldout.json").read_text())
    records = []
    for example in examples:
        sources = "\n".join(
            f"[C-{i + 1}] {line}"
            for i, line in enumerate(example["context"].splitlines())
            if line.strip()
        )
        prompt = (
            "根据所提供病史、检查结果和原始胸片，给出最可能的具体疾病和至多两个鉴别诊断。"
            "用中文解释支持和反对理由，引用对应C编号或I-original。未提供的资料不是阴性结果；不要编造数值或影像尺寸。"
            "诊断为疑似，assessment写有限。这里只考察诊断推理，不生成确认检查建议；"
            "recommended_checks=[]、next_checks=[]、answer_choice=null。本次没有任何K来源，不可引用K编号。"
            "不要把影像现象当成具体病因，说明关键病史如何影响判断。\n病史：\n"
            + sources
            + "\n[I-original] 随请求附上的原始胸片，可直接观察。"
        )
        try:
            report = backend.json(
                prompt,
                Report,
                str((config.DATA / example["image_path"]).resolve()),
                max_tokens=1400,
            )
            status = "completed"
            error = None
        except Exception as exception:
            report = None
            status = "failed"
            error = str(exception)
        records.append(
            {
                "title": example["title"],
                "reference": example["reference_diagnosis"],
                "status": status,
                "error": error,
                "report": report.model_dump() if report else None,
            }
        )
        args.output.write_text(
            json.dumps(
                {
                    "purpose": "known-development diagnostic isolation probe",
                    "model": config.MODEL.name,
                    "clinical_accuracy_validated": False,
                    "records": records,
                    "complete": len(records) == len(examples),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(
            example["title"],
            status,
            report.most_likely.name if report else error,
            flush=True,
        )
