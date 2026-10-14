"""Run the local reviewer on synthetic contradictions and matched correct controls.

This checks review behavior, not diagnostic accuracy. No benchmark labels are used.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from chest_agent import agents, workflow
from chest_agent.schemas import CaseProfile, Evidence, Report


def main():
    root = Path(__file__).resolve().parent.parent
    output = root / "data/review_regression"
    output.mkdir(parents=True, exist_ok=True)
    pairs = [
        (
            "smoking",
            "A 45-year-old patient with fever and cough. Never smoked.",
            "患者长期吸烟。",
            "患者从不吸烟。",
        ),
        (
            "age",
            "A 45-year-old patient with fever and cough.",
            "患者80岁。",
            "患者45岁。",
        ),
        (
            "fever",
            "A 45-year-old patient reports fever and cough for three days.",
            "患者无发热。",
            "患者发热三天。",
        ),
    ]
    records = []
    original_log = workflow.log
    workflow.log = lambda *args: None
    try:
        for name, history, wrong, correct in pairs:
            for injected, claim in [(True, wrong), (False, correct)]:
                draft = Report.model_validate(
                    {
                        "most_likely": {
                            "name": "肺炎待排",
                            "support": [
                                {"text": "发热及咳嗽", "evidence_ids": ["C-1"]}
                            ],
                        },
                        "findings": [{"text": claim, "evidence_ids": ["C-1"]}],
                    }
                )
                state = {
                    "mode": "verified",
                    "case": {"context": history},
                    "question": "依据病例检查报告事实是否与原文一致。",
                    "evidence": [
                        Evidence(
                            id="C-1",
                            kind="case",
                            title="synthetic history",
                            content=history,
                        )
                    ],
                    "report": draft,
                    "profile": CaseProfile(facts=[]),
                    "repairs": 0,
                }
                result = workflow.verify(state)
                message = result["agent_messages"][-1]
                accepted = message.get("output", {}).get("accepted_challenges", [])
                target_detected = any(i["report_quote"] == claim for i in accepted)
                record = {
                    "pair": name,
                    "injected": injected,
                    "completed": message["status"] == "completed",
                    "target_detected": target_detected,
                    "accepted_count": len(accepted),
                    "matched_expected_behavior": target_detected
                    if injected
                    else not accepted,
                    "message": message,
                }
                records.append(record)
                (output / "results.json").write_text(
                    json.dumps(records, ensure_ascii=False, indent=2)
                )
                print(
                    json.dumps({k: v for k, v in record.items() if k != "message"}),
                    flush=True,
                )
    finally:
        workflow.log = original_log
        agents.reviewer.model.release()
    summary = {
        "date": datetime.now(timezone.utc).isoformat(),
        "type": "synthetic contradiction review regression",
        "model": agents.reviewer.identity(),
        "cases": len(records),
        "completed": sum(r["completed"] for r in records),
        "injected_errors_detected": sum(
            r["injected"] and r["target_detected"] for r in records
        ),
        "injected_errors": len(pairs),
        "correct_controls_flagged": sum(
            not r["injected"] and r["accepted_count"] > 0 for r in records
        ),
        "correct_controls": len(pairs),
        "limitations": [
            "Small synthetic engineering check, not diagnostic accuracy.",
            "Only literal patient-fact contradictions are tested.",
            "Frozen 60-patient evaluation remains unchanged.",
        ],
    }
    (root / "evals/review-regression.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
