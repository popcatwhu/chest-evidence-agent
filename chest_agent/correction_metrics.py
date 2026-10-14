"""Separate diagnostic correction, damage and completion from final MCQ accuracy."""


def summarize(records):
    valid = lambda value: value in set("ABCDEF") if isinstance(value, str) else False
    wrong, right, corrected, harmed, unavailable, final_correct = 0, 0, 0, 0, 0, 0
    for row in records:
        target = row["reference_answer"]
        draft = row.get("draft_choice")
        final = row.get("final_choice") if row["status"] == "completed" else None
        final_correct += int(valid(final) and final == target)
        if not valid(draft):
            unavailable += 1
        elif draft == target:
            right += 1
            harmed += int(final != target)
        else:
            wrong += 1
            corrected += int(final == target)
    return {
        "cases": len(records),
        "completed": sum(r["status"] == "completed" for r in records),
        "draft_correct": right,
        "draft_wrong": wrong,
        "draft_unavailable": unavailable,
        "wrong_drafts_corrected": corrected,
        "correct_drafts_harmed": harmed,
        "correction_rate": corrected / wrong if wrong else None,
        "harm_rate": harmed / right if right else None,
        "net_correct_answers_changed": corrected - harmed,
        "final_correct": final_correct,
        "final_accuracy": final_correct / len(records) if records else None,
    }
