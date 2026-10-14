from chest_agent.correction_metrics import summarize


def test_correction_and_damage_are_separate_and_failures_count_as_wrong():
    rows = [
        {
            "reference_answer": "A",
            "draft_choice": "B",
            "final_choice": "A",
            "status": "completed",
        },
        {
            "reference_answer": "A",
            "draft_choice": "A",
            "final_choice": "B",
            "status": "completed",
        },
        {
            "reference_answer": "A",
            "draft_choice": "A",
            "final_choice": "A",
            "status": "failed",
        },
        {
            "reference_answer": "A",
            "draft_choice": None,
            "final_choice": None,
            "status": "failed",
        },
    ]
    result = summarize(rows)
    assert result["correction_rate"] == 1
    assert result["harm_rate"] == 1
    assert result["net_correct_answers_changed"] == -1
    assert result["draft_unavailable"] == 1
    assert result["final_accuracy"] == 0.25


def test_empty_denominators_are_unavailable_not_perfect_scores():
    assert summarize([])["correction_rate"] is None
    assert summarize([])["final_accuracy"] is None
