from scripts.evaluate_http import summarize


def test_failures_stay_in_denominator_and_pairing_uses_patients():
    records = [
        {
            "case_id": "a",
            "mode": "direct",
            "status": "completed",
            "prediction": "B",
            "correct": True,
        },
        {
            "case_id": "a",
            "mode": "verified",
            "status": "failed",
            "prediction": None,
            "correct": False,
        },
        {
            "case_id": "b",
            "mode": "direct",
            "status": "completed",
            "prediction": "A",
            "correct": False,
        },
        {
            "case_id": "b",
            "mode": "verified",
            "status": "completed",
            "prediction": "C",
            "correct": True,
        },
    ]
    result = summarize(records, ["direct", "verified"])
    assert result["modes"]["verified"]["accuracy"] == 0.5
    assert result["modes"]["verified"]["tasks"] == 2
    assert result["modes"]["verified"]["completed"] == 1
    assert result["modes"]["verified"]["invalid_or_missing_choices"] == 1
    assert result["paired"] == {
        "patients": 2,
        "both_correct": 0,
        "direct_only_correct": 1,
        "verified_only_correct": 1,
        "both_wrong": 0,
    }
    assert result["modes"]["direct"]["wilson_95_ci"] == [0.0945, 0.9055]
