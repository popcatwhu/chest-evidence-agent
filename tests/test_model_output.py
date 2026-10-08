import pytest
from pydantic import ValidationError
from chest_agent.schemas import Review


def test_schema_echo_is_not_a_successful_review():
    with pytest.raises(ValidationError):
        Review.model_validate({'properties':{'issues':[]},'title':'Review','type':'object'})


def test_missing_issues_cannot_default_to_passed():
    with pytest.raises(ValidationError):
        Review.model_validate({})
