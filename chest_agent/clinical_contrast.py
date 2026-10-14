"""Patient-grounded evidence contrasts; intermediate hypotheses are never new patient facts."""

import json
from pydantic import Field
from chest_agent.schemas import (
    StrictModel,
    Fact,
    Claim,
    Candidate,
    CaseProfile,
    Report,
    validate_evidence,
    EvidenceID,
)
from chest_agent.quality import ground_facts, normalized
from chest_agent.diagnosis_checks import canonical_name
from chest_agent.independent_review import (
    SelectedFact,
    SelectedCaseProfile,
    resolve_facts,
)


class ConciseFact(Fact):
    name: str = Field(min_length=1, max_length=70)
    value: str = Field(min_length=1, max_length=200)
    source_quote: str = Field(min_length=1, max_length=220)


class ConciseClaim(Claim):
    text: str = Field(min_length=1, max_length=240)
    evidence_ids: list[EvidenceID] = Field(min_length=1, max_length=3)


class Hypothesis(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    support: list[ConciseClaim] = Field(min_length=1, max_length=2)
    against: list[ConciseClaim] = Field(default_factory=list, max_length=2)
    still_unknown: list[str] = Field(default_factory=list, max_length=2)


class ObservationConflict(StrictModel):
    first_source: str
    first_quote: str = Field(min_length=1, max_length=220)
    second_source: str
    second_quote: str = Field(min_length=1, max_length=220)
    description: str = Field(min_length=1, max_length=240)


class ClinicalContrast(StrictModel):
    key_case_clues: list[ConciseFact] = Field(default_factory=list, max_length=5)
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=3)
    observation_conflicts: list[ObservationConflict] = Field(
        default_factory=list, max_length=3
    )
    remaining_questions: list[str] = Field(default_factory=list, max_length=3)


class IndexedClinicalContrast(ClinicalContrast):
    key_case_clues: list[SelectedFact] = Field(default_factory=list, max_length=5)


def resolve_contrast(raw, source_rows):
    profile, notes = resolve_facts(
        SelectedCaseProfile(facts=raw.key_case_clues), source_rows
    )
    data = raw.model_dump(exclude={"key_case_clues"})
    references = {r["id"]: r["evidence_id"] for r in source_rows}
    for candidate in data["hypotheses"]:
        for claim in candidate["support"] + candidate["against"]:
            claim["evidence_ids"] = list(
                dict.fromkeys(references.get(ref, ref) for ref in claim["evidence_ids"])
            )
    result = ClinicalContrast.model_validate(
        {
            **data,
            "key_case_clues": [fact.model_dump() for fact in profile.facts],
        }
    )
    return result, notes


def source_text(evidence):
    """Compare quotes to actual source values, including JSON-encoded observer text."""
    try:
        data = json.loads(evidence.content)
    except (json.JSONDecodeError, TypeError):
        return evidence.content

    def strings(value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [s for item in value for s in strings(item)]
        if isinstance(value, dict):
            return [s for item in value.values() for s in strings(item)]
        return []

    return "\n".join(strings(data))


def ground_contrast(contrast, evidence, allow_original=True):
    available = {e.id: e for e in evidence if e.status == "completed"}
    grounded, rejected = ground_facts(
        CaseProfile(facts=contrast.key_case_clues), evidence
    )
    notes = ["临床线索未通过原文核对：" + n for n in rejected]
    literal = []
    for fact in grounded.facts:
        if normalized(fact.value) not in normalized(fact.source_quote):
            notes.append("临床线索值不是引用原文中的直接片段：" + fact.name)
        else:
            literal.append(fact)
    hypotheses = []
    seen = set()
    for item in contrast.hypotheses:
        candidate = Candidate.model_validate(item.model_dump(exclude={"still_unknown"}))
        issues = validate_evidence(
            Report(most_likely=candidate), evidence, allow_original=allow_original
        )
        key = canonical_name(item.name)
        if key in seen:
            issues.append("同义或重复的候选疾病")
        if issues:
            notes.append("未采用临床对照候选“" + item.name + "”：" + "；".join(issues))
        else:
            hypotheses.append(item)
            seen.add(key)
    conflicts = []
    for conflict in contrast.observation_conflicts:
        first = available.get(conflict.first_source)
        second = available.get(conflict.second_source)
        if not first or not second or first.id == second.id:
            notes.append("观察冲突没有两项独立、有效的来源")
            continue
        if not (
            normalized(conflict.first_quote) in normalized(source_text(first))
            and normalized(conflict.second_quote) in normalized(source_text(second))
        ):
            notes.append("观察冲突引用的原文无法匹配")
            continue
        conflicts.append(conflict)
    if len(hypotheses) < 2:
        notes.append("有效候选不足两项，未完成充分的临床对照")
    return ClinicalContrast(
        key_case_clues=literal,
        hypotheses=hypotheses,
        observation_conflicts=conflicts,
        remaining_questions=contrast.remaining_questions,
    ), notes


def contrast_prompt(context):
    return (
        "Before drafting a diagnostic report, make a concise clinical evidence contrast. "
        "Use Chinese disease names, short English explanations (at most25words per claim), "
        "Select up to5 distinctive GIVEN clinical clues, prioritizing "
        "time course, recurrence, associations, risks, and existing test results. A clue is valid "
        "only by selecting an indexed C-source snippet: key_case_clues contains name and source_id "
        "(the full C-...-S... snippet ID), no value or source_quote. The program copies its exact text. "
        "Generic K-criteria are not patient facts. "
        "Hypothesis support/against evidence_ids must reference evidence IDs such as C-1 or I-original, "
        "not snippet IDs; snippet IDs are only for key_case_clues. "
        "Compare2 or3 genuinely distinct plausible disease causes: patient-specific support, "
        "opposing evidence, and unknown confirmation evidence. Explain distinctive history rather "
        "than only repeating a broad radiographic finding. Do not infer unprovided exposure, "
        "pathology, physiology or negative symptoms. Clinical associations suggest hypotheses "
        "without proving histology. A low classifier score cannot exclude disease. "
        "If two observations disagree, quote both actual sources; do not vote or silently merge them. "
        "Do not fabricate conflicts or force a rare subtype without support. I-original can support "
        "direct image observations but has no textual quotation; conflicts require two quotable "
        "sources. This intermediate contrast is fallible model analysis, not new evidence.\n"
        + context
    )
