import copy
import json
import os

import pytest

from validate_dataset import validate

CHUNKS = [
    {"document": "A.pdf", "page": 1, "section": "Section 1", "text": "x"},
    {"document": "A.pdf", "page": 2, "section": "Section 2", "text": "y"},
    {"document": "A.pdf", "page": 3, "section": "Section 2", "text": "y2"},
]

GOOD = {"question_id": "Q001", "question": "What is Section 1?", "source_document": "A.pdf",
        "source_section": "Section 1", "source_page": 1, "expected_answer": "It is x.",
        "relevant_sections": ["Section 1"], "difficulty": "easy", "category": "definition", "answerable": True}
OOS = {"question_id": "Q002", "question": "Who is the PM?", "source_document": None, "source_section": None,
       "source_page": None, "expected_answer": "Insufficient evidence", "relevant_sections": [],
       "difficulty": "easy", "category": "out_of_scope", "answerable": False}


def check(*questions, pages=None):
    return validate(list(questions), CHUNKS, pages)


def variant(**changes):
    q = copy.deepcopy(GOOD)
    q.update(changes)
    return q


def test_valid_dataset_has_no_errors():
    errors, warnings = check(GOOD, OOS, pages={"A.pdf": 3})
    assert errors == [] and warnings == []


def test_missing_field_is_reported():
    q = copy.deepcopy(GOOD)
    del q["expected_answer"]
    errors, _ = check(q)
    assert any("missing fields" in e and "expected_answer" in e for e in errors)


def test_duplicate_question_ids():
    errors, _ = check(GOOD, variant(question="Another question?"))
    assert any("duplicate question_id Q001" in e for e in errors)


def test_empty_expected_answer():
    errors, _ = check(variant(expected_answer="  "))
    assert any("missing expected_answer" in e for e in errors)


def test_invalid_document_name():
    errors, _ = check(variant(source_document="Nope.pdf"))
    assert any("invalid document" in e for e in errors)


def test_section_that_does_not_exist():
    errors, _ = check(variant(source_section="Section 99", relevant_sections=["Section 99"]))
    assert any("does not exist" in e for e in errors)


def test_relevant_section_that_does_not_exist():
    errors, _ = check(variant(relevant_sections=["Section 1", "Section 42"]))
    assert any("Section 42" in e for e in errors)


def test_invalid_page_numbers():
    assert any("positive integer" in e for e in check(variant(source_page=0))[0])
    assert any("beyond the" in e for e in check(variant(source_page=50), pages={"A.pdf": 3})[0])
    assert any("has no text on page" in e for e in check(variant(source_page=2))[0])   # Section 1 is on page 1


def test_null_page_is_allowed():
    errors, _ = check(variant(source_page=None))
    assert errors == []


def test_bad_difficulty_and_category():
    errors, _ = check(variant(difficulty="impossible", category="made_up"))
    assert any("difficulty" in e for e in errors) and any("category" in e for e in errors)


def test_unanswerable_question_must_not_have_source():
    bad = dict(OOS, source_document="A.pdf")
    errors, _ = check(bad)
    assert any("unanswerable" in e for e in errors)


def test_empty_or_non_list_input():
    assert validate([], CHUNKS)[0]
    assert validate({"a": 1}, CHUNKS)[0]


@pytest.mark.skipif(not (os.path.exists("evaluation/questions.json") and os.path.exists("chunks.json")),
                    reason="run ingest.py first")
def test_real_dataset_is_valid_and_large_enough():
    questions = json.load(open("evaluation/questions.json", encoding="utf-8"))
    chunks = json.load(open("chunks.json"))
    errors, _ = validate(questions, chunks)
    assert errors == []
    assert len(questions) >= 30
    assert any(not q["answerable"] for q in questions)       # out-of-scope questions are included


def test_evidence_phrase_must_exist_in_a_chunk_of_that_section():
    chunks = [dict(c, text="The penalty is ONE crore rupees") if c["section"] == "Section 1" else c for c in CHUNKS]
    ok = variant(evidence_phrases={"Section 1": "one crore rupees"})
    bad = variant(evidence_phrases={"Section 1": "ten crore rupees"})
    assert validate([ok], chunks)[0] == []
    assert any("evidence phrase not found" in e for e in validate([bad], chunks)[0])


def test_evidence_phrase_for_unlisted_section_is_an_error():
    errors, _ = check(variant(evidence_phrases={"Section 2": "y"}))
    assert any("not in relevant_sections" in e for e in errors)
