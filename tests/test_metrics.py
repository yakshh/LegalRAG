import math

import pytest

from evaluation.metrics import (describe, hit_at_k, ndcg_at_k, paired_permutation_test, precision_at_k,
                                recall_at_k, reciprocal_rank, retrieval_metrics, token_recall)

Q = {"source_document": "A.pdf", "relevant_sections": ["Section 1", "Section 2"]}


def chunk(section, doc="A.pdf"):
    return {"document": doc, "page": 1, "section": section, "text": "t"}


RETRIEVED = [chunk("Section 9"), chunk("Section 1"), chunk("Section 1"), chunk("Section 7"), chunk("Section 2")]
CORPUS = RETRIEVED + [chunk("Section 1"), chunk("Section 2")]
REL = [False, True, True, False, True]


def test_precision_at_k():
    assert precision_at_k(REL, 3) == pytest.approx(2 / 3)
    assert precision_at_k(REL, 5) == pytest.approx(3 / 5)


def test_recall_at_k_is_section_level():
    assert recall_at_k(RETRIEVED, Q, 3) == 0.5      # only Section 1 found of {1, 2}
    assert recall_at_k(RETRIEVED, Q, 5) == 1.0


def test_recall_ignores_other_documents():
    other = [chunk("Section 1", doc="B.pdf"), chunk("Section 2", doc="B.pdf")]
    assert recall_at_k(other, Q, 5) == 0.0


def test_hit_at_k():
    assert hit_at_k(REL, 1) == 0.0 and hit_at_k(REL, 2) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(REL) == 0.5
    assert reciprocal_rank([False, False]) == 0.0
    assert reciprocal_rank([True]) == 1.0


def test_ndcg_perfect_and_imperfect():
    assert ndcg_at_k([True, True, False], 3, n_relevant_in_corpus=2) == pytest.approx(1.0)
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([False, True, True], 3, 2) == pytest.approx(expected)
    assert ndcg_at_k([False, False], 2, 0) == 0.0


def test_retrieval_metrics_bundle():
    m = retrieval_metrics(RETRIEVED, Q, CORPUS, ks=(3, 5))
    assert m["precision@3"] == pytest.approx(2 / 3) and m["recall@5"] == 1.0
    assert m["hit@3"] == 1.0 and m["mrr"] == 0.5 and 0 < m["ndcg@5"] <= 1


def test_describe_mean_median_std():
    d = describe([1, 2, 3, 4])
    assert d["mean"] == 2.5 and d["median"] == 2.5 and d["std"] == pytest.approx(1.2909944)
    assert describe([])["n"] == 0 and describe([5])["std"] == 0.0


def test_permutation_test_detects_clear_difference_but_not_noise():
    a = [1.0] * 30
    b = [0.0] * 30
    assert paired_permutation_test(a, b, trials=2000) < 0.01
    same = [0.5, 0.7, 0.2, 0.9] * 5
    assert paired_permutation_test(same, same, trials=500) == 1.0


def test_token_recall():
    assert token_recall("one crore rupees", "up to one crore rupees") == 1.0
    assert token_recall("one crore rupees", "nothing") == 0.0


def test_evidence_phrase_makes_only_the_matching_chunk_relevant():
    from evaluation.metrics import is_relevant
    q = {"source_document": "A.pdf", "relevant_sections": ["Section 2"],
         "evidence_phrases": {"Section 2": '"Consumer" means any person'}}
    right = {"document": "A.pdf", "page": 4, "section": "Section 2", "text": "(7) \u201cconsumer\u201d  means any person who buys"}
    other = {"document": "A.pdf", "page": 5, "section": "Section 2", "text": "(8) \u201ccomplaint\u201d means any allegation"}
    assert is_relevant(right, q) and not is_relevant(other, q)
    assert not is_relevant(dict(right, document="B.pdf"), q)


def test_recall_needs_a_relevant_chunk_not_just_the_section():
    q = {"source_document": "A.pdf", "relevant_sections": ["Section 2"],
         "evidence_phrases": {"Section 2": "the right definition"}}
    wrong_chunk = {"document": "A.pdf", "page": 1, "section": "Section 2", "text": "another definition"}
    right_chunk = {"document": "A.pdf", "page": 2, "section": "Section 2", "text": "the right definition is here"}
    assert recall_at_k([wrong_chunk], q, 5) == 0.0
    assert recall_at_k([wrong_chunk, right_chunk], q, 5) == 1.0
