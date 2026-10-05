import math

import pytest

from retrieve import Retriever, interleave, rrf, tokenize

CHUNKS = [
    {"id": 0, "document": "Act.pdf", "page": 1, "section": "Section 1",
     "text": "Whoever tampers with computer source documents shall be punished with imprisonment up to three years."},
    {"id": 1, "document": "Act.pdf", "page": 2, "section": "Section 2",
     "text": "A consumer may file a complaint with the District Commission within two years of the cause of action."},
    {"id": 2, "document": "Act.pdf", "page": 3, "section": "Section 3",
     "text": "The Controller of Certifying Authorities shall supervise the licensed certifying authorities."},
    {"id": 3, "document": "Act.pdf", "page": 4, "section": "Section 4",
     "text": "A product manufacturer is liable when the product contains a manufacturing defect."},
    {"id": 4, "document": "Act.pdf", "page": 5, "section": "Section 5",
     "text": "Penalty for damage to a computer system without the permission of its owner is compensation."},
    {"id": 5, "document": "Act.pdf", "page": 6, "section": "Section 6",
     "text": "The Central Government may make rules by notification in the Official Gazette."},
]


@pytest.fixture(scope="module")
def retriever(cfg):
    small = dict(cfg, top_k=3, n_candidates=6, rerank_candidates=6)
    return Retriever(CHUNKS, small)


def test_tokenize_lowercases_words():
    assert tokenize("Section 43: Computer-System!") == ["section", "43", "computer", "system"]


def test_rrf_scores_match_the_formula():
    scores = rrf([[10, 20, 30], [20, 10]], k=60)
    assert scores[10] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[20] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[30] == pytest.approx(1 / 63)
    assert max(scores, key=scores.get) in (10, 20)


def test_rrf_rewards_chunks_found_by_both_retrievers():
    scores = rrf([[1, 2, 3], [9, 2, 8]], k=60)
    assert max(scores, key=scores.get) == 2          # rank 2 in both beats rank 1 in only one


def test_interleave_removes_duplicates_and_keeps_order():
    assert interleave([1, 2, 3], [3, 4, 5]) == [1, 3, 2, 4, 5]


def test_bm25_finds_exact_keyword(retriever):
    ids, scores = retriever.bm25_search("Controller of Certifying Authorities", 3)
    assert ids[0] == 2
    assert scores[0] >= scores[1]


def test_dense_retrieval_understands_paraphrase(retriever):
    ids, scores = retriever.dense_search("how long do I have to bring a case as a buyer", 3)
    assert ids[0] == 1
    assert all(-1.0 <= s <= 1.0 + 1e-6 for s in scores)   # cosine similarity range


def test_dense_method_returns_k_chunks_with_scores(retriever):
    out = retriever.retrieve("damage to a computer system", "dense")
    assert len(out) == 3
    assert {"document", "page", "section", "text", "score", "score_type"} <= set(out[0])
    assert out[0]["score_type"] == "cosine similarity"


def test_hybrid_method_uses_rrf(retriever):
    out = retriever.retrieve("damage to a computer system without permission", "hybrid")
    assert out[0]["score_type"] == "RRF score"
    assert out[0]["section"] in ("Section 5", "Section 1")
    scores = [c["score"] for c in out]
    assert scores == sorted(scores, reverse=True)


def test_reranker_reorders_candidates(retriever):
    q = "What is the time limit for a consumer to file a complaint?"
    out = retriever.retrieve(q, "hybrid_rerank")
    assert out[0]["section"] == "Section 2"
    assert out[0]["score_type"] == "cross-encoder score"
    scores = [c["score"] for c in out]
    assert scores == sorted(scores, reverse=True)


def test_all_methods_return_chunks_from_the_corpus(retriever):
    for method in ("dense", "bm25", "dense_bm25", "hybrid", "hybrid_rerank"):
        out = retriever.retrieve("computer source documents", method, k=2)
        assert len(out) == 2
        assert all(c["text"] in {x["text"] for x in CHUNKS} for c in out)


def test_unknown_method_raises(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("x", "magic")
