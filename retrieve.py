"""Retrieval methods.

Main experiment (same chunks, same settings for all three):
    dense          Method 1: sentence embeddings + FAISS cosine search
    hybrid         Method 2: dense + BM25, merged with Reciprocal Rank Fusion (RRF)
    hybrid_rerank  Method 3: hybrid candidates re-scored by a cross-encoder

Extra methods used only for the ablation study (evaluate.py --ablation):
    bm25           BM25 only
    dense_bm25     dense + BM25 merged WITHOUT RRF (simple interleaving)

Every returned chunk is a dict with the extra keys "score" and "score_type".
"""
import json
import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
import yaml
from rank_bm25 import BM25Okapi

MAIN_METHODS = ["dense", "hybrid", "hybrid_rerank"]
ABLATION_METHODS = ["dense", "bm25", "dense_bm25", "hybrid", "hybrid_rerank"]
METHOD_NAMES = {
    "dense": "Dense RAG",
    "bm25": "BM25 only",
    "dense_bm25": "Dense + BM25 (no RRF)",
    "hybrid": "Hybrid RAG",
    "hybrid_rerank": "Hybrid + Reranking",
}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def rrf(ranked_lists: Sequence[Sequence[int]], k: int = 60) -> Dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over the lists of 1 / (k + rank(d)). Needs no score scaling."""
    score: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, i in enumerate(ranked, start=1):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank)
    return score


def interleave(*ranked_lists: Sequence[int]) -> List[int]:
    """Naive merge for the ablation: take rank 1 of each list, then rank 2, ... (duplicates removed)."""
    merged: List[int] = []
    for row in zip(*ranked_lists):
        for i in row:
            if i not in merged:
                merged.append(i)
    return merged


class Retriever:
    """Builds the dense (FAISS) and BM25 indexes once; the cross-encoder is loaded on first use."""

    def __init__(self, chunks: List[dict], cfg: dict):
        from sentence_transformers import SentenceTransformer  # imported here: heavy

        self.cfg = cfg
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.embedder = SentenceTransformer(cfg["embedding_model"])
        vectors = self.embedder.encode(self.texts, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(vectors.shape[1])   # normalised vectors: inner product = cosine
        self.index.add(np.asarray(vectors, dtype="float32"))
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts], k1=cfg["bm25_k1"], b=cfg["bm25_b"])
        self.reranker = None

    # ---- individual retrievers: return (chunk ids, scores), best first ----
    def dense_search(self, question: str, n: int):
        q = self.embedder.encode([question], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), n)
        keep = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        return [i for i, _ in keep], [s for _, s in keep]

    def bm25_search(self, question: str, n: int):
        scores = self.bm25.get_scores(tokenize(question))
        ids = sorted(range(len(self.texts)), key=lambda i: -scores[i])[:n]
        return ids, [float(scores[i]) for i in ids]

    def rerank(self, question: str, candidate_ids: List[int]):
        from sentence_transformers import CrossEncoder

        if self.reranker is None:
            self.reranker = CrossEncoder(self.cfg["reranker_model"])
        scores = self.reranker.predict([(question, self.texts[i]) for i in candidate_ids])
        ranked = sorted(zip(candidate_ids, (float(s) for s in scores)), key=lambda p: -p[1])
        return [i for i, _ in ranked], [s for _, s in ranked]

    # ---- the methods ----
    def retrieve(self, question: str, method: str, k: Optional[int] = None) -> List[dict]:
        """Return the top-k chunks (dicts with 'score' and 'score_type' added) for the given method."""
        k = k or self.cfg["top_k"]
        n = self.cfg["n_candidates"]
        if method == "dense":
            ids, scores = self.dense_search(question, n)
            stype = "cosine similarity"
        elif method == "bm25":
            ids, scores = self.bm25_search(question, n)
            stype = "BM25 score"
        elif method == "dense_bm25":
            ids = interleave(self.dense_search(question, n)[0], self.bm25_search(question, n)[0])
            scores, stype = [float("nan")] * len(ids), "none (interleaved)"
        elif method in ("hybrid", "hybrid_rerank"):
            fused = rrf([self.dense_search(question, n)[0], self.bm25_search(question, n)[0]],
                        self.cfg["rrf_k"])
            ids = sorted(fused, key=lambda i: -fused[i])
            scores, stype = [fused[i] for i in ids], "RRF score"
            if method == "hybrid_rerank":
                ids, scores = self.rerank(question, ids[: self.cfg["rerank_candidates"]])
                stype = "cross-encoder score"
        else:
            raise ValueError(f"Unknown retrieval method: {method!r}")
        return [dict(self.chunks[i], score=s, score_type=stype) for i, s in zip(ids[:k], scores[:k])]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Shared Retriever built from config.yaml and chunks.json (run ingest.py first)."""
    cfg = load_config()
    try:
        with open(cfg["chunks_file"], encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"{cfg['chunks_file']} not found. Run 'python ingest.py' first.")
    return Retriever(chunks, cfg)


def retrieve(question: str, method: str, k: Optional[int] = None) -> List[dict]:
    return get_retriever().retrieve(question, method, k)
