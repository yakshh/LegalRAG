"""The three retrieval methods: dense, hybrid (BM25 + dense with RRF), hybrid + reranker."""
import json
import re

import faiss
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

cfg = yaml.safe_load(open("config.yaml"))
chunks = json.load(open("chunks.json"))
texts = [c["text"] for c in chunks]

# dense index (normalised vectors + inner product = cosine similarity)
embedder = SentenceTransformer(cfg["embedding_model"])
index = faiss.IndexFlatIP(embedder.get_sentence_embedding_dimension())
index.add(embedder.encode(texts, normalize_embeddings=True))

# BM25 index
tokenize = lambda s: re.findall(r"\w+", s.lower())
bm25 = BM25Okapi([tokenize(t) for t in texts], k1=cfg["bm25_k1"], b=cfg["bm25_b"])

reranker = None  # loaded the first time it is needed


def dense_search(question, n):
    q = embedder.encode([question], normalize_embeddings=True)
    _, ids = index.search(q, n)
    return [int(i) for i in ids[0] if i >= 0]


def bm25_search(question, n):
    scores = bm25.get_scores(tokenize(question))
    return sorted(range(len(texts)), key=lambda i: -scores[i])[:n]


def rrf(*ranked_lists):
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank) over the lists."""
    score = {}
    for ranked in ranked_lists:
        for rank, i in enumerate(ranked, start=1):
            score[i] = score.get(i, 0) + 1 / (cfg["rrf_k"] + rank)
    return sorted(score, key=lambda i: -score[i])


def retrieve(question, method, k=None):
    """method is 'dense', 'hybrid' or 'hybrid_rerank'. Returns a list of chunk dicts."""
    global reranker
    k = k or cfg["top_k"]
    n = cfg["n_candidates"]
    dense = dense_search(question, n)

    if method == "dense":
        ids = dense[:k]
    else:
        fused = rrf(dense, bm25_search(question, n))
        if method == "hybrid":
            ids = fused[:k]
        else:
            if reranker is None:
                reranker = CrossEncoder(cfg["reranker_model"])
            candidates = fused[:cfg["rerank_candidates"]]
            scores = reranker.predict([(question, texts[i]) for i in candidates])
            ranked = sorted(zip(scores, candidates), reverse=True)
            ids = [i for _, i in ranked][:k]
    return [chunks[i] for i in ids]
