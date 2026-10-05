"""Retrieval and citation metrics (no models or API needed).

A retrieved chunk is RELEVANT for a question when it comes from the question's source document, its
section is one of the question's relevant_sections, and, if the question gives an "evidence_phrases" entry
for that section, the chunk text contains that phrase. (The phrase is needed for long sections that are
split into many chunks, for example the definitions in Section 2: only the chunk that holds the asked
definition is relevant, not every chunk of the section.)

Retrieval metrics (computed on the ranked list of retrieved chunks, best first):
  Precision@K  relevant chunks in the top K / K
  Recall@K     distinct relevant sections that have a relevant chunk in the top K / number of relevant
               sections (section level, because one section may need several chunks)
  Hit@K        1 if at least one relevant chunk is in the top K, else 0
  MRR          1 / rank of the first relevant chunk (0 if none is retrieved)
  nDCG@K       rank-weighted gain with binary relevance: sum(rel_i / log2(i + 1)) / ideal value

These are ranking metrics, not classification metrics, so Accuracy / F1 are not used for retrieval.
"""
import math
import statistics
from typing import Dict, List, Sequence


def normalize(text: str) -> str:
    """Lower-case, straight quotes, single spaces (so a phrase can be matched against PDF text)."""
    for curly, straight in (("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")):
        text = text.replace(curly, straight)
    return " ".join(text.lower().split())


def is_relevant(chunk: dict, question: dict) -> bool:
    if chunk["document"] != question.get("source_document") or chunk["section"] not in question.get("relevant_sections", []):
        return False
    phrase = question.get("evidence_phrases", {}).get(chunk["section"])
    return phrase is None or normalize(phrase) in normalize(chunk["text"])


def precision_at_k(rel: Sequence[bool], k: int) -> float:
    return sum(rel[:k]) / k


def recall_at_k(retrieved: List[dict], question: dict, k: int) -> float:
    wanted = set(question["relevant_sections"])
    found = {c["section"] for c in retrieved[:k] if is_relevant(c, question)} & wanted
    return len(found) / len(wanted)


def hit_at_k(rel: Sequence[bool], k: int) -> float:
    return 1.0 if any(rel[:k]) else 0.0


def reciprocal_rank(rel: Sequence[bool]) -> float:
    for i, r in enumerate(rel, start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rel: Sequence[bool], k: int, n_relevant_in_corpus: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(rel[:k], start=1) if r)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, n_relevant_in_corpus) + 1))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(retrieved: List[dict], question: dict, all_chunks: List[dict],
                      ks: Sequence[int] = (3, 5)) -> Dict[str, float]:
    """All retrieval metrics for ONE question. `retrieved` is the ranked chunk list (use depth >= max(ks))."""
    rel = [is_relevant(c, question) for c in retrieved]
    n_rel = sum(is_relevant(c, question) for c in all_chunks)
    out: Dict[str, float] = {}
    for k in ks:
        out[f"precision@{k}"] = precision_at_k(rel, k)
        out[f"recall@{k}"] = recall_at_k(retrieved, question, k)
        out[f"hit@{k}"] = hit_at_k(rel, k)
    out["mrr"] = reciprocal_rank(rel)
    out["ndcg@5"] = ndcg_at_k(rel, 5, n_rel)
    return out


# ---------------- citations ----------------
def citation_metrics(citations: List[dict], retrieved: List[dict], question: dict) -> Dict[str, float]:
    """Deterministic citation checks for one answer.

    cite_present  1 if the answer contains at least one citation
    cite_valid    share of citations that match a retrieved chunk (document + page + section);
                  a citation of something that was not retrieved is invalid
    cite_precision  share of citations that point to a relevant section of the right document
    cite_recall     share of the relevant sections that are cited (completeness)
    """
    retrieved_keys = {(c["document"], c["page"], c["section"]) for c in retrieved}
    wanted = set(question.get("relevant_sections", []))
    present = 1.0 if citations else 0.0
    if not citations:
        return {"cite_present": 0.0, "cite_valid": 0.0, "cite_precision": 0.0, "cite_recall": 0.0}
    valid = sum((c["document"], c["page"], c["section"]) in retrieved_keys for c in citations) / len(citations)
    right = sum(c["document"] == question.get("source_document") and c["section"] in wanted
                for c in citations) / len(citations)
    cited = {c["section"] for c in citations if c["document"] == question.get("source_document")}
    recall = len(cited & wanted) / len(wanted) if wanted else 0.0
    return {"cite_present": present, "cite_valid": valid, "cite_precision": right, "cite_recall": recall}


# ---------------- statistics ----------------
def describe(values: Sequence[float]) -> Dict[str, float]:
    """mean, median and (sample) standard deviation."""
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def paired_permutation_test(a: Sequence[float], b: Sequence[float], trials: int = 10000, seed: int = 42) -> float:
    """Two-sided paired permutation (sign-flip) test on the mean difference of two methods over the same
    questions. Returns a p-value. Exploratory only: with few questions it has little power."""
    import random

    diffs = [x - y for x, y in zip(a, b)]
    if not diffs or all(d == 0 for d in diffs):
        return 1.0
    observed = abs(sum(diffs) / len(diffs))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(trials):
        total = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(total / len(diffs)) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (trials + 1)


def token_recall(expected: str, answer: str) -> float:
    """Share of the distinct words of the expected answer that also occur in the generated answer
    (a ROUGE-1-recall style proxy; it ignores meaning and is only a rough secondary signal)."""
    import re

    words = set(re.findall(r"\w+", expected.lower()))
    return len(words & set(re.findall(r"\w+", answer.lower()))) / len(words) if words else 0.0


def update_json(path: str, key: str, value) -> None:
    """Write value under `key` in a JSON file, keeping the other keys (used for results/summary.json)."""
    import json
    import os

    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data[key] = value
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
