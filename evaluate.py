"""Retrieval evaluation: runs the retrieval methods on the evaluation questions and measures quality + latency.

    python evaluate.py              Dense vs Hybrid vs Hybrid + Reranking   (the main experiment)
    python evaluate.py --ablation   also BM25-only and Dense+BM25 without RRF (component analysis)

Needs no API key. Writes:
    evaluation/retrieval_results.csv   one row per (question, method)
    results/summary.json               aggregated metrics (key "retrieval", and "ablation" with --ablation)
Metric definitions are in evaluation/metrics.py.
"""
import argparse
import csv
import json
import os
import time
from typing import Dict, List

from evaluation.metrics import describe, paired_permutation_test, retrieval_metrics, update_json
from retrieve import ABLATION_METHODS, MAIN_METHODS, get_retriever

METRICS = ["precision@3", "precision@5", "recall@3", "recall@5", "hit@3", "hit@5", "mrr", "ndcg@5"]


def load_questions(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. See README.md (evaluation dataset).")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_method(retriever, method: str, questions: List[dict], depth: int, ks) -> List[dict]:
    """Retrieve for every answerable question and compute the metrics. The first query is a warm-up
    (loads the cross-encoder) and is not timed."""
    retriever.retrieve("warm-up question", method, depth)
    rows = []
    for q in questions:
        start = time.perf_counter()
        found = retriever.retrieve(q["question"], method, depth)
        latency = time.perf_counter() - start
        row = {"question_id": q["question_id"], "method": method, "category": q["category"],
               "difficulty": q["difficulty"], **retrieval_metrics(found, q, retriever.chunks, ks),
               "latency_s": latency,
               "retrieved": " | ".join(f"{c['section']} (p{c['page']})" for c in found[:5])}
        rows.append(row)
    return rows


def summarise(rows: List[dict]) -> Dict[str, dict]:
    """mean / median / std of every metric and of the latency, per method; plus mean MRR and Hit@5 by category."""
    out: Dict[str, dict] = {}
    for method in dict.fromkeys(r["method"] for r in rows):
        mine = [r for r in rows if r["method"] == method]
        entry = {m: describe([r[m] for r in mine]) for m in METRICS + ["latency_s"]}
        entry["by_category"] = {}
        for cat in sorted({r["category"] for r in mine}):
            sub = [r for r in mine if r["category"] == cat]
            entry["by_category"][cat] = {"n": len(sub), "mrr": sum(r["mrr"] for r in sub) / len(sub),
                                         "hit@5": sum(r["hit@5"] for r in sub) / len(sub)}
        out[method] = entry
    return out


def paired_tests(rows: List[dict], pairs, trials: int, seed: int) -> List[dict]:
    result = []
    for a, b in pairs:
        ra = {r["question_id"]: r for r in rows if r["method"] == a}
        rb = {r["question_id"]: r for r in rows if r["method"] == b}
        for metric in ("mrr", "hit@5", "recall@5", "precision@5"):
            ids = sorted(set(ra) & set(rb))
            va, vb = [ra[i][metric] for i in ids], [rb[i][metric] for i in ids]
            result.append({"a": a, "b": b, "metric": metric, "mean_a": sum(va) / len(va),
                           "mean_b": sum(vb) / len(vb),
                           "p_value": paired_permutation_test(va, vb, trials, seed), "n": len(ids)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ablation", action="store_true", help="also run BM25-only and Dense+BM25 (no RRF)")
    args = parser.parse_args()

    retriever = get_retriever()
    cfg = retriever.cfg
    questions = [q for q in load_questions(cfg["questions_file"]) if q["answerable"]]
    methods = ABLATION_METHODS if args.ablation else MAIN_METHODS
    print(f"{len(questions)} answerable questions, {len(retriever.chunks)} chunks, methods: {methods}")

    rows: List[dict] = []
    for method in methods:
        rows += run_method(retriever, method, questions, cfg["eval_depth"], cfg["retrieval_ks"])
        s = summarise([r for r in rows if r["method"] == method])[method]
        print(f"{method:14s} P@3={s['precision@3']['mean']:.3f} P@5={s['precision@5']['mean']:.3f} "
              f"R@3={s['recall@3']['mean']:.3f} R@5={s['recall@5']['mean']:.3f} "
              f"Hit@3={s['hit@3']['mean']:.3f} Hit@5={s['hit@5']['mean']:.3f} MRR={s['mrr']['mean']:.3f} "
              f"nDCG@5={s['ndcg@5']['mean']:.3f} latency={s['latency_s']['mean'] * 1000:.0f} ms")

    summary = summarise(rows)
    pairs = [("hybrid", "dense"), ("hybrid_rerank", "dense"), ("hybrid_rerank", "hybrid")]
    if args.ablation:
        update_json(os.path.join(cfg["results_dir"], "summary.json"), "ablation", summary)
    else:
        update_json(os.path.join(cfg["results_dir"], "summary.json"), "retrieval",
                    {"n_questions": len(questions), "n_chunks": len(retriever.chunks), "methods": summary,
                     "paired_tests": paired_tests(rows, pairs, cfg["permutation_trials"], cfg["random_seed"])})
        os.makedirs(cfg["evaluation_dir"], exist_ok=True)
        path = os.path.join(cfg["evaluation_dir"], "retrieval_results.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            for r in rows:
                writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
        print(f"Saved {path} and results/summary.json")


if __name__ == "__main__":
    main()
