"""Run the three methods on questions.json and print retrieval metrics and latency.

    python evaluate.py             retrieval metrics only
    python evaluate.py --answers   also save generated answers to answers.json
                                   (answer quality is then scored with a rubric / LLM judge)
"""
import json
import os
import sys
import time

from generate import answer
from retrieve import cfg, chunks, retrieve

METHODS = ["dense", "hybrid", "hybrid_rerank"]
if not os.path.exists(cfg["questions_file"]):
    sys.exit(f"{cfg['questions_file']} not found. Create it with your test questions (format in README.md, Part B).")
questions = json.load(open(cfg["questions_file"]))
with_answers = "--answers" in sys.argv
saved = []


def is_relevant(chunk, q):
    """A chunk is relevant if it is from the labelled document and section."""
    return chunk["document"] == q["source_document"] and chunk["section"] == q["section"]


for method in METHODS:
    p3 = p5 = r3 = r5 = hit5 = mrr = seconds = 0.0
    n = 0
    for q in questions:
        if not q["section"]:  # unanswerable question: no retrieval label
            continue
        start = time.time()
        found = retrieve(q["question"], method, k=5)
        seconds += time.time() - start

        rel = [is_relevant(c, q) for c in found]
        total_rel = sum(is_relevant(c, q) for c in chunks) or 1
        p3 += sum(rel[:3]) / 3
        p5 += sum(rel[:5]) / 5
        r3 += sum(rel[:3]) / total_rel
        r5 += sum(rel[:5]) / total_rel
        hit5 += any(rel[:5])
        mrr += next((1 / (i + 1) for i, r in enumerate(rel) if r), 0)
        n += 1

    print(f"{method:14s} P@3={p3/n:.3f} P@5={p5/n:.3f} R@3={r3/n:.3f} R@5={r5/n:.3f} "
          f"Hit@5={hit5/n:.3f} MRR={mrr/n:.3f} retrieval_s={seconds/n:.3f}")

    if with_answers:  # all questions, including the unanswerable ones
        for q in questions:
            found = retrieve(q["question"], method, k=cfg["top_k"])
            saved.append({"method": method, "question": q["question"], "answer": answer(q["question"], found)})

if with_answers:
    json.dump(saved, open("answers.json", "w"), indent=1)
