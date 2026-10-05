"""Answer-quality evaluation: generate an answer for every question with each retrieval method, then score it.

    python evaluate_answers.py                 generate answers + score them with the LLM judge
    python evaluate_answers.py --no-judge      only generate answers (no judging)
    python evaluate_answers.py --judge-only    only judge answers that are already saved
    python evaluate_answers.py --limit 5       use only the first 5 questions (quick test)
    python evaluate_answers.py --manual-sheet  write evaluation/manual_scoring_sheet.csv for hand scoring

Needs GEMINI_API_KEY (see .env.example). The script resumes: records already saved in
evaluation/answers.json are not generated or judged again, so it is safe to re-run after a rate-limit error.

Same corpus, chunks, questions, LLM, prompt and judge for every method; only retrieval differs.
Judge scores are LLM-based (evaluation/judge.py), not human scores.
"""
import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List

from evaluation.judge import SCORE_KEYS, judge_answer
from evaluation.metrics import citation_metrics, describe, token_recall, update_json
from generate import INSUFFICIENT, QuotaExceeded, answer_question
from retrieve import MAIN_METHODS, get_retriever

CSV_FIELDS = ["question_id", "method", "category", "difficulty", "answerable", "status", "correctness_score",
              "faithfulness_score", "relevance_score", "citation_score", "cite_present", "cite_valid",
              "cite_precision", "cite_recall", "answer_token_recall", "retrieval_s", "generation_s",
              "judge_reason", "answer"]


def load_json(path: str, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def generate_records(questions: List[dict], methods: List[str], records: List[dict], path: str, cfg: dict):
    retriever = get_retriever()
    done = {(r["question_id"], r["method"]) for r in records}
    for method in methods:
        for q in questions:
            if (q["question_id"], method) in done:
                continue
            t0 = time.perf_counter()
            chunks = retriever.retrieve(q["question"], method, cfg["top_k"])
            t1 = time.perf_counter()
            try:
                result = answer_question(q["question"], chunks)
            except QuotaExceeded as e:  # nothing more can be generated today: stop, resume later
                print(f"  ! STOPPED: {e}")
                print("  Re-run the same command later; saved answers are kept.")
                return records
            except Exception as e:  # keep going; the missing record is retried on the next run
                print(f"  ! {q['question_id']} {method}: {e}")
                continue
            t2 = time.perf_counter()
            records.append({"question_id": q["question_id"], "method": method, "question": q["question"],
                            "answer": result["answer"], "status": result["status"],
                            "citations": result["citations"], "retrieved_context": chunks,
                            "expected_answer": q["expected_answer"],
                            "retrieval_s": t1 - t0, "generation_s": t2 - t1, "generator_model": cfg["llm_model"]})
            save_json(path, records)
            print(f"  generated {q['question_id']} {method}: {result['status']}")
            time.sleep(cfg["llm_delay_seconds"])
    return records


def judge_records(records: List[dict], path: str, cfg: dict) -> None:
    for r in records:
        if r.get("correctness_score") is not None or "judged" in r:
            continue
        try:
            j = judge_answer(r)
        except QuotaExceeded as e:
            print(f"  ! STOPPED: {e}")
            print("  Re-run the same command later; saved scores are kept.")
            return
        except Exception as e:
            print(f"  ! judge {r['question_id']} {r['method']}: {e}")
            continue
        r.update(correctness_score=j["correctness"], faithfulness_score=j["faithfulness"],
                 relevance_score=j["relevance"], citation_score=j["citation_correctness"],
                 judge_reason=j["reason"], judged=True)
        save_json(path, records)
        print(f"  judged {r['question_id']} {r['method']}: {j['correctness']}/{j['faithfulness']}/"
              f"{j['relevance']}/{j['citation_correctness']}")
        time.sleep(cfg["llm_delay_seconds"])


def add_deterministic_metrics(records: List[dict], by_id: Dict[str, dict]) -> None:
    for r in records:
        q = by_id[r["question_id"]]
        r.update(citation_metrics(r["citations"], r["retrieved_context"], q))
        r["answer_token_recall"] = token_recall(r["expected_answer"], r["answer"])


def summarise(records: List[dict], by_id: Dict[str, dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for method in dict.fromkeys(r["method"] for r in records):
        mine = [r for r in records if r["method"] == method]
        ans = [r for r in mine if by_id[r["question_id"]]["answerable"]]
        oos = [r for r in mine if not by_id[r["question_id"]]["answerable"]]
        supported = [r for r in ans if r["status"] != INSUFFICIENT]
        entry = {
            "n_answerable": len(ans), "n_out_of_scope": len(oos),
            "judge_scores_answerable": {k: describe([r.get(f"{k}_score") for r in ans if r.get(f"{k}_score") is not None])
                                        for k in ("correctness", "faithfulness", "relevance", "citation")},
            "judge_scores_out_of_scope": {k: describe([r.get(f"{k}_score") for r in oos if r.get(f"{k}_score") is not None])
                                          for k in ("correctness", "faithfulness")},
            "correct_refusal_rate_out_of_scope": (sum(r["status"] == INSUFFICIENT for r in oos) / len(oos)) if oos else None,
            "false_refusal_rate_answerable": (sum(r["status"] == INSUFFICIENT for r in ans) / len(ans)) if ans else None,
            "citations_on_supported_answers": {k: describe([r[k] for r in supported])
                                               for k in ("cite_present", "cite_valid", "cite_precision", "cite_recall")},
            "answer_token_recall_answerable": describe([r["answer_token_recall"] for r in ans]),
            "retrieval_s": describe([r["retrieval_s"] for r in mine]),
            "generation_s": describe([r["generation_s"] for r in mine]),
        }
        out[method] = entry
    return out


def write_csv(records: List[dict], by_id: Dict[str, dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            q = by_id[r["question_id"]]
            row = dict(r, category=q["category"], difficulty=q["difficulty"], answerable=q["answerable"])
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


def write_manual_sheet(records: List[dict], path: str) -> None:
    """A sheet for scoring answers by hand with the same 0-3 rubric as the LLM judge."""
    fields = ["question_id", "method", "question", "expected_answer", "answer",
              "correctness_0_3", "faithfulness_0_3", "relevance_0_3", "citation_0_3", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(records, key=lambda r: (r["question_id"], r["method"])):
            writer.writerow({k: r.get(k, "") for k in fields[:5]})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--manual-sheet", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from generate import cfg
    questions = json.load(open(cfg["questions_file"], encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    by_id = {q["question_id"]: q for q in questions}
    path = os.path.join(cfg["evaluation_dir"], "answers.json")
    records = load_json(path, [])

    if not args.judge_only and not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is not set, so answers cannot be generated. Answer evaluation is PENDING. "
                 "Copy .env.example to .env and add your key.")

    if not args.judge_only:
        print("Generating answers...")
        records = generate_records(questions, MAIN_METHODS, records, path, cfg)
    if not args.no_judge:
        print("Judging answers...")
        judge_records(records, path, cfg)

    records = [r for r in records if r["question_id"] in by_id]
    add_deterministic_metrics(records, by_id)
    write_csv(records, by_id, os.path.join(cfg["evaluation_dir"], "answer_results.csv"))
    if args.manual_sheet:
        write_manual_sheet(records, os.path.join(cfg["evaluation_dir"], "manual_scoring_sheet.csv"))
    summary = summarise(records, by_id)
    update_json(os.path.join(cfg["results_dir"], "summary.json"), "answers",
                {"judge": "LLM-based (" + cfg["judge_model"] + "), not human evaluation",
                 "generator": cfg["llm_model"], "methods": summary})
    for method, s in summary.items():
        sc = s["judge_scores_answerable"]
        print(f"{method:14s} correctness={sc['correctness']['mean']:.2f} faithfulness={sc['faithfulness']['mean']:.2f} "
              f"relevance={sc['relevance']['mean']:.2f} citation={sc['citation']['mean']:.2f} "
              f"refusal(oos)={s['correct_refusal_rate_out_of_scope']} false_refusal={s['false_refusal_rate_answerable']}")
    print("Saved evaluation/answers.json, evaluation/answer_results.csv, results/summary.json")


if __name__ == "__main__":
    main()
