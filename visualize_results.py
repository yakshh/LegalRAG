"""Draw the comparison charts from the real result files (results/summary.json). Nothing is invented:
a chart is skipped, and reported as pending, when its data does not exist yet.

    python visualize_results.py

Run `python evaluate.py` (retrieval charts) and `python evaluate_answers.py` (answer charts) first.
Charts are saved in results/figures/. Bars show the mean over questions; thin whiskers show +/- 1 standard error of the mean.
"""
import json
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import yaml

NAMES = {"dense": "Dense RAG", "bm25": "BM25 only", "dense_bm25": "Dense + BM25\n(no RRF)",
         "hybrid": "Hybrid RAG", "hybrid_rerank": "Hybrid +\nReranking"}
COLORS = {"dense": "#4C78A8", "bm25": "#9D9D9D", "dense_bm25": "#B279A2", "hybrid": "#F58518",
          "hybrid_rerank": "#54A24B"}


def bar_chart(methods: List[str], means: List[float], sds: List[float], title: str, ylabel: str, path: str,
              ylim: Optional[tuple] = None, fmt: str = "{:.3f}") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = range(len(methods))
    ax.bar(xs, means, color=[COLORS[m] for m in methods], width=0.6)
    ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor="#333333", elinewidth=1, capsize=3)
    for x, m, e in zip(xs, means, sds):
        ax.text(x, m + e, fmt.format(m), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([NAMES[m] for m in methods], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


def grouped_chart(methods: List[str], series: Dict[str, List[float]], title: str, ylabel: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(methods)
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for j, (label, vals) in enumerate(series.items()):
        ax.bar([i + j * width for i in range(len(methods))], vals, width=width, label=label,
               color=palette[j % len(palette)])
    ax.set_xticks([i + width * (len(series) - 1) / 2 for i in range(len(methods))])
    ax.set_xticklabels([NAMES[m] for m in methods], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.4)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


def se(entry: dict) -> float:
    """Standard error of the mean from a describe() entry."""
    return entry["std"] / entry["n"] ** 0.5 if entry["n"] > 1 else 0.0


def main() -> None:
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    summary_path = os.path.join(cfg["results_dir"], "summary.json")
    if not os.path.exists(summary_path):
        raise SystemExit(f"{summary_path} not found. Run evaluate.py and evaluate_answers.py first.")
    summary = json.load(open(summary_path, encoding="utf-8"))
    out = os.path.join(cfg["results_dir"], "figures")
    os.makedirs(out, exist_ok=True)

    retrieval = summary.get("retrieval", {}).get("methods")
    if retrieval:
        methods = [m for m in ("dense", "hybrid", "hybrid_rerank") if m in retrieval]
        for key, title in [("precision@5", "Precision@5"), ("recall@5", "Recall@5 (section level)"),
                           ("mrr", "Mean Reciprocal Rank (MRR)"), ("hit@5", "Hit@5")]:
            bar_chart(methods, [retrieval[m][key]["mean"] for m in methods],
                      [se(retrieval[m][key]) for m in methods], f"{title} by retrieval method", title,
                      os.path.join(out, key.replace("@", "_at_") + ".png"), ylim=(0, 1.1))
        bar_chart(methods, [retrieval[m]["latency_s"]["mean"] * 1000 for m in methods],
                  [se(retrieval[m]["latency_s"]) * 1000 for m in methods],
                  "Average retrieval latency per question", "milliseconds",
                  os.path.join(out, "retrieval_latency.png"), fmt="{:.0f} ms")
    else:
        print("retrieval charts: PENDING (run python evaluate.py)")

    ablation = summary.get("ablation")
    if ablation:
        methods = [m for m in ("dense", "bm25", "dense_bm25", "hybrid", "hybrid_rerank") if m in ablation]
        bar_chart(methods, [ablation[m]["mrr"]["mean"] for m in methods], [se(ablation[m]["mrr"]) for m in methods],
                  "Ablation: MRR as components are added", "MRR", os.path.join(out, "ablation_mrr.png"), ylim=(0, 1.1))
        bar_chart(methods, [ablation[m]["recall@5"]["mean"] for m in methods],
                  [se(ablation[m]["recall@5"]) for m in methods], "Ablation: Recall@5 as components are added",
                  "Recall@5", os.path.join(out, "ablation_recall_at_5.png"), ylim=(0, 1.1))
    else:
        print("ablation charts: PENDING (run python evaluate.py --ablation)")

    answers = summary.get("answers", {}).get("methods")
    judged = answers and all(answers[m]["judge_scores_answerable"]["correctness"]["n"] for m in answers)
    if judged:
        methods = [m for m in ("dense", "hybrid", "hybrid_rerank") if m in answers]
        judge = summary["answers"]["judge"]
        for key, title in [("correctness", "Answer correctness"), ("faithfulness", "Faithfulness (groundedness)")]:
            sc = [answers[m]["judge_scores_answerable"][key] for m in methods]
            bar_chart(methods, [s["mean"] for s in sc], [se(s) for s in sc],
                      f"{title} (0-3, LLM judge)", "mean score (0-3)",
                      os.path.join(out, f"answer_{key}.png"), ylim=(0, 3.3), fmt="{:.2f}")
        cite = lambda m, k: answers[m]["citations_on_supported_answers"][k]["mean"]
        grouped_chart(methods, {
            "citation valid (matches retrieved chunk)": [cite(m, "cite_valid") for m in methods],
            "citation precision (relevant section)": [cite(m, "cite_precision") for m in methods],
            "citation recall (relevant sections cited)": [cite(m, "cite_recall") for m in methods],
            "judge citation score / 3 (all answerable)": [answers[m]["judge_scores_answerable"]["citation"]["mean"] / 3 for m in methods],
        }, "Citation quality (first three bars: answers that gave an answer)", "score (0-1)", os.path.join(out, "citation_quality.png"))
        print("answer judge:", judge)
    else:
        print("answer charts: PENDING (run python evaluate_answers.py with a valid GEMINI_API_KEY)")


if __name__ == "__main__":
    main()
