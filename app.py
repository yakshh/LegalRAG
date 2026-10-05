"""LegalRAG web demo.  Run:  streamlit run app.py"""
import json
import os
import subprocess
import sys
import time

import pandas as pd
import streamlit as st

from generate import INSUFFICIENT, answer_question, cfg
from retrieve import METHOD_NAMES, MAIN_METHODS, get_retriever

DISCLAIMER = "This system is developed for academic/research purposes and does not constitute legal advice."

st.set_page_config(page_title="LegalRAG", page_icon="⚖️", layout="wide")
st.title("LEGALRAG")
st.caption("Legal Question Answering using Retrieval-Augmented Generation")
st.warning(DISCLAIMER)


@st.cache_resource(show_spinner="Loading models and building indexes (first run only)...")
def load_retriever():
    return get_retriever()


def read_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


ask_tab, eval_tab = st.tabs(["Ask", "Evaluation"])

# ------------------------------------------------------------------ Ask
with ask_tab:
    method = st.radio("Method", MAIN_METHODS, format_func=METHOD_NAMES.get, horizontal=True)
    question = st.text_input("Question", placeholder="e.g. What is the time limit for filing a consumer complaint?")
    if st.button("Ask Question", type="primary") and question.strip():
        try:
            retriever = load_retriever()
        except SystemExit as e:
            st.error(str(e))
            st.stop()
        t0 = time.perf_counter()
        with st.spinner("Searching the Acts..."):
            chunks = retriever.retrieve(question, method, cfg["top_k"])
        t1 = time.perf_counter()

        result = None
        try:
            with st.spinner("Writing the answer..."):
                result = answer_question(question, chunks)
        except Exception as e:  # no internet, bad key, model busy ...
            st.error(f"Could not get an answer from Gemini: {e}. The retrieved evidence is shown below.")
        t2 = time.perf_counter()

        if result:
            st.subheader("Answer")
            if result["status"] == INSUFFICIENT:
                st.info("Evidence status: INSUFFICIENT EVIDENCE. The retrieved documents do not support an answer.")
            else:
                st.success("Evidence status: SUPPORTED by the retrieved text (check the citations below).")
            st.markdown(result["answer"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval time", f"{t1 - t0:.2f} s")
        c2.metric("Generation time", f"{t2 - t1:.2f} s")
        c3.metric("Total time", f"{t2 - t0:.2f} s")

        st.subheader("Retrieved evidence")
        for i, c in enumerate(chunks, start=1):
            score = c["score"]
            score_text = f"{score:.3f}" if score == score else "n/a"  # n/a for NaN
            with st.expander(f"{i}. {c['document']} | page {c['page']} | {c['section']} | "
                             f"{c['score_type']}: {score_text}"):
                st.text(c["text"])

# ------------------------------------------------------------------ Evaluation
with eval_tab:
    summary = read_json(os.path.join(cfg["results_dir"], "summary.json")) or {}
    questions = read_json(cfg["questions_file"]) or []
    chunks_all = read_json(cfg["chunks_file"]) or []

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluation questions", len(questions))
    c2.metric("Answerable / out-of-scope", f"{sum(q['answerable'] for q in questions)} / "
                                           f"{sum(not q['answerable'] for q in questions)}")
    c3.metric("Documents", len({c['document'] for c in chunks_all}))
    c4.metric("Chunks", len(chunks_all))

    st.subheader("Retrieval metrics")
    ret = summary.get("retrieval", {}).get("methods")
    if ret:
        keys = ["precision@3", "precision@5", "recall@3", "recall@5", "hit@3", "hit@5", "mrr", "ndcg@5"]
        df = pd.DataFrame({METHOD_NAMES[m]: {k: ret[m][k]["mean"] for k in keys} for m in MAIN_METHODS if m in ret})
        df.loc["avg latency (ms)"] = [ret[m]["latency_s"]["mean"] * 1000 for m in MAIN_METHODS if m in ret]
        st.dataframe(df.style.format("{:.3f}"))
    else:
        st.info("Retrieval results are pending. Click 'Run retrieval evaluation' below.")

    st.subheader("Answer metrics (LLM judge, scores 0-3)")
    ans = summary.get("answers", {}).get("methods")
    if ans:
        st.caption(summary["answers"]["judge"])
        rows = {}
        for m in MAIN_METHODS:
            if m in ans:
                s = ans[m]
                rows[METHOD_NAMES[m]] = {
                    "correctness": s["judge_scores_answerable"]["correctness"]["mean"],
                    "faithfulness": s["judge_scores_answerable"]["faithfulness"]["mean"],
                    "relevance": s["judge_scores_answerable"]["relevance"]["mean"],
                    "citation": s["judge_scores_answerable"]["citation"]["mean"],
                    "correct refusal (out-of-scope)": s["correct_refusal_rate_out_of_scope"],
                    "false refusal (answerable)": s["false_refusal_rate_answerable"],
                    "generation time (s)": s["generation_s"]["mean"],
                }
        st.dataframe(pd.DataFrame(rows).style.format("{:.3f}"))
    else:
        st.info("Answer evaluation is pending. It needs a Gemini API key (see README).")

    figures = os.path.join(cfg["results_dir"], "figures")
    if os.path.isdir(figures) and os.listdir(figures):
        st.subheader("Charts")
        cols = st.columns(2)
        for i, name in enumerate(sorted(os.listdir(figures))):
            if name.endswith(".png"):
                cols[i % 2].image(os.path.join(figures, name), caption=name)

    st.subheader("Run evaluation")
    st.caption("Runs the same scripts as the command line. The API key is never shown.")

    def run_script(args):
        with st.spinner("Running " + " ".join(args) + " ..."):
            proc = subprocess.run([sys.executable] + args, capture_output=True, text=True)
        st.code((proc.stdout + proc.stderr)[-3000:] or "(no output)")

    b1, b2, b3 = st.columns(3)
    if b1.button("Run retrieval evaluation"):
        run_script(["evaluate.py"])
    if b2.button("Run answer evaluation (uses Gemini API)"):
        run_script(["evaluate_answers.py"])
    if b3.button("Redraw charts"):
        run_script(["visualize_results.py"])
