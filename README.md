# LegalRAG

**LegalRAG: Comparative Evaluation of Dense Retrieval and Hybrid Retrieval with Reranking for Legal Question Answering**
GTU B.E. Semester VII, Natural Language Processing (3170723), CIPAT.

A research prototype that answers questions about Indian Acts using Retrieval-Augmented Generation and compares three retrieval methods on the same documents, questions, LLM and metrics:

| Method | Retrieval |
|---|---|
| **A. Dense RAG** | sentence embeddings + FAISS |
| **B. Hybrid RAG** | dense + BM25 merged with Reciprocal Rank Fusion (RRF) |
| **C. Hybrid + Reranking** | method B, then a cross-encoder re-scores the candidates |

No model is trained. We do not assume that hybrid or reranking is better; the results decide (`results/summary.json`).

> This system is developed for academic/research purposes and does not constitute legal advice.

---

## 1. Requirements

- Python 3.10 or newer (developed on Python 3.14, Windows)
- Internet for the first run (downloads two small Hugging Face models, about 90 MB each; no account or token needed) and for answer generation (Gemini API)
- A free Gemini API key for generation and answer evaluation. Retrieval evaluation, the tests and dataset validation need **no** key.

## 2. Installation

```bash
pip install -r requirements.txt
```

## 3. Environment setup (API key)

```bash
copy .env.example .env        # Windows   (macOS/Linux: cp .env.example .env)
```

Open `.env` and set `GEMINI_API_KEY=<your key>`. The file is in `.gitignore`; never commit it. The code never contains a key.

The Gemini model names are in `config.yaml` (`llm_model` for answers, `judge_model` for the judge). If Google retires a model you get a clear error; change the name there. The free tier limits how many calls a model allows (some allow ~20 per day, the "-lite" models 15 per minute), so `evaluate_answers.py` saves progress and can be re-run to continue.

## 4. Adding the legal PDFs

Put the Acts in `data/` (details and download links in `data/README.md`):

```
data/ITAct2000.pdf
data/ConsumerProtectionAct2019.pdf
```

## 5. Commands

Run everything from this folder.

| Step | Command | Needs key? | Output |
|---|---|---|---|
| Ingestion (chunks) | `python ingest.py` | no | `chunks.json` |
| Validate the question set | `python validate_dataset.py` | no | validation report |
| Retrieval evaluation | `python evaluate.py` | no | `evaluation/retrieval_results.csv`, `results/summary.json` |
| Ablation (optional) | `python evaluate.py --ablation` | no | `results/summary.json` (key `ablation`) |
| Answer evaluation | `python evaluate_answers.py` | **yes** | `evaluation/answers.json`, `evaluation/answer_results.csv`, `results/summary.json` |
| Manual scoring sheet (optional) | `python evaluate_answers.py --manual-sheet` | no (after answers exist) | `evaluation/manual_scoring_sheet.csv` |
| Charts | `python visualize_results.py` | no | `results/figures/*.png` |
| Web demo | `streamlit run app.py` | yes (for answers) | browser |
| Tests | `python -m pytest` | no | test report |

Typical order: `ingest.py`, `validate_dataset.py`, `evaluate.py`, `evaluate.py --ablation`, `evaluate_answers.py`, `visualize_results.py`.

`evaluate_answers.py` options: `--no-judge` (only generate), `--judge-only`, `--limit N` (first N questions), `--manual-sheet`.

### The web demo

`streamlit run app.py` has two tabs.
- **Ask**: choose Dense, Hybrid or Hybrid + Reranking, type a question, and see the answer, the evidence status (SUPPORTED or INSUFFICIENT EVIDENCE), the retrieved evidence (document, page, section, score) and retrieval / generation / total time. The disclaimer is always shown.
- **Evaluation**: dataset and corpus size, the retrieval and answer metric tables, latency, the charts, and buttons to run the evaluations (the API key is never shown).

## 6. Project structure

```
app.py                  Streamlit demo (Ask + Evaluation tabs)
ingest.py               PDFs -> section-aware chunks (chunks.json)
retrieve.py             Dense, BM25, RRF, cross-encoder reranking (the three methods + ablation variants)
generate.py             Grounded prompt, Gemini call, citation extraction, SUPPORTED / INSUFFICIENT status
evaluate.py             Retrieval evaluation (Precision@K, Recall@K, Hit@K, MRR, nDCG@5, latency)
evaluate_answers.py     Answer generation + judging + citation / refusal metrics
validate_dataset.py     Checks evaluation/questions.json
visualize_results.py    Charts from the real result files
config.yaml             All settings (paths, models, chunking, K/N/M, RRF k, BM25, evaluation)
requirements.txt        Dependencies
.env.example            Template for GEMINI_API_KEY
data/                   Act PDFs + README.md
evaluation/             questions.json (dataset), metrics.py, judge.py, results (answers.json, *.csv)
research/               paper_analysis.md, comparison.md, research_gap.md, methodology.md, error_analysis.md, limitations.md
results/                summary.json, figures/
docs/                   CIPAT_REPORT_CONTENT.md, PPT_CONTENT.md, VIVA_QUESTIONS.md
tests/                  pytest tests
FINAL_AUDIT.md          Requirement checklist with evidence
```

## 7. Interpreting the results

- `results/summary.json` is the single source of numbers. Quote results from it.
- **Retrieval** metrics (key `retrieval`) are computed on the answerable questions. Relevance is judged at section level. Precision@K has a low ceiling when one section is split into several chunks, so read it with Recall@K and Hit@K.
- **Answers** (key `answers`) hold judge scores on a 0-3 scale (0 poor, 1 partially correct, 2 mostly correct, 3 fully correct), the refusal rates on out-of-scope and answerable questions, and the deterministic citation checks. The judge is an LLM, **not** a human.
- `retrieval.paired_tests` are exploratory permutation tests. Do not describe a difference as significant on that basis alone.
- Latency changes with the machine; compare methods, not absolute numbers.
- If `answers` is missing from `summary.json`, answer evaluation has not been run yet (pending).

## 8. Limitations

Two Acts, 50 hand-written questions, small general-purpose retrieval models, an LLM judge, the IT Act in its 2000 version, English only, free-tier API limits, and no query rewriting or fine-tuning. Full list: `research/limitations.md`.

## 9. Documentation for the CIPAT

- `research/` for the literature review, comparison, research gap, methodology, error analysis and limitations
- `docs/CIPAT_REPORT_CONTENT.md`, `docs/PPT_CONTENT.md`, `docs/VIVA_QUESTIONS.md`
- `FINAL_AUDIT.md` for what is complete, partial or pending
