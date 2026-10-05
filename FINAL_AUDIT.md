# Final audit

Status values: **COMPLETE** = implemented and tested in this project; **PARTIAL** = done with a stated gap; **PENDING** = needs something only the team can do.

## Requirements checklist

| CIPAT requirement | Status | Evidence / file |
|---|---|---|
| Dense RAG (Method A) | COMPLETE | `retrieve.py` (`dense`), `tests/test_retrieval.py` |
| Hybrid RAG: BM25 + dense (Method B) | COMPLETE | `retrieve.py` (`hybrid`) |
| Reciprocal Rank Fusion | COMPLETE | `retrieve.py` (`rrf`), tested against the formula |
| Cross-encoder reranking (Method C) | COMPLETE | `retrieve.py` (`hybrid_rerank`) |
| Fair comparison (same corpus, chunks, questions, LLM, prompt, judge) | COMPLETE | `config.yaml`, `research/methodology.md` section 1 |
| Multi-document ingestion with page and section metadata | COMPLETE | `ingest.py`, `data/README.md`, `tests/test_ingest.py` |
| Ingestion logs (documents, pages, sections, chunks) | COMPLETE | `python ingest.py` output |
| Legal corpus: IT Act 2000 and Consumer Protection Act 2019 | COMPLETE | `data/` (both PDFs present; IT Act is the 2000 text without later amendments) |
| Evaluation dataset (50 questions, fields as specified, no invented page numbers) | COMPLETE | `evaluation/questions.json` (44 answerable, 6 out-of-scope; pages come from the ingested chunks) |
| Dataset validation script | COMPLETE | `validate_dataset.py` (reports VALID), `tests/test_dataset.py` |
| Retrieval metrics: Precision@3/5, Recall@3/5, Hit@3/5, MRR, nDCG@5, latency | COMPLETE | `evaluate.py`, `evaluation/metrics.py`, `evaluation/retrieval_results.csv` |
| Answer evaluation: correctness, faithfulness, relevance, citation | COMPLETE (LLM judge) | `evaluate_answers.py`, `evaluation/answers.json`, `evaluation/answer_results.csv` |
| Human evaluation | PENDING | not performed; `evaluation/manual_scoring_sheet.csv` is provided for it |
| LLM-as-judge (fixed prompt, same for all methods, labelled as LLM-based) | COMPLETE | `evaluation/judge.py` (judge `gemini-3.1-flash-lite`, generator `gemini-3.5-flash-lite`) |
| Citation evaluation (presence, validity, correctness, completeness) | COMPLETE | `evaluation/metrics.py` (`citation_metrics`), `tests/test_generation.py` |
| Hallucination / groundedness handling, SUPPORTED vs INSUFFICIENT EVIDENCE | COMPLETE | `generate.py` |
| Out-of-scope questions measured | COMPLETE | 6 questions; correct-refusal rate in `results/summary.json` |
| Research gap | COMPLETE | `research/research_gap.md` |
| Paper analysis (HyPA-RAG, CBR-RAG, LegalRAG, plus one more) | COMPLETE | `research/paper_analysis.md` (the three requested papers were read from the arXiv full texts; the entries taken from the report draft are marked as not re-verified) |
| Feature-wise comparison | COMPLETE | `research/comparison.md` |
| Methodology and architecture description | COMPLETE | `research/methodology.md` (ASCII and Mermaid) |
| Results files | COMPLETE | `evaluation/*.csv`, `results/summary.json` |
| Graphs from real data | COMPLETE | `results/figures/*.png` (`visualize_results.py`) |
| Statistics (mean, median, SD; permutation tests) | COMPLETE (exploratory) | `results/summary.json`; no claim of proven significance |
| Error analysis from real outputs | COMPLETE | `research/error_analysis.md`, `results/error_cases.md` (`analyze_errors.py`) |
| Ablation (dense, BM25, dense + BM25, + RRF, + cross-encoder) | COMPLETE (retrieval only) | `python evaluate.py --ablation`, `results/summary.json` key `ablation` |
| Streamlit UI (method choice, answer, evidence status, evidence with scores, timings, disclaimer) | COMPLETE | `app.py`; started and exercised headlessly with Streamlit's AppTest; not inspected by eye in a browser |
| Evaluation dashboard tab (dataset, corpus, metrics, charts, run buttons) | COMPLETE | `app.py` (Evaluation tab) |
| Configuration | COMPLETE | `config.yaml` |
| API key handling (`.env`, `.env.example`, not committed) | COMPLETE | `.gitignore`; the key is read from the environment only |
| Requirements | PARTIAL | `requirements.txt`; verified in the development environment, not in a fresh virtual environment |
| Reproduction guide | COMPLETE | `README.md` (only commands that exist) |
| Tests | COMPLETE | `tests/` (63 tests pass: ingestion, chunking, dense, BM25, RRF, reranking, dataset validation, citations, insufficient evidence, metrics, judge) |
| Report-ready content | COMPLETE | `docs/CIPAT_REPORT_CONTENT.md` |
| PPT content | COMPLETE | `docs/PPT_CONTENT.md` |
| Viva questions | COMPLETE | `docs/VIVA_QUESTIONS.md` (44 questions) |
| Diagrams, screenshots, hardware description for the report | PENDING | marked **[INSERT]** in the docs; only the team can produce them |
| The submitted Word report (`LegalRAG_Report.docx`) | PARTIAL | still the earlier draft (placeholders, "pending" tables). Copy the results from `docs/CIPAT_REPORT_CONTENT.md` into it |
| Optional "Improvement 3" (relevance check) | NOT IMPLEMENTED | listed as future work; not in the report scope |
| Final ZIP | COMPLETE | `LegalRAG-CIPAT-FINAL.zip` |

## Things to know before submitting

1. **Quota-driven model choice.** The free Gemini tier allowed only about 20 requests per day for the `gemini-3.5/3.6-flash` models, so generation uses `gemini-3.5-flash-lite` and judging `gemini-3.1-flash-lite` (15 requests per minute). `gemini-2.5-flash` returned "no longer available to new users". The names are in `config.yaml` and saved with the results.
2. **The judge is an LLM, not a human.** It gave faithfulness 3 to all 150 answers, so faithfulness does not separate the methods. Score a sample by hand with `evaluation/manual_scoring_sheet.csv`.
3. **Relevance labels** use section plus an optional evidence phrase. This was added after a first run showed that section-only labels overstated retrieval for long sections such as the definitions. The reported retrieval numbers use the stricter labels.
4. **Statistics are exploratory.** 44 questions, many metrics, no multiple-comparison correction.
5. **Papers from the report draft** (Wagle, Keisha, Kulkarni, Zhao) were not re-checked against their sources. The three papers named in the brief were.
6. **IT Act = 2000 text.** Questions about later amendments are out-of-scope by design.
7. `evaluation/questions.json` was found cut to 30 questions at one point during development (cause unknown). It was regenerated to the full 50 and validated; the answers were produced from the full set.
