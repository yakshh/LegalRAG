# CIPAT report content

Report-ready text for the GTU NLP (3170723) CIPAT. All results below come from `results/summary.json`; if you re-run the experiments, update the numbers from that file. Items marked **[INSERT]** need a figure or a screenshot that only the team can produce.

---

## 1. Title

**LegalRAG: Comparative Evaluation of Dense Retrieval and Hybrid Retrieval with Reranking for Legal Question Answering**

## 2. Abstract

Retrieval-Augmented Generation (RAG) grounds the answers of a large language model in documents retrieved at question time, which matters in the legal domain, where an invented provision or citation can cause real harm. The retrieval stage largely decides quality. Dense retrieval captures meaning but can miss exact legal terms; BM25 captures exact words but can miss paraphrases. Hybrid retrieval and reranking are proposed as remedies, but the published evidence is mixed. This project reviews recent legal RAG studies and builds LegalRAG, a research prototype that compares three configurations under identical conditions: Dense RAG, Hybrid RAG (BM25 + dense with Reciprocal Rank Fusion) and Hybrid RAG with a cross-encoder reranker. All retrieval components are pretrained and open-source; no model is trained. The corpus is the Information Technology Act, 2000 and the Consumer Protection Act, 2019, split into section-aware chunks; the evaluation set has 50 hand-written questions (44 answerable, 6 out-of-scope).

On the 44 answerable questions, Hybrid + Reranking had the best retrieval quality (MRR 0.781, Recall@5 0.917, Hit@5 0.932) compared with Hybrid (0.679, 0.830, 0.864) and Dense (0.648, 0.705, 0.750), at a retrieval time of about 0.61 s per question against about 0.014 s. An LLM judge gave mean correctness scores of 2.61, 2.39 and 1.98 (scale 0 to 3) for Hybrid + Reranking, Hybrid and Dense; the difference mainly comes from false refusals (13.6%, 20.5% and 31.8% of answerable questions). All three methods refused all six out-of-scope questions and no answer contained a citation that was not retrieved. The comparison is limited by a small question set, an LLM-based judge and exploratory statistics. The system is for academic use and is not legal advice.

## 3. Introduction

Legal text is long, structured into Acts, chapters and sections, and precise: one word such as "shall" or "may" changes a provision. LLMs write fluent answers but may hallucinate provisions or citations. RAG first retrieves passages from a trusted collection and then asks the LLM to answer only from them. Retrieval can be lexical (BM25), dense (embedding vectors) or hybrid, and a cross-encoder can re-score the candidates by reading the question and each passage together.

## 4. Problem statement

Legal RAG depends on retrieving the correct provision, yet studies disagree on which retrieval strategy works best, and none of the reviewed papers compares dense, hybrid (BM25 + dense with RRF) and hybrid + cross-encoder reranking under identical conditions with open components on Indian statutes. Practical challenges: exact terms versus paraphrase, preserving section and page structure for citation, hallucination, refusing questions the documents cannot answer, and cost.

## 5. Motivation

Citizens and students need to find what an Act says. A system that retrieves the right section, answers only from it, cites document, page and section, and admits when the documents are insufficient is more trustworthy than a general chatbot. Choosing the retrieval strategy on evidence rather than assumption is the point of the project.

## 6. Objectives

1. Review recent papers on legal RAG and legal question answering.
2. Implement Dense RAG, Hybrid RAG and Hybrid + Reranking on the same documents, questions and LLM.
3. Generate answers only from retrieved context, with citations and a fixed fallback when evidence is insufficient.
4. Evaluate retrieval (Precision@K, Recall@K, Hit@K, MRR, nDCG@5, latency) and answers (correctness, faithfulness, relevance, citation quality, refusal behaviour).
5. Report the results honestly, including cases where hybrid or reranking did not help.

## 7. Literature review

Full entries are in `research/paper_analysis.md`; the feature table is in `research/comparison.md`.

- **HyPA-RAG** (Kalra et al., 2024/25): adaptive retrieval parameters chosen by a query-complexity classifier, hybrid BM25 + dense retrieval with RRF, a knowledge graph, optional reranker; evaluated on NYC Local Law 144 with RAGAS-style metrics. Reports that reranking may slightly reduce correctness in some settings.
- **CBR-RAG** (Wiratunga et al., 2024): case-based reasoning to form RAG context, comparing BERT, LegalBERT and AnglEBERT embeddings with intra, inter and hybrid (weighted embedding) similarity; its "hybrid" is not BM25 + dense.
- **LegalRAG, multilingual** (Kabir et al., 2025): English/Bangla gazettes; a relevance-check and query-refinement step improves a vanilla pipeline; out-of-context questions remain a weakness.
- **CanLegalRAGBench** (Zhao et al., 2026, as summarised in the report draft): BM25, dense, hybrid and reranked retrieval on Canadian case law; hybrid with RRF was below dense alone, and a reranker recovered the difference.
- Other studies from the report: Wagle et al. (Nepali case law, BM25 beat dense), Keisha et al. (open-source embeddings, reranking inconclusive), Kulkarni et al. (layout-aware chunking and hybrid retrieval).

We do not claim to reproduce any of them.

## 8. Research gap

The literature shows the value of retrieval augmentation, hybrid retrieval, domain-specific retrieval and reranking, but a **controlled comparison of dense, hybrid and hybrid + cross-encoder retrieval under the same corpus, questions, generation model and evaluation protocol** is missing, especially on Indian statutes, with citation quality and out-of-scope questions measured. Details in `research/research_gap.md`. Our contribution is the controlled comparison and evaluation, not a new algorithm.

## 9. Proposed system

LegalRAG ingests legal PDFs into section-aware chunks, retrieves with one of three methods, and generates a grounded answer with citations. The research question is: *How does hybrid lexical-semantic retrieval with reranking affect retrieval quality and legal QA performance compared with a dense vector-based RAG system?* We do not assume the answer.

## 10. System architecture

See the diagram in `research/methodology.md` (ASCII and Mermaid). **[INSERT]** a drawn version for the report.

```
Legal PDFs -> extraction -> preprocessing -> section-aware chunks -> embeddings -> FAISS
                                                              \-> BM25 index
Method A: FAISS top-K
Method B: FAISS top-N + BM25 top-N -> RRF -> top-K
Method C: Method B candidates -> cross-encoder -> top-K
Top-K context -> grounded LLM -> answer + citations + status (SUPPORTED / INSUFFICIENT EVIDENCE)
```

## 11. Dataset

- **Corpus:** `ITAct2000.pdf` (34 pages) and `ConsumerProtectionAct2019.pdf` (39 pages), used unaltered. The IT Act is the 2000 text without the 2008 amendments.
- **After ingestion:** 73 pages, 353 chunks (IT Act 161, Consumer Protection Act 192), 207 sections (including schedules and preambles).
- **Evaluation set:** 50 hand-written questions: 44 answerable (22 per Act) and 6 out-of-scope; 10 categories (definition, section-specific, legal obligation, penalty, applicability, scenario-based, comparison, multi-section, exact terminology, semantic understanding); 17 easy, 21 medium, 12 hard. Each has source document, section, page, expected answer, relevant sections and optional evidence phrases. Checked by `validate_dataset.py`.

## 12. Preprocessing

PyMuPDF extracts text page by page. The table of contents is dropped; gazette footnotes (such as "1. Ins. by Act 33 of 2021") are not treated as headings; broken dash characters are fixed; whitespace is normalised.

## 13. Chunking

Sections are detected from numbered headings, and schedules are labelled separately. Each section is cut into chunks of about 1000 characters with 150 overlap, only at spaces. A chunk stores document, page (where the chunk starts), section and text.

## 14. Dense retrieval

`all-MiniLM-L6-v2` embeds chunks and questions; FAISS `IndexFlatIP` over normalised vectors gives cosine similarity.

## 15. BM25

`BM25Okapi` (k1 = 1.5, b = 0.75) over lower-cased word tokens.

## 16. Hybrid retrieval

The top N = 20 chunks from dense search and from BM25 are merged into one candidate list.

## 17. Reciprocal Rank Fusion

`score = sum of 1 / (60 + rank)` over the two lists. No score scaling is needed.

## 18. Cross-encoder reranking

`ms-marco-MiniLM-L-6-v2` scores each (question, chunk) pair for the 20 best fused candidates; the top K = 5 are kept.

## 19. LLM generation

Gemini (`gemini-3.5-flash-lite`, temperature 0), the same for all methods, with one fixed prompt.

## 20. Legal grounding

The prompt requires answers only from the context, no invented sections, cases or penalties, citations in the form `[document, p. N, Section X]`, and the exact fallback "Insufficient evidence in the retrieved legal documents to answer this question reliably." Each answer is labelled SUPPORTED or INSUFFICIENT EVIDENCE.

## 21. Experimental setup

| Item | Value |
|---|---|
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Generator | gemini-3.5-flash-lite |
| Judge | gemini-3.1-flash-lite (a different model, LLM-based) |
| Chunk size / overlap | 1000 / 150 characters |
| K / N / M / RRF k | 5 / 20 / 20 / 60 |
| BM25 (k1, b) | 1.5, 0.75 |
| Questions | 50 (44 answerable, 6 out-of-scope) |
| Hardware | the team's Windows laptop (CPU only) **[INSERT exact CPU/RAM]** |

The models were chosen for being small, free and open; the generator and judge were chosen for what the free API tier allowed (models with a daily quota of about 20 requests were not usable for 300 calls). Settings were fixed before the final runs.

## 22. Evaluation metrics

Precision@K, Recall@K (section level), Hit@K, MRR, nDCG@5 and latency for retrieval; LLM-judged correctness, faithfulness, relevance and citation correctness (0 to 3) for answers; deterministic citation checks (present, valid, precise, complete); correct refusal on out-of-scope questions and false refusal on answerable questions. Definitions in `research/methodology.md`. A chunk is relevant if it comes from a labelled section and, for long sections, contains the evidence phrase.

## 23. Results

### 23.1 Retrieval (44 answerable questions)

| Metric | Dense RAG | Hybrid RAG | Hybrid + Reranking |
|---|---|---|---|
| Precision@3 | 0.288 | 0.326 | 0.386 |
| Precision@5 | 0.191 | 0.218 | 0.245 |
| Recall@3 | 0.652 | 0.742 | 0.875 |
| Recall@5 | 0.705 | 0.830 | 0.917 |
| Hit@3 | 0.682 | 0.773 | 0.909 |
| Hit@5 | 0.750 | 0.864 | 0.932 |
| MRR | 0.648 | 0.679 | 0.781 |
| nDCG@5 | 0.634 | 0.705 | 0.813 |
| Retrieval latency, mean (ms) | 13.1 | 14.0 | 611.9 |

Precision@K is low for every method because most questions need one or two chunks, so at most 1 to 2 of 5 results can be relevant.

Exploratory paired permutation tests (44 questions, no correction for multiple comparisons): Hybrid vs Dense, MRR p = 0.46, Hit@5 p = 0.13, Recall@5 p = 0.07; Hybrid + Reranking vs Dense, MRR p = 0.036, Hit@5 p = 0.021, Recall@5 p = 0.005, Precision@5 p = 0.002; Hybrid + Reranking vs Hybrid, MRR p = 0.045, Hit@5 p = 0.38, Recall@5 p = 0.14.

### 23.2 Ablation (retrieval only)

| Variant | MRR | Recall@5 | Hit@5 | nDCG@5 |
|---|---|---|---|---|
| Dense | 0.648 | 0.705 | 0.750 | 0.634 |
| BM25 only | 0.687 | 0.777 | 0.795 | 0.695 |
| Dense + BM25 (no RRF) | 0.688 | 0.811 | 0.818 | 0.699 |
| Dense + BM25 + RRF (Hybrid) | 0.679 | 0.830 | 0.864 | 0.705 |
| + cross-encoder (Hybrid + Reranking) | 0.781 | 0.917 | 0.932 | 0.813 |

### 23.3 Answer quality (LLM judge, 0 to 3; 44 answerable questions)

| Metric | Dense RAG | Hybrid RAG | Hybrid + Reranking |
|---|---|---|---|
| Correctness (mean, SD) | 1.98 (1.36) | 2.39 (1.22) | 2.61 (0.97) |
| Faithfulness | 3.00 | 3.00 | 3.00 |
| Relevance | 2.45 | 2.45 | 2.66 |
| Citation correctness | 2.18 | 2.39 | 2.66 |
| False refusals (answerable) | 14 (31.8%) | 9 (20.5%) | 6 (13.6%) |
| Correct refusals (6 out-of-scope) | 6 / 6 | 6 / 6 | 6 / 6 |
| Citation valid / precision / recall (answered questions) | 1.00 / 1.00 / 0.98 | 1.00 / 0.97 / 0.99 | 1.00 / 0.98 / 0.99 |
| Generation time, mean (s) | 1.23 | 1.22 | 1.24 |

The standard error of a mean correctness score is about 0.15 to 0.20, so the Dense vs Hybrid + Reranking gap (0.64) is larger than two standard errors, while the Hybrid vs Hybrid + Reranking gap (0.23) is not clearly outside the noise.

### 23.4 Charts

**[INSERT]** from `results/figures/`: `precision_at_5.png`, `recall_at_5.png`, `mrr.png`, `hit_at_5.png`, `retrieval_latency.png`, `answer_correctness.png`, `answer_faithfulness.png`, `citation_quality.png`, `ablation_mrr.png`.

## 24. Comparative analysis

- **Retrieval quality.** Hybrid + Reranking was best on every retrieval metric. Hybrid was better than Dense on every metric, but the paired tests cannot separate them reliably. In the ablation BM25 alone beat Dense alone, so statute wording favours lexical matching here; the reranker produced the largest single gain.
- **Answer quality.** Mean correctness and citation score follow the same order. Most of the difference is the number of false refusals, which falls from 14 to 9 to 6 as retrieval improves. Faithfulness was 3.00 for every method, so it does not distinguish them.
- **Trade-off.** Reranking costs about 0.6 s extra per question on a CPU (about 44 times the retrieval time), a small absolute cost compared with LLM generation of 1.2 s here, but it would grow with more candidates or larger corpora.
- **Safety.** Every method refused all out-of-scope questions and produced no citation outside the retrieved chunks.
- **Hypothesis.** The hypothesis that hybrid retrieval with reranking differs from dense-only retrieval is supported on this corpus. We do not claim this transfers to other corpora, models or larger question sets.

## 25. Error analysis

See `research/error_analysis.md`. Main points: BM25 wins when the question uses the Act's own words; dense wins on some definition and paraphrase questions; the reranker fixes many ranks but demoted the key chunk for Q001; multi-section and definition questions cause most false refusals (two definition questions were not retrieved by any method).

## 26. Advantages

Controlled comparison with identical inputs; open retrieval components and no training; section, page and document metadata kept for citation; automatic citation checks; out-of-scope questions; one configuration file; tests; every number reproducible with one command.

## 27. Limitations

Two Acts; 50 questions written by the team; section and phrase-level relevance labels; small general-purpose models; LLM-based judging with a ceiling effect on faithfulness; IT Act in the 2000 version; one run per method; free-tier API limits; latency depends on the machine. See `research/limitations.md`.

## 28. Future work

More Acts and case law; a relevance-check and query-refinement step (as in the multilingual LegalRAG paper); definition-level chunking; larger K or per-section retrieval for multi-section questions; tuning fusion weights; legal-domain or larger embedding and reranking models; Hindi and Gujarati; human evaluation by legal experts.

## 29. Conclusion

On two Indian Acts and 50 questions, hybrid retrieval with cross-encoder reranking gave the best retrieval and answer quality, hybrid retrieval gave a smaller and less certain improvement over dense retrieval, and reranking added about 0.6 s of retrieval time per question. Questions that the documents cannot answer were refused by all methods. The results are specific to this corpus, these small models and an LLM judge, and should be confirmed on a larger, expert-reviewed question set.

## 30. References

1. N. Wiratunga, R. Abeyratne, L. Jayawardena, K. Martin, S. Massie, I. Nkisi-Orji, R. Weerasinghe, A. Liret, B. Fleisch, "CBR-RAG: Case-Based Reasoning for Retrieval Augmented Generation in LLMs for Legal Question Answering," arXiv:2404.04302, 2024.
2. R. Kalra, Z. Wu, A. Gulley, A. Hilliard, X. Guan, A. Koshiyama, P. Treleaven, "HyPA-RAG: A Hybrid Parameter Adaptive Retrieval-Augmented Generation System for AI Legal and Policy Applications," arXiv:2409.09046, 2024 (v2, 2025).
3. M. R. Kabir et al., "LegalRAG: A Hybrid RAG System for Multilingual Legal Information Retrieval," arXiv:2504.16121, 2025.
4. E. Zhao, M. Taranukhin, W. Cui, M. Aikenhead, V. Shwartz, "CanLegalRAGBench: Evaluating Retrieval-Augmented Generation on Canadian Case Law," arXiv:2605.30497, 2026.
5. S. Wagle et al., "Retrieval Augmented Generation Framework for the Nepali Legal Domain Question Answering," arXiv:2606.07523, 2026.
6. F. Keisha et al., "All for law and law for all: Adaptive RAG Pipeline for Legal Research," arXiv:2508.13107, 2025.
7. A. Kulkarni et al., "Long-Context Long-Form Question Answering for Legal Domain," EACL Industry Track 2026, arXiv:2602.07190.
8. N. Pipitone and G. H. Alami, "LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain," arXiv:2408.10343, 2024.
9. P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS, 2020.
10. S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," Foundations and Trends in Information Retrieval, 2009.
11. G. V. Cormack, C. L. A. Clarke, S. Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods," SIGIR, 2009.
12. R. Nogueira and K. Cho, "Passage Re-ranking with BERT," arXiv:1901.04085, 2019.
13. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP, 2019.

Items 1 to 3 were checked against the arXiv texts; check items 4 to 7 against the original papers before submission (they were taken from the team's report draft).
