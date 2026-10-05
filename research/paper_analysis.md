# Paper analysis

This file reviews the research papers that motivate the project. In every entry:

- **PAPER METHOD** describes what the authors did.
- **OUR IMPLEMENTATION** says what LegalRAG does and, where it differs, says so.
- **Verification.** Part A (HyPA-RAG, CBR-RAG, LegalRAG) was written from the full texts of the arXiv versions named below, which were read while preparing this file. Part B (the papers of the CIPAT report draft) was written from the team's own report draft (chapters 2 and 3). Those four entries were **not re-checked against the original papers** in this work, so check them against the papers before quoting numbers.
- We **do not claim to reproduce any of these papers.** LegalRAG is a controlled comparison of three retrieval configurations built from standard components (see `research/methodology.md`).

---

## Part A. Papers requested for the project

### Paper 1. HyPA-RAG

| Field | Content |
|---|---|
| **Title** | HyPA-RAG: A Hybrid Parameter Adaptive Retrieval-Augmented Generation System for AI Legal and Policy Applications |
| **Authors / year** | R. Kalra, Z. Wu, A. Gulley, A. Hilliard, X. Guan, A. Koshiyama, P. Treleaven (Holistic AI, University College London). arXiv:2409.09046v2, submitted Aug 2024, revised Feb 2025 |
| **Problem** | LLMs have outdated knowledge, hallucinate and reason poorly in legal and policy settings. RAG helps but suffers from retrieval errors, poor integration of context and high operating cost. |
| **Objective** | Build a RAG system for the AI legal domain that (1) adapts retrieval parameters to query complexity, (2) combines dense, sparse and knowledge-graph retrieval, and (3) comes with an evaluation framework. |
| **Dataset** | One corpus: New York City Local Law 144 (a 15-page version combining the law and enforcement rules). A "gold" question set was generated with GPT-3.5-Turbo and Giskard, de-duplicated and expert-reviewed. Question types: simple, complex, situational, plus comparative, complex situational, vague and rule-conclusion. The size of the test set was not captured in the sections reviewed. |
| **Retrieval technique** | BM25 (sparse) and vector (dense) retrieval, fused with Reciprocal Rank Fusion; a knowledge-graph retriever (triplets extracted with GPT-4o); an optional query rewriter; an optional reranker (bge-reranker-large in the ablation). Three chunking methods compared (sentence, semantic, pattern-based); sentence-level chunks of 512 tokens with 200 overlap were used afterwards. |
| **NLP / LLM technique** | A DistilBERT query-complexity classifier (fine-tuned on synthetic queries; macro F1 0.90 for 3 classes, 0.92 for 2 classes) chooses top-k and the number of query rewrites. LLM-as-judge evaluation with RAGAS metrics. |
| **Methodology** | Compare LLM-only baselines, fixed top-k RAG, parameter-adaptive RAG (PA) and the version with the knowledge graph (HyPA), plus an ablation of the reranker and query rewriter. Temperature 0. |
| **Evaluation metrics** | RAGAS faithfulness, answer relevancy, context precision and context recall, plus an adapted correctness score (1-5; counted as correct when the score is at least 4). |
| **Results (as reported)** | LLM only: GPT-3.5-Turbo faithfulness 0.2856, correctness 0.1973; GPT-4o-Mini 0.3463 and 0.4572. Fixed k=10: 0.8480 and 0.7658. Adaptive PA (3-class): 0.8971 and 0.8141. HyPA (3-class): 0.8465 and 0.7918. Ablation: adding a reranker to adaptive k and rewrites gave the highest faithfulness (0.9098); the best correctness (0.8402) came from HyPA with reranker and rewriter. Pattern-based chunking gave the best context recall (0.9046). The authors note that reranking may slightly reduce overall correctness, a trade-off. |
| **Advantages** | Adaptive cost control; combines several retrieval signals; a careful evaluation framework with human-judge agreement analysis; clear ablation. |
| **Limitations (stated)** | Correctness judged against one evaluator; knowledge-graph construction could be better; parameter mappings not rigorously validated; classifier trained on synthetic data; the law is already in the GPT models' training data. Only one small corpus. |
| **Relevance to our project** | Uses the same family of components we compare (BM25 + dense with RRF, with and without a reranker), and shows that the benefit of a reranker is not uniform. Its evaluation of faithfulness separately from correctness is the model for our answer evaluation. Pattern-based (section-delimited) chunking supports our section-aware chunking. |
| **PAPER METHOD vs OUR IMPLEMENTATION** | We implement hybrid BM25 + dense + RRF and a cross-encoder reranker. We do **not** implement the query-complexity classifier, adaptive top-k, query rewriting or the knowledge graph. We use a fixed top-k = 5 for every method. |

### Paper 2. CBR-RAG

| Field | Content |
|---|---|
| **Title** | CBR-RAG: Case-Based Reasoning for Retrieval Augmented Generation in LLMs for Legal Question Answering |
| **Authors / year** | N. Wiratunga et al. arXiv:2404.04302v1, April 2024 (ICCBR 2024) |
| **Problem** | Plain RAG does not exploit the structure of earlier cases; legal QA needs context that is matched on the right parts of a question and of the evidence. |
| **Objective** | Use the retrieve step of the case-based reasoning cycle to build the context for an LLM, and compare embedding models and similarity methods for case retrieval. |
| **Dataset** | Open Australian Legal QA (ALQA): 2,124 LLM-generated question-answer pairs with supporting snippets (2,084 cases kept after removing 40). Test set: 35 questions generated by Mistral-7B from pairs of cases that share a legal act, 32 kept after manual review. |
| **Retrieval technique** | k-nearest-neighbour search over embeddings. Case = question, support text, entities, answer. Three comparison strategies: intra-embedding (question to question), inter-embedding (question to support or entities, using a retrieval cue prefix) and a weighted hybrid of the two. Embeddings from BERT, LegalBERT and AnglEBERT. |
| **NLP / LLM technique** | Mistral-7B generates the answer; GPT-3.5 and GPT-4 were used to build and annotate data. |
| **Methodology** | Retrieval analysis with F1@k (a case pair is relevant), then generation with k = 1 and k = 3 for support-only and full-case context, against a no-RAG baseline. |
| **Evaluation metrics** | F1 at k for retrieval; cosine similarity between the generated answer and the reference answer (using Mistral embeddings); ANOVA and one-tailed t-tests. |
| **Results (as reported)** | Best retriever: hybrid AnglEBERT with weights [0.25, 0.40, 0.35], k = 3. Cosine score: no-RAG 0.8967, best CBR-RAG 0.9141 (about 1.94% higher); significantly better than no-RAG and than hybrid LegalBERT (k = 3) at the 95% level. |
| **Advantages** | Formal link between CBR and RAG; compares embeddings and similarity types; reports significance tests; code released. |
| **Limitations (stated)** | No qualitative expert evaluation yet; no fine-tuning of the embeddings; combining several neighbours into a coherent prompt is hard. The test set is small (32 questions) and synthetic. |
| **Relevance to our project** | Shows that retrieval design matters and that a general-purpose embedding model can beat a legal-domain one. Supports measuring retrieval and generation separately, and testing significance rather than only comparing means. |
| **PAPER METHOD vs OUR IMPLEMENTATION** | Note the word *hybrid*: in CBR-RAG it means a weighted mix of **embedding** similarities (intra and inter), **not** lexical + semantic retrieval. We have no case base, no entities, no CBR cycle and no AnglEBERT; our retrieval units are statute sections. |

### Paper 3. LegalRAG: A Hybrid RAG System for Multilingual Legal Information Retrieval

| Field | Content |
|---|---|
| **Title** | LegalRAG: A Hybrid RAG System for Multilingual Legal Information Retrieval |
| **Authors / year** | M. R. Kabir et al. arXiv:2504.16121v1, April 2025 (accepted at IJCNN 2025 according to the arXiv page) |
| **Problem** | Legal and regulatory documents are unstructured and, for low-resource languages such as Bangla, hard to search. A vanilla RAG pipeline performs clearly worse on this legal data than on finance or science data. |
| **Objective** | Build a bilingual (English and Bangla) question-answering system for the Bangladesh Police Gazettes and improve the vanilla pipeline. |
| **Dataset** | 13 Bangladesh Police Gazettes (2016-2023), 81 pages, text obtained by Tesseract OCR; about 65% English and 35% Bangla text. 168 question-answer pairs generated with GPT-4o and checked by two authors; categories include factual, temporal changes, statistical, gazette search, Bangla dialect, spelling errors and **out-of-context** questions. |
| **Retrieval technique** | Vanilla: recursive character chunks, BAAI/bge-m3 embeddings, ChromaDB, MMR retrieval. Advanced: the same retriever followed by a Llama 3.2 (3B) **relevance check** of the retrieved chunks and **query refinement** (up to three iterations); 3 chunks are used. |
| **NLP / LLM technique** | Generators: Mixtral 8x7B, Llama 3.1 (8B) and Gemma 2 (9B), temperature 0.1. |
| **Methodology** | Compare vanilla and advanced pipelines on all 168 questions with each generator; ablation of temperature and instruction language. |
| **Evaluation metrics** | Human rating 1-5 by three evaluators (correctness and quality) and cosine similarity between generated and reference answers. |
| **Results (as reported)** | Human score, vanilla to advanced: Mixtral 2.77 to 3.09, Llama 3.1 3.41 to 3.70, Gemma 2 3.02 to 3.28. Cosine: 0.70 to 0.76, 0.76 to 0.82, 0.74 to 0.81. Vanilla RAG scores 0.760 on the legal data against 0.895 (finance) and 0.905 (science). Out-of-context questions score lowest (about 0.41 to 0.58) for both pipelines. |
| **Advantages** | Real low-resource legal documents; human evaluation; includes out-of-context and noisy questions; the extra LLM step is small (3B). |
| **Limitations (stated)** | LLMs have little exposure to complex legal text; **both** pipelines still answer out-of-context questions with confidence and get them wrong; fine-tuning left as future work. Small test set. |
| **Relevance to our project** | Confirms that legal RAG needs care with questions the documents cannot answer, which is why LegalRAG includes out-of-scope questions and a refusal message. The relevance-check step is the idea behind the optional "Improvement 3" of our roadmap. |
| **PAPER METHOD vs OUR IMPLEMENTATION** | Again, *hybrid* means a **pipeline** hybrid (retrieval plus an LLM relevance filter and query refiner). It has **no BM25, no RRF and no cross-encoder reranker.** We do not implement the relevance check or query refinement; we do not handle Bangla or OCR; we use English statutes only. |

### Paper 4 (additional): CanLegalRAGBench

Chosen because it is the closest published study to our experiment: it compares BM25, dense, hybrid (RRF) and reranked retrieval for legal RAG.

| Field | Content (from the team's report draft; not re-verified here) |
|---|---|
| **Title / source** | E. Zhao, M. Taranukhin, W. Cui, M. Aikenhead, V. Shwartz, "CanLegalRAGBench: Evaluating Retrieval-Augmented Generation on Canadian Case Law", arXiv:2605.30497, 2026 (preprint) |
| **Problem** | Legal RAG assistants still hallucinate; existing benchmarks use artificial queries; Canadian law is under-represented. |
| **Objective** | A benchmark of realistic queries with expert-annotated answers, and an evaluation of retrieval and generation. |
| **Dataset** | 532 queries, 3,193 query-document gold pairs over 588 case-law documents. |
| **Retrieval technique** | BM25L, dense retrieval with open and closed embedders, hybrid (RRF or reranker), a proprietary Kanon-2 reranker. |
| **Evaluation metrics** | Recall@10 and @25, MRR, nDCG; FActScore-style claim-level groundedness and accuracy. |
| **Results (as reported)** | Average Recall@10: BM25 0.133, dense 0.366, hybrid 0.308, reranking 0.390. With the same embedder and chunk size: dense 0.359, hybrid (RRF) 0.299, dense + reranker 0.363, hybrid + reranker 0.364. Groundedness 0.71 to 0.80. |
| **Limitations** | English case law only; one annotator per query; gold sets miss relevant documents; hybrid tested with one embedder; proprietary reranker. |
| **Relevance** | Hybrid with RRF was *worse* than dense alone there, and the reranker brought it back. This is why we do not assume Method B or C is better. |
| **PAPER METHOD vs OUR IMPLEMENTATION** | They use case law, a proprietary reranker and document-level metrics. We use statutes, an open reranker (ms-marco-MiniLM-L-6-v2) and section-level metrics. |

---

## Part B. Other papers in the CIPAT report draft

These are summarised from the report draft. The full entries (problem, objectives, datasets, limitations) are in the report chapter 2.

| Paper | Main idea | Key reported result | Gap relative to our work |
|---|---|---|---|
| Wagle et al., 2026, "Retrieval Augmented Generation Framework for the Nepali Legal Domain Question Answering" (arXiv:2606.07523) | First RAG baseline for Nepali Supreme Court case law; BM25 vs dense (e5, LaBSE) | Precision@1: BM25 on chunks 0.91, e5-large 0.75 | BM25 beat dense; hybrid and reranking were not tested |
| Keisha et al., 2025, "All for law and law for all: Adaptive RAG Pipeline for Legal Research" (arXiv:2508.13107) | Open-source embeddings, cosine vs BM25, reranking on LegalBench-RAG-mini | Cosine beat BM25; reranking inconclusive | BM25 and dense were compared, not fused |
| Kulkarni et al., 2026, "Long-Context Long-Form Question Answering for Legal Domain" (arXiv:2602.07190) | Layout-aware chunking, query rewriting, hybrid BM25 + dense with RRF | Claim recall 0.5369 (RAG) to 0.6798 (LCLF-QA) | The effect of hybrid retrieval alone was not isolated |

## What the literature says and what we test

- The best retriever depends on the corpus and the setup: dense beat BM25 in some studies and lost clearly in another.
- Hybrid retrieval is not automatically better (CanLegalRAGBench), and reranking depends on the configuration (HyPA-RAG, Keisha et al.).
- Questions that the corpus cannot answer are a known weak point (LegalRAG, multilingual).

LegalRAG tests these questions on two Indian statutes with fully open retrieval components and identical inputs for all methods.
