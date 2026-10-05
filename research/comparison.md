# Feature-wise comparison

Papers: **P1** HyPA-RAG (Kalra et al., 2024/25), **P2** CBR-RAG (Wiratunga et al., 2024), **P3** LegalRAG for multilingual legal retrieval (Kabir et al., 2025), **P4** CanLegalRAGBench (Zhao et al., 2026) and **Ours** = this project.

A feature is marked **Yes** only if the paper describes it. **No** means the paper does not use it. "Not described" means the reviewed text does not say. P1 to P3 were checked in their full text; P4 comes from the team's report draft (see `research/paper_analysis.md`).

| Feature | P1 HyPA-RAG | P2 CBR-RAG | P3 LegalRAG (multilingual) | P4 CanLegalRAGBench | **Ours (LegalRAG)** |
|---|---|---|---|---|---|
| Dense retrieval | Yes (vector search) | Yes (embedding kNN) | Yes (bge-m3 + MMR) | Yes | **Yes** (MiniLM + FAISS) |
| Sparse retrieval | Yes | No | No | Yes | **Yes** |
| BM25 | Yes | No | No | Yes (BM25L) | **Yes** (BM25Okapi) |
| Hybrid retrieval | Yes (BM25 + dense) | Different meaning: weighted mix of embedding similarities | Different meaning: retrieval + LLM relevance filter (a pipeline hybrid) | Yes (BM25 + dense, with Qwen only) | **Yes** (BM25 + dense) |
| RRF | Yes | No | No | Yes | **Yes** (k = 60) |
| Query refinement | Yes (query rewriter, adaptive number of rewrites) | No (a retrieval cue prefix is added to the query embedding) | Yes (LLM refines the query, up to 3 times) | No | **No** (future work) |
| Reranking | Yes (optional; bge-reranker-large in the ablation) | No | No (an LLM relevance filter, not a ranking step) | Yes (proprietary Kanon-2) | **Yes** |
| Cross-encoder | Reranker model used; its architecture is not stated in the reviewed text | No | No | Yes (per the report draft) | **Yes** (ms-marco-MiniLM-L-6-v2) |
| Knowledge graph | Yes (GPT-4o triplets) | No | No | No | **No** |
| Domain adaptation | Partial: a classifier trained on synthetic legal queries; embeddings not fine-tuned | Compares LegalBERT with general encoders; no fine-tuning | No fine-tuning | A legal-specific closed embedder was compared | **No** (general pretrained models, nothing fine-tuned) |
| Multilingual support | No | No | **Yes** (English and Bangla) | No | **No** (English only) |
| Legal citations | Not described | Not evaluated | Page numbers appear in sample answers; citation quality not evaluated | Case-citation strings; no section or page | **Yes**: document, page and section in every context and answer; checked automatically |
| Grounded generation | Yes (RAG context; faithfulness measured) | Yes (case context in the prompt) | Yes | Yes | **Yes** (answer only from context, fixed fallback message) |
| Retrieval evaluation | Yes (RAGAS context precision and recall) | Yes (F1@k) | Not reported separately | Yes (Recall@K, MRR, nDCG) | **Yes** (Precision@K, Recall@K, Hit@K, MRR, nDCG@5, latency) |
| Answer evaluation | Yes (faithfulness, relevancy, correctness) | Yes (cosine similarity to reference) | Yes (human 1-5 score and cosine similarity) | Yes (claim-level groundedness and accuracy) | **Yes** (LLM judge 0-3: correctness, faithfulness, relevance, citation; plus deterministic citation checks) |
| Hallucination handling | Measured through faithfulness; refusal not described | Not addressed specifically | Out-of-context questions tested; the paper reports confident wrong answers | Measured through groundedness | **Yes**: strict prompt, SUPPORTED / INSUFFICIENT EVIDENCE status, 6 out-of-scope questions |
| Statistical testing | Not described | Yes (ANOVA, t-tests) | Reports mean and standard deviation | Not described | **Partial**: paired permutation test, exploratory only |
| Open-source components only | No (GPT models) | Partly (Mistral-7B; GPT used for data) | Yes for the pipeline (GPT-4o used to create questions) | No (proprietary reranker, closed models) | **Retrieval: yes. Generation and judging: Gemini API** |

## Reading the table

- No paper above compares **dense, hybrid (BM25 + dense with RRF) and hybrid + cross-encoder reranking under identical conditions with open retrieval components**. P1 includes a reranker ablation, and P4 is the closest, but P4 uses one embedder for hybrid and a proprietary reranker.
- P2 and P3 use the word "hybrid" for something else. Do not cite them as evidence for or against BM25 + dense fusion.
- Citation quality and out-of-scope handling are rarely measured. LegalRAG measures both.
- Features in the "Ours" column marked **No** (query refinement, knowledge graph, domain adaptation, multilingual) are deliberately out of scope; see `research/limitations.md`.
