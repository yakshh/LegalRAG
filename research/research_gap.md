# Research gap

## The gap

Existing legal RAG approaches demonstrate the value of retrieval augmentation, hybrid retrieval, domain-specific retrieval and reranking, but there is a need for a **controlled comparative evaluation of dense retrieval against hybrid lexical-semantic retrieval and hybrid retrieval with cross-encoder reranking under the same corpus, questions, generation model and evaluation protocol.**

## Evidence from the literature

| Observation | Source |
|---|---|
| Studies disagree on the best retriever. Dense beat BM25 in some, BM25 beat dense in another. | CanLegalRAGBench; Keisha et al.; Wagle et al. (see `paper_analysis.md`) |
| Hybrid retrieval is not automatically better. In one study hybrid (RRF) scored below dense alone and the reranker recovered the difference. | CanLegalRAGBench (as reported in the report draft) |
| Reranking helps in some configurations and slightly lowers correctness in others. | HyPA-RAG (ablation), Keisha et al. |
| Two well-known "legal RAG" papers use *hybrid* to mean something else (weighted embedding similarities; an LLM relevance filter), so they do not answer the BM25 + dense question. | CBR-RAG, LegalRAG (multilingual) |
| Questions that the documents cannot answer lead to confident wrong answers. | LegalRAG (multilingual) |
| Citation quality (document, page, section) is rarely evaluated. | none of the reviewed papers evaluates it |
| Several studies depend on proprietary models or APIs. | CanLegalRAGBench (reranker), HyPA-RAG (GPT models) |
| No reviewed study uses Indian statutory text. | all |

## What this project contributes

The contribution is **not a new algorithm**. LegalRAG does not claim to be state of the art and does not reproduce any paper.

1. **A controlled comparison** of Dense RAG, Hybrid RAG (BM25 + dense with RRF) and Hybrid RAG + cross-encoder reranking. Same Acts, same chunks, same questions, same LLM, same prompt, same metrics. Only the retrieval step changes.
2. **A legal-domain implementation on Indian statutes**: the Information Technology Act, 2000 and the Consumer Protection Act, 2019, chunked by section with document, page and section kept.
3. **Retrieval evaluation**: Precision@K, Recall@K (section level), Hit@K, MRR, nDCG@5 and latency, and a small ablation (BM25 only, dense + BM25 without RRF).
4. **Answer-quality evaluation**: an LLM judge with a fixed rubric (correctness, faithfulness, relevance, citation), clearly labelled as LLM-based, plus automatic checks.
5. **Citation and grounding evaluation**: does the answer cite only retrieved sections, and the right ones?
6. **Out-of-scope behaviour**: six questions the corpus cannot answer, to test refusal.
7. **An analysis of the trade-offs** between retrieval quality, answer quality and latency, with the numbers from the real runs (`results/summary.json`).

## Research question

How does hybrid lexical-semantic retrieval with reranking affect retrieval quality and legal question-answering performance compared with a dense vector-based RAG system?

## Hypothesis and how it is tested

Hybrid retrieval with reranking gives *different* retrieval quality and answer groundedness than dense-only retrieval on Indian statutory text with open-source components. The direction is not assumed. A null or negative result is acceptable and is reported as it is.

The retrieval comparison uses per-question paired differences with an exploratory permutation test (`results/summary.json`, key `retrieval.paired_tests`). The answer comparison is descriptive: with a small question set and an LLM judge it cannot support strong significance claims.
