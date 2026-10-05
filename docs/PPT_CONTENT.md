# PPT content (15 slides)

What to put on each slide. Numbers come from `results/summary.json`. Use the charts in `results/figures/` and the diagram from `research/methodology.md`. Keep text short; speak the details.

---

### Slide 1. Title
- LegalRAG: Comparative Evaluation of Dense Retrieval and Hybrid Retrieval with Reranking for Legal Question Answering
- NLP (3170723), B.E. VII Computer Engineering, C. K. Pithawalla College of Engineering and Technology
- Team: Nisarg Patel, Prince Mistry, Yaksh Vaidya, Mahek Batavia. Guide: Prof. Pooja Pariyani
- Footer: "Academic/research use only, not legal advice"

### Slide 2. Problem
- LLMs can invent legal sections and citations
- RAG fixes this only if the right provision is retrieved
- Studies disagree: dense vs BM25 vs hybrid vs reranking
- Show: one example of a wrong/hallucinated legal answer (**[INSERT]** your own example)

### Slide 3. Motivation
- People need to find what an Act actually says
- Trustworthy = right section + citation + "I don't know" when unsure
- Choose the retrieval method by evidence, not assumption

### Slide 4. Objectives
- Review recent legal RAG papers
- Compare 3 retrieval methods under identical conditions
- Grounded answers with page and section citations
- Measure retrieval, answers, citations, refusals and latency
- Report honestly, even if hybrid or reranking does not help

### Slide 5. Literature review
- HyPA-RAG (hybrid BM25 + dense + RRF, KG, adaptive parameters)
- CBR-RAG ("hybrid" = weighted embedding similarities)
- LegalRAG multilingual (relevance check + query refinement)
- CanLegalRAGBench (hybrid was below dense; reranker recovered)
- Table from `research/comparison.md` (short version). Say: we do not reproduce them.

### Slide 6. Research gap
- No study compares dense, hybrid and hybrid + cross-encoder under identical, open conditions
- No Indian statutes; citations and out-of-scope questions rarely measured
- Our contribution: a controlled comparison and evaluation, not a new algorithm
- Research question (one line)

### Slide 7. Proposed architecture
- Diagram: PDFs -> chunks -> embeddings/FAISS + BM25 -> three paths -> same LLM -> answer + citations
- Mark where methods A, B, C differ (only retrieval)

### Slide 8. Method A: Dense RAG
- MiniLM embeddings, FAISS cosine search, top K = 5
- Strength: paraphrase. Weakness: exact legal terms
- Example from the demo (**[INSERT]** screenshot)

### Slide 9. Method B: Hybrid RAG
- Dense + BM25, Reciprocal Rank Fusion: score = sum 1 / (60 + rank)
- Small worked example of RRF with two lists
- Strength: both meaning and exact words

### Slide 10. Method C: Hybrid + Reranking
- Cross-encoder reads (question, chunk) together, re-scores 20 candidates, keep top 5
- More accurate, slower (about 0.6 s)
- Bi-encoder vs cross-encoder picture

### Slide 11. Dataset
- IT Act 2000 + Consumer Protection Act 2019: 73 pages, 353 section-aware chunks
- 50 questions: 44 answerable, 6 out-of-scope; 10 categories; 3 difficulty levels
- Written from the Acts, validated by script; chunk keeps document, page, section

### Slide 12. Experimental setup
- Same chunks, questions, LLM (gemini-3.5-flash-lite), prompt, judge (gemini-3.1-flash-lite)
- Settings: K = 5, N = M = 20, RRF k = 60, BM25 k1 = 1.5, b = 0.75
- Metrics: Precision@K, Recall@K, Hit@K, MRR, nDCG, latency; judge 0-3; citation checks; refusals

### Slide 13. Results
- Table: Dense / Hybrid / Hybrid + Reranking
  - MRR 0.648 / 0.679 / 0.781
  - Recall@5 0.705 / 0.830 / 0.917
  - Hit@5 0.750 / 0.864 / 0.932
  - Latency 13 ms / 14 ms / 612 ms
  - Answer correctness (0-3) 1.98 / 2.39 / 2.61
  - False refusals 31.8% / 20.5% / 13.6%
  - Out-of-scope refused 6/6 for all; invalid citations: none
- Charts: `mrr.png`, `recall_at_5.png`, `answer_correctness.png`, `retrieval_latency.png`
- One bullet on honesty: Hybrid vs Dense is not clearly separable (p = 0.07 to 0.46); reranking vs dense is (exploratory p = 0.002 to 0.036)
- Ablation chart: `ablation_mrr.png`

### Slide 14. Limitations and future work
- 2 Acts, 50 questions, LLM judge, small general models, 2000 version of the IT Act
- Error analysis: definitions inside very long sections and multi-section questions cause most failures
- Future: relevance check + query refinement, definition-level chunking, more Acts, human evaluation, Hindi/Gujarati

### Slide 15. Conclusion
- Reranked hybrid retrieval was best here: better retrieval and answers, about 0.6 s extra per question
- Hybrid alone helped a little; BM25 alone beat dense alone on this statute text
- Out-of-scope questions refused; no invented citations
- Results hold for this corpus; confirm with a larger, expert-reviewed set
- Live demo: `streamlit run app.py` (one answerable and one out-of-scope question)
