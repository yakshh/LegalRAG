# Viva questions and answers

Simple answers for the GTU viva. Numbers about our own results are in `results/summary.json`; quote them from there, not from memory.

## A. Basics: NLP, LLM, RAG

**1. What is NLP?**
Natural Language Processing is the field of making computers read, understand and produce human language. Question answering is one NLP task.

**2. What is an LLM?**
A Large Language Model is a neural network (a transformer) trained on huge amounts of text that can write fluent answers. It stores knowledge in its weights, so that knowledge can be old or wrong.

**3. What is hallucination?**
When an LLM writes something that sounds right but is false or not supported, for example an invented section number. In law this can cause real harm.

**4. What is RAG?**
Retrieval-Augmented Generation. First we *retrieve* the most relevant passages from our documents, then we give them to the LLM and tell it to answer only from them. This grounds the answer in real text.

**5. Why use RAG instead of only asking the LLM?**
It reduces hallucination, lets us use our own documents (the two Acts), allows citations (page and section), and needs no training.

**6. Why the legal domain?**
Law is precise (the word "shall" or "may" changes the meaning), the text is long and structured into sections, and a wrong answer is costly. So retrieval quality and citations matter a lot.

**7. What is our project in one sentence?**
A controlled experiment that compares dense retrieval, hybrid retrieval (BM25 + dense with RRF) and hybrid retrieval with a cross-encoder reranker for legal question answering on two Indian Acts.

## B. Text processing and embeddings

**8. What is chunking? Why do we chunk?**
Splitting a document into small pieces. The retriever returns pieces, not whole Acts, and the LLM has a limited context. Our chunks follow section boundaries and are about 1000 characters with 150 characters of overlap so a sentence at a border is not lost.

**9. Why section-aware chunking?**
So every chunk knows its document, page and section. Then the answer can cite "Section 43, page 15" and we can check retrieval at section level.

**10. What is an embedding?**
A list of numbers (a vector) that represents the meaning of a text. Similar meanings give similar vectors. We use `all-MiniLM-L6-v2` (384 numbers per chunk).

**11. What is cosine similarity?**
A measure of how close two vectors point in the same direction, from -1 to 1. With normalised vectors it equals the inner product, which FAISS computes quickly.

**12. What is a vector database / FAISS?**
A structure that stores embeddings and finds the nearest ones fast. FAISS is a Facebook/Meta library for this. We use the exact index `IndexFlatIP`.

## C. Retrieval methods

**13. What is dense retrieval? Its strength and weakness?**
Embed the question and return the chunks with the nearest vectors. It understands paraphrases ("how long to bring a case" matches "limitation period") but can miss exact legal terms and numbers.

**14. What is sparse retrieval and BM25?**
Sparse retrieval matches words. BM25 scores a chunk by how often the query words appear in it, how rare the words are, and the chunk length. It is strong on exact terms such as "Certifying Authority" but misses paraphrases.

**15. What is hybrid retrieval and why use it?**
Run dense and BM25 together and merge the lists, to get the benefits of both meaning and exact words.

**16. What is Reciprocal Rank Fusion (RRF)?**
A way to merge ranked lists: each chunk gets the sum of 1/(k + rank) over the lists (k = 60). A chunk that is high in both lists wins. It needs no score scaling, which is useful because BM25 scores and cosine scores are on different scales.

**17. What is a cross-encoder? How is it different from the embedding model?**
The embedding model (bi-encoder) encodes the question and the chunk separately. A cross-encoder reads the question and the chunk *together* and outputs one relevance score, so it is more accurate but much slower. It cannot pre-compute anything, so we only use it on a short candidate list.

**18. What is reranking and why use it?**
Re-ordering the candidates with a better (but slower) model so that the best chunks come first, because the LLM sees only the top K chunks.

**19. Is hybrid or reranking always better?**
No. Published studies disagree, and one found hybrid worse than dense. That is why we did not assume it and tested it. (Our own outcome is in `results/summary.json`.)

**20. What is the ablation study in your project?**
We compare BM25 only, dense only, dense + BM25 without RRF, hybrid with RRF and hybrid + reranker, to see what each component adds.

## D. Generation and grounding

**21. How do you make the LLM stay grounded?**
The prompt says: use only the context, do not invent sections or penalties, cite document/page/section, and say "Insufficient evidence..." if the context is not enough. We also check the citations automatically.

**22. What happens for a question that the Acts cannot answer?**
The system should answer with the fixed insufficient-evidence message, and the status shows INSUFFICIENT EVIDENCE. We test this with 6 out-of-scope questions, for example "Who is the current Prime Minister of India?".

**23. Why temperature 0?**
For repeatable, factual answers. Higher temperature gives more random text, which is bad for law.

## E. Evaluation

**24. What is Precision@K?**
Of the K chunks returned, the fraction that are relevant.

**25. What is Recall@K?**
Of the relevant sections, the fraction found in the top K. We count sections, not chunks, because one section is split into several chunks.

**26. What is Hit@K?**
1 if at least one relevant chunk is in the top K, otherwise 0, averaged over questions.

**27. What is MRR?**
Mean Reciprocal Rank: for each question take 1 divided by the rank of the first relevant chunk (1, 1/2, 1/3, ...), then average. It rewards putting the right chunk first.

**28. What is nDCG?**
A rank-weighted score: relevant chunks near the top count more. 1.0 means a perfect order.

**29. Why do you not use accuracy or F1?**
Retrieval returns a ranked list, not a class label, so ranking metrics are the right ones. For free-form generated answers there is no single correct label either, so we score them with a rubric.

**30. How do you evaluate answer quality?**
An LLM judge (a different Gemini model) scores each answer from 0 to 3 for correctness, faithfulness, relevance and citation correctness with a fixed prompt, the same for all methods. We also check citations automatically (are they valid, right and complete?) and measure correct refusals. We never claim this is human evaluation.

**31. What is faithfulness?**
Whether every claim in the answer is supported by the retrieved text.

**32. What are the weaknesses of an LLM judge?**
It can be biased (for example prefer its own style), inconsistent, or wrong. We use a different model from the generator, a fixed rubric, and we state the limitation. A manual scoring sheet is provided for human checks.

**33. What is latency and why measure it?**
The time taken. Reranking improves ranking but costs time; we measure retrieval time per question to see the trade-off.

**34. Can you say that Method C is significantly better?**
Only if the data and a proper test support it. Our permutation tests are exploratory (44 questions, many metrics), so we report differences with caution.

## F. Research and design

**35. What is the research gap?**
Studies disagree on dense vs hybrid vs reranking for legal text, no study compares all three with open components under identical conditions, none uses Indian statutes, and citation quality and out-of-scope questions are rarely tested.

**36. Is your system a new algorithm?**
No. The contribution is a controlled comparison, the Indian-statute implementation, the evaluation of retrieval, answers, citations and refusals, and an honest trade-off analysis.

**37. Did you train any model?**
No. The embedding model, cross-encoder and LLM are all pretrained and used as they are.

**38. Which papers did you study?**
HyPA-RAG, CBR-RAG, LegalRAG (multilingual) and CanLegalRAGBench, plus the papers in the report (Wagle, Keisha, Kulkarni). We note that "hybrid" means different things in different papers.

**39. What are the limitations?**
Only two Acts; 50 hand-written questions; small general-purpose models; an LLM judge; the IT Act is the 2000 version; English only; free-tier API limits. See `research/limitations.md`.

**40. What is future work?**
Add more Acts and case law, a relevance check and query refinement (as in the multilingual LegalRAG paper), tune fusion weights, legal-domain or larger models, Hindi and Gujarati, and human evaluation by legal experts.

**41. Is this legal advice?**
No. It is for academic and research use only; the interface shows a disclaimer.

**42. How did you make the comparison fair?**
Same documents, chunks, questions, LLM, prompt, metrics and judge for all three methods. Only the retrieval step changes.

**43. How do you avoid data leakage?**
The questions are never used to build the index and nothing is trained or tuned on them; settings were fixed before the final run.

**44. What would you do if hybrid were worse than dense?**
Report it. A null or negative result is a valid outcome, and published studies have found it too.
